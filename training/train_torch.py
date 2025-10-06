"""
PyTorch-based Training Script for Safe Multi-Agent Poker with GNE convergence
macOS compatible without threading issues

Usage:
    python train_torch.py --n_players 3 --episodes 10000 --save_dir checkpoints/
"""

import os
import sys

# Disable threading BEFORE importing PyTorch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import argparse
import logging
from pathlib import Path
import numpy as np
import torch

# Disable PyTorch threading
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from env import SafePokerEnv, BankrollConstraints
from algorithms.madac_poker_torch import MADACPokerTorch
from evaluation.gne_metrics import GNEEvaluator


def setup_logging(log_dir: str):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_torch_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('SafePokerTrainingTorch')


def parse_args():
    parser = argparse.ArgumentParser(description='Train Safe Multi-Agent Poker with GNE (PyTorch)')

    # Environment
    parser.add_argument('--n_players', type=int, default=3, help='Number of players')
    parser.add_argument('--starting_bankroll', type=float, default=1000.0)
    parser.add_argument('--small_blind', type=float, default=5.0)
    parser.add_argument('--big_blind', type=float, default=10.0)
    parser.add_argument('--game_type', type=str, default='limit', choices=['limit', 'no-limit'])

    # Safety constraints
    parser.add_argument('--min_bankroll', type=float, default=100.0)
    parser.add_argument('--max_variance', type=float, default=500.0)
    parser.add_argument('--max_drawdown', type=float, default=0.5)

    # Training
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--eval_interval', type=int, default=1000, help='Evaluation interval (increased for speed)')
    parser.add_argument('--save_interval', type=int, default=2000, help='Checkpoint save interval')
    parser.add_argument('--batch_size', type=int, default=512, help='Larger batch for efficiency')
    parser.add_argument('--buffer_size', type=int, default=100000)
    parser.add_argument('--update_epochs', type=int, default=5, help='Reduced epochs for speed')

    # Algorithm hyperparameters
    parser.add_argument('--lr_actor', type=float, default=3e-4)
    parser.add_argument('--lr_critic', type=float, default=1e-3)
    parser.add_argument('--lr_lambda', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--lambda_init', type=float, default=1.0)
    parser.add_argument('--safety_threshold', type=float, default=0.0)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--value_loss_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)

    # Paths
    parser.add_argument('--save_dir', type=str, default='checkpoints_torch/')
    parser.add_argument('--log_dir', type=str, default='logs_torch/')
    parser.add_argument('--load_checkpoint', type=str, default=None)

    # PyTorch specific
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='mps' if torch.backends.mps.is_available() else 'cpu',
                       choices=['cpu', 'cuda', 'mps'], help='Auto-detect MPS on Apple Silicon')

    return parser.parse_args()


def collect_trajectories(env, trainer, n_steps: int = 512):
    """
    Collect trajectories from environment (optimized with larger buffer)

    Returns:
        Dictionary with states, actions, rewards, dones, values, log_probs
    """
    states_list = []
    actions_list = []
    rewards_list = []
    dones_list = []
    values_list = []
    log_probs_list = []
    safety_costs_list = []

    obs = env.reset()
    done = {i: False for i in range(env.n_players)}

    for step in range(n_steps):
        # Convert observations to array
        obs_array = np.array([obs[i] for i in range(env.n_players)])

        # Select actions
        actions_array, infos = trainer.select_actions(obs_array, deterministic=False)

        # Convert to dict for env
        actions = {i: actions_array[i] for i in range(env.n_players)}

        # Step environment
        next_obs, rewards, dones, env_infos = env.step(actions)

        # Store transitions
        for i in range(env.n_players):
            if not done[i]:
                states_list.append(obs[i])
                actions_list.append(actions[i])
                rewards_list.append(rewards[i])
                dones_list.append(dones[i])
                values_list.append(infos[i]['task_value'])
                log_probs_list.append(0.0)  # Will be recomputed
                safety_costs_list.append(env_infos[i]['risk_metrics']['constraint_value'])

        obs = next_obs
        done = dones

        if all(done.values()):
            obs = env.reset()
            done = {i: False for i in range(env.n_players)}

    return {
        'states': np.array(states_list),
        'actions': np.array(actions_list),
        'rewards': np.array(rewards_list),
        'dones': np.array(dones_list),
        'values': np.array(values_list),
        'safety_costs': np.array(safety_costs_list)
    }


def train(args):
    """Main training loop with PyTorch"""

    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting Safe Multi-Agent Poker training (PyTorch) with args: {args}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Device: {args.device}")

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    constraints = BankrollConstraints(
        min_bankroll=args.min_bankroll,
        max_variance=args.max_variance,
        max_drawdown=args.max_drawdown
    )

    env = SafePokerEnv(
        n_players=args.n_players,
        starting_bankroll=args.starting_bankroll,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        constraints=constraints,
        game_type=args.game_type
    )

    logger.info(f"Environment created: {args.n_players} players, {args.game_type} poker")
    logger.info(f"State dim: {env.observation_space.shape[0]}")

    # Create MADAC trainer (PyTorch version)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n if args.game_type == 'limit' else 1

    trainer = MADACPokerTorch(
        state_dim=state_dim,
        action_dim=action_dim,
        n_agents=args.n_players,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_lambda=args.lr_lambda,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        continuous_actions=(args.game_type == 'no-limit'),
        safety_threshold=args.safety_threshold,
        lambda_init=args.lambda_init,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device
    )

    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load(args.load_checkpoint)
        logger.info(f"Loaded checkpoint from {args.load_checkpoint}")

    # Training history
    history = {
        'rewards': [],
        'safety_values': [],
        'lambdas': [],
        'bankrolls': [],
        'violations': [],
        'actor_loss': [],
        'critic_loss': []
    }

    # Training loop
    logger.info("Starting training...")
    os.makedirs(args.save_dir, exist_ok=True)

    pbar = tqdm(range(args.episodes), desc="Training (PyTorch)")

    for episode in pbar:
        # Collect trajectories (increased to 512 steps)
        trajectories = collect_trajectories(env, trainer, n_steps=512)

        # Train on collected data
        if len(trajectories['states']) > args.batch_size:
            metrics = trainer.train_step(trajectories, args.update_epochs)

            if metrics and episode % 50 == 0:  # Update display less frequently
                pbar.set_postfix({
                    'actor_loss': f"{metrics['actor_loss']:.4f}",
                    'critic_loss': f"{metrics['critic_loss']:.4f}",
                    'avg_reward': f"{np.mean(trajectories['rewards']):.2f}",
                    'lambda': f"{trainer.get_avg_lambda():.4f}"
                })

            # Record history
            history['actor_loss'].append(metrics['actor_loss'])
            history['critic_loss'].append(metrics['critic_loss'])

        history['rewards'].append(np.mean(trajectories['rewards']))
        history['safety_values'].append(np.mean(trajectories['safety_costs']))
        history['lambdas'].append(trainer.get_avg_lambda())
        history['bankrolls'].append(np.mean([env.game_state.players[i].bankroll for i in range(args.n_players)]))

        # Evaluation (reduced frequency and episodes)
        if episode % args.eval_interval == 0 and episode > 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Episode {episode} Evaluation")
            logger.info(f"{'='*60}")

            # Evaluate performance (reduced to 50 episodes for speed)
            eval_rewards = []
            eval_safety = []
            for _ in range(50):
                obs = env.reset()
                done = {i: False for i in range(args.n_players)}
                episode_reward = 0

                while not all(done.values()):
                    obs_array = np.array([obs[i] for i in range(args.n_players)])
                    actions_array, infos = trainer.select_actions(obs_array, deterministic=True)
                    actions = {i: actions_array[i] for i in range(args.n_players)}
                    obs, rewards, done, _ = env.step(actions)
                    episode_reward += sum(rewards.values())

                eval_rewards.append(episode_reward)

            logger.info(f"Avg reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
            logger.info(f"Avg lambda: {trainer.get_avg_lambda():.4f}")
            logger.info(f"Avg bankroll: {history['bankrolls'][-1]:.2f}")

            # Check convergence
            if len(history['rewards']) >= 100:
                recent_rewards = history['rewards'][-100:]
                reward_variance = np.var(recent_rewards)
                logger.info(f"Reward variance (last 100): {reward_variance:.4f}")

                if reward_variance < 1.0:
                    logger.info("Training appears to have converged!")

        # Save checkpoint
        if episode % args.save_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_torch_ep{episode}')
            trainer.save(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Final evaluation
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80)

    eval_rewards = []
    for _ in range(1000):
        obs = env.reset()
        done = {i: False for i in range(args.n_players)}
        episode_reward = 0

        while not all(done.values()):
            obs_array = np.array([obs[i] for i in range(args.n_players)])
            actions_array, _ = trainer.select_actions(obs_array, deterministic=True)
            actions = {i: actions_array[i] for i in range(args.n_players)}
            obs, rewards, done, _ = env.step(actions)
            episode_reward += sum(rewards.values())

        eval_rewards.append(episode_reward)

    logger.info(f"Final avg reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
    logger.info(f"Final lambda: {trainer.get_avg_lambda():.4f}")

    # Save final model
    final_path = os.path.join(args.save_dir, 'final_model_torch')
    trainer.save(final_path)
    logger.info(f"Final model saved: {final_path}")

    logger.info("Training complete!")


if __name__ == '__main__':
    args = parse_args()
    train(args)
