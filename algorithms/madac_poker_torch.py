"""
PyTorch-based MADAC (Multi-Agent Dual Actor-Critic) for Poker
Optimized for macOS compatibility without threading issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import pickle
import os
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.actor_critic_torch import (
    TaskActorCritic, SafetyActorCritic, NetworkConfig,
    sample_action_discrete, sample_action_continuous,
    compute_gae, compute_entropy_discrete, compute_entropy_continuous
)


class MADACPokerTorch:
    """
    Multi-Agent Dual Actor-Critic for Poker (PyTorch version)

    Implements the MADAC algorithm with:
    - Task policy: Maximize expected reward
    - Safety policy: Satisfy bankroll constraints
    - Lagrange multipliers: Balance task and safety objectives
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        lr_lambda: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        continuous_actions: bool = False,
        safety_threshold: float = 0.0,
        lambda_init: float = 1.0,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        hidden_dims: Tuple[int, ...] = (256, 256, 128),
        device: str = 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.continuous_actions = continuous_actions
        self.safety_threshold = safety_threshold
        self.lr_lambda = lr_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        # Create network config
        config = NetworkConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            continuous_actions=continuous_actions
        )

        # Initialize agents
        self.agents = []
        for i in range(n_agents):
            agent = self._create_agent(config, lr_actor, lr_critic, lambda_init)
            self.agents.append(agent)

    def _create_agent(self, config: NetworkConfig, lr_actor: float, lr_critic: float, lambda_init: float):
        """Create a single agent with task and safety networks"""

        # Task actor-critic
        task_ac = TaskActorCritic(config).to(self.device)
        task_optimizer = optim.Adam(task_ac.parameters(), lr=lr_actor)

        # Safety actor-critic
        safety_ac = SafetyActorCritic(config).to(self.device)
        safety_optimizer = optim.Adam(safety_ac.parameters(), lr=lr_critic)

        # Lagrange multiplier
        log_lambda = torch.tensor(np.log(lambda_init), dtype=torch.float32, device=self.device)

        return {
            'task_ac': task_ac,
            'task_optimizer': task_optimizer,
            'safety_ac': safety_ac,
            'safety_optimizer': safety_optimizer,
            'log_lambda': log_lambda
        }

    def select_actions(self, states: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, List[dict]]:
        """
        Select actions for all agents

        Args:
            states: States for all agents [n_agents, state_dim]
            deterministic: If True, use deterministic policy

        Returns:
            actions: Actions for all agents [n_agents]
            infos: List of info dicts for each agent
        """
        actions = []
        infos = []

        with torch.no_grad():
            for i, agent in enumerate(self.agents):
                state = torch.FloatTensor(states[i:i+1]).to(self.device)

                # Get task policy
                task_dist, task_value = agent['task_ac'](state)

                # Get safety value
                _, safety_value = agent['safety_ac'](state)

                # Sample action
                if self.continuous_actions:
                    mean, log_std = task_dist
                    action, log_prob = sample_action_continuous(mean, log_std, deterministic)
                else:
                    logits = task_dist
                    action, log_prob = sample_action_discrete(logits, deterministic)

                lambda_ = torch.exp(agent['log_lambda'])
                alpha = torch.sigmoid(-safety_value * lambda_)

                actions.append(int(action[0]) if not self.continuous_actions else float(action[0].item()))
                infos.append({
                    'task_value': float(task_value[0].item()),
                    'safety_value': float(safety_value[0].item()),
                    'lambda': float(lambda_.item()),
                    'alpha': float(alpha[0].item())
                })

        return np.array(actions), infos

    def train_step(self, trajectories: Dict[str, torch.Tensor], epochs: int = 10) -> Dict[str, float]:
        """
        Train on collected trajectories

        Args:
            trajectories: Dictionary with states, actions, rewards, dones, values, safety_costs
            epochs: Number of training epochs

        Returns:
            metrics: Training metrics
        """
        states = torch.FloatTensor(trajectories['states']).to(self.device)
        actions = torch.LongTensor(trajectories['actions']).to(self.device) if not self.continuous_actions else torch.FloatTensor(trajectories['actions']).to(self.device)
        rewards = torch.FloatTensor(trajectories['rewards']).to(self.device)
        dones = torch.FloatTensor(trajectories['dones']).to(self.device)
        safety_costs = torch.FloatTensor(trajectories['safety_costs']).to(self.device)

        n_samples = len(states)
        agent_metrics = []

        # Train each agent
        for agent_idx, agent in enumerate(self.agents):
            # Get agent-specific data (simple approach: all agents see all data)
            with torch.no_grad():
                _, task_values = agent['task_ac'](states)
                _, safety_values = agent['safety_ac'](states)

            # Compute advantages
            task_advantages, task_returns = compute_gae(
                rewards, task_values, dones, self.gamma, self.gae_lambda
            )
            safety_advantages, safety_returns = compute_gae(
                safety_costs, safety_values, dones, self.gamma, self.gae_lambda
            )

            # Normalize advantages
            task_advantages = (task_advantages - task_advantages.mean()) / (task_advantages.std() + 1e-8)
            safety_advantages = (safety_advantages - safety_advantages.mean()) / (safety_advantages.std() + 1e-8)

            # Training loop
            total_actor_loss = 0
            total_critic_loss = 0
            total_lambda_update = 0

            for epoch in range(epochs):
                # Forward pass for task network
                task_dist, task_values_new = agent['task_ac'](states)

                # Compute log probs
                if self.continuous_actions:
                    mean, log_std = task_dist
                    _, task_log_probs = sample_action_continuous(mean, log_std, False)
                    task_entropy = compute_entropy_continuous(log_std).mean()
                else:
                    logits = task_dist
                    dist = torch.distributions.Categorical(logits=logits)
                    task_log_probs = dist.log_prob(actions)
                    task_entropy = compute_entropy_discrete(logits).mean()

                # Task policy loss
                lambda_ = torch.exp(agent['log_lambda']).detach()
                combined_advantages = task_advantages - lambda_ * safety_advantages

                actor_loss = -(task_log_probs * combined_advantages).mean()
                actor_loss = actor_loss - self.entropy_coef * task_entropy

                # Task value loss
                task_value_loss = F.mse_loss(task_values_new, task_returns)
                total_task_loss = actor_loss + self.value_loss_coef * task_value_loss

                # Update task network
                agent['task_optimizer'].zero_grad()
                total_task_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent['task_ac'].parameters(), self.max_grad_norm)
                agent['task_optimizer'].step()

                # Forward pass for safety network (separate)
                safety_dist, safety_values_new = agent['safety_ac'](states)
                safety_value_loss = F.mse_loss(safety_values_new, safety_returns)
                critic_loss = self.value_loss_coef * safety_value_loss

                # Update safety network
                agent['safety_optimizer'].zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent['safety_ac'].parameters(), self.max_grad_norm)
                agent['safety_optimizer'].step()

                # Update Lagrange multiplier
                with torch.no_grad():
                    constraint_violation = safety_values_new.mean() - self.safety_threshold
                    lambda_update = self.lr_lambda * constraint_violation
                    agent['log_lambda'] = agent['log_lambda'] + lambda_update
                    agent['log_lambda'] = torch.clamp(agent['log_lambda'], -5, 5)

                total_actor_loss += actor_loss.item()
                total_critic_loss += (task_value_loss.item() + safety_value_loss.item())
                total_lambda_update += lambda_update.item()

            agent_metrics.append({
                'actor_loss': total_actor_loss / epochs,
                'critic_loss': total_critic_loss / epochs,
                'lambda_update': total_lambda_update / epochs
            })

        # Average metrics across agents
        avg_metrics = {
            'actor_loss': np.mean([m['actor_loss'] for m in agent_metrics]),
            'critic_loss': np.mean([m['critic_loss'] for m in agent_metrics]),
            'lambda_update': np.mean([m['lambda_update'] for m in agent_metrics])
        }

        return avg_metrics

    def get_avg_lambda(self) -> float:
        """Get average Lagrange multiplier across agents"""
        lambdas = [torch.exp(agent['log_lambda']).item() for agent in self.agents]
        return np.mean(lambdas)

    def save(self, path: str):
        """Save agent models"""
        os.makedirs(path, exist_ok=True)
        for i, agent in enumerate(self.agents):
            torch.save({
                'task_ac': agent['task_ac'].state_dict(),
                'safety_ac': agent['safety_ac'].state_dict(),
                'log_lambda': agent['log_lambda'].item(),
                'task_optimizer': agent['task_optimizer'].state_dict(),
                'safety_optimizer': agent['safety_optimizer'].state_dict()
            }, os.path.join(path, f'agent_{i}.pt'))

    def load(self, path: str):
        """Load agent models"""
        for i, agent in enumerate(self.agents):
            checkpoint = torch.load(os.path.join(path, f'agent_{i}.pt'), map_location=self.device)
            agent['task_ac'].load_state_dict(checkpoint['task_ac'])
            agent['safety_ac'].load_state_dict(checkpoint['safety_ac'])
            agent['log_lambda'] = torch.tensor(checkpoint['log_lambda'], device=self.device)
            agent['task_optimizer'].load_state_dict(checkpoint['task_optimizer'])
            agent['safety_optimizer'].load_state_dict(checkpoint['safety_optimizer'])
