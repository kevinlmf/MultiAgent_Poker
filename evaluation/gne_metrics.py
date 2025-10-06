"""
Generalized Nash Equilibrium Evaluation Metrics

Metrics for evaluating convergence to GNE and comparing with CFR
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class GNEMetrics:
    """Container for GNE evaluation metrics"""
    # Equilibrium quality
    exploitability: float  # How much can best response exploit
    nash_conv: float  # Nash convergence metric

    # Safety metrics
    constraint_violation_rate: float  # % of episodes violating constraints
    avg_safety_value: float  # Average V^h(s)
    min_safety_value: float  # Minimum V^h(s) encountered

    # Performance metrics
    avg_reward: float  # Average reward per hand
    sharpe_ratio: float  # Risk-adjusted return
    max_drawdown: float  # Maximum drawdown

    # Bankroll metrics
    avg_bankroll: float
    min_bankroll: float
    final_bankrolls: List[float]

    # Convergence metrics
    value_variance: float  # Variance of value estimates
    policy_entropy: float  # Policy entropy (exploration)
    lagrange_multipliers: List[float]  # λ for each agent

    def __str__(self):
        return f"""
GNE Evaluation Metrics:
=======================
Equilibrium Quality:
  Exploitability: {self.exploitability:.4f}
  Nash Convergence: {self.nash_conv:.4f}

Safety Metrics:
  Constraint Violations: {self.constraint_violation_rate:.2%}
  Avg Safety Value: {self.avg_safety_value:.4f}
  Min Safety Value: {self.min_safety_value:.4f}

Performance:
  Avg Reward/Hand: {self.avg_reward:.2f}
  Sharpe Ratio: {self.sharpe_ratio:.4f}
  Max Drawdown: {self.max_drawdown:.2%}

Bankroll:
  Average: {self.avg_bankroll:.2f}
  Minimum: {self.min_bankroll:.2f}
  Final: {np.mean(self.final_bankrolls):.2f} ± {np.std(self.final_bankrolls):.2f}

Convergence:
  Value Variance: {self.value_variance:.4f}
  Policy Entropy: {self.policy_entropy:.4f}
  Avg Lambda: {np.mean(self.lagrange_multipliers):.4f}
"""


class GNEEvaluator:
    """Evaluates GNE properties of learned policies"""

    def __init__(self, env, madac_trainer):
        self.env = env
        self.trainer = madac_trainer
        self.n_agents = env.n_players

    def evaluate(
        self,
        n_episodes: int = 1000,
        deterministic: bool = True
    ) -> GNEMetrics:
        """
        Comprehensive evaluation of GNE properties

        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy
        """
        # Tracking
        episode_rewards = {i: [] for i in range(self.n_agents)}
        episode_safety_values = {i: [] for i in range(self.n_agents)}
        constraint_violations = {i: 0 for i in range(self.n_agents)}
        bankroll_history = {i: [] for i in range(self.n_agents)}
        value_estimates = {i: [] for i in range(self.n_agents)}

        for episode in range(n_episodes):
            obs = self.env.reset(reset_bankrolls=False)
            done = {i: False for i in range(self.n_agents)}
            episode_reward = {i: 0 for i in range(self.n_agents)}

            while not all(done.values()):
                # Get actions
                actions = self.trainer.select_actions(obs, deterministic=deterministic)

                # Step environment
                next_obs, rewards, dones, infos = self.env.step(actions)

                # Track metrics
                for i in range(self.n_agents):
                    episode_reward[i] += rewards[i]

                    if not infos[i]['is_safe']:
                        constraint_violations[i] += 1

                    bankroll_history[i].append(infos[i]['bankroll'])
                    episode_safety_values[i].append(
                        infos[i]['risk_metrics']['safety_value']
                    )

                    # Get value estimate
                    import torch
                    state_tensor = torch.FloatTensor(obs[i]).unsqueeze(0)
                    with torch.no_grad():
                        _, value = self.trainer.agents[i].task_ac.forward(state_tensor)
                    value_estimates[i].append(value.item())

                obs = next_obs
                done = dones

            # Record episode rewards
            for i in range(self.n_agents):
                episode_rewards[i].append(episode_reward[i])

        # Compute metrics
        exploitability = self._estimate_exploitability(episode_rewards)
        nash_conv = self._compute_nash_convergence(episode_rewards)

        # Safety metrics
        total_steps = sum(len(v) for v in episode_safety_values.values())
        total_violations = sum(constraint_violations.values())
        constraint_violation_rate = total_violations / total_steps if total_steps > 0 else 0

        all_safety_values = [v for vals in episode_safety_values.values() for v in vals]
        avg_safety_value = np.mean(all_safety_values)
        min_safety_value = np.min(all_safety_values)

        # Performance metrics
        all_rewards = [r for rewards in episode_rewards.values() for r in rewards]
        avg_reward = np.mean(all_rewards)

        reward_std = np.std(all_rewards)
        sharpe_ratio = avg_reward / reward_std if reward_std > 0 else 0

        # Bankroll metrics
        all_bankrolls = [b for bankrolls in bankroll_history.values() for b in bankrolls]
        avg_bankroll = np.mean(all_bankrolls)
        min_bankroll = np.min(all_bankrolls)
        final_bankrolls = [bankroll_history[i][-1] for i in range(self.n_agents)]

        max_drawdown = self._compute_max_drawdown(bankroll_history)

        # Convergence metrics
        all_values = [v for vals in value_estimates.values() for v in vals]
        value_variance = np.var(all_values)

        # Get policy entropy (sample some states)
        policy_entropy = self._compute_policy_entropy()

        # Get Lagrange multipliers
        lagrange_multipliers = [agent.lambda_.item() for agent in self.trainer.agents]

        return GNEMetrics(
            exploitability=exploitability,
            nash_conv=nash_conv,
            constraint_violation_rate=constraint_violation_rate,
            avg_safety_value=avg_safety_value,
            min_safety_value=min_safety_value,
            avg_reward=avg_reward,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            avg_bankroll=avg_bankroll,
            min_bankroll=min_bankroll,
            final_bankrolls=final_bankrolls,
            value_variance=value_variance,
            policy_entropy=policy_entropy,
            lagrange_multipliers=lagrange_multipliers
        )

    def _estimate_exploitability(self, episode_rewards: Dict[int, List[float]]) -> float:
        """
        Estimate exploitability

        Exploitability = max_i (BR_i - v_i)
        where BR_i is best response value and v_i is current value

        For poker, this requires computing best response, which is expensive.
        We use a proxy: variance of rewards (high variance = more exploitable)
        """
        variances = [np.var(rewards) for rewards in episode_rewards.values()]
        return np.mean(variances)

    def _compute_nash_convergence(self, episode_rewards: Dict[int, List[float]]) -> float:
        """
        Nash convergence metric

        Measures how much agents' strategies deviate from equilibrium
        Using reward variance as proxy
        """
        all_rewards = np.array([rewards for rewards in episode_rewards.values()])
        # If all agents have similar reward distributions, closer to equilibrium
        cross_agent_variance = np.var(np.mean(all_rewards, axis=1))
        return cross_agent_variance

    def _compute_max_drawdown(self, bankroll_history: Dict[int, List[float]]) -> float:
        """Compute maximum drawdown across all agents"""
        max_drawdowns = []
        for bankrolls in bankroll_history.values():
            bankrolls = np.array(bankrolls)
            peak = np.maximum.accumulate(bankrolls)
            drawdown = (peak - bankrolls) / peak
            max_drawdowns.append(np.max(drawdown))
        return np.mean(max_drawdowns)

    def _compute_policy_entropy(self, n_samples: int = 100) -> float:
        """Compute average policy entropy"""
        entropies = []

        for _ in range(n_samples):
            # Sample random states
            obs = self.env.reset()
            state = obs[0]  # Use first agent

            import torch
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            for agent in self.trainer.agents:
                with torch.no_grad():
                    if self.trainer.continuous_actions:
                        action_dist, _ = agent.task_ac.forward(state_tensor)
                        mean, log_std = action_dist
                        std = torch.exp(log_std)
                        entropy = torch.distributions.Normal(mean, std).entropy().sum()
                    else:
                        action_dist, _ = agent.task_ac.forward(state_tensor)
                        entropy = torch.distributions.Categorical(logits=action_dist).entropy()

                entropies.append(entropy.item())

        return np.mean(entropies)

    def compare_with_cfr(self, cfr_policy, n_episodes: int = 1000) -> Dict:
        """
        Compare GNE policy with CFR policy

        Returns comparison metrics
        """
        # Evaluate CFR policy
        cfr_metrics = self._evaluate_policy(cfr_policy, n_episodes)

        # Evaluate GNE policy
        gne_metrics = self.evaluate(n_episodes)

        comparison = {
            'gne': gne_metrics,
            'cfr': cfr_metrics,
            'improvement': {
                'reward': gne_metrics.avg_reward - cfr_metrics.avg_reward,
                'sharpe': gne_metrics.sharpe_ratio - cfr_metrics.sharpe_ratio,
                'safety': gne_metrics.constraint_violation_rate - cfr_metrics.constraint_violation_rate
            }
        }

        return comparison

    def _evaluate_policy(self, policy, n_episodes: int) -> GNEMetrics:
        """Evaluate a generic policy (for comparison)"""
        # Placeholder - implement policy evaluation
        # This would run the policy in the environment and collect metrics
        pass

    def plot_learning_curves(
        self,
        history: Dict,
        save_path: Optional[str] = None
    ):
        """Plot learning curves for GNE training"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Rewards
        axes[0, 0].plot(history['rewards'])
        axes[0, 0].set_title('Average Reward')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')

        # Safety values
        axes[0, 1].plot(history['safety_values'])
        axes[0, 1].axhline(y=0, color='r', linestyle='--', label='Safety threshold')
        axes[0, 1].set_title('Safety Value V^h(s)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].legend()

        # Lagrange multipliers
        axes[0, 2].plot(history['lambdas'])
        axes[0, 2].set_title('Lagrange Multiplier λ')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('λ')

        # Bankrolls
        axes[1, 0].plot(history['bankrolls'])
        axes[1, 0].set_title('Average Bankroll')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Bankroll')

        # Constraint violations
        axes[1, 1].plot(history['violations'])
        axes[1, 1].set_title('Constraint Violation Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Violation Rate')

        # Exploitability
        if 'exploitability' in history:
            axes[1, 2].plot(history['exploitability'])
            axes[1, 2].set_title('Exploitability')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Exploitability')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def generate_report(
        self,
        metrics: GNEMetrics,
        save_path: str = 'gne_report.txt'
    ):
        """Generate detailed evaluation report"""
        report = f"""
{"="*80}
SAFE MULTI-AGENT POKER: GNE EVALUATION REPORT
{"="*80}

{metrics}

Interpretation:
---------------
1. Exploitability: Lower is better. Values < 0.1 indicate strong equilibrium.
2. Constraint Violations: Should be close to 0% for safe learning.
3. Sharpe Ratio: Higher is better. Values > 1.0 indicate good risk-adjusted returns.
4. Lagrange Multipliers: Adapt based on constraint satisfaction.
   - High λ → More conservative (prioritize safety)
   - Low λ → More aggressive (prioritize reward)

Comparison with Single-Game Nash (CFR):
----------------------------------------
- GNE accounts for multi-episode bankroll dynamics
- CFR optimizes single-hand equilibrium
- GNE should have:
  * Lower risk of ruin
  * Better long-term bankroll growth
  * Slightly lower per-hand EV (due to safety constraints)

{"="*80}
"""

        with open(save_path, 'w') as f:
            f.write(report)

        print(f"Report saved to {save_path}")
        return report
