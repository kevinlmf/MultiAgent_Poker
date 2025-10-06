"""
PyTorch-based Actor-Critic Networks for Task and Safety Policies
Based on Li & Azizan (2024) MADAC framework
Compatible with macOS without threading issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class NetworkConfig:
    """Configuration for actor-critic networks"""
    state_dim: int
    action_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256, 128)
    activation: str = 'relu'
    continuous_actions: bool = False


class TaskActorCritic(nn.Module):
    """
    Task Actor-Critic Network in PyTorch

    Learns policy π_θ to maximize expected value:
        J(θ) = E[∑ γ^t r_t]

    Components:
    - Actor: π_θ(a|s) - policy network
    - Critic: V_θ(s) - value function for task reward
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config

        # Shared feature extractor
        layers = []
        prev_dim = config.state_dim
        for hidden_dim in config.hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        # Actor head
        self.actor_hidden = nn.Sequential(
            nn.Linear(prev_dim, config.hidden_dims[-1]),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dims[-1])
        )

        if config.continuous_actions:
            self.actor_mean = nn.Linear(config.hidden_dims[-1], config.action_dim)
            self.actor_log_std = nn.Linear(config.hidden_dims[-1], config.action_dim)
        else:
            self.actor_logits = nn.Linear(config.hidden_dims[-1], config.action_dim)

        # Critic head
        self.critic_hidden = nn.Sequential(
            nn.Linear(prev_dim, config.hidden_dims[-1]),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dims[-1])
        )
        self.critic_value = nn.Linear(config.hidden_dims[-1], 1)

    def _get_activation(self):
        if self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'tanh':
            return nn.Tanh()
        elif self.config.activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            action_dist: Action distribution params (logits or mean/log_std)
            value: State value V(s)
        """
        x = self.shared(state)

        # Actor
        actor_h = self.actor_hidden(x)
        if self.config.continuous_actions:
            mean = self.actor_mean(actor_h)
            log_std = torch.clamp(self.actor_log_std(actor_h), -20, 2)
            action_dist = (mean, log_std)
        else:
            logits = self.actor_logits(actor_h)
            action_dist = logits

        # Critic
        critic_h = self.critic_hidden(x)
        value = self.critic_value(critic_h).squeeze(-1)

        return action_dist, value


class SafetyActorCritic(nn.Module):
    """
    Safety Actor-Critic Network in PyTorch

    Learns safety policy π_h and safety value V^h(s):
        V^h(s) = E[∑ γ^t h(s_t)]

    where h(s) >= 0 is the safety constraint
    """

    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.config = config

        # Shared feature extractor
        layers = []
        prev_dim = config.state_dim
        for hidden_dim in config.hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*layers)

        # Safety actor head
        self.safety_actor_hidden = nn.Sequential(
            nn.Linear(prev_dim, config.hidden_dims[-1]),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dims[-1])
        )

        if config.continuous_actions:
            self.safety_actor_mean = nn.Linear(config.hidden_dims[-1], config.action_dim)
            self.safety_actor_log_std = nn.Linear(config.hidden_dims[-1], config.action_dim)
        else:
            self.safety_actor_logits = nn.Linear(config.hidden_dims[-1], config.action_dim)

        # Safety critic head
        self.safety_critic_hidden = nn.Sequential(
            nn.Linear(prev_dim, config.hidden_dims[-1]),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dims[-1])
        )
        self.safety_critic_value = nn.Linear(config.hidden_dims[-1], 1)

    def _get_activation(self):
        if self.config.activation == 'relu':
            return nn.ReLU()
        elif self.config.activation == 'tanh':
            return nn.Tanh()
        elif self.config.activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {self.config.activation}")

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Returns:
            safety_action_dist: Safety policy distribution params
            safety_value: Safety value V^h(s)
        """
        x = self.shared(state)

        # Safety actor
        safety_h = self.safety_actor_hidden(x)
        if self.config.continuous_actions:
            mean = self.safety_actor_mean(safety_h)
            log_std = torch.clamp(self.safety_actor_log_std(safety_h), -20, 2)
            safety_action_dist = (mean, log_std)
        else:
            logits = self.safety_actor_logits(safety_h)
            safety_action_dist = logits

        # Safety critic
        safety_critic_h = self.safety_critic_hidden(x)
        safety_value = self.safety_critic_value(safety_critic_h).squeeze(-1)

        return safety_action_dist, safety_value


def sample_action_discrete(logits: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample discrete action from categorical distribution

    Args:
        logits: Action logits [batch_size, action_dim]
        deterministic: If True, select argmax

    Returns:
        action: Sampled action [batch_size]
        log_prob: Log probability of action [batch_size]
    """
    dist = Categorical(logits=logits)

    if deterministic:
        action = torch.argmax(logits, dim=-1)
    else:
        action = dist.sample()

    log_prob = dist.log_prob(action)
    return action, log_prob


def sample_action_continuous(mean: torch.Tensor, log_std: torch.Tensor,
                             deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample continuous action from Gaussian distribution with tanh squashing

    Args:
        mean: Action mean [batch_size, action_dim]
        log_std: Log std of action [batch_size, action_dim]
        deterministic: If True, use mean

    Returns:
        action: Sampled action [batch_size, action_dim]
        log_prob: Log probability of action [batch_size]
    """
    std = torch.exp(log_std)
    dist = Normal(mean, std)

    if deterministic:
        action = mean
    else:
        action = dist.rsample()

    # Compute log prob before tanh
    log_prob = dist.log_prob(action).sum(dim=-1)

    # Apply tanh squashing
    action = torch.tanh(action)

    # Adjust log prob for tanh transformation
    log_prob = log_prob - torch.sum(torch.log(1 - action**2 + 1e-6), dim=-1)

    return action, log_prob


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE)

    Args:
        rewards: Rewards [T]
        values: Value estimates [T]
        dones: Done flags [T]
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: GAE advantages [T]
        returns: Value targets [T]
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def compute_entropy_discrete(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of categorical distribution"""
    dist = Categorical(logits=logits)
    return dist.entropy()


def compute_entropy_continuous(log_std: torch.Tensor) -> torch.Tensor:
    """Compute entropy of Gaussian distribution"""
    return 0.5 + 0.5 * np.log(2 * np.pi) + log_std.sum(dim=-1)
