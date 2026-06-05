"""
Residual RL — 在 OR 决策基础上学习微调

思路: OR (MIP/LP/DP) 给出基线 → RL 学习 residual 调整 (补货/产量/维护)
      最终决策 = OR + RL_adjustment
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent.risk.control_agent import RiskAdjustment


@dataclass
class RLAdjustment:
    reorder_boost: int = 0
    production_scale: float = 1.0
    budget_scale: float = 1.0
    force_maintain: bool = False
    action_id: int = 0


# 5 档 residual 动作
RL_ACTIONS: List[RLAdjustment] = [
    RLAdjustment(0, 1.00, 1.00, False, 0),   # 完全信任 OR
    RLAdjustment(25, 1.05, 1.00, False, 1),  # 略增补货
    RLAdjustment(60, 1.10, 1.02, False, 2),  # 积极补货+增产
    RLAdjustment(40, 1.05, 1.00, True, 3),   # 维护优先
    RLAdjustment(100, 1.15, 1.05, True, 4),  # 危机模式
]


class ResidualRLPolicy:
    """
    轻量 Q-learning：状态 → 在 OR 之上的 residual 动作。
    训练目标：最大化 profit + 满足率奖励 - 缺货/维护惩罚。
    """

    def __init__(self, n_actions: int = 5, lr: float = 0.15, gamma: float = 0.95, epsilon: float = 0.1):
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q: Dict[Tuple[int, ...], np.ndarray] = {}

    def _discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        # [库存比, 需求趋势, 设备健康, backlog比, 场景压力] → 各 0-2
        bins = []
        for i, v in enumerate(state[:5]):
            if i == 2:  # equipment 0-1
                bins.append(min(2, int(v * 3)))
            else:
                bins.append(min(2, int(v * 3)))
        return tuple(bins)

    def select(self, state: np.ndarray, explore: bool = False) -> RLAdjustment:
        key = self._discretize(state)
        if key not in self.q:
            self.q[key] = np.zeros(self.n_actions)
        if explore and np.random.random() < self.epsilon:
            aid = np.random.randint(self.n_actions)
        else:
            aid = int(np.argmax(self.q[key]))
        return RL_ACTIONS[aid]

    def update(self, state: np.ndarray, action_id: int, reward: float, next_state: np.ndarray) -> None:
        key = self._discretize(state)
        nkey = self._discretize(next_state)
        if key not in self.q:
            self.q[key] = np.zeros(self.n_actions)
        if nkey not in self.q:
            self.q[nkey] = np.zeros(self.n_actions)
        td = reward + self.gamma * np.max(self.q[nkey]) - self.q[key][action_id]
        self.q[key][action_id] += self.lr * td

    @staticmethod
    def build_state(
        inventory: float,
        demand_history: List[float],
        equipment_health: float,
        backlog_size: float,
        scenario_stress: float = 0.0,
        demand_today: float = 250.0,
    ) -> np.ndarray:
        ma7 = np.mean(demand_history[-7:]) if len(demand_history) >= 7 else demand_today
        ma30 = np.mean(demand_history[-30:]) if len(demand_history) >= 30 else ma7
        if ma30 <= 0 or not np.isfinite(ma30):
            ma30 = max(demand_today, 1.0)
        if ma7 <= 0 or not np.isfinite(ma7):
            ma7 = max(demand_today, 1.0)
        inv_ratio = inventory / max(ma7 * 7, 1.0)
        trend = ma7 / max(ma30, 1.0)
        backlog_ratio = backlog_size / max(ma7, 1.0)
        return np.array([inv_ratio, trend, equipment_health, backlog_ratio, scenario_stress], dtype=float)

    @staticmethod
    def merge(or_adj: RiskAdjustment, rl_adj: RLAdjustment) -> RiskAdjustment:
        """OR/risk 基线 + RL residual → 最终调整"""
        merged = RiskAdjustment(
            risk_level=or_adj.risk_level,
            demand_forecast_multiplier=or_adj.demand_forecast_multiplier,
            capacity_multiplier=or_adj.capacity_multiplier,
            reorder_boost=or_adj.reorder_boost + rl_adj.reorder_boost,
            extra_inventory_injection=or_adj.extra_inventory_injection,
            budget_scale=or_adj.budget_scale * rl_adj.budget_scale,
            production_scale=or_adj.production_scale * rl_adj.production_scale,
            contingency_plan_id=or_adj.contingency_plan_id,
            contingency_cost=or_adj.contingency_cost,
            anomalies=list(or_adj.anomalies),
            recommended_actions=list(or_adj.recommended_actions),
        )
        merged.force_maintain = rl_adj.force_maintain
        return merged

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_actions": self.n_actions,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "q": {json.dumps(list(k)): v.tolist() for k, v in self.q.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResidualRLPolicy":
        policy = cls(
            n_actions=int(data.get("n_actions", 5)),
            lr=float(data.get("lr", 0.15)),
            gamma=float(data.get("gamma", 0.95)),
            epsilon=float(data.get("epsilon", 0.05)),
        )
        for key_str, values in data.get("q", {}).items():
            key = tuple(json.loads(key_str))
            policy.q[key] = np.array(values, dtype=float)
        return policy

    def load_checkpoint(self, data: Dict[str, Any]) -> None:
        loaded = self.from_dict(data)
        self.q = loaded.q
        self.epsilon = max(self.epsilon, loaded.epsilon)

    @property
    def num_states(self) -> int:
        return len(self.q)


def train_residual_policy(
    simulate_fn,
    episodes: int = 40,
    days_per_episode: int = 90,
) -> ResidualRLPolicy:
    """在短周期模拟上预训练 RL residual。"""
    policy = ResidualRLPolicy(epsilon=0.25)
    for ep in range(episodes):
        total_r = 0.0
        state_vec, done = simulate_fn(policy, train=True, days=days_per_episode)
        total_r += state_vec.get("episode_reward", 0)
        if (ep + 1) % 10 == 0:
            pass  # silent train
    policy.epsilon = 0.05
    return policy
