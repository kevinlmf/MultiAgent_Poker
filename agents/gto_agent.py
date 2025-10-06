"""
GTO (Game Theory Optimal) Baseline Agent
Implements approximated GTO strategy for poker

This agent serves as a baseline opponent that our exploitative agent
needs to beat to achieve 3 BB/100 hands profit.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class HandStrength(Enum):
    """Simplified hand strength categories"""
    PREMIUM = 5  # AA, KK, QQ, AK
    STRONG = 4   # JJ, TT, AQ, AJs
    MEDIUM = 3   # 99-77, AJ, KQ, suited connectors
    WEAK = 2     # 66-22, Ax, Kx
    TRASH = 1    # Everything else


class GTOAgent:
    """
    Approximated GTO Agent using preflop ranges and postflop heuristics

    GTO Strategy Components:
    1. Balanced preflop ranges by position
    2. Continuation betting with proper frequency
    3. Balanced bluffing ratios
    4. Position-aware play
    """

    def __init__(self, player_id: int, n_players: int = 3,
                 small_blind: float = 5.0, big_blind: float = 10.0):
        self.player_id = player_id
        self.n_players = n_players
        self.small_blind = small_blind
        self.big_blind = big_blind

        # GTO frequencies (simplified)
        self.preflop_ranges = self._initialize_preflop_ranges()
        self.cbet_frequency = 0.65  # C-bet 65% of the time
        self.bluff_ratio = 0.33     # 1 bluff for every 2 value bets

    def _initialize_preflop_ranges(self) -> Dict[str, Dict[str, float]]:
        """
        Initialize GTO preflop ranges by position
        Returns dict of {position: {action: frequency}}
        """
        ranges = {
            'button': {
                'premium': {'raise': 0.95, 'call': 0.05, 'fold': 0.0},
                'strong': {'raise': 0.80, 'call': 0.20, 'fold': 0.0},
                'medium': {'raise': 0.40, 'call': 0.35, 'fold': 0.25},
                'weak': {'raise': 0.10, 'call': 0.20, 'fold': 0.70},
                'trash': {'raise': 0.05, 'call': 0.05, 'fold': 0.90}
            },
            'cutoff': {
                'premium': {'raise': 0.95, 'call': 0.05, 'fold': 0.0},
                'strong': {'raise': 0.75, 'call': 0.25, 'fold': 0.0},
                'medium': {'raise': 0.30, 'call': 0.30, 'fold': 0.40},
                'weak': {'raise': 0.05, 'call': 0.15, 'fold': 0.80},
                'trash': {'raise': 0.02, 'call': 0.03, 'fold': 0.95}
            },
            'early': {
                'premium': {'raise': 0.90, 'call': 0.10, 'fold': 0.0},
                'strong': {'raise': 0.60, 'call': 0.30, 'fold': 0.10},
                'medium': {'raise': 0.15, 'call': 0.20, 'fold': 0.65},
                'weak': {'raise': 0.0, 'call': 0.05, 'fold': 0.95},
                'trash': {'raise': 0.0, 'call': 0.0, 'fold': 1.0}
            }
        }
        return ranges

    def _classify_hand_strength(self, observation: np.ndarray) -> str:
        """
        Classify hand strength from observation

        Observation typically includes:
        - Hand cards
        - Board cards
        - Position
        - Pot size
        - Stack sizes
        """
        # Extract hand strength from observation (simplified)
        # In real implementation, this would use hand evaluator

        # Assume observation has hand strength indicator
        # This is a placeholder - real implementation needs proper hand evaluation
        if len(observation) > 10:
            hand_value = observation[0:2].sum()  # Simplified

            if hand_value > 0.8:
                return 'premium'
            elif hand_value > 0.6:
                return 'strong'
            elif hand_value > 0.4:
                return 'medium'
            elif hand_value > 0.2:
                return 'weak'
            else:
                return 'trash'
        else:
            # Random fallback
            return np.random.choice(['premium', 'strong', 'medium', 'weak', 'trash'],
                                   p=[0.05, 0.15, 0.30, 0.30, 0.20])

    def _get_position_category(self, observation: np.ndarray) -> str:
        """Determine position category from observation"""
        # Simplified position determination
        # In real env, this should come from observation

        position_idx = self.player_id % self.n_players

        if position_idx == 0:
            return 'button'
        elif position_idx == 1:
            return 'cutoff'
        else:
            return 'early'

    def _get_betting_round(self, observation: np.ndarray) -> str:
        """Determine current betting round"""
        # Simplified - in real implementation, extract from observation
        # Assume observation encodes this
        if len(observation) > 20:
            round_indicator = observation[10]
            if round_indicator < 0.25:
                return 'preflop'
            elif round_indicator < 0.5:
                return 'flop'
            elif round_indicator < 0.75:
                return 'turn'
            else:
                return 'river'
        return 'preflop'

    def select_action(self, observation: np.ndarray) -> int:
        """
        Select action using GTO strategy

        Returns:
            action: 0 = fold, 1 = call, 2 = raise
        """
        hand_strength = self._classify_hand_strength(observation)
        position = self._get_position_category(observation)
        betting_round = self._get_betting_round(observation)

        if betting_round == 'preflop':
            return self._preflop_action(hand_strength, position)
        else:
            return self._postflop_action(hand_strength, betting_round, observation)

    def _preflop_action(self, hand_strength: str, position: str) -> int:
        """GTO preflop action based on hand strength and position"""

        if position not in self.preflop_ranges:
            position = 'early'  # Default to tight

        action_probs = self.preflop_ranges[position].get(hand_strength,
                                                         {'fold': 0.9, 'call': 0.05, 'raise': 0.05})

        # Sample action from probabilities
        actions = ['fold', 'call', 'raise']
        probs = [action_probs.get('fold', 0.3),
                action_probs.get('call', 0.3),
                action_probs.get('raise', 0.4)]

        # Normalize
        probs = np.array(probs)
        probs = probs / probs.sum()

        action_name = np.random.choice(actions, p=probs)

        # Map to action index
        action_map = {'fold': 0, 'call': 1, 'raise': 2}
        return action_map[action_name]

    def _postflop_action(self, hand_strength: str, betting_round: str,
                        observation: np.ndarray) -> int:
        """
        GTO postflop action

        Strategy:
        - Strong hands: Bet for value (80%)
        - Medium hands: Check/call (70%)
        - Weak hands: Bluff or fold based on position
        """

        # Extract pot odds and position from observation
        is_in_position = self._is_in_position(observation)
        pot_size = self._get_pot_size(observation)
        facing_bet = self._is_facing_bet(observation)

        if hand_strength in ['premium', 'strong']:
            # Value betting
            if facing_bet:
                return np.random.choice([1, 2], p=[0.3, 0.7])  # Mostly raise
            else:
                return np.random.choice([1, 2], p=[0.2, 0.8])  # Mostly bet

        elif hand_strength == 'medium':
            # Showdown value
            if facing_bet:
                return 1  # Call
            else:
                return np.random.choice([1, 2], p=[0.7, 0.3])  # Mostly check

        else:  # weak or trash
            # Bluff or fold
            if facing_bet:
                # Bluff raise occasionally
                if is_in_position and np.random.random() < 0.15:
                    return 2
                else:
                    return 0  # Fold
            else:
                # Bluff bet occasionally (GTO bluff ratio)
                if np.random.random() < self.bluff_ratio:
                    return 2
                else:
                    return 1  # Check

    def _is_in_position(self, observation: np.ndarray) -> bool:
        """Check if agent is in position"""
        # Simplified
        return self.player_id == 0

    def _get_pot_size(self, observation: np.ndarray) -> float:
        """Extract pot size from observation"""
        # Simplified - should extract from actual observation
        if len(observation) > 15:
            return observation[12] * 100
        return 50.0

    def _is_facing_bet(self, observation: np.ndarray) -> bool:
        """Check if facing a bet"""
        # Simplified - should extract from actual observation
        if len(observation) > 18:
            return observation[15] > 0.5
        return np.random.random() < 0.4


class GTOAgentPool:
    """
    Pool of GTO agents with slight variations
    Creates a realistic multi-player GTO environment
    """

    def __init__(self, n_agents: int, small_blind: float = 5.0, big_blind: float = 10.0):
        self.n_agents = n_agents
        self.agents = []

        for i in range(n_agents):
            agent = GTOAgent(i, n_agents, small_blind, big_blind)

            # Add slight variations to make agents less exploitable
            variation = np.random.normal(0, 0.05)
            agent.cbet_frequency = np.clip(agent.cbet_frequency + variation, 0.55, 0.75)
            agent.bluff_ratio = np.clip(agent.bluff_ratio + variation, 0.25, 0.40)

            self.agents.append(agent)

    def select_actions(self, observations: np.ndarray) -> np.ndarray:
        """Get actions from all GTO agents"""
        actions = []
        for i, agent in enumerate(self.agents):
            if i < len(observations):
                action = agent.select_action(observations[i])
                actions.append(action)
        return np.array(actions)

    def get_agent(self, player_id: int) -> GTOAgent:
        """Get specific agent by ID"""
        return self.agents[player_id]


def create_mixed_agent_pool(n_total_agents: int, n_learning_agents: int,
                            small_blind: float = 5.0, big_blind: float = 10.0):
    """
    Create a mixed pool of learning agents and GTO agents

    Args:
        n_total_agents: Total number of agents in the game
        n_learning_agents: Number of learning (exploitative) agents

    Returns:
        agent_pool: Mixed pool with learning and GTO agents
        learning_indices: Indices of learning agents
    """
    gto_pool = GTOAgentPool(n_total_agents - n_learning_agents, small_blind, big_blind)

    # Indices 0 to n_learning_agents-1 are learning agents
    # Rest are GTO agents
    learning_indices = list(range(n_learning_agents))

    return gto_pool, learning_indices
