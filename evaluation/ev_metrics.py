"""
Expected Value (EV) Metrics for Poker Agents
Measures profitability in BB/100 hands (big blinds per 100 hands)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from collections import defaultdict


@dataclass
class HandResult:
    """Result of a single poker hand"""
    player_id: int
    profit: float  # Net profit/loss for this hand
    position: int  # Position at table (0=dealer, 1=small blind, etc.)
    action_sequence: List[str]  # Actions taken (fold, call, raise)
    final_hand_strength: Optional[int] = None  # Hand ranking if went to showdown
    won: bool = False


@dataclass
class EVMetrics:
    """Expected Value metrics for poker performance"""
    # Core EV metrics
    total_hands: int = 0
    total_profit: float = 0.0
    ev_per_hand: float = 0.0
    ev_bb_per_100: float = 0.0  # Target: 3 BB/100

    # Variance metrics
    variance: float = 0.0
    std_dev: float = 0.0
    sharpe_ratio: float = 0.0

    # Win rate metrics
    hands_won: int = 0
    hands_lost: int = 0
    hands_folded: int = 0
    win_rate: float = 0.0  # % of hands won

    # Profitability by position
    ev_by_position: Dict[int, float] = field(default_factory=dict)
    hands_by_position: Dict[int, int] = field(default_factory=dict)

    # Action statistics
    fold_rate: float = 0.0
    call_rate: float = 0.0
    raise_rate: float = 0.0

    # Safety metrics
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    bankroll_trajectory: List[float] = field(default_factory=list)

    def __str__(self):
        return f"""
EV Performance Metrics:
======================
Total Hands: {self.total_hands}
Total Profit: {self.total_profit:.2f} BB
EV per Hand: {self.ev_per_hand:.4f} BB
EV per 100 Hands: {self.ev_bb_per_100:.2f} BB/100  {'✓' if self.ev_bb_per_100 >= 3.0 else '✗ (Target: 3.0)'}

Win Rate: {self.win_rate:.2%}
  - Hands Won: {self.hands_won}
  - Hands Lost: {self.hands_lost}
  - Hands Folded: {self.hands_folded}

Variance Metrics:
  - Std Dev: {self.std_dev:.2f} BB
  - Sharpe Ratio: {self.sharpe_ratio:.4f}
  - Max Drawdown: {self.max_drawdown:.2f} BB

Action Distribution:
  - Fold Rate: {self.fold_rate:.2%}
  - Call Rate: {self.call_rate:.2%}
  - Raise Rate: {self.raise_rate:.2%}
"""


class EVEvaluator:
    """
    Evaluates poker agent performance using EV metrics
    Target: Achieve consistent 3 BB/100 hands profit
    """

    def __init__(self, big_blind: float = 10.0, n_players: int = 3):
        self.big_blind = big_blind
        self.n_players = n_players
        self.hand_results: Dict[int, List[HandResult]] = defaultdict(list)

    def record_hand(self, player_id: int, result: HandResult):
        """Record result of a single hand"""
        self.hand_results[player_id].append(result)

    def compute_metrics(self, player_id: int, window: Optional[int] = None) -> EVMetrics:
        """
        Compute EV metrics for a player

        Args:
            player_id: Player to evaluate
            window: If specified, only consider last N hands
        """
        results = self.hand_results[player_id]
        if window:
            results = results[-window:]

        if not results:
            return EVMetrics()

        metrics = EVMetrics()

        # Basic statistics
        metrics.total_hands = len(results)
        profits = np.array([r.profit for r in results])
        metrics.total_profit = profits.sum()
        metrics.ev_per_hand = profits.mean()

        # Convert to BB/100 hands
        metrics.ev_bb_per_100 = (metrics.ev_per_hand / self.big_blind) * 100

        # Variance metrics
        metrics.variance = profits.var()
        metrics.std_dev = profits.std()

        # Sharpe ratio (risk-adjusted return)
        if metrics.std_dev > 0:
            metrics.sharpe_ratio = metrics.ev_per_hand / metrics.std_dev

        # Win rate
        metrics.hands_won = sum(1 for r in results if r.won)
        metrics.hands_folded = sum(1 for r in results if 'fold' in r.action_sequence)
        metrics.hands_lost = metrics.total_hands - metrics.hands_won - metrics.hands_folded
        metrics.win_rate = metrics.hands_won / metrics.total_hands if metrics.total_hands > 0 else 0

        # Position analysis
        position_profits = defaultdict(list)
        for r in results:
            position_profits[r.position].append(r.profit)

        for pos, profs in position_profits.items():
            metrics.ev_by_position[pos] = (np.mean(profs) / self.big_blind) * 100
            metrics.hands_by_position[pos] = len(profs)

        # Action statistics
        total_actions = sum(len(r.action_sequence) for r in results)
        if total_actions > 0:
            fold_count = sum(r.action_sequence.count('fold') for r in results)
            call_count = sum(r.action_sequence.count('call') for r in results)
            raise_count = sum(r.action_sequence.count('raise') for r in results)

            metrics.fold_rate = fold_count / total_actions
            metrics.call_rate = call_count / total_actions
            metrics.raise_rate = raise_count / total_actions

        # Drawdown analysis
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        metrics.max_drawdown = drawdown.max()
        metrics.current_drawdown = drawdown[-1]
        metrics.bankroll_trajectory = cumulative.tolist()

        return metrics

    def evaluate_all_players(self, window: Optional[int] = None) -> Dict[int, EVMetrics]:
        """Compute metrics for all players"""
        return {player_id: self.compute_metrics(player_id, window)
                for player_id in self.hand_results.keys()}

    def plot_performance(self, player_id: int, save_path: Optional[str] = None):
        """
        Plot comprehensive performance analysis
        """
        metrics = self.compute_metrics(player_id)
        results = self.hand_results[player_id]

        if not results:
            print(f"No data for player {player_id}")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Player {player_id} Performance Analysis (EV: {metrics.ev_bb_per_100:.2f} BB/100)',
                     fontsize=16, fontweight='bold')

        # 1. Cumulative profit
        ax1 = axes[0, 0]
        cumulative = np.cumsum([r.profit for r in results])
        ax1.plot(cumulative, linewidth=2)
        ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax1.set_title('Cumulative Profit')
        ax1.set_xlabel('Hands Played')
        ax1.set_ylabel('Profit (chips)')
        ax1.grid(True, alpha=0.3)

        # Add target line (3 BB/100)
        target_line = np.arange(len(results)) * (3 * self.big_blind / 100)
        ax1.plot(target_line, 'g--', alpha=0.5, label='Target (3 BB/100)')
        ax1.legend()

        # 2. Rolling EV (per 100 hands)
        ax2 = axes[0, 1]
        window_size = min(100, len(results) // 5)
        if window_size >= 10:
            profits = np.array([r.profit for r in results])
            rolling_ev = np.convolve(profits, np.ones(window_size)/window_size, mode='valid')
            rolling_ev_bb100 = (rolling_ev / self.big_blind) * 100
            ax2.plot(rolling_ev_bb100, linewidth=2)
            ax2.axhline(y=3.0, color='g', linestyle='--', alpha=0.7, label='Target (3 BB/100)')
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax2.set_title(f'Rolling EV (window={window_size} hands)')
            ax2.set_xlabel('Hand Number')
            ax2.set_ylabel('EV (BB/100)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. EV by Position
        ax3 = axes[1, 0]
        if metrics.ev_by_position:
            positions = sorted(metrics.ev_by_position.keys())
            evs = [metrics.ev_by_position[p] for p in positions]
            colors = ['red' if ev < 0 else 'green' for ev in evs]
            ax3.bar(positions, evs, color=colors, alpha=0.6)
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax3.axhline(y=3.0, color='g', linestyle='--', alpha=0.5, label='Target (3 BB/100)')
            ax3.set_title('EV by Position')
            ax3.set_xlabel('Position (0=Button)')
            ax3.set_ylabel('EV (BB/100)')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')

        # 4. Win Rate and Action Distribution
        ax4 = axes[1, 1]
        categories = ['Win Rate', 'Fold Rate', 'Call Rate', 'Raise Rate']
        values = [metrics.win_rate, metrics.fold_rate, metrics.call_rate, metrics.raise_rate]
        colors_pie = ['green', 'red', 'yellow', 'blue']
        ax4.bar(categories, values, color=colors_pie, alpha=0.6)
        ax4.set_title('Win Rate & Action Distribution')
        ax4.set_ylabel('Percentage')
        ax4.set_ylim([0, 1])
        ax4.grid(True, alpha=0.3, axis='y')

        # Format y-axis as percentage
        ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def check_convergence(self, player_id: int, min_hands: int = 1000,
                         window: int = 500, target_ev: float = 3.0,
                         tolerance: float = 0.5) -> Tuple[bool, str]:
        """
        Check if player has converged to target EV

        Args:
            player_id: Player to check
            min_hands: Minimum hands required
            window: Window size for recent performance
            target_ev: Target EV in BB/100 (default 3.0)
            tolerance: Acceptable deviation (default ±0.5 BB/100)

        Returns:
            (converged, message)
        """
        results = self.hand_results[player_id]

        if len(results) < min_hands:
            return False, f"Need {min_hands - len(results)} more hands"

        # Check recent performance
        recent_metrics = self.compute_metrics(player_id, window=window)
        overall_metrics = self.compute_metrics(player_id)

        recent_ev = recent_metrics.ev_bb_per_100
        overall_ev = overall_metrics.ev_bb_per_100

        # Check if both recent and overall EV are within tolerance
        recent_ok = abs(recent_ev - target_ev) <= tolerance
        overall_ok = abs(overall_ev - target_ev) <= tolerance

        if recent_ok and overall_ok:
            return True, f"✓ Converged! EV: {overall_ev:.2f} BB/100 (recent: {recent_ev:.2f})"
        elif recent_ok:
            return False, f"Recent EV good ({recent_ev:.2f}), but overall {overall_ev:.2f} BB/100"
        else:
            return False, f"Current EV: {recent_ev:.2f} BB/100 (target: {target_ev:.2f} ± {tolerance})"

    def get_leaderboard(self, window: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Get leaderboard of players sorted by EV

        Returns:
            List of (player_id, ev_bb_per_100) tuples
        """
        leaderboard = []
        for player_id in self.hand_results.keys():
            metrics = self.compute_metrics(player_id, window)
            leaderboard.append((player_id, metrics.ev_bb_per_100))

        return sorted(leaderboard, key=lambda x: x[1], reverse=True)

    def save_results(self, filepath: str):
        """Save evaluation results to file"""
        import json

        results = {}
        for player_id in self.hand_results.keys():
            metrics = self.compute_metrics(player_id)
            results[f'player_{player_id}'] = {
                'total_hands': metrics.total_hands,
                'ev_bb_per_100': metrics.ev_bb_per_100,
                'win_rate': metrics.win_rate,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'ev_by_position': metrics.ev_by_position,
                'converged': metrics.ev_bb_per_100 >= 3.0
            }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {filepath}")
