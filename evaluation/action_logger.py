"""
Action Logger for Poker Hands
Exports detailed action sequences to CSV for analysis
"""

import csv
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ActionRecord:
    """Record of a single action in a poker hand"""
    hand_id: int
    player_id: int
    position: str  # Button, SB, BB, UTG, etc.
    betting_round: str  # Preflop, Flop, Turn, River
    action: str  # fold, call, raise
    amount: float  # Amount bet/raised
    pot_before: float  # Pot size before action
    pot_after: float  # Pot size after action
    stack_before: float  # Player stack before action
    stack_after: float  # Player stack after action
    hand_strength: Optional[str]  # Premium, Strong, Medium, Weak, Trash
    is_gto_player: bool  # True if GTO agent, False if learning agent
    ev_at_action: Optional[float] = None  # Estimated EV at this decision point
    timestamp: Optional[str] = None


class ActionLogger:
    """
    Logs all actions taken during poker hands to CSV

    Useful for:
    - Analyzing learned strategy
    - Comparing GTO vs exploitative play
    - Debugging training
    - Creating training data
    """

    def __init__(self, output_dir: str = "action_logs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.actions: List[ActionRecord] = []
        self.current_hand_id = 0

        # Position names
        self.position_names = {
            0: "BTN",  # Button
            1: "SB",   # Small Blind
            2: "BB",   # Big Blind
            3: "UTG",  # Under the Gun
            4: "MP",   # Middle Position
            5: "CO",   # Cutoff
        }

    def start_new_hand(self):
        """Start logging a new hand"""
        self.current_hand_id += 1

    def log_action(self,
                   player_id: int,
                   position: int,
                   betting_round: str,
                   action: str,
                   amount: float,
                   pot_before: float,
                   pot_after: float,
                   stack_before: float,
                   stack_after: float,
                   hand_strength: Optional[str] = None,
                   is_gto_player: bool = False,
                   ev_at_action: Optional[float] = None):
        """Log a single action"""

        record = ActionRecord(
            hand_id=self.current_hand_id,
            player_id=player_id,
            position=self.position_names.get(position, f"P{position}"),
            betting_round=betting_round,
            action=action,
            amount=amount,
            pot_before=pot_before,
            pot_after=pot_after,
            stack_before=stack_before,
            stack_after=stack_after,
            hand_strength=hand_strength,
            is_gto_player=is_gto_player,
            ev_at_action=ev_at_action,
            timestamp=datetime.now().isoformat()
        )

        self.actions.append(record)

    def save_to_csv(self, filename: Optional[str] = None):
        """Save all logged actions to CSV"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"actions_{timestamp}.csv"

        filepath = os.path.join(self.output_dir, filename)

        if not self.actions:
            print("No actions to save")
            return

        # Write CSV
        with open(filepath, 'w', newline='') as f:
            fieldnames = [
                'hand_id', 'player_id', 'position', 'betting_round',
                'action', 'amount', 'pot_before', 'pot_after',
                'stack_before', 'stack_after', 'hand_strength',
                'is_gto_player', 'ev_at_action', 'timestamp'
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for action in self.actions:
                writer.writerow(asdict(action))

        print(f"Saved {len(self.actions)} actions to {filepath}")
        return filepath

    def get_action_statistics(self) -> Dict:
        """Get summary statistics of logged actions"""

        if not self.actions:
            return {}

        stats = {
            'total_hands': self.current_hand_id,
            'total_actions': len(self.actions),
            'actions_per_hand': len(self.actions) / self.current_hand_id if self.current_hand_id > 0 else 0
        }

        # Action frequencies
        action_counts = {}
        for action in self.actions:
            action_counts[action.action] = action_counts.get(action.action, 0) + 1

        stats['action_frequencies'] = {
            k: v / len(self.actions) for k, v in action_counts.items()
        }

        # By player type
        gto_actions = [a for a in self.actions if a.is_gto_player]
        learning_actions = [a for a in self.actions if not a.is_gto_player]

        stats['gto_action_count'] = len(gto_actions)
        stats['learning_action_count'] = len(learning_actions)

        # By position
        position_counts = {}
        for action in self.actions:
            position_counts[action.position] = position_counts.get(action.position, 0) + 1

        stats['actions_by_position'] = position_counts

        return stats

    def export_hand_history(self, hand_id: int, filename: Optional[str] = None) -> str:
        """Export a single hand as readable text"""

        hand_actions = [a for a in self.actions if a.hand_id == hand_id]

        if not hand_actions:
            return f"No actions found for hand {hand_id}"

        # Sort by timestamp
        hand_actions.sort(key=lambda x: x.timestamp if x.timestamp else "")

        # Format hand history
        history = f"Hand #{hand_id}\n"
        history += "=" * 50 + "\n\n"

        current_round = None
        for action in hand_actions:
            # Print round header
            if action.betting_round != current_round:
                current_round = action.betting_round
                history += f"\n*** {action.betting_round.upper()} ***\n"
                history += f"Pot: {action.pot_before:.2f}\n\n"

            # Print action
            player_type = "[GTO]" if action.is_gto_player else "[AI] "
            history += f"{player_type} Player {action.player_id} ({action.position}): "

            if action.action == "fold":
                history += "Folds\n"
            elif action.action == "call":
                history += f"Calls {action.amount:.2f}\n"
            elif action.action == "raise":
                history += f"Raises to {action.amount:.2f}\n"
            else:
                history += f"{action.action}\n"

        history += f"\nFinal Pot: {hand_actions[-1].pot_after:.2f}\n"

        if filename:
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                f.write(history)
            print(f"Hand history saved to {filepath}")

        return history

    def compare_strategies(self) -> Dict:
        """Compare GTO vs Learning agent action distributions"""

        gto_actions = [a for a in self.actions if a.is_gto_player]
        learning_actions = [a for a in self.actions if not a.is_gto_player]

        def get_distribution(actions):
            total = len(actions)
            if total == 0:
                return {}

            counts = {}
            for a in actions:
                counts[a.action] = counts.get(a.action, 0) + 1

            return {k: v / total for k, v in counts.items()}

        comparison = {
            'gto_distribution': get_distribution(gto_actions),
            'learning_distribution': get_distribution(learning_actions),
            'total_gto_actions': len(gto_actions),
            'total_learning_actions': len(learning_actions)
        }

        # Calculate differences
        all_actions = set(list(comparison['gto_distribution'].keys()) +
                         list(comparison['learning_distribution'].keys()))

        differences = {}
        for action in all_actions:
            gto_freq = comparison['gto_distribution'].get(action, 0)
            learning_freq = comparison['learning_distribution'].get(action, 0)
            differences[action] = learning_freq - gto_freq

        comparison['frequency_differences'] = differences

        return comparison

    def clear(self):
        """Clear all logged actions"""
        self.actions = []
        self.current_hand_id = 0


class ActionAnalyzer:
    """Analyze actions from CSV files"""

    @staticmethod
    def load_from_csv(filepath: str) -> List[ActionRecord]:
        """Load actions from CSV file"""
        actions = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert types
                row['hand_id'] = int(row['hand_id'])
                row['player_id'] = int(row['player_id'])
                row['amount'] = float(row['amount'])
                row['pot_before'] = float(row['pot_before'])
                row['pot_after'] = float(row['pot_after'])
                row['stack_before'] = float(row['stack_before'])
                row['stack_after'] = float(row['stack_after'])
                row['is_gto_player'] = row['is_gto_player'].lower() == 'true'

                if row['ev_at_action']:
                    row['ev_at_action'] = float(row['ev_at_action'])
                else:
                    row['ev_at_action'] = None

                actions.append(ActionRecord(**row))

        return actions

    @staticmethod
    def analyze_position_tendencies(actions: List[ActionRecord]) -> Dict:
        """Analyze action tendencies by position"""

        position_stats = {}

        for action in actions:
            pos = action.position
            if pos not in position_stats:
                position_stats[pos] = {'fold': 0, 'call': 0, 'raise': 0, 'total': 0}

            position_stats[pos][action.action] = position_stats[pos].get(action.action, 0) + 1
            position_stats[pos]['total'] += 1

        # Convert to frequencies
        for pos, stats in position_stats.items():
            total = stats['total']
            position_stats[pos] = {
                'fold_rate': stats.get('fold', 0) / total,
                'call_rate': stats.get('call', 0) / total,
                'raise_rate': stats.get('raise', 0) / total,
                'total_actions': total
            }

        return position_stats

    @staticmethod
    def find_exploitable_patterns(actions: List[ActionRecord]) -> List[str]:
        """Identify potentially exploitable patterns in play"""

        patterns = []

        # Separate by player type
        gto_actions = [a for a in actions if a.is_gto_player]
        learning_actions = [a for a in actions if not a.is_gto_player]

        # Check fold frequency
        if gto_actions:
            gto_fold_rate = sum(1 for a in gto_actions if a.action == 'fold') / len(gto_actions)
            if gto_fold_rate > 0.7:
                patterns.append(f"GTO players fold {gto_fold_rate:.1%} of the time - too passive!")

        if learning_actions:
            learning_fold_rate = sum(1 for a in learning_actions if a.action == 'fold') / len(learning_actions)
            if learning_fold_rate > 0.7:
                patterns.append(f"Learning agent folds {learning_fold_rate:.1%} - too tight!")
            elif learning_fold_rate < 0.3:
                patterns.append(f"Learning agent only folds {learning_fold_rate:.1%} - too loose!")

        # Check raise aggression
        if learning_actions:
            learning_raise_rate = sum(1 for a in learning_actions if a.action == 'raise') / len(learning_actions)
            if learning_raise_rate > 0.5:
                patterns.append(f"Learning agent raises {learning_raise_rate:.1%} - very aggressive!")
            elif learning_raise_rate < 0.1:
                patterns.append(f"Learning agent raises only {learning_raise_rate:.1%} - too passive!")

        return patterns
