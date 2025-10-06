# Game Theory Optimal (GTO) Poker Strategy Explained

## What is GTO?

**Game Theory Optimal (GTO)** strategy is a balanced strategy in poker that cannot be exploited.

### Core Concepts

1. **Nash Equilibrium**
   - GTO strategy forms a Nash equilibrium
   - No player can improve their expected value by unilaterally changing their strategy
   - Even if opponents know your strategy, they cannot exploit you

2. **Unexploitability**
   ```
   If you perfectly execute GTO strategy:
   - Opponents can at most break even
   - Regardless of how opponents adjust, your EV won't become negative
   - But you also cannot exploit opponents' mistakes
   ```

3. **Mixed Strategy**
   - GTO is not a fixed action
   - It's a probability distribution: e.g., 65% bet, 35% check
   - Randomization makes you unpredictable to opponents

## GTO vs Exploitative Play

### GTO Strategy
```python
Advantages:
✓ Cannot be exploited
✓ Guaranteed not to lose long-term
✓ Safe in high-level games
✓ No need to read opponents

Disadvantages:
✗ Cannot maximize profit
✗ Cannot exploit opponent mistakes
✗ Usually low EV (0-2 BB/100)
✗ Computationally complex
```

### Exploitative Strategy
```python
Advantages:
✓ Can exploit opponent weaknesses
✓ Maximize EV (can reach 3-10+ BB/100)
✓ Adapt to different opponents
✓ High profit in low-level games

Disadvantages:
✗ Can be counter-exploited
✗ Requires accurate opponent modeling
✗ Will lose if opponent read is wrong
✗ Higher variance
```

### Our Strategy
```
┌────────────────────────────────────────┐
│        Exploitative Learning           │
│                                        │
│  Target: GTO opponents (predictable)   │
│  Method: Learn to exploit fixed        │
│          patterns                      │
│  Goal: Achieve 3+ BB/100 hands         │
│                                        │
│  Key: GTO is balanced but cannot       │
│       adjust to suboptimal play, which │
│       we can exploit                   │
└────────────────────────────────────────┘
```

## Core Components of GTO Strategy

### 1. Preflop Ranges

GTO uses different starting hand ranges based on position:

#### Button (BTN - Best Position)
```
Premium Hands (AA, KK, QQ, AKs):
  - Raise: 95%
  - Call: 5%
  - Fold: 0%

Strong Hands (JJ, TT, AQs, AJs):
  - Raise: 80%
  - Call: 20%
  - Fold: 0%

Medium Hands (99-77, AJ, KQ, suited connectors):
  - Raise: 40%
  - Call: 35%
  - Fold: 25%

Weak Hands (66-22, Ax, Kx):
  - Raise: 10%
  - Call: 20%
  - Fold: 70%
```

#### Early Position (UTG - Worst Position)
```
Premium Hands:
  - Raise: 90%
  - Call: 10%
  - Fold: 0%

Strong Hands:
  - Raise: 60%
  - Call: 30%
  - Fold: 10%

Medium Hands:
  - Raise: 15%
  - Call: 20%
  - Fold: 65%

Weak Hands:
  - Raise: 0%
  - Call: 5%
  - Fold: 95%
```

**Key**: Better position allows wider range of playable hands

### 2. Postflop Strategy

#### C-Bet Frequency (Continuation Bet)
```
GTO Standard:
- Overall c-bet frequency: 65%
- Balance value bets and bluffs
- Not betting every time
```

#### Bluff-to-Value Ratio
```
GTO Principle:
- Pot-sized bet: 1:2 bluff-to-value
  (1 bluff for every 2 value bets)

- 1/2 pot bet: 1:1 bluff-to-value

Mathematical principle:
- Makes opponent's call and fold expectations equal
- Prevents opponent from exploiting through frequency reads
```

#### Positional Advantage
```
In Position:
- Can bluff more frequently
- Control pot size
- Get more information

Out of Position:
- More conservative
- More check-calling
- Avoid large pots
```

### 3. Bet Sizing

GTO uses multiple bet sizes:

```
Bet Size           Usage                 Frequency
────────────────────────────────────────────────
1/3 pot           Thin value, inducing   25%
1/2 pot           Standard value+bluff   50%
2/3-3/4 pot       Strong value           20%
Pot or larger     Polarized range        5%
```

**Key**: Different sizes contain different ranges, making it hard for opponents to exploit

## How to Exploit GTO Players?

While GTO is theoretically unexploitable, in practice GTO agents have the following weaknesses:

### 1. Over-Predictability
```python
Problem: Strictly follows 65% c-bet frequency
Exploit: Aggressively bluff when they don't c-bet (35% of time)
```

### 2. No Opponent Adaptation
```python
Problem: Strategy doesn't change regardless of opponent adjustments
Exploit: Find fixed patterns and adjust accordingly
```

### 3. Rigid Positional Play
```python
Problem: Too tight in early position, too loose on button
Exploit:
  - Give more respect to early position raises
  - 3-bet button more frequently
```

### 4. Lack of Dynamic Adjustment
```python
Problem: Doesn't consider historical information
Exploit: Leverage session dynamics
```

## Our Exploitative Strategy

### Learning Objectives

```python
Objective 1: Identify GTO patterns
- Track opponent frequency distributions
- Identify fixed betting patterns
- Find deviations from optimal

Objective 2: Dynamic adjustment
- Adjust strategy based on observations
- Use different strategies for GTO players in different positions
- Leverage historical information

Objective 3: Maximize EV
- Don't pursue balance
- Focus on exploiting opponent weaknesses
- Target: 3+ BB/100 (vs GTO's 0-2 BB/100)
```

### Specific Exploitation Techniques

#### 1. Over-fold Exploit
```python
if opponent_fold_rate > 0.7:
    increase_bluff_frequency()
    # GTO too passive -> we bluff more
```

#### 2. Under-defense Exploit
```python
if opponent_3bet_defense_rate < 0.5:
    increase_3bet_frequency()
    # GTO folds too much to 3-bets -> we 3-bet more
```

#### 3. Position-based Exploit
```python
if position == "Button" and opponent_from_early:
    # Early position GTO range is strong
    respect_their_range()
    fold_marginal_hands()

if position == "BB" and opponent_from_button:
    # Button GTO range is wide
    defend_wider()
    check_raise_more()
```

#### 4. Bet Size Tell
```python
if opponent_bet_size == "2/3 pot":
    # GTO always uses 2/3 pot for strong value
    fold_weak_hands()

if opponent_bet_size == "1/3 pot":
    # GTO uses 1/3 pot for thin value/weak
    call_or_raise_more()
```

## Implementation Details

### GTO Agent Implementation (`agents/gto_agent.py`)

```python
class GTOAgent:
    def __init__(self):
        # Preflop ranges by position
        self.preflop_ranges = {
            'button': {...},
            'cutoff': {...},
            'early': {...}
        }

        # Postflop frequencies
        self.cbet_frequency = 0.65
        self.bluff_ratio = 0.33  # 1:2

    def select_action(self, obs):
        # Based on position and hand strength
        # Randomly select action according to GTO frequencies
        return sample_from_distribution(
            actions=['fold', 'call', 'raise'],
            probs=self.compute_gto_probs(obs)
        )
```

### Exploitative Agent Learning

```python
class ExploitativeLearner:
    def learn(self, opponent_history):
        # 1. Analyze GTO opponent frequencies
        opponent_stats = analyze_frequencies(opponent_history)

        # 2. Find deviations
        exploits = find_exploitable_patterns(opponent_stats)

        # 3. Adjust strategy
        for exploit in exploits:
            if exploit.type == "over_fold":
                self.increase_bluff_freq()
            elif exploit.type == "under_3bet_defense":
                self.increase_3bet_freq()

        # 4. Evaluate if 3 BB/100 is reached
        if self.ev_bb100 >= 3.0:
            convergence = True
```

## Mathematics of GTO vs Exploitative

### GTO Expected Value
```
E[GTO] = 0 to +2 BB/100

Reason:
- Opponents cannot exploit you
- But you don't exploit opponents either
- Break even against perfect opponents
- Slight profit against imperfect opponents (+1-2 BB/100)
```

### Exploitative Expected Value
```
E[Exploitative] = -5 to +10 BB/100

Depends on:
✓ Accuracy of opponent modeling
✓ Actual opponent weaknesses
✓ Whether opponent counter-exploits

vs GTO opponents:
E[Exploitative] = +3 to +5 BB/100
(because GTO won't counter-exploit)
```

### Why Can We Achieve 3 BB/100?

```
1. Inherent GTO weaknesses:
   - Must maintain balance → fixed frequencies
   - Cannot adjust to opponents → predictable
   - Same strategy for all opponents → suboptimal

2. Our advantages:
   - Don't need balance → flexible adjustment
   - Specifically target GTO → highly optimized
   - Can exploit fixed patterns → increase EV

3. Mathematical proof:
   If GTO player fold rate = 0.7
   Our bluff frequency = 0.5
   → EV = 0.7 * pot + 0.3 * (-bet)
   → If pot = 2BB, bet = 1BB
   → EV = 0.7*2 - 0.3*1 = 1.1 BB per hand
   → 1.1 BB/hand * 100 = 110 BB/100 ❌ (too high)

   In practice:
   - Not every hand can be exploited
   - GTO won't fold 70% (more like 30-40%)
   - Must consider position, hand strength, etc.
   - Actual EV = 3-5 BB/100 ✓
```

## Practical Recommendations

### Against GTO Players
1. **Observe at least 200 hands** before exploiting
2. **Track key frequencies**: fold%, c-bet%, 3-bet%
3. **Analyze by position**, don't mix them together
4. **Make small adjustments**, don't over-exploit
5. **Maintain sample size** awareness

### Training Targets
```
Checkpoint 1: 500 hands, EV > 1 BB/100
Checkpoint 2: 1000 hands, EV > 2 BB/100
Checkpoint 3: 2000 hands, EV > 3 BB/100 ✓
Final: 5000+ hands, stable 3+ BB/100
```

## Further Reading

### Classic GTO Resources
- **"The Mathematics of Poker"** by Bill Chen & Jerrod Ankenman
- **"Applications of No-Limit Hold'em"** by Matthew Janda
- **"Modern Poker Theory"** by Michael Acevedo

### GTO Solvers
- **PioSOLVER** - Most popular GTO solver
- **GTO+** - Simple and easy to use
- **MonkerSolver** - Feature-rich and powerful

### Theoretical Extensions
- **Counterfactual Regret Minimization (CFR)** - GTO algorithm
- **Nash Equilibrium in Extensive-Form Games**
- **Exploitability Measurement**

---

## Summary

```
┌─────────────────────────────────────────────────────┐
│                   GTO vs Exploitative               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  GTO:                                              │
│  ✓ Safe, unbeatable                               │
│  ✗ Low profit (0-2 BB/100)                        │
│                                                     │
│  Exploitative (our approach):                      │
│  ✓ High profit (3-5+ BB/100)                      │
│  ✓ vs GTO: still safe (they won't counter)       │
│  ✗ vs adaptive opponents: risky                   │
│                                                     │
│  Best of both worlds:                              │
│  → Start from GTO foundation                       │
│  → Learn to exploit specific opponents             │
│  → Fall back to GTO vs unknowns                   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

Our goal is to train an agent that can identify fixed patterns in GTO players and achieve stable 3 BB/100 profit through precise exploitation. This leverages the theoretical foundation of GTO while surpassing its conservative strategy.
