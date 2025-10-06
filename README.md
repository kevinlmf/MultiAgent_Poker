# Safe Multi-Agent Poker with GTO Exploitation

This repository implements a **multi-agent reinforcement learning system** for **exploitative poker AI** using **PyTorch**.
The goal is to train agents that achieve **3 BB/100 hands expected value** against **GTO (Game Theory Optimal) players** through adaptive exploitation.

The system demonstrates a **research-grade poker AI architecture** with:
- MADAC (Multi-Agent Dual Actor-Critic) algorithm
- GTO baseline opponent modeling
- Real-time EV tracking with BB/100 metrics
- Comprehensive performance evaluation (Sharpe ratio, drawdown analysis)
- Detailed action logging for strategy analysis

---

##  Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/Safe_Multi_Agent_Poker
cd Safe_Multi_Agent_Poker

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run training (recommended test run first)
python training/train_exploitative.py \
  --n_players 3 \
  --max_episodes 100 \
  --eval_interval 50 \
  --eval_hands 100 \
  --target_ev 2.0

# Analyze results
# Training logs and action CSVs are stored in logs/
```

---

## 📁 Project Structure

```
Safe_Multi_Agent_Poker/
├── agents/                   # RL agents
│   ├── madac_agent.py            # MADAC algorithm implementation
│   └── gto_agent.py              # GTO baseline opponent
│
├── environment/              # Poker environment
│   ├── poker_env.py              # Multi-agent poker game logic
│   ├── hand_evaluator.py         # Hand ranking and comparison
│   └── game_state.py             # Game state representation
│
├── evaluation/               # Performance evaluation
│   ├── ev_calculator.py          # EV and BB/100 tracking
│   ├── metrics.py                # Sharpe ratio, drawdown, position analysis
│   └── convergence_detector.py   # Automatic convergence detection
│
├── training/                 # Training scripts
│   ├── train_exploitative.py     # Main exploitative training
│   └── train_gto.py              # GTO baseline training
│
├── logging/                  # Action logging
│   ├── action_logger.py          # CSV export for actions
│   └── strategy_analyzer.py      # GTO vs Learning comparison
│
├── docs/                     # Documentation
│   ├── gto_theory.md             # GTO explanation
│   ├── exploitative_strategy.md  # Exploitation techniques
│   └── training_guide.md         # Training best practices
│
├── logs/                     # Training outputs
│   ├── actions/                  # Action CSV files
│   └── checkpoints/              # Model checkpoints
│
├── requirements.txt          # Python dependencies
└── README.md
```

---

## Current Implementation

### 1. **Multi-Agent Training System**
- PyTorch-based implementation
- MADAC (Multi-Agent Dual Actor-Critic) algorithm
- GTO baseline opponents for exploitation learning
- Self-play and opponent modeling

### 2. **EV Evaluation Framework**
- Real-time BB/100 (Big Blinds per 100 hands) tracking
- Sharpe ratio calculation for risk-adjusted performance
- Maximum drawdown monitoring
- Position-based analysis (BTN, SB, BB)
- Automatic convergence detection

### 3. **Action Logging & Analysis**
- CSV format export for all actions
- Detailed game state recording
- GTO vs Exploitative strategy comparison
- Statistical analysis tools

### 4. **Theory & Documentation**
- Comprehensive GTO theory explanation
- Exploitative strategy techniques
- Training guide and best practices

---

##  Future Directions

### 1. **JAX Integration**
- Migrate from PyTorch to JAX for improved computational efficiency
- Leverage hardware acceleration (GPU/TPU) and automatic differentiation
- Reduce training time through optimized tensor operations
- Enable larger-scale multi-agent experiments

### 2. **Representation Learning**
- Deep feature extraction from poker game states
- Advanced information abstraction for hand ranges


### 3. **Extended Game Types**
- Support for different poker variants (Hold'em, Omaha, Stud)
- Variable player counts (2-10 players) and stack sizes
- Tournament structures and ICM (Independent Chip Model) considerations
- Multi-table tournament (MTT) scenarios
- Cash game vs tournament dynamics

---

##  References

This project implements techniques from:

**MADAC (Multi-Agent Dual Actor-Critic)**:
Mao et al. (2020). Multi-Agent Dual Actor-Critic for Cooperative Multi-Agent Reinforcement Learning.

**GTO Poker Theory**:
Mathematics of Poker by Bill Chen and Jerrod Ankenman.

**Exploitative Play**:
Brown & Sandholm (2019). Superhuman AI for multiplayer poker. Science, 365(6456).

---



