# Safe Multi-Agent Poker with GTO Exploitation

## Motivation

Most poker AIs — such as PioSOLVER — focus on heads-up GTO convergence.  This project moves beyond GTO solvers by applying multi-agent reinforcement learning (MARL) to model adaptive, exploitative play in multi-player settings, aiming for +3 BB/100 EV against near-GTO opponents.


---

##  Quick Start

```bash
# Clone repository
git clone https://github.com/kevinlmf/Safe_Multi_Agent_Poker
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

### 4. **Toward Realistic Competitive Environments**
Model elite tournaments (e.g., Triton London) where most participants adhere to near-GTO strategies, enabling the study of adaptive counter-strategies and meta-level reasoning in realistic high-stakes settings.
 

---

## References

This project implements techniques from:

- Li, Zeyang, and Navid Azizan. *Safe Multi-Agent Reinforcement Learning with Convergence to Generalized Nash Equilibrium.* arXiv preprint arXiv:2411.15036, 2024.  
  🔗 [https://arxiv.org/abs/2411.15036](https://arxiv.org/abs/2411.15036)






---



