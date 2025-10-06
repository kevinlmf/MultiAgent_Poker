# Safe Multi-Agent Poker with GTO Exploitation

## Project Goal

Train a poker AI to achieve an expected value of **3 BB/100 hands** against **GTO players**.

---

## Completed Features

### 1. **Complete Training System**
- PyTorch implementation (avoiding JAX macOS issues)
- MADAC algorithm (Multi-Agent Dual Actor-Critic)
- GTO baseline opponents
- Exploitative learning

### 2. **EV Evaluation System**
- Real-time BB/100 tracking
- Sharpe ratio calculation
- Maximum drawdown monitoring
- Position analysis
- Automatic convergence detection

### 3. **Action Logging**
- CSV format export
- Detailed action recording
- GTO vs Learning comparison
- Strategy analysis tools

### 4. **Theory Documentation**
- GTO explanation
- Exploitative strategy description
- Training guide

---

## Quick Start

```bash
# Test run (recommended to run this first)
python training/train_exploitative.py \
  --n_players 3 \
  --max_episodes 100 \
  --eval_interval 50 \
  --eval_hands 100 \
  --target_ev 2.0
```

---


