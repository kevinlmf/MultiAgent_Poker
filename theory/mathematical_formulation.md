# Mathematical Formulation: Safe Multi-Agent Poker with GNE

## 1. Problem Statement

We extend single-game poker equilibrium computation (e.g., CFR, Pluribus) to a **multi-game, bankroll-constrained framework** where agents must maintain safety constraints across episodes while converging to a Generalized Nash Equilibrium (GNE).

---

## 2. Core Definitions

### 2.1 Multi-Game Poker as Constrained Markov Game

Define the system as:

$$\mathcal{M} = \langle N, \mathcal{X}, \{\mathcal{U}_i\}_{i \in N}, f, \{r_i\}_{i \in N}, \{h_i\}_{i \in N}, \gamma \rangle$$

Where:

- **$N = \{1, 2, ..., n\}$**: Set of players
- **$\mathcal{X} = \mathcal{B} \times \mathcal{S} \times \mathcal{P}$**: State space
  - $\mathcal{B} \subset \mathbb{R}^n$: Bankroll states $b = (b_1, ..., b_n)$
  - $\mathcal{S}$: Hidden card states (private information)
  - $\mathcal{P}$: Public board states (community cards, pot size)

- **$\mathcal{U}_i$**: Action space for player $i$
  - Discrete: $\{\text{fold}, \text{call}, \text{raise}, \text{all-in}\}$
  - Or continuous: $\{\text{fold}, \text{bet}(a)\}$ where $a \in [0, b_i]$

- **$f: \mathcal{X} \times \mathcal{U} \to \mathcal{X}$**: State transition
  $$b_i^{t+1} = b_i^t + \Delta r_i^t(u^t, s^t)$$
  where $\Delta r_i^t$ is the net reward from hand $t$

- **$r_i: \mathcal{X} \times \mathcal{U} \to \mathbb{R}$**: Immediate reward (expected value per hand)

- **$h_i: \mathcal{X} \to \mathbb{R}$**: Safety constraint function
  $$h_i(x) = \begin{cases}
  b_i - b_{\min} & \text{(bankroll constraint)} \\
  \sigma_{\max}^2 - \text{Var}(\pi_i) & \text{(variance constraint)} \\
  \text{CVaR}_\alpha(b_i) - \text{threshold} & \text{(risk constraint)}
  \end{cases}$$

- **$\gamma \in (0,1)$**: Discount factor

---

### 2.2 Policy and Value Functions

**Policy**: $\pi_i: \mathcal{X} \times \mathcal{U}_i \to [0,1]$

**Task Value Function** (EV maximization):
$$V_i^{\pi}(x) = \mathbb{E}_{\pi}\left[\sum_{t=0}^\infty \gamma^t r_i(x^t, u^t) \mid x^0 = x\right]$$

**Safety Value Function** (constraint satisfaction):
$$V_i^h(x) = \mathbb{E}_{\pi}\left[\sum_{t=0}^\infty \gamma^t h_i(x^t) \mid x^0 = x\right]$$

---

## 3. Generalized Nash Equilibrium (GNE) Definition

A policy profile $\pi^* = (\pi_1^*, ..., \pi_n^*)$ is a **Generalized Nash Equilibrium** if:

1. **Best Response Property**:
   $$\forall i \in N, \forall x \in \mathcal{X}: \quad V_i^{\pi^*}(x) = \max_{\pi_i \in \Pi_i(x)} V_i^{(\pi_i, \pi_{-i}^*)}(x)$$

2. **Feasibility (Safety Constraint Satisfaction)**:
   $$\forall i \in N, \forall x \in \mathcal{X}: \quad V_i^h(x) \geq 0$$

3. **Coupled Constraint Set** (CIS):
   $$\Pi_i(x) = \{\pi_i : V_i^h(x) \geq 0, \text{ and } \mathbb{E}_{\pi_i}[b_j^{t+1}] \geq b_{\min}, \forall j \neq i\}$$

   This captures the key insight: **player $i$'s feasible policy depends on other players' bankroll states**.

---

## 4. Constrained Optimization Formulation

For each player $i$, solve:

$$
\begin{align}
\max_{\pi_i} \quad & V_i^{\pi}(x) \\
\text{s.t.} \quad & V_i^h(x) \geq 0 \quad \forall x \in \mathcal{X} \quad \text{(local safety)} \\
& \mathbb{E}_{\pi}[b_j^{t+1} \mid x] \geq b_{\min} \quad \forall j \neq i \quad \text{(coupled safety)}
\end{align}
$$

**Lagrangian Form**:
$$\mathcal{L}_i(\pi_i, \lambda_i) = V_i^{\pi}(x) - \lambda_i \cdot \left(-V_i^h(x)\right)$$

Where $\lambda_i \geq 0$ is the dual variable (risk-aversion parameter).

---

## 5. Key Theoretical Results (from Li & Azizan 2024)

### Theorem 1: Convergence to GNE
Under mild regularity conditions (Lipschitz continuity of $f, r, h$), the **Multi-Agent Dual Actor-Critic (MADAC)** algorithm converges to a GNE with:

- **Linear convergence rate**: $O(\gamma^k)$ for strongly monotone games
- **Sample complexity**: $\tilde{O}(\epsilon^{-2})$ to reach $\epsilon$-approximate GNE

### Theorem 2: Safety Guarantees
If initial states satisfy $V_i^h(x^0) \geq 0$ for all $i$, then:

$$\mathbb{P}(h_i(x^t) \geq 0, \forall t) \geq 1 - \delta$$

for any $\delta > 0$, with appropriate choice of safety value threshold.

---

## 6. Connection to Poker Theory

### 6.1 Single-Game Nash (CFR) vs Multi-Game GNE

| Aspect | CFR/DeepStack | Our GNE Framework |
|--------|---------------|-------------------|
| Equilibrium Concept | Nash Equilibrium | Generalized Nash Equilibrium |
| Constraint | None (fixed game tree) | State-dependent (bankroll, risk) |
| Objective | Min exploitability | Max long-term EV under safety |
| Information | Perfect recall | Imperfect + bankroll history |
| Convergence | Regret minimization | Dual policy gradient |

### 6.2 Bankroll Dynamics as Control-Invariant Set

The feasible region $\mathcal{C} = \{x \in \mathcal{X} : h(x) \geq 0\}$ forms a **control-invariant set**:

$$x \in \mathcal{C} \implies f(x, \pi^*(x)) \in \mathcal{C}$$

This ensures that once bankroll is above threshold, the GNE policy keeps it safe.

---

## 7. Exploitability under GNE

Define **GNE exploitability**:

$$\epsilon_{\text{GNE}}(\pi^*) = \max_{i \in N} \left[ \max_{\pi_i \in \Pi_i} V_i^{(\pi_i, \pi_{-i}^*)} - V_i^{\pi^*} \right]$$

subject to $\pi_i$ maintaining safety constraints.

**Key difference from CFR**: Best response must also respect bankroll constraints.

---

## 8. Open Research Questions

1. **Complexity**: What is the computational cost of finding GNE in large poker games (e.g., No-Limit Hold'em)?

2. **Approximation**: Can we use linear value approximation (as in Li & Azizan) for poker's combinatorial action space?

3. **Equilibrium Selection**: When multiple GNE exist, which one is "better" for poker?

4. **Population Dynamics**: How does GNE evolve when playing against non-equilibrium opponents (exploitative players)?

5. **Partial Observability**: How to extend to imperfect information games with hidden states?

---

## 9. Next Steps

1. Implement poker environment with bankroll state tracking
2. Adapt MADAC algorithm to poker action space
3. Compare GNE-based policy vs CFR-based policy on:
   - Long-term EV
   - Risk of ruin
   - Exploitability
4. Theoretical analysis of convergence rate in poker setting

---

## References

- Li, Z., & Azizan, N. (2024). "Safe Multi-Agent Reinforcement Learning with Convergence to Generalized Nash Equilibrium"
- Brown, N., & Sandholm, T. (2019). "Superhuman AI for multiplayer poker" (Pluribus)
- Zinkevich, M., et al. (2007). "Regret minimization in games with incomplete information" (CFR)
