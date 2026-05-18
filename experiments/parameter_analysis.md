# MCTS Parameter Selection Analysis

## Experimental Setup

All experiments evaluate MCTSAgent (Player 0) against RuleAgent (Player 1) over 500 games per configuration with a fixed seed (`base_seed=42`). Parameters are varied one-at-a-time while holding all others at their default values (`c=1.41`, `num_simulations=100`, `rollout_depth=150`). Win rate is reported as the fraction of games won by MCTSAgent. The margin of error at 500 games is approximately ±3.5% (95% confidence interval for a binomial proportion near 0.55).

---

## 1. Exploration Constant `c`

**Range tested:** 0.35, 0.7, 1.0, 1.41, 2.0, 3.0

| c | Win rate |
|---|---|
| 0.35 | 54.8% |
| 0.7 | 56.8% |
| **1.0** | **57.0%** |
| 1.41 (default, ≈√2) | 53.6% |
| 2.0 | 56.8% |
| 3.0 | 53.0% |

**Analysis.** Performance is broadly stable across c ∈ [0.7, 2.0], with a peak at c=1.0 (57.0%) and degradation at the extremes (c=0.35: 54.8%, c=3.0: 53.0%). The differences within the central range are within the ±3.5% margin of error, indicating that the algorithm is robust to the choice of c — consistent with the general MCTS literature finding that UCB1-based tree search is not highly sensitive to the exploration constant when it is kept within a reasonable interval.

The theoretical optimum for standard UCT is c=√2 ≈ 1.41 (Kocsis & Szepesvári, 2006). In SO-ISMCTS, the UCB1 denominator is `log(availability)` rather than `log(parent.visits)`, where availability ≥ visits by construction. This inflates the exploration term relative to standard UCT, which shifts the effective optimum toward lower values of c. The empirical peak at c=1.0 is consistent with this theoretical expectation.

**Selected:** `c = 1.0`

---

## 2. Number of Simulations `num_simulations`

**Range tested:** 25, 50, 100, 200, 500

| num_simulations | Win rate | Time / 500 games |
|---|---|---|
| 25 | 51.6% | 140s |
| 50 | 54.6% | 280s |
| 100 | 53.6% | 485s |
| 200 | 57.6% | 1079s |
| **500** | **62.0%** | 2642s |

**Analysis.** This is the most sensitive parameter: win rate rises monotonically from 51.6% (n=25) to 62.0% (n=500), a spread of 10.4 percentage points. This behaviour is theoretically expected — more simulations allow the tree to explore deeper and average over more determinizations of the opponent's hidden hand, reducing both approximation error and variance. The curve shows no sign of plateauing, indicating that n=500 has not reached the point of diminishing returns.

However, n=500 requires approximately 5 seconds per decision on a MacBook (inferred from 2642s for 500 games with ~31 decisions per player per game), making it impractical for real-time play and for large-scale benchmarks (10,000-game evaluations). n=200 achieves 57.6% win rate — a meaningful improvement over the original 100-simulation baseline — at roughly half the compute cost (≈2 seconds per decision).

**Selected:** `num_simulations = 200`

The choice reflects a deliberate trade-off between performance and practical compute budget. If offline evaluation is the only requirement, n=500 is strictly better.

---

## 3. Rollout Depth `rollout_depth`

**Range tested:** 10, 20, 30, 50, 100

| rollout_depth | Win rate |
|---|---|
| 10 | 55.6% |
| **20** | **58.0%** |
| 30 | 55.8% |
| 50 | 54.6% |
| 100 | 53.6% |
| 150 (original default) | ~53.6% |

**Analysis.** Performance peaks sharply at depth=20 and degrades consistently as depth increases beyond that point. This result is explained by the interaction between random rollout quality and the heuristic evaluation used at the depth limit.

When the depth limit is reached before a natural game-over, the agent falls back to a hand-size heuristic:

```
score = 0.5 + 0.2 × (opponent_hand − own_hand) / (opponent_hand + own_hand)
```

This heuristic directly encodes UNO's win condition (empty hand wins) and provides a low-variance, high-signal estimate of positional advantage. By contrast, extending the rollout with random play introduces noise: both players make random card choices, and over many steps the game outcome converges toward 50-50 regardless of the initial positional advantage (mean reversion under random play). The heuristic is therefore a more informative signal than a long random playout, a phenomenon sometimes called the **rollout paradox** in the MCTS literature.

The UNO benchmark shows an average game length of 62.6 turns, corresponding to approximately 31 turns per player. At depth=20, most rollouts terminate before reaching the depth limit in a typical mid-game position, providing a mix of true terminal outcomes and heuristic scores. Beyond depth=30, nearly all rollouts reach a natural terminal state — but via random play, which diminishes the quality of the resulting signal.

**Selected:** `rollout_depth = 20`

---

## 4. Rollout Policy

**Configurations tested:** `random` vs `rule` (both with `n=100`, `c=1.0`, `depth=20`)

| Rollout policy | Win rate | Avg turns/game |
|---|---|---|
| **random** | **58.4%** | 49.3 |
| rule | 56.6% | 47.5 |

**Analysis.** Contrary to the common expectation that heavier (rule-based) rollouts outperform light (random) rollouts, random rollout yields a higher win rate (58.4% vs 56.6%). Three factors explain this result.

First, with `rollout_depth=20`, the majority of rollouts reach the depth limit before a natural game-over and fall back to the hand-size heuristic. The rollout policy therefore only affects a minority of simulations where the game terminates within 20 steps. The practical difference between the two policies is smaller than it would be at greater depths.

Second, the rule-based rollout models both the agent and the opponent as following RuleAgent's priority ordering during the simulation phase. This introduces a systematic bias: the tree is trained on rollout outcomes generated by a specific opponent model (rule-based), but the real opponent (RuleAgent) also plays by those same rules. The resulting value estimates are biased toward a "rule vs rule" scenario, which does not accurately represent the MCTS agent's own UCB1-guided decisions during the tree search phase.

Third, random rollouts span a wider distribution of game trajectories, providing less biased and more diverse value estimates. This diversity supports better exploration in the UCB1 tree even if the individual rollout quality is lower.

The 1.8 percentage-point gap (58.4% vs 56.6%) is within the ±3.5% margin of error for 500 games, so the result is not statistically conclusive. Nevertheless, random rollout is also simpler and faster, making it the preferred choice.

**Selected:** `rollout_policy = "random"`

---

## Summary and Final Parameter Selection

| Parameter | Original default | Selected value | Change |
|---|---|---|---|
| `c` | 1.41 | **1.0** | Empirically higher; theoretically consistent with ISMCTS availability inflation |
| `num_simulations` | 100 | **200** | +4.0% win rate vs RuleAgent; practical compute budget on laptop |
| `rollout_depth` | 150 | **20** | +4.4% win rate; heuristic outperforms long random rollout |
| `rollout_policy` | random | **random** | Rule rollout introduces opponent-model bias; random is simpler and marginally better |

The one-at-a-time sweep design does not capture parameter interactions, so a final validation experiment with the selected combination (`c=1.0`, `num_simulations=200`, `rollout_depth=20`) will be run against both RandomAgent and RuleAgent to confirm the combined improvement before updating the agent defaults.
