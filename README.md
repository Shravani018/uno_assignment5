## 🃏 [UNO Agent Benchmark](https://uno-matchup.streamlit.app/)


A 2-player UNO engine for comparing agent strategies: rule-based, tree search (MCTS), and deep reinforcement learning. 

---

**Agents**

| Agent | Description |
|---|---|
| **Random Agent** | Plays any legal card at random. Performance floor for every other agent must beat this. |
| **Rule Agent** | Greedy heuristics with threat detection. Primary baseline for all stronger agents. |
| **Anti-Rule Agent** | Deliberately anti-optimal. Inverts every RuleAgent decision. Used as a validation sanity check. |
| **RL Agent** | DQN agent trained via self-play. Scores each valid card independently with a 50-input MLP (state 54 + card 21 floats). Wild color selection falls back to the RuleAgent heuristic. |
| **MCTS Agent** | Information Set MCTS (SO-ISMCTS, Cowling et al. 2012). Re-samples the opponent hand each iteration. UCB1 uses an `availability` count as the denominator rather than parent visits. |

---

**Benchmark Results (10,000 games each)**

| Matchup | P0 Wins | P1 Wins |
|---|---:|---:|
| Random vs Random | 50.2% | 49.8% |
| Rule vs Random | **56.4%** | 43.6% |
| Rule vs Rule | 52.1% | 47.9% |
| Rule vs Rule Bad | **59.1%** | 40.9% |
| MCTS vs Random | **65.0%** | 35.0% |
| MCTS vs Rule | **59.2%** | 40.8% |
| MCTS vs RL | **63.5%** | 36.5% |

---

**How to Run**

```bash

# Run matchups 
python main.py

# Launch Streamlit UI (Human vs any agent)
streamlit run app.py

# Run tests
python -m unittest discover tests/ -v

```

---

**References**

- Cowling, P. I., Powley, E. J., & Whitehouse, D. (2012). Information Set Monte Carlo Tree Search. *IEEE TCIAIG*, 4(2), 120–143.
- Kocsis, L., & Szepesvári, C. (2006). Bandit Based Monte-Carlo Planning. *ECML 2006*, LNCS 4212, pp. 282–293.
- [Rules](https://www.unorules.com/​)
- [Strategies](https://www.unorules.com/best-strategies-to-win-uno/)
