## 🃏 [UNO Agent Arena](https://uno-matchup.streamlit.app/)


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
**Final Rankings**

| Rank | Agent              | Summary                                                               |
| ---- | ------------------ | --------------------------------------------------------------------- |
| 🥇 1 | MCTS Agent        | Dominates every agent with the highest win margins and shortest games |
| 🥈 2 | Rule Agent      | Strong consistent heuristic player; beats RL, Random, and Anti-Rule   |
| 🥉 3 | RL Bot          | Better than Random overall, but still weaker than Rule and MCTS       |
| 4    | Random Agent      | Surprisingly competitive against Anti-Rule                            |
| 5    | Anti-Rule Agent | Intentionally poor strategy; weakest overall                     |

**Insights**

- MCTS Agent emerged as the strongest overall agent, consistently outperforming every other strategy with the highest win rates and shortest average game lengths, suggesting more decisive and efficient play.
- The Rule Agent proved surprisingly competitive, outperforming the RL agent and serving as a strong handcrafted baseline against all opponents.
- The RL Agent successfully learned strategies better than random play, but still struggled against stronger heuristic and search-based methods.
- Random Agent beats Anti-Rule Agent outright in both player-order configurations (53.7% and 51.6%), confirming that intentionally poor strategy performs worse than pure chance.
- No significant player-order advantage was observed, suggesting the UNO environment remains fairly balanced despite stochastic card draws and random game dynamics.
  
---

**Matchup Results (10,000 games each)**

| Match | Games | P0 Wins | P1 Wins | Draws | Avg Turns | Diff | Winner |
|---|---:|---:|---:|---:|---:|---:|:---:|
| Random vs Random | 10,000 | 50.2% | 49.8% | 0 | 62.6 | 0.4% | Draw |
| Rule Agent vs Rule Agent | 10,000 | 52.1% | 47.9% | 1 | 60.8 | 4.2% | Draw |
| Rule Agent vs Random | 10,000 | **56.4%** | 43.6% | 0 | 60.2 | 12.8% |  Rule Agent |
| Random vs Rule Agent | 10,000 | 45.2% | **54.8%** | 0 | 61.0 | 9.6% |  Rule Agent |
| Rule Agent vs Anti-Rule Agent | 10,000 | **59.1%** | 40.9% | 0 | 59.2 | 18.2% |  Rule Agent |
| Anti-Rule Agent vs Rule Agent | 10,000 | 43.9% | **56.1%** | 0 | 59.2 | 12.2% |  Rule Agent |
| Anti-Rule Agent vs Random | 10,000 | 48.4% | **51.6%** | 0 | 59.8 | 3.2% |  Random |
| Random vs Anti-Rule Agent | 10,000 | **53.7%** | 46.3% | 1 | 60.0 | 7.4% |  Random |
| MCTS vs Random | 10,000 | **65.0%** | 35.0% | 0 | 49.1 | 30.0% |  MCTS |
| Random vs MCTS | 10,000 | 42.7% | **57.3%** | 0 | 48.1 | 14.6% |  MCTS |
| MCTS vs Rule Agent | 10,000 | **59.2%** | 40.8% | 0 | 47.7 | 18.4% |  MCTS |
| Rule Agent vs MCTS | 10,000 | 42.7% | **57.3%** | 0 | 48.1 | 14.6% |  MCTS |
| MCTS vs RL | 10,000 | **63.5%** | 36.5% | 0 | 46.6 | 27.0% |  MCTS |
| RL vs MCTS | 10,000 | 40.0% | **60.0%** | 0 | 49.8 | 20.0% |  MCTS |
| Random vs RL | 10,000 | 47.8% | **52.2%** | 0 | 62.4 | 4.4% |  RL |
| RL vs Random | 10,000 | **55.5%** | 44.5% | 0 | 61.3 | 11.0% |  RL |
| Rule Agent vs RL | 10,000 | **53.0%** | 47.0% | 0 | 62.7 | 6.0% |  Rule Agent |
| RL vs Rule Agent | 10,000 | 47.1% | **52.9%** | 0 | 57.7 | 5.8% |  Rule Agent |
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

**Project Structure**

```text
agents/             Agent implementations
core/               UNO engine and game logic
data/               Models, logs, datasets
tests/              Unit tests
app.py              Streamlit UI
matchups.py         Tournament runner
```

---

**References**

- [Rules](https://www.unorules.com/)
- [UNO (card game). *Wikipedia*.](https://en.wikipedia.org/wiki/Uno_(card_game))
- [Winning UNO With Reinforcement Learning. *Stanford University AA228 Final Report* (2020).](https://web.stanford.edu/class/aa228/reports/2020/final79.pdf)
- [Reinforcement learning compared to rule-based play in UNO. *DiVA Portal*.](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1955734&dswid=-5242)
- [Demaine, E. D., Demaine, M. L., Uehara, R., Uno, T., & Uno, Y. (2010). The Complexity of UNO.](https://www.researchgate.net/publication/45906238_The_complexity_of_UNO)
- [Cowling, P. I., Powley, E. J., & Whitehouse, D. (2012). Information Set Monte Carlo Tree Search. *IEEE TCIAIG*, 4(2), 120–143.](https://www.researchgate.net/publication/254060888_Information_Set_Monte_Carlo_Tree_Search)
- [Kocsis, L., & Szepesvári, C. (2006). Bandit Based Monte-Carlo Planning. *ECML 2006*, LNCS 4212, pp. 282–293.](https://www.researchgate.net/publication/221112399_Bandit_Based_Monte-Carlo_Planning)
- [Strategies](https://www.unorules.com/best-strategies-to-win-uno/)
  
