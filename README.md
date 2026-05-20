## UNO Agents

A 2-player UNO game engine built to compare agent strategies: rule-based, tree search (MCTS), and reinforcement learning. 
The project generates its own training data via self-play.

---

**What Has Been Built**

**Game Engine**
A fully rule-compliant 2-player UNO engine. All game logic lives in one place. Agents interact with the engine through a clean interface -- they receive a state snapshot and return a card to play. No agent can access opponent cards or manipulate the engine directly.

**Agents**
- **RandomAgent** - plays any legal card at random. Serves as the performance floor. Every other agent must beat this to demonstrate any strategic value.
- **RuleAgent** - plays greedily using hand-coded heuristics. The primary baseline all stronger agents are measured against.
- **MCTSAgent** - uses Information Set MCTS (ISMCTS) with UCB1, following the SO-ISMCTS algorithm (Cowling et al., 2012). Each simulation re-samples the opponent hand independently to handle partial observability, building a single shared tree across all samples (Single Observer variant). Nodes track an `availability` count (times the node was reachable across samples) used as the UCB1 denominator instead of parent visit count — this availability normalisation is the key distinction of ISMCTS over standard UCT. The Single Observer variant was chosen over Multi-Observer ISMCTS (also described in Cowling et al.) because re-sampling the opponent hand each iteration already captures the main source of uncertainty in 2-player UNO without the added complexity of maintaining separate per-player trees. Defaults: 100 simulations per decision, exploration constant c=1.41 (≈√2, the theoretical optimum from Kocsis & Szepesvári 2006), rollout depth 150.
- **RL** - todo

**Self-Play Pipeline**
Runs any two agents against each other for N games, logging every turn as a `(state, action, reward)` tuple to a JSONL dataset file. Used to generate training data for future RL training.

**Feature Encoder**
Converts a game state into a 29-length float vector for use in ML models. One-hot encodes top card color, type, and value. Normalises hand sizes and draw pile count to [0, 1].

---

**UNO Rules Implemented**

**A card can be played if:**
- It shares the active color with the top card
- It shares the number or symbol with the top card
- It is a Wild or Wild Draw 4 (always playable)

**If no valid play exists:**
- The player draws one card and their turn ends

**Special card effects:**

| Card | Effect |
|---|---|
| Skip | Opponent loses their next turn |
| Reverse | Acts as Skip in 2-player mode |
| Draw Two (+2) | Opponent draws 2 cards and loses their turn |
| Wild | Player chooses the new active color |
| Wild Draw 4 (+4) | Opponent draws 4 cards and loses their turn; player chooses new color |

**Starting card rule:**
The first card placed on the discard pile must be a plain colored number card. 
Wild, Skip, Reverse, Draw Two, and Wild Draw 4 are rejected and returned to the deck until a valid starting card is found.

**Win condition:**
First player to empty their hand wins. If 500 turns pass without a winner the game is recorded as a draw.

---

**RuleAgent Strategy**

RuleAgent uses a two-phase decision process each turn.

**Phase 1 - Threat detection**

If the opponent holds 2 or fewer cards, threat mode activates. The agent plays its strongest card immediately without conserving wilds.

**Phase 2 - Normal play**

Outside threat mode the agent avoids spending wilds unnecessarily, working through this priority ladder on non-wild cards first:

```
Wild Draw 4 > Draw Two > Skip > Reverse > highest number
```

If no non-wild card is available, it falls through to wilds:

```
Wild Draw 4 > Wild
```

**Wild color selection**

When a wild is played, the agent counts the colors in its current valid plays and picks the most frequent. Falls back to red if all available cards are wild.

---

**Benchmark Order**

Run matchups in this order. Each step answers a specific question before moving to the next.

```
Step 1 -- Random vs Random      Verify engine fairness (~50/50 expected)
Step 2 -- Rule vs Random        Measure how much heuristics gain over random play
Step 3 -- Rule vs Rule          Confirm baseline parity (~50/50 expected)
Step 4 -- MCTS vs Random        Confirm tree search dominates random 
Step 5 -- MCTS vs Rule          Key test: does lookahead beat heuristics
Step 6 -- RL vs Random          Confirm RL dominates random
Step 7 -- RL vs Rule            Key test: does learned policy beat heuristics
Step 8 -- RL vs MCTS            Final comparison: best vs best
```

If an agent barely beats Random there is no point running it against stronger opponents.

---

**Results So Far**

**Self-Play Evaluation Results**

| Matchup | Games | P0 Wins | P1 Wins | Draws | Avg Turns |
|---|---:|---:|---:|---:|---:|
| Random vs Random | 10,000 | 50.2% | 49.8% | 0 | 62.6 |
| Rule vs Random | 10,000 | **56.4%** | 43.6% | 0 | 60.2 |
| Random vs Rule | 10,000 | 45.2% | **54.8%** | 0 | 61.0 |
| Rule vs Rule | 10,000 | 52.1% | 47.9% | 1 | 60.8 |
| Rule Good vs Rule Bad | 10,000 | **59.1%** | 40.9% | 0 | 59.2 |
| Rule Bad vs Rule Good | 10,000 | 43.9% | **56.1%** | 0 | 59.2 |
| Rule Bad vs Random | 10,000 |48.4% |**51.6%** | 0| 59.8|
| Random vs Rule Bad | 10,000 | **53.7%** |46.3% | 1|60.0 |
| MCTS vs Random | 10,000 | **65%** | 35% | 0 | 49.1 |
| MCTS vs Rule | 10,000 | **59.2%** | 40.8% | 0 | 47.7 |
| MCTS vs RL | 10,000 |**63.5%** |36.5% |0 |46.6 |
| Random vs MCTS | 10,000 | 42.7% | **57.3%** | 0 | 48.1 |
| Rule vs MCTS | 10,000 | 42.7% | **57.3%** | 0 | 48.1 |
| RL vs MCTS | 10,000 | 40% | **60%** | 0 | 49.8 |


*MCTS parameters: `num_simulations=200`, `c=1.0`, `rollout_depth=20`. Position-averaged win rates: 64.1% vs Random, 58.8% vs Rule.*

**MCTS Parameter Tuning**

A one-at-a-time sensitivity analysis was run over three key SO-ISMCTS hyperparameters (500 games per configuration vs RuleAgent). Results are summarised in `experiments/results/mcts_param_sweep.png` and detailed in `experiments/parameter_analysis.md`.

| Parameter | Default | Tuned | Key finding |
|---|---|---|---|
| `c` | 1.41 | **1.0** | Performance robust across c ∈ [0.7, 2.0]; lower optimum consistent with ISMCTS availability inflation |
| `num_simulations` | 100 | **200** | Most sensitive parameter; monotonic gain from 51.6% (n=25) to 62.0% (n=500); n=200 chosen as compute budget |
| `rollout_depth` | 150 | **20** | Short rollout + hand-size heuristic outperforms long random playout; peak at depth=20 |
| `rollout_policy` | random | **random** | Rule-based rollout (56.6%) marginally worse than random (58.4%); difference within noise |

**Verdict**

- **Engine is fair.** Random vs Random lands at 50.2% vs 49.8%, well within noise. No positional bias exists in the engine.

- **RuleAgent holds a genuine skill edge over Random.** Winning 56.4% going first and 54.8% going second, the 1.6 point gap between positions is within the margin of error. The honest conclusion is RuleAgent wins approximately 55-56% against RandomAgent regardless of who goes first. This is a real strategy advantage, not a positional artifact.

- **First-mover advantage exists but is small.** Rule vs Rule gives the first player 52.1% with identical agents on both sides. This isolates position as worth roughly 2 percentage points. It is statistically real at this sample size but practically minor.

- **RuleGood convincingly beats RuleBad.** Winning 59.1% going first and 56.1% going second, the average skill edge is approximately 57-58% across both positions. The 3 point gap between positions is larger than in any other matchup, suggesting the bad strategy is especially vulnerable to being exploited by whichever player acts first.

- **RuleBad performs worse than Random.** RandomAgent wins 43.6-45.2% against RuleAgent. RuleBad wins only 40.9-43.9% against RuleGood. The confidence intervals barely touch, meaning actively bad strategy is measurably worse than random play when facing a competent opponent. This validates the bad agent design: it is not just different from RuleAgent, it is genuinely inferior.

- **MCTSAgent convincingly beats both baselines across both positions.** Averaged over first and second player, MCTSAgent wins 64.1% against RandomAgent and 58.8% against RuleAgent. The win rate is stable across positions (64.2% vs 64.0% against Random; 59.9% vs 57.7% against Rule), confirming the advantage is not a positional artifact. The 2.2 percentage-point gap vs Rule is consistent with the ~2-point first-mover advantage observed in Rule vs Rule, leaving the position-corrected skill edge intact.

**Baselines for RL.** Any new agent must exceed 64% against RandomAgent and 59% against RuleAgent (position-averaged) before a result is considered a genuine strategic improvement over MCTSAgent.

---

**How to Run**

**Play against RuleAgent**

```bash
python play_human.py
```

Your full hand is displayed each turn. Cards marked `[* N]` are playable -- enter `N` to play. Cards marked `[ ]` cannot be played this turn. Enter `d` to draw.

**Example turn display:**

```
=======================================================
  Turn 4  --  YOUR TURN
-------------------------------------------------------
  Top card    : red 5
  Active color: RED
  RuleAgent holds 5 card(s)
-------------------------------------------------------
  YOUR HAND:

    [* 0]  red 7
    [   ]  blue 3  (not playable)
    [* 1]  red skip
    [   ]  green 9  (not playable)
    [* 2]  wild

  Playable cards marked [* N] -- enter N to play.
=======================================================
  Select [0-2] or 'd' to draw:
```

**Run self-play and generate dataset**

```bash
python main.py
```

Runs all three current matchups (Random vs Random, Rule vs Random, Rule vs Rule) and saves the Rule vs Rule dataset to `data/datasets/`.


**Run tests**

```bash
python -m unittest discover tests/ -v
```

Runs 47 tests covering deck composition, card validity rules, special card effects, and agent behaviour. All must pass before any new agent is added.

**User Interface**
```bash
streamlit run app.py
```
*WIP*: Player 1 is always **Human** and Player 2 can be either of the agents **(Greedy, Anti-Greedy, MCTS, RL)**

---

**References**

- Cowling, P. I., Powley, E. J., & Whitehouse, D. (2012). Information Set Monte Carlo Tree Search. *IEEE Transactions on Computational Intelligence and AI in Games*, 4(2), 120–143. https://doi.org/10.1109/TCIAIG.2012.2200894

- Kocsis, L., & Szepesvári, C. (2006). Bandit Based Monte-Carlo Planning. In *Proceedings of the 17th European Conference on Machine Learning (ECML 2006)*, Lecture Notes in Computer Science, vol. 4212, pp. 282–293.

---

**Dataset Format**

Each row in `data/datasets/rule-a_vs_rule-b.jsonl` represents one turn:

```json
{
  "player": 0,
  "state": [1.0, 0.0, 0.0, ...],
  "action_index": 0,
  "reward": 1.0,
  "next_state": [1.0, 0.0, 0.0, ...],
  "done": false
}
```

| Field | Description |
|---|---|
| `player` | Which player took this turn (0 or 1) |
| `state` | 29-length encoded game state vector |
| `action_index` | Index of the card played within `valid_plays` |
| `reward` | +1.0 win, -1.0 loss, -0.5 draw (assigned at game end) |
| `next_state` | Encoded state after the turn, `null` on terminal turns |
| `done` | True on the final turn of the game |
