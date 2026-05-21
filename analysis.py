"""
UNO Agent Performance Analysis
================================
Usage : python3 analysis.py
Output: printed statistics + charts saved to the results/ folder

Analyses performed
------------------
1. Win rates with 95% Wilson confidence intervals and binomial p-values
   (win/loss outcomes → Binomial test)
2. Surviving rounds with Mann-Whitney U test vs Random vs Random baseline
   (turn counts → Mann-Whitney U test)
3. First-mover advantage with binomial p-values (P0 vs P1 win rate)
   (win/loss outcomes → Binomial test)
4. Overall agent ranking by cumulative win rate across all standard matchups

MCTS is used as the near-GTO baseline for all comparisons.
"""

import sys
sys.path.insert(0, ".")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from agents.mcts_agent import MCTSAgent
from agents.rl_agent import RLAgent
from core.game import UnoGame
from config import Paths

# Output directory
os.makedirs("results", exist_ok=True)

# Set to 1000 for a quick test run; change to 10000 for final submission
NUM_GAMES = 10000


# ══════════════════════════════════════════════════════════════
# 1. Core: run games and collect per-game data
# ══════════════════════════════════════════════════════════════

def run_matchup_detailed(agent0, agent1, num_games=NUM_GAMES, base_seed=42):
    """Run num_games between agent0 and agent1 and collect per-game statistics.

    For each game records:
      - winner (0, 1, or -1 for draw/timeout)
      - total turn count
      - loser's surviving rounds (number of turns the losing agent acted)

    Args:
        agent0: First player agent (Player 0, moves first).
        agent1: Second player agent (Player 1).
        num_games: Number of games to simulate.
        base_seed: Starting seed for reproducibility (None disables seeding).

    Returns:
        dict with wins, draws, avg_turns, all_turns, loser_surviving lists.
    """
    wins = [0, 0]
    draws = 0
    all_turns = []
    loser_surviving = []

    for i in range(num_games):
        seed = base_seed + i if base_seed is not None else None
        game = UnoGame(agent0, agent1, seed=seed)
        winner, game_log = game.run()

        all_turns.append(game.turn_count)

        if winner == -1:
            draws += 1
        else:
            wins[winner] += 1
            loser = 1 - winner
            loser_turns = sum(1 for t in game_log if t["player"] == loser)
            loser_surviving.append(loser_turns)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{num_games} games done...")

    return {
        "wins":            wins,
        "draws":           draws,
        "total_games":     num_games,
        "avg_turns":       np.mean(all_turns),
        "all_turns":       all_turns,
        "loser_surviving": loser_surviving,
    }


# ══════════════════════════════════════════════════════════════
# 2. Run all matchups
# ══════════════════════════════════════════════════════════════

def _fresh_agent(name, rl_instance):
    """Return a fresh agent instance for the given name.

    Args:
        name: One of 'Random', 'Rule', 'MCTS', 'RL'.
        rl_instance: Pre-loaded RLAgent instance.

    Returns:
        Agent instance.
    """
    if name == "Random":
        return RandomAgent("Random")
    if name == "Rule":
        return RuleAgent("Rule")
    if name == "MCTS":
        return MCTSAgent("MCTS")
    if name == "RL":
        return rl_instance
    raise ValueError(f"Unknown agent name: {name}")


def run_all_matchups():
    """Load agents and run every benchmark matchup.

    MCTS is used as the near-GTO baseline.
    Standard matchups: MCTS vs all others + remaining pairwise comparisons.
    First-mover matchups: each agent as P0 then P1 vs MCTS baseline.

    Returns:
        List of result dicts, one per matchup.
    """
    rl = RLAgent("RL", epsilon=0.0)
    rl.load(str(Paths.MODELS / "rl_agent.pt"))

    # MCTS leads all baseline comparisons (P0 position)
    standard = [
        ("MCTS vs Random",   "MCTS",   "Random"),
        ("MCTS vs Rule",     "MCTS",   "Rule"),
        ("MCTS vs RL",       "MCTS",   "RL"),
        ("Random vs Random", "Random", "Random"),
        ("Rule vs Random",   "Rule",   "Random"),
        ("RL vs Random",     "RL",     "Random"),
        ("RL vs Rule",       "RL",     "Rule"),
    ]

    # First-mover: each agent swapped into P0 then P1 vs MCTS baseline
    first_mover = [
        ("Random as P0 vs MCTS", "Random", "MCTS"),
        ("Random as P1 vs MCTS", "MCTS",   "Random"),
        ("Rule as P0 vs MCTS",   "Rule",   "MCTS"),
        ("Rule as P1 vs MCTS",   "MCTS",   "Rule"),
        ("RL as P0 vs MCTS",     "RL",     "MCTS"),
        ("RL as P1 vs MCTS",     "MCTS",   "RL"),
    ]

    fm_labels = {x[0] for x in first_mover}
    results = []

    for label, a0_name, a1_name in standard + first_mover:
        a0 = _fresh_agent(a0_name, rl)
        a1 = _fresh_agent(a1_name, rl)
        print(f"\n{'=' * 55}\nRunning: {label}")
        r = run_matchup_detailed(a0, a1, num_games=NUM_GAMES)
        r["label"]    = label
        r["agent0"]   = a0_name
        r["agent1"]   = a1_name
        r["category"] = "first_mover" if label in fm_labels else "standard"
        ls = r["loser_surviving"]
        avg_surv = f"{np.mean(ls):.1f}" if ls else "N/A"
        print(f"  Done. W0={r['wins'][0]}  W1={r['wins'][1]}  "
              f"Avg turns={r['avg_turns']:.1f}  "
              f"Loser surviving={avg_surv}")
        results.append(r)

    return results


# ══════════════════════════════════════════════════════════════
# 3. Statistical analysis
# ══════════════════════════════════════════════════════════════

def _wilson_ci(wins, total, z=1.96):
    """Compute 95% Wilson confidence interval for a proportion.

    Args:
        wins: Number of successes.
        total: Total trials.
        z: Z-score for desired confidence level (default 1.96 = 95%).

    Returns:
        Tuple (ci_low_pct, ci_high_pct) as percentages (rounded to 1 dp).
    """
    p      = wins / total
    denom  = 1 + z ** 2 / total
    center = (p + z ** 2 / (2 * total)) / denom
    margin = (z * np.sqrt(p * (1 - p) / total
                           + z ** 2 / (4 * total ** 2))) / denom
    return (round(max(0.0, center - margin) * 100, 1),
            round(min(1.0, center + margin) * 100, 1))


def compute_stats(results):
    """Compute win rates, CIs, binomial p-values, and surviving rounds.

    Win/loss outcomes use Binomial test.
    Surviving rounds raw data stored for Mann-Whitney U test downstream.

    Args:
        results: List of matchup result dicts from run_all_matchups().

    Returns:
        pandas DataFrame with one row per matchup.
    """
    rows = []
    for r in results:
        w0, w1   = r["wins"]
        total    = r["total_games"]
        win_rate = w0 / total
        ci_low, ci_high = _wilson_ci(w0, total)

        # Binomial test  H0: win rate == 0.50
        p_value = stats.binomtest(
            w0, total, p=0.5, alternative="two-sided"
        ).pvalue

        ls = r["loser_surviving"]
        rows.append({
            "Matchup":        r["label"],
            "Category":       r["category"],
            "Agent0":         r["agent0"],
            "Agent1":         r["agent1"],
            "W0":             w0,
            "W1":             w1,
            "Draws":          r["draws"],
            "Total":          total,
            "WinRate_A0":     round(win_rate * 100, 1),
            "CI_low":         ci_low,
            "CI_high":        ci_high,
            "p_value":        round(p_value, 4),
            "Significant":    "Yes" if p_value < 0.05 else "No",
            "Avg_Turns":      round(r["avg_turns"], 1),
            "Loser_Surv_Avg": round(np.mean(ls), 1) if ls else None,
            "Loser_Surv_Std": round(np.std(ls), 1)  if ls else None,
            "Loser_Surv_Raw": ls,   # raw list for Mann-Whitney U test
        })

    return pd.DataFrame(rows)


def compute_ranking(df):
    """Derive an overall agent ranking from standard matchup results.

    Args:
        df: DataFrame from compute_stats().

    Returns:
        DataFrame with columns Rank, Agent, AvgWinRate sorted descending.
    """
    standard = df[
        (df["Category"] == "standard") &
        (df["Matchup"] != "Random vs Random")
        ].copy()
    agent_wins  = {}
    agent_games = {}

    for _, row in standard.iterrows():
        for agent, wins, total in [
            (row["Agent0"], row["W0"], row["Total"]),
            (row["Agent1"], row["W1"], row["Total"]),
        ]:
            agent_wins[agent]  = agent_wins.get(agent, 0)  + wins
            agent_games[agent] = agent_games.get(agent, 0) + total

    ranking = pd.DataFrame([
        {
            "Agent":      a,
            "TotalWins":  agent_wins[a],
            "TotalGames": agent_games[a],
            "AvgWinRate": round(agent_wins[a] / agent_games[a] * 100, 1),
        }
        for a in agent_wins
    ]).sort_values("AvgWinRate", ascending=False).reset_index(drop=True)
    ranking.index += 1
    return ranking


def compute_first_mover(df):
    """Summarise first-mover advantage for each agent vs MCTS baseline.

    Compares win rate as Player 0 (first mover) against win rate as
    Player 1 (second mover) when facing MCTS.
    Uses Binomial test to assess whether the positional difference is
    statistically significant.

    Win/loss outcomes → Binomial test.

    Args:
        df: DataFrame from compute_stats().

    Returns:
        DataFrame with Agent, WinRate_P0, WinRate_P1, Advantage,
        p_value, Significant.
    """
    fm = df[df["Category"] == "first_mover"].copy()
    rows = []
    for agent in ["Rule", "Random", "RL"]:
        p0_row = fm[fm["Matchup"] == f"{agent} as P0 vs MCTS"]
        p1_row = fm[fm["Matchup"] == f"{agent} as P1 vs MCTS"]
        if p0_row.empty or p1_row.empty:
            continue

        # Agent as P0: WinRate_A0 is the agent's own win rate
        wr_p0  = p0_row.iloc[0]["WinRate_A0"]
        w0_p0  = p0_row.iloc[0]["W0"]
        total  = p0_row.iloc[0]["Total"]

        # Agent as P1: agent win rate = 100 - MCTS win rate
        wr_p1  = round(100.0 - p1_row.iloc[0]["WinRate_A0"], 1)

        # Binomial test: is P0 win rate significantly different from P1?
        # H0: true win rate as P0 == observed win rate as P1
        p_value = stats.binomtest(
            w0_p0, total, p=wr_p1 / 100.0, alternative="two-sided"
        ).pvalue

        rows.append({
            "Agent":       agent,
            "WinRate_P0":  wr_p0,
            "WinRate_P1":  wr_p1,
            "Advantage":   round(wr_p0 - wr_p1, 1),
            "p_value":     round(p_value, 4),
            "Significant": "Yes" if p_value < 0.05 else "No",
        })
    return pd.DataFrame(rows)


def compute_surviving_stats(df):
    """Compare each matchup's surviving rounds vs Random vs Random baseline.

    Surviving rounds are non-normally distributed count data, so uses the
    non-parametric Mann-Whitney U test.

    Surviving rounds (turn counts) → Mann-Whitney U test.

    Args:
        df: DataFrame from compute_stats() containing Loser_Surv_Raw column.

    Returns:
        DataFrame with Matchup, Surv_Avg, Surv_Std, U_statistic,
        p_value, Significant.
    """
    standard = df[df["Category"] == "standard"].copy()

    # Random vs Random is the no-strategy baseline
    baseline_row = standard[standard["Matchup"] == "Random vs Random"]
    if baseline_row.empty:
        print("Warning: Random vs Random baseline not found.")
        return pd.DataFrame()
    baseline = baseline_row.iloc[0]["Loser_Surv_Raw"]

    rows = []
    for _, row in standard.iterrows():
        ls = row["Loser_Surv_Raw"]
        if not ls or row["Matchup"] == "Random vs Random":
            continue

        u_stat, p_value = stats.mannwhitneyu(
            ls, baseline, alternative="two-sided"
        )
        rows.append({
            "Matchup":     row["Matchup"],
            "Surv_Avg":    row["Loser_Surv_Avg"],
            "Surv_Std":    row["Loser_Surv_Std"],
            "U_statistic": round(u_stat, 1),
            "p_value":     round(p_value, 4),
            "Significant": "Yes" if p_value < 0.05 else "No",
        })

    return pd.DataFrame(rows)


def print_stats(df, ranking, first_mover_df, surv_stats_df):
    """Print formatted summaries of all four analyses to stdout.

    Args:
        df: DataFrame from compute_stats().
        ranking: DataFrame from compute_ranking().
        first_mover_df: DataFrame from compute_first_mover().
        surv_stats_df: DataFrame from compute_surviving_stats().
    """
    # ── 1. Win rates (Binomial test) ──────────────────────────
    print("\n" + "=" * 80)
    print("1. WIN RATES & STATISTICAL SIGNIFICANCE  (Binomial test)")
    print("=" * 80)
    standard = df[df["Category"] == "standard"]
    for _, row in standard.iterrows():
        sig = "significant (p < 0.05)" if row["Significant"] == "Yes" \
              else "not significant"
        print(f"\n  {row['Matchup']}")
        print(f"    {row['Agent0']:<10}  win rate : {row['WinRate_A0']:>5.1f}%  "
              f"95% CI: [{row['CI_low']:.1f}%, {row['CI_high']:.1f}%]")
        print(f"    p-value  : {row['p_value']:.4f}  — {sig}")
        print(f"    Avg game length        : {row['Avg_Turns']} turns")
        if row["Loser_Surv_Avg"] is not None:
            print(f"    Loser surviving rounds : "
                  f"{row['Loser_Surv_Avg']} ± {row['Loser_Surv_Std']} turns")

    # ── 2. Surviving rounds (Mann-Whitney U test) ─────────────
    print("\n" + "=" * 80)
    print("2. SURVIVING ROUNDS — Mann-Whitney U test vs Random baseline")
    print("=" * 80)
    print(f"\n  {'Matchup':<25}  {'Avg':>6}  {'Std':>6}  "
          f"{'p-value':>8}  {'Significant':>12}")
    print("  " + "-" * 65)
    for _, row in surv_stats_df.iterrows():
        print(f"  {row['Matchup']:<25}  {row['Surv_Avg']:>6.1f}  "
              f"{row['Surv_Std']:>6.1f}  "
              f"{row['p_value']:>8.4f}  {row['Significant']:>12}")

    # ── 3. First-mover advantage (Binomial test) ──────────────
    print("\n" + "=" * 80)
    print("3. FIRST-MOVER ADVANTAGE vs MCTS  (Binomial test, P0 vs P1)")
    print("=" * 80)
    print(f"\n  {'Agent':<8}  {'As P0 (first)':>14}  "
          f"{'As P1 (second)':>15}  {'Advantage':>10}  "
          f"{'p-value':>8}  {'Significant':>12}")
    print("  " + "-" * 75)
    for _, row in first_mover_df.iterrows():
        sig = row["Significant"]
        print(f"  {row['Agent']:<8}  {row['WinRate_P0']:>13.1f}%  "
              f"{row['WinRate_P1']:>14.1f}%  "
              f"{row['Advantage']:>+10.1f}%  "
              f"{row['p_value']:>8.4f}  {sig:>12}")

    # ── 4. Overall ranking ────────────────────────────────────
    print("\n" + "=" * 80)
    print("4. OVERALL AGENT RANKING  (cumulative win rate, standard matchups)")
    print("=" * 80)
    print(f"\n  {'Rank':<6}  {'Agent':<10}  {'Avg Win Rate':>12}  "
          f"{'Total Wins':>11}  {'Total Games':>12}")
    print("  " + "-" * 56)
    for rank, row in ranking.iterrows():
        print(f"  {rank:<6}  {row['Agent']:<10}  "
              f"{row['AvgWinRate']:>11.1f}%  "
              f"{int(row['TotalWins']):>11}  "
              f"{int(row['TotalGames']):>12}")

    print("\n" + "=" * 80)


# ══════════════════════════════════════════════════════════════
# 4. Visualisation (5 charts)
# ══════════════════════════════════════════════════════════════

def plot_winrates(df):
    """Chart 1: Win-rate bar chart with 95% CI error bars (standard matchups).

    Red bars = statistically significant (p < 0.05), grey = not significant.
    Statistical test: Binomial test.

    Args:
        df: DataFrame from compute_stats().
    """
    data     = df[df["Category"] == "standard"]
    labels   = data["Matchup"].tolist()
    winrates = data["WinRate_A0"].tolist()
    err_low  = [w - l for w, l in zip(data["WinRate_A0"], data["CI_low"])]
    err_high = [h - w for w, h in zip(data["WinRate_A0"], data["CI_high"])]
    colors   = ["#e74c3c" if s == "Yes" else "#95a5a6"
                for s in data["Significant"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, winrates, color=colors,
           yerr=[err_low, err_high], capsize=5, edgecolor="white")
    ax.axhline(50, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Win Rate of Player 0 (%)", fontsize=11)
    ax.set_title("Agent Win Rates with 95% Confidence Intervals\n"
                 "(Binomial test, H\u2080: win rate = 50%)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(30, 80)
    ax.tick_params(axis="x", rotation=20)

    red_patch  = mpatches.Patch(color="#e74c3c", label="Significant (p < 0.05)")
    grey_patch = mpatches.Patch(color="#95a5a6", label="Not significant")
    baseline   = plt.Line2D([0], [0], color="black", linestyle="--",
                            label="50% baseline")
    ax.legend(handles=[red_patch, grey_patch, baseline])

    plt.tight_layout()
    plt.savefig("results/winrates.png", dpi=150)
    print("Saved: results/winrates.png")
    plt.close()


def plot_surviving_rounds(results, surv_stats_df=None):
    """Chart 2: Box plot of the loser's surviving rounds for every matchup.

    Annotates each box with Mann-Whitney U significance vs Random baseline.

    Args:
        results: List of result dicts from run_all_matchups().
        surv_stats_df: DataFrame from compute_surviving_stats().
    """
    standard_results = [r for r in results if r["category"] == "standard"
                        and r["loser_surviving"]]
    if not standard_results:
        print("No surviving-rounds data — skipping chart.")
        return

    data_to_plot = [r["loser_surviving"] for r in standard_results]
    labels       = [r["label"] for r in standard_results]
    palette      = ["#3498db", "#2ecc71", "#e67e22",
                    "#9b59b6", "#1abc9c", "#e74c3c", "#f39c12"]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data_to_plot, patch_artist=True)
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Surviving Rounds (Loser)", fontsize=11)
    ax.set_title("Loser's Surviving Rounds by Matchup\n"
                 "(Mann-Whitney U test vs Random vs Random baseline)",
                 fontsize=13, fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Annotate significance above each box
    if surv_stats_df is not None and not surv_stats_df.empty:
        y_top = ax.get_ylim()[1] * 0.93
        for i, label in enumerate(labels):
            match = surv_stats_df[surv_stats_df["Matchup"] == label]
            if not match.empty:
                sig_text = "p<0.05*" if match.iloc[0]["Significant"] == "Yes" \
                           else "n.s."
                color = "red" if match.iloc[0]["Significant"] == "Yes" \
                        else "grey"
                ax.text(i + 1, y_top, sig_text,
                        ha="center", fontsize=8, color=color,
                        fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/surviving_rounds.png", dpi=150)
    print("Saved: results/surviving_rounds.png")
    plt.close()


def plot_mcts_focus(df, results):
    """Chart 3: MCTS-focused panel — win rate (left) and opponent surviving
    rounds (right). MCTS is the near-GTO baseline.

    Args:
        df: DataFrame from compute_stats().
        results: List of result dicts from run_all_matchups().
    """
    mcts_df      = df[(df["Category"] == "standard") &
                      (df["Matchup"].str.contains("MCTS"))]
    mcts_results = [r for r in results
                    if r["category"] == "standard" and "MCTS" in r["label"]]

    if mcts_df.empty:
        print("No MCTS matchups found — skipping MCTS focus chart.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: MCTS win rates with CI
    labels   = mcts_df["Matchup"].tolist()
    winrates = mcts_df["WinRate_A0"].tolist()
    err_low  = [w - l for w, l in zip(mcts_df["WinRate_A0"], mcts_df["CI_low"])]
    err_high = [h - w for w, h in zip(mcts_df["WinRate_A0"], mcts_df["CI_high"])]

    ax1.bar(labels, winrates, color="#2ecc71",
            yerr=[err_low, err_high], capsize=6, edgecolor="white")
    ax1.axhline(50, color="black", linestyle="--", linewidth=1)
    ax1.set_ylabel("MCTS Win Rate (%)", fontsize=11)
    ax1.set_title("MCTS Win Rate vs Other Agents\n(near-GTO Baseline)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylim(30, 85)
    ax1.tick_params(axis="x", rotation=15)

    # Right: surviving rounds of MCTS's opponents
    surv_data   = [r["loser_surviving"] for r in mcts_results
                   if r["loser_surviving"]]
    surv_labels = [r["label"] for r in mcts_results
                   if r["loser_surviving"]]

    if surv_data:
        bp = ax2.boxplot(surv_data, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3498db")
            patch.set_alpha(0.7)
        ax2.set_xticks(range(1, len(surv_labels) + 1))
        ax2.set_xticklabels(surv_labels, rotation=15, ha="right")
        ax2.set_ylabel("Opponent's Surviving Rounds", fontsize=11)
        ax2.set_title("How Long Does MCTS's Opponent Survive?",
                      fontsize=12, fontweight="bold")
        ax2.yaxis.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig("results/mcts_focus.png", dpi=150)
    print("Saved: results/mcts_focus.png")
    plt.close()


def plot_first_mover(first_mover_df):
    """Chart 4: Grouped bar chart comparing P0 vs P1 win rate per agent vs MCTS.

    Annotates each agent with Binomial test significance.
    Statistical test: Binomial test (win/loss outcomes).

    Args:
        first_mover_df: DataFrame from compute_first_mover().
    """
    if first_mover_df.empty:
        print("No first-mover data — skipping chart.")
        return

    agents  = first_mover_df["Agent"].tolist()
    wr_p0   = first_mover_df["WinRate_P0"].tolist()
    wr_p1   = first_mover_df["WinRate_P1"].tolist()
    x       = np.arange(len(agents))
    width   = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars0 = ax.bar(x - width / 2, wr_p0, width, label="As P0 (first mover)",
                   color="#3498db", edgecolor="white")
    bars1 = ax.bar(x + width / 2, wr_p1, width, label="As P1 (second mover)",
                   color="#e67e22", edgecolor="white")

    ax.axhline(50, color="black", linestyle="--", linewidth=1,
               label="50% baseline")
    ax.set_ylabel("Win Rate vs MCTS (%)", fontsize=11)
    ax.set_title("First-Mover Advantage by Agent vs MCTS\n"
                 "(Binomial test, H\u2080: P0 win rate = P1 win rate)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, fontsize=11)
    ax.set_ylim(20, 85)
    ax.legend()

    # Annotate bars with win-rate values
    for bar in list(bars0) + list(bars1):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{bar.get_height():.1f}%",
                ha="center", va="bottom", fontsize=9)

    # Annotate significance above each agent group
    for i, (_, row) in enumerate(first_mover_df.iterrows()):
        sig_text = "p<0.05*" if row["Significant"] == "Yes" else "n.s."
        color    = "red" if row["Significant"] == "Yes" else "grey"
        ax.text(i, max(wr_p0[i], wr_p1[i]) + 5,
                sig_text, ha="center", fontsize=9,
                color=color, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/first_mover.png", dpi=150)
    print("Saved: results/first_mover.png")
    plt.close()


def plot_ranking(ranking):
    """Chart 5: Horizontal bar chart of overall agent ranking.

    Args:
        ranking: DataFrame from compute_ranking().
    """
    if ranking.empty:
        print("No ranking data — skipping chart.")
        return

    agents   = ranking["Agent"].tolist()[::-1]
    winrates = ranking["AvgWinRate"].tolist()[::-1]
    colors   = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"][::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(agents, winrates, color=colors[:len(agents)],
                   edgecolor="white")
    ax.axvline(50, color="black", linestyle="--", linewidth=1,
               label="50% baseline")
    ax.set_xlabel("Cumulative Win Rate (%)", fontsize=11)
    ax.set_title("Overall Agent Ranking",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(30, 70)
    ax.legend()

    for bar, val in zip(bars, winrates):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig("results/ranking.png", dpi=150)
    print("Saved: results/ranking.png")
    plt.close()


# ══════════════════════════════════════════════════════════════
# 5. Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Starting UNO Agent Analysis...")
    print(f"Games per matchup: {NUM_GAMES}")

    # Run all matchups
    results = run_all_matchups()

    # Compute statistics
    df             = compute_stats(results)
    ranking        = compute_ranking(df)
    first_mover_df = compute_first_mover(df)
    surv_stats_df  = compute_surviving_stats(df)

    # Print to terminal
    print_stats(df, ranking, first_mover_df, surv_stats_df)

    # Save CSVs
    df.drop(columns=["Loser_Surv_Raw"]).to_csv(
        "results/stats_summary.csv", index=False)
    ranking.to_csv("results/ranking.csv", index=True)
    first_mover_df.to_csv("results/first_mover.csv", index=False)
    surv_stats_df.to_csv("results/surviving_stats.csv", index=False)
    print("Saved: results/stats_summary.csv")
    print("Saved: results/ranking.csv")
    print("Saved: results/first_mover.csv")
    print("Saved: results/surviving_stats.csv")

    # Generate charts
    plot_winrates(df)
    plot_surviving_rounds(results, surv_stats_df)
    plot_mcts_focus(df, results)
    plot_first_mover(first_mover_df)
    plot_ranking(ranking)

    print("\nAnalysis complete. All outputs are in the results/ folder.")
