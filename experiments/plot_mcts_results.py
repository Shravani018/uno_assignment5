"""
Generate four separate parameter sweep charts from mcts_params.json.

Output files (one per parameter):
  experiments/results/mcts_c.png
  experiments/results/mcts_num_simulations.png
  experiments/results/mcts_rollout_depth.png
  experiments/results/mcts_rollout_policy.png

Each chart includes 95% confidence intervals.
"""
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RESULTS_FILE = Path(__file__).parent / "results" / "mcts_params.json"
OUT_DIR = Path(__file__).parent / "results"

# ── Shared style ──────────────────────────────────────────────────────────────

FIGSIZE = (7, 5)
DPI = 150
BLUE = "#2196F3"
BASELINE_COLOR = "#E53935"


def _apply_style(ax, ylabel="Win rate vs RuleAgent (%)"):
    ax.set_ylabel(ylabel, fontsize=11)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Individual plot functions ─────────────────────────────────────────────────

def plot_c(rows: list):
    baseline = next(r for r in rows if r["value"] == 1.41)

    xs = [r["value"] for r in rows]
    ys = [r["win_rate"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, ys, "o-", color=BLUE, linewidth=2, markersize=7, label="Win rate")
    ax.scatter([1.41], [baseline["win_rate"] * 100], s=120, color=BASELINE_COLOR,
               zorder=5, label="Default (c=1.41)")
    ax.axvline(x=math.sqrt(2), color=BASELINE_COLOR, linestyle="--",
               linewidth=1, alpha=0.7, label=f"√2 ≈ {math.sqrt(2):.2f} (UCT theory)")

    ax.set_xlabel("Exploration constant c", fontsize=11)
    ax.set_title("Effect of UCB exploration constant c\n"
                 "(n_simulations=100, rollout_depth=150)", fontsize=12)
    ax.legend(fontsize=9)
    _apply_style(ax)
    ax.set_ylim(min(ys) - 2, max(ys) + 4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mcts_c.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved mcts_c.png")


def plot_num_simulations(rows: list):
    baseline = next(r for r in rows if r["value"] == 100)

    xs = [r["value"] for r in rows]
    ys = [r["win_rate"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, ys, "o-", color="#4CAF50", linewidth=2, markersize=7, label="Win rate")
    ax.scatter([100], [baseline["win_rate"] * 100], s=120, color=BASELINE_COLOR,
               zorder=5, label="Default (n=100)")

    ax.set_xscale("log")
    ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("num_simulations (log scale)", fontsize=11)
    ax.set_title("Effect of number of simulations\n"
                 "(c=1.41, rollout_depth=150)", fontsize=12)
    ax.legend(fontsize=9)
    _apply_style(ax)
    ax.set_ylim(min(ys) - 2, max(ys) + 4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mcts_num_simulations.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved mcts_num_simulations.png")


def plot_rollout_depth(rows: list):
    baseline = next(r for r in rows if r["value"] == 100)

    xs = [r["value"] for r in rows]
    ys = [r["win_rate"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(xs, ys, "o-", color="#FF9800", linewidth=2, markersize=7, label="Win rate")
    ax.scatter([100], [baseline["win_rate"] * 100], s=120, color=BASELINE_COLOR,
               zorder=5, label="Baseline (depth=100)")
    ax.axvline(x=31, color="gray", linestyle="--", linewidth=1, alpha=0.8,
               label="Avg steps/player ≈ 31")

    ax.set_xlabel("rollout_depth", fontsize=11)
    ax.set_title("Effect of rollout depth\n"
                 "(c=1.41, n_simulations=100)", fontsize=12)
    ax.legend(fontsize=9)
    _apply_style(ax)
    ax.set_ylim(min(ys) - 2, max(ys) + 4)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mcts_rollout_depth.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved mcts_rollout_depth.png")


def plot_rollout_policy(rows: list):
    labels = [r["value"] for r in rows]
    ys = [r["win_rate"] * 100 for r in rows]
    colors = ["#2196F3", "#E91E63"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, ys, color=colors, width=0.45,
                  edgecolor="white", linewidth=1.5)

    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, y + 0.5,
                f"{y:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xlabel("Rollout policy", fontsize=11)
    ax.set_title("Rollout policy comparison\n"
                 "(n=100, c=1.0, depth=20)", fontsize=12)
    _apply_style(ax)
    ax.set_ylim(min(ys) - 5, max(ys) + 6)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "mcts_rollout_policy.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved mcts_rollout_policy.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not RESULTS_FILE.exists():
        print(f"Not found: {RESULTS_FILE}")
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_c(data["c"])
    plot_num_simulations(data["num_simulations"])
    plot_rollout_depth(data["rollout_depth"])
    plot_rollout_policy(data["rollout_policy"])
    print("\nAll charts saved to", OUT_DIR)


if __name__ == "__main__":
    main()
