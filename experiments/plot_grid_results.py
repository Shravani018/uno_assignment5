"""
Generate heatmaps from mcts_grid.json.

Two side-by-side heatmaps:
  Left  -- win rate (%) vs RuleAgent
  Right -- avg decision time (s)

Output: experiments/results/mcts_grid_heatmap.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_FILE = Path(__file__).parent / "results" / "mcts_grid.json"
OUT_FILE = Path(__file__).parent / "results" / "mcts_grid_heatmap.png"

DPI = 150


def main():
    if not RESULTS_FILE.exists():
        print(f"Not found: {RESULTS_FILE}")
        sys.exit(1)

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    c_vals     = sorted(set(r["c"] for r in results))
    depth_vals = sorted(set(r["depth"] for r in results))

    # Build 2D arrays: rows=depth, cols=c
    wr_grid  = np.full((len(depth_vals), len(c_vals)), np.nan)
    dt_grid  = np.full((len(depth_vals), len(c_vals)), np.nan)

    for r in results:
        row = depth_vals.index(r["depth"])
        col = c_vals.index(r["c"])
        wr_grid[row, col] = r["win_rate"] * 100
        dt_grid[row, col] = r["avg_decision_time_s"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Win rate heatmap ─────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(wr_grid, cmap="YlGn", aspect="auto",
                   vmin=np.nanmin(wr_grid) - 1, vmax=np.nanmax(wr_grid) + 1)
    plt.colorbar(im, ax=ax, label="Win rate (%)")

    ax.set_xticks(range(len(c_vals)))
    ax.set_xticklabels([str(c) for c in c_vals])
    ax.set_yticks(range(len(depth_vals)))
    ax.set_yticklabels([str(d) for d in depth_vals])
    ax.set_xlabel("c", fontsize=11)
    ax.set_ylabel("rollout_depth", fontsize=11)
    ax.set_title(f"Win rate vs RuleAgent (%)\n(num_simulations={results[0]['num_simulations']}, "
                 f"n={results[0]['total_games']} games)", fontsize=11)

    best_idx = np.unravel_index(np.nanargmax(wr_grid), wr_grid.shape)
    for row in range(len(depth_vals)):
        for col in range(len(c_vals)):
            val = wr_grid[row, col]
            if not np.isnan(val):
                weight = "bold" if (row, col) == best_idx else "normal"
                outline = dict(boxstyle="round,pad=0.2", facecolor="white",
                               edgecolor="black", linewidth=1.5) if (row, col) == best_idx else None
                txt = ax.text(col, row, f"{val:.1f}%", ha="center", va="center",
                              fontsize=10, fontweight=weight)
                if outline:
                    txt.set_bbox(outline)

    # ── Decision time heatmap ─────────────────────────────────────────────────
    ax = axes[1]
    im2 = ax.imshow(dt_grid, cmap="OrRd", aspect="auto",
                    vmin=0, vmax=np.nanmax(dt_grid) * 1.1)
    plt.colorbar(im2, ax=ax, label="Avg decision time (s)")

    ax.set_xticks(range(len(c_vals)))
    ax.set_xticklabels([str(c) for c in c_vals])
    ax.set_yticks(range(len(depth_vals)))
    ax.set_yticklabels([str(d) for d in depth_vals])
    ax.set_xlabel("c", fontsize=11)
    ax.set_ylabel("rollout_depth", fontsize=11)
    ax.set_title("Avg decision time (s)\n(lower is faster)", fontsize=11)

    for row in range(len(depth_vals)):
        for col in range(len(c_vals)):
            val = dt_grid[row, col]
            if not np.isnan(val):
                ax.text(col, row, f"{val:.3f}s", ha="center", va="center", fontsize=10)

    fig.suptitle("MCTS grid search: c × rollout_depth", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
