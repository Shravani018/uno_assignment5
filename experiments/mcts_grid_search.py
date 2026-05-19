"""
MCTS grid search: c x rollout_depth (num_simulations=200 fixed).

Grid:
  c             : [0.7, 1.0, 1.41, 2.0]
  rollout_depth : [10, 20, 30]
  num_simulations: 200 (fixed)

500 games per config vs RuleAgent, base_seed=42.
Results saved incrementally to experiments/results/mcts_grid.json.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.mcts_agent import MCTSAgent
from agents.rule_agent import RuleAgent
from training.self_play import run_self_play

RESULTS_FILE = Path(__file__).parent / "results" / "mcts_grid.json"
RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

NUM_GAMES   = 500
BASE_SEED   = 42
FIXED_SIMS  = 200

C_VALUES     = [0.7, 1.0, 1.41, 2.0, 3.0]
DEPTH_VALUES = [10, 20, 30, 40, 50]


def run_config(c: float, depth: int) -> dict:
    label = f"c={c}, depth={depth}"
    print(f"  {label} ...", flush=True)
    mcts = MCTSAgent(num_simulations=FIXED_SIMS, c=c, rollout_depth=depth)
    t0 = time.time()
    result = run_self_play(
        mcts, RuleAgent(),
        num_games=NUM_GAMES,
        save_logs=False,
        save_dataset=False,
        base_seed=BASE_SEED,
    )
    elapsed = time.time() - t0
    win_rate = result["wins"][0] / result["total_games"]
    avg_dec_time = elapsed / (result["total_games"] * result["avg_turns"] / 2)
    print(f"    win_rate={win_rate:.3f}  elapsed={elapsed:.1f}s  "
          f"dec_time={avg_dec_time:.3f}s", flush=True)
    return {
        "c": c,
        "depth": depth,
        "num_simulations": FIXED_SIMS,
        "win_rate": win_rate,
        "wins": result["wins"],
        "draws": result["draws"],
        "total_games": result["total_games"],
        "avg_turns": result["avg_turns"],
        "elapsed_s": round(elapsed, 1),
        "avg_decision_time_s": round(avg_dec_time, 4),
    }


def main():
    # Resume from existing results if interrupted
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        done = {(r["c"], r["depth"]) for r in results}
        print(f"Resuming: {len(done)} configs already done.")
    else:
        results = []
        done = set()

    total = len(C_VALUES) * len(DEPTH_VALUES)
    print(f"Grid search: {total} configs x {NUM_GAMES} games "
          f"(num_simulations={FIXED_SIMS})\n")

    for c in C_VALUES:
        for depth in DEPTH_VALUES:
            if (c, depth) in done:
                print(f"  Skipping c={c}, depth={depth} (already done)")
                continue
            row = run_config(c, depth)
            results.append(row)
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {RESULTS_FILE}")

    best = max(results, key=lambda r: r["win_rate"])
    print(f"Best config: c={best['c']}, depth={best['depth']}  "
          f"win_rate={best['win_rate']:.3f}")


if __name__ == "__main__":
    main()
