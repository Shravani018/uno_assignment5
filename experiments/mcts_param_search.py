"""
MCTS parameter sweep experiment.

Tests three key ISMCTS parameters one-at-a-time against RuleAgent (500 games each),
plus a rollout policy comparison. Results saved to experiments/results/mcts_params.json.

Theory-driven parameter ranges:
  c             -- centred on sqrt(2) ≈ 1.41 (UCT theoretical optimum, K&S 2006)
  num_simulations -- doubling from 25 to 500 to find the elbow of the gain curve
  rollout_depth   -- scaled around UNO average ~31 steps/player (62.6 turns / 2)
"""
import json
import sys
import time
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.mcts_agent import MCTSAgent
from agents.rule_agent import RuleAgent
from training.self_play import run_self_play

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "mcts_params.json"

NUM_GAMES = 500
BASE_SEED = 42

# Defaults used when a parameter is NOT the one being varied
DEFAULT_SIMS = 100
DEFAULT_C = 1.41
DEFAULT_DEPTH = 150

# ── Parameter grids ──────────────────────────────────────────────────────────

C_VALUES = [0.35, 0.7, 1.0, 1.41, 2.0, 3.0]
SIM_VALUES = [25, 50, 100, 200, 500]
DEPTH_VALUES = [10, 20, 30, 50, 100]


def run_matchup(agent0, agent1, label: str) -> dict:
    print(f"  {label} ... ", end="", flush=True)
    t0 = time.time()
    result = run_self_play(
        agent0, agent1,
        num_games=NUM_GAMES,
        save_logs=False,
        save_dataset=False,
        base_seed=BASE_SEED,
    )
    elapsed = time.time() - t0
    win_rate = result["wins"][0] / result["total_games"]
    print(f"{win_rate:.3f}  ({elapsed:.1f}s)")
    return {
        "label": label,
        "win_rate": win_rate,
        "wins": result["wins"],
        "draws": result["draws"],
        "total_games": result["total_games"],
        "avg_turns": result["avg_turns"],
        "elapsed_s": round(elapsed, 1),
    }


def sweep_c():
    print("\n── Sweeping c (num_simulations={}, rollout_depth={}) ──".format(
        DEFAULT_SIMS, DEFAULT_DEPTH))
    results = []
    for c in C_VALUES:
        mcts = MCTSAgent(name=f"MCTS_c{c}", num_simulations=DEFAULT_SIMS,
                         c=c, rollout_depth=DEFAULT_DEPTH)
        r = run_matchup(mcts, RuleAgent(), f"c={c}")
        r["param"] = "c"
        r["value"] = c
        results.append(r)
    return results


def sweep_simulations():
    print("\n── Sweeping num_simulations (c={}, rollout_depth={}) ──".format(
        DEFAULT_C, DEFAULT_DEPTH))
    results = []
    for n in SIM_VALUES:
        mcts = MCTSAgent(name=f"MCTS_n{n}", num_simulations=n,
                         c=DEFAULT_C, rollout_depth=DEFAULT_DEPTH)
        r = run_matchup(mcts, RuleAgent(), f"n_sims={n}")
        r["param"] = "num_simulations"
        r["value"] = n
        results.append(r)
    return results


def sweep_depth():
    print("\n── Sweeping rollout_depth (c={}, num_simulations={}) ──".format(
        DEFAULT_C, DEFAULT_SIMS))
    results = []
    for d in DEPTH_VALUES:
        mcts = MCTSAgent(name=f"MCTS_d{d}", num_simulations=DEFAULT_SIMS,
                         c=DEFAULT_C, rollout_depth=d)
        r = run_matchup(mcts, RuleAgent(), f"depth={d}")
        r["param"] = "rollout_depth"
        r["value"] = d
        results.append(r)
    return results


def sweep_rollout_policy(best_sims: int, best_c: float, best_depth: int):
    """Compare random vs rule-based rollout using the best parameters found so far."""
    print("\n── Rollout policy comparison (n={}, c={}, depth={}) ──".format(
        best_sims, best_c, best_depth))

    results = []

    # random rollout (default)
    mcts_random = MCTSAgent(name="MCTS_random_rollout",
                            num_simulations=best_sims, c=best_c,
                            rollout_depth=best_depth,
                            rollout_policy="random")
    r = run_matchup(mcts_random, RuleAgent(), "rollout=random")
    r["param"] = "rollout_policy"
    r["value"] = "random"
    results.append(r)

    # rule-based rollout
    mcts_rule = MCTSAgent(name="MCTS_rule_rollout",
                          num_simulations=best_sims, c=best_c,
                          rollout_depth=best_depth,
                          rollout_policy="rule")
    r = run_matchup(mcts_rule, RuleAgent(), "rollout=rule")
    r["param"] = "rollout_policy"
    r["value"] = "rule"
    results.append(r)

    return results


def best_value(sweep_results: list) -> float:
    """Return the parameter value with the highest win rate."""
    return max(sweep_results, key=lambda r: r["win_rate"])["value"]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("MCTS parameter sweep — {} games per config vs RuleAgent".format(NUM_GAMES))

    all_results = {}
    all_results["c"] = sweep_c()
    all_results["num_simulations"] = sweep_simulations()
    all_results["rollout_depth"] = sweep_depth()

    best_c = best_value(all_results["c"])
    best_sims = best_value(all_results["num_simulations"])
    best_depth = best_value(all_results["rollout_depth"])

    print(f"\nBest so far: c={best_c}, n_sims={best_sims}, depth={best_depth}")

    # Use n=100 for rollout policy comparison (n=500 is too slow on a laptop)
    all_results["rollout_policy"] = sweep_rollout_policy(
        100, float(best_c), int(best_depth)
    )

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
