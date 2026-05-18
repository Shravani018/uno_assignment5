"""
Finish the parameter sweep: run only the rollout policy comparison,
then combine with completed sweep results and save mcts_params.json.

The c / num_simulations / rollout_depth sweeps were completed previously;
their results are hardcoded here from the recorded output.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.mcts_agent import MCTSAgent
from agents.rule_agent import RuleAgent
from training.self_play import run_self_play

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "mcts_params.json"
NUM_GAMES = 500
BASE_SEED = 42

# ── Completed sweep results (from earlier run) ───────────────────────────────

COMPLETED = {
    "c": [
        {"param": "c", "value": 0.35, "win_rate": 0.548, "wins": [274, 226], "draws": 0, "total_games": 500, "avg_turns": 53.9},
        {"param": "c", "value": 0.7,  "win_rate": 0.568, "wins": [284, 216], "draws": 0, "total_games": 500, "avg_turns": 56.3},
        {"param": "c", "value": 1.0,  "win_rate": 0.570, "wins": [285, 215], "draws": 0, "total_games": 500, "avg_turns": 51.6},
        {"param": "c", "value": 1.41, "win_rate": 0.536, "wins": [268, 232], "draws": 0, "total_games": 500, "avg_turns": 51.9},
        {"param": "c", "value": 2.0,  "win_rate": 0.568, "wins": [284, 216], "draws": 0, "total_games": 500, "avg_turns": 52.2},
        {"param": "c", "value": 3.0,  "win_rate": 0.530, "wins": [265, 235], "draws": 0, "total_games": 500, "avg_turns": 51.9},
    ],
    "num_simulations": [
        {"param": "num_simulations", "value": 25,  "win_rate": 0.516, "wins": [258, 242], "draws": 0, "total_games": 500, "avg_turns": 54.9},
        {"param": "num_simulations", "value": 50,  "win_rate": 0.546, "wins": [273, 227], "draws": 0, "total_games": 500, "avg_turns": 52.9},
        {"param": "num_simulations", "value": 100, "win_rate": 0.536, "wins": [268, 232], "draws": 0, "total_games": 500, "avg_turns": 51.9},
        {"param": "num_simulations", "value": 200, "win_rate": 0.576, "wins": [288, 212], "draws": 0, "total_games": 500, "avg_turns": 50.6},
        {"param": "num_simulations", "value": 500, "win_rate": 0.620, "wins": [310, 190], "draws": 0, "total_games": 500, "avg_turns": 48.6},
    ],
    "rollout_depth": [
        {"param": "rollout_depth", "value": 10,  "win_rate": 0.556, "wins": [278, 222], "draws": 0, "total_games": 500, "avg_turns": 48.2},
        {"param": "rollout_depth", "value": 20,  "win_rate": 0.580, "wins": [290, 210], "draws": 0, "total_games": 500, "avg_turns": 48.3},
        {"param": "rollout_depth", "value": 30,  "win_rate": 0.558, "wins": [279, 221], "draws": 0, "total_games": 500, "avg_turns": 50.8},
        {"param": "rollout_depth", "value": 50,  "win_rate": 0.546, "wins": [273, 227], "draws": 0, "total_games": 500, "avg_turns": 51.1},
        {"param": "rollout_depth", "value": 100, "win_rate": 0.536, "wins": [268, 232], "draws": 0, "total_games": 500, "avg_turns": 53.3},
    ],
}

# ── Rollout policy comparison ─────────────────────────────────────────────────

# Use c=1.0, depth=20 (best from sweeps), n=100 (practical for Mac)
POLICY_N = 100
POLICY_C = 1.0
POLICY_DEPTH = 20


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
        "param": "rollout_policy",
        "value": label.split("=")[1],
        "win_rate": win_rate,
        "wins": result["wins"],
        "draws": result["draws"],
        "total_games": result["total_games"],
        "avg_turns": result["avg_turns"],
        "elapsed_s": round(elapsed, 1),
        "config": {"n": POLICY_N, "c": POLICY_C, "depth": POLICY_DEPTH},
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Rollout policy comparison — {NUM_GAMES} games each")
    print(f"Config: n={POLICY_N}, c={POLICY_C}, depth={POLICY_DEPTH}\n")

    policy_results = []

    mcts_random = MCTSAgent(name="MCTS_random_rollout",
                            num_simulations=POLICY_N, c=POLICY_C,
                            rollout_depth=POLICY_DEPTH, rollout_policy="random")
    policy_results.append(run_matchup(mcts_random, RuleAgent(), "rollout=random"))

    mcts_rule = MCTSAgent(name="MCTS_rule_rollout",
                          num_simulations=POLICY_N, c=POLICY_C,
                          rollout_depth=POLICY_DEPTH, rollout_policy="rule")
    policy_results.append(run_matchup(mcts_rule, RuleAgent(), "rollout=rule"))

    all_results = dict(COMPLETED)
    all_results["rollout_policy"] = policy_results

    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to {RESULTS_FILE}")
    print("\nSummary:")
    print(f"  random rollout: {policy_results[0]['win_rate']:.1%}")
    print(f"  rule rollout:   {policy_results[1]['win_rate']:.1%}")


if __name__ == "__main__":
    main()
