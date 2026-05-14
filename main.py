# This script runs self-play matches between different agents and collects data for training.
import sys

sys.path.insert(0, ".")
from agents.bad_rule_agent import RuleBadAgent
from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from agents.rl_agent import RLAgent
from training.self_play import run_self_play
from config import Paths

max = 10_000  # no of games to run for each matchup
print("=== 1. Random vs Random ===")
run_self_play(
    RandomAgent("Random-A"),
    RandomAgent("Random-B"),
    num_games=max,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

print("\n=== 2. Rule vs Random ===")
run_self_play(
    RuleAgent("Rule"),
    RandomAgent("Random"),
    num_games=max,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

print("\n=== 3. Rule vs Rule ===")
run_self_play(
    RuleAgent("Rule-A"),
    RuleAgent("Rule-B"),
    num_games=max,
    save_logs=False,
    save_dataset=True,
    base_seed=0,
)

print("\n=== 4. Random vs Rule ===")
run_self_play(
    RandomAgent("Random"),
    RuleAgent("Rule"),
    num_games=max,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

print("\n=== 5. Rule Good vs Rule Bad ===")
run_self_play(
    RuleAgent("Rule-Good"),
    RuleBadAgent("Rule-Bad"),
    num_games=max,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

print("\n=== 6. Rule Bad vs Rule Good ===")
run_self_play(
    RuleBadAgent("Rule-Bad"),
    RuleAgent("Rule-Good"),
    num_games=max,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

rl = RLAgent("RL", epsilon=0.0)
rl.load(str(Paths.MODELS / "rl_agent.pt"))

print("\n=== 7. RL vs Random ===")
run_self_play(
    rl,
    RandomAgent("Random"),
    num_games=max,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

print("\n=== 8. RL vs Rule ===")
run_self_play(
    rl,
    RuleAgent("Rule"),
    num_games=max,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

print("\n=== 9. MCTS vs Random ===")
run_self_play(
    MCTSAgent("MCTS"),
    RandomAgent("Random"),
    num_games=10_000,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

print("\n=== 10. MCTS vs Rule ===")
run_self_play(
    MCTSAgent("MCTS"),
    RuleAgent("Rule"),
    num_games=10_000,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)
