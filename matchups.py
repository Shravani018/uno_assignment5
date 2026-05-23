# This script runs matches between different agents.
import sys

sys.path.insert(0, ".")
from agents.bad_rule_agent import RuleBadAgent
from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from agents.rl_agent import RLAgent
from training.self_play import run_self_play
from config import Paths

num_games = 10_000


print("=== 1. Random vs Random ===")
run_self_play(
    RandomAgent("Random-A"),
    RandomAgent("Random-B"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)


print("\n=== 2. Rule vs Rule ===")
run_self_play(
    RuleAgent("Rule-A"),
    RuleAgent("Rule-B"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)


print("\n=== 3. Rule vs Random ===")
run_self_play(
    RuleAgent("Rule"),
    RandomAgent("Random"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

print("\n=== 4. Random vs Rule ===")
run_self_play(
    RandomAgent("Random"),
    RuleAgent("Rule"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)
print("\n=== 5. Rule vs Anti-Rule ===")
run_self_play(
    RuleAgent("Rule-Good"),
    RuleBadAgent("Rule-Bad"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

print("\n=== 6. Anti-Rule vs Rule ===")
run_self_play(
    RuleBadAgent("Rule-Bad"),
    RuleAgent("Rule-Good"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

print("\n=== 7. Anti-Rule vs Random ===")
run_self_play(
    RuleBadAgent("Rule-Bad"),
    RandomAgent("Random"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)


print("\n=== 8. Random vs Anti-Rule ===")
run_self_play(
    RandomAgent("Random"),
    RuleBadAgent("Rule-Bad"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

# Loading RL model
rl = RLAgent("RL", epsilon=0.0)
rl.load(str(Paths.MODELS / "rl_agent.pt"))


print("\n=== 9. Random vs RL ===")
run_self_play(
    RandomAgent("Random"),
    rl,
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)


print("\n=== 10. RL vs Random ===")
run_self_play(
    rl,
    RandomAgent("Random"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)


print("\n=== 11. Rule vs RL ===")
run_self_play(
    RuleAgent("Rule"),
    rl,
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)


print("\n=== 12. RL vs Rule ===")
run_self_play(
    rl,
    RuleAgent("Rule"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)


print("\n=== 13. MCTS vs Random ===")
run_self_play(
    MCTSAgent("MCTS"),
    RandomAgent("Random"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)

print("\n=== 14. Random vs MCTS ===")
run_self_play(
    RandomAgent("Random"),
    MCTSAgent("MCTS"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)


print("\n=== 15. MCTS vs Rule ===")
run_self_play(
    MCTSAgent("MCTS"),
    RuleAgent("Rule"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)


print("\n=== 16. Rule vs MCTS ===")
run_self_play(
    RuleAgent("Rule"),
    MCTSAgent("MCTS"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)


print("\n=== 17. MCTS vs RL ===")
run_self_play(
    MCTSAgent("MCTS"),
    rl,
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)


print("\n=== 18. RL vs MCTS ===")
run_self_play(
    rl,
    MCTSAgent("MCTS"),
    num_games=num_games,
    save_logs=False,
    save_dataset=False,
    base_seed=0,
)
