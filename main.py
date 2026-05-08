# This script runs self-play matches between different agents and collects data for training.
import sys
sys.path.insert(0, '.')
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from training.self_play import run_self_play
max=10_000 #no of games to run for each matchup
print("=== 1. Random vs Random ===")
run_self_play(RandomAgent("Random-A"), RandomAgent("Random-B"),
              num_games=max, save_logs=False, save_dataset=False, base_seed=0)

print("\n=== 2. Rule vs Random ===")
run_self_play(RuleAgent("Rule"), RandomAgent("Random"),
              num_games=max, save_logs=False, save_dataset=False, base_seed=0)

print("\n=== 3. Rule vs Rule ===")
run_self_play(RuleAgent("Rule-A"), RuleAgent("Rule-B"),
              num_games=max, save_logs=False, save_dataset=True, base_seed=0)