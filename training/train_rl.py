"""
Training loop for the DQN reinforcement learning agent.

ROLE IN SYSTEM
--------------
train_rl.py is the only script that trains RLAgent.  It runs self-play
games, fills a replay buffer with (state, action_vec, reward, next_state,
done) rows, and calls agent.train_step() in batches.

After training completes the final weights are saved to
data/models/rl_agent.pt, which is the file main.py loads for benchmarking.

WARM-START (recommended)
------------------------
If data/models/rl_agent_pretrained.pt exists (produced by pretrain_rl.py),
this script loads those weights before starting RL training.  The agent
then begins with RuleAgent's strategy already encoded and improves from
there rather than learning from scratch.

Run order:
    1. python training/pretrain_rl.py    (creates rl_agent_pretrained.pt)
    2. python training/train_rl.py       (loads pretrained weights, fine-tunes)
    3. python main.py                    (benchmarks rl_agent.pt)

If rl_agent_pretrained.pt does not exist, training starts from random
weights as before.

HOW TO RUN
----------
    python training/train_rl.py

Progress is printed every LOG_EVERY episodes.  The checkpoint is written
at the end of training and also every CHECKPOINT_EVERY episodes so a
crash does not lose all progress.

CURRICULUM
----------
Training proceeds in three phases against progressively stronger opponents.
Switching opponent mid-training is the single most reliable way to avoid
the RL agent learning a narrow policy that only works against one strategy.

    Phase 1 (episodes 0      - PHASE2_START-1) : vs RandomAgent
    Phase 2 (episodes PHASE2_START - PHASE3_START-1) : vs RuleBadAgent
    Phase 3 (episodes PHASE3_START - NUM_EPISODES-1) : vs RuleAgent

When warm-starting, Phase 1 is shortened (PHASE2_START is lower) because
the agent already beats Random from the first episode.

TRAINING LOOP (per episode)
----------------------------
1. Instantiate a fresh UnoGame with RLAgent as player 0.
2. Call game.play_turn() until the game ends or max turns is reached.
3. For every turn where RLAgent played a card:
       a. Encode the played card via encode_card().
       b. Store (state, action_vec, reward=0, next_state, done) in buffer.
4. Backfill the terminal reward (+1/-1/-0.5) into all rows for this game.
5. Once buffer has >= BATCH_SIZE rows, sample a batch and call train_step().
6. Every TARGET_SYNC_EVERY steps, sync the target network.
7. Decay epsilon linearly from EPSILON_START to EPSILON_END.

REPLAY BUFFER
-------------
A fixed-size deque (collections.deque with maxlen=BUFFER_SIZE).
When full, the oldest experience is evicted automatically.
Sampling is uniform random -- no prioritised replay.

ACTION_VEC FIELD
----------------
self_play.py stores action_index (an integer) but train_step() needs the
21-float card encoding.  This script adds "action_vec" to each row so
train_step() has direct access without re-encoding.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
from collections import deque
from typing import Deque, List, Optional

from core.cards import CardType
from core.game import UnoGame, TurnResult
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent
from agents.bad_rule_agent import RuleBadAgent
from agents.rl_agent import RLAgent
from training.feature_encoder import encode_state, encode_card
from config import TrainingConfig, Paths

# ── Hyperparameters ───────────────────────────────────────────────────────────

NUM_EPISODES: int = 30_000
BATCH_SIZE: int = TrainingConfig.BATCH_SIZE
BUFFER_SIZE: int = 50_000
TARGET_SYNC_EVERY: int = 500
CHECKPOINT_EVERY: int = 5_000
LOG_EVERY: int = 1_000

EPSILON_START: float = 1.0
EPSILON_END: float = 0.05

# Curriculum phase boundaries
# When warm-starting, Phase 1 is shorter because the agent already plays well
PHASE2_START_COLD: int = 10_000  # used when starting from random weights
PHASE2_START_WARM: int = 5_000  # used when warm-starting from pretrained
PHASE3_START: int = 20_000

LEARNING_RATE: float = TrainingConfig.LEARNING_RATE
GAMMA: float = TrainingConfig.GAMMA

# When warm-starting, start with lower epsilon -- the agent already has a
# reasonable policy so less random exploration is needed early on
EPSILON_START_WARM: float = 0.3

CHECKPOINT_PATH: str = str(Paths.MODELS / "rl_agent.pt")
PRETRAINED_PATH: str = str(Paths.MODELS / "rl_agent_pretrained.pt")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _pick_opponent(episode: int, phase2_start: int):
    """Returning the appropriate opponent agent for the current episode.

    Args:
        episode:      Current episode index (0-based).
        phase2_start: Episode at which to switch from Random to RuleBad.
                      Lower when warm-starting (agent needs less Random time).

    Returning an initialised opponent BaseAgent.
    """
    if episode < phase2_start:
        return RandomAgent("Random")
    if episode < PHASE3_START:
        return RuleBadAgent("RuleBad")
    return RuleAgent("Rule")


def _terminal_reward(winner: int, player: int) -> float:
    """Returning terminal reward for a player given the game winner.

    Args:
        winner: Winning player index, or -1 for draw/timeout.
        player: Player index to compute reward for.

    Returning float reward value from TrainingConfig.
    """
    if winner == -1:
        return TrainingConfig.REWARD_DRAW
    if winner == player:
        return TrainingConfig.REWARD_WIN
    return TrainingConfig.REWARD_LOSE


def _intermediate_reward(result: TurnResult, pre_hand: int, post_hand: int) -> float:
    """Computing a small shaped reward for the current turn.

    Intermediate rewards give the agent a denser learning signal than
    the terminal reward alone.  They are added on top of the terminal
    reward, not instead of it.

    Args:
        result:    TurnResult from game.play_turn().
        pre_hand:  Agent's hand size before this turn.
        post_hand: Agent's hand size after this turn.

    Returning a small float reward (typically in [-0.2, 0.3]).
    """
    r = 0.0
    card = result.card_played
    if card is None:
        return r

    # Reward for playing aggressive action cards
    if card.card_type == CardType.WILD_DRAW_FOUR:
        r += 0.3
    elif card.card_type == CardType.DRAW_TWO:
        r += 0.2
    elif card.card_type in (CardType.SKIP, CardType.REVERSE):
        r += 0.1

    # Reward for reducing hand size
    r += 0.05 * (pre_hand - post_hand)

    return r


def _collect_episode(
    agent: RLAgent,
    opponent,
    episode_seed: Optional[int],
) -> tuple:
    """Playing one game and collecting experience rows for the RL agent only.

    Only turns where RLAgent (player 0) played a card are stored.
    Rows are returned with the terminal reward already backfilled.

    Each row contains:
        "player"     : int
        "state"      : List[float]
        "action_vec" : List[float]   -- encode_card(card_played)
        "reward"     : float         -- terminal + intermediate reward
        "next_state" : List[float] | None
        "done"       : bool

    Args:
        agent:        RLAgent instance (player 0).
        opponent:     Any BaseAgent instance (player 1).
        episode_seed: Optional random seed for this game.

    Returning (winner, list of experience rows).
    """
    game = UnoGame(agent, opponent, seed=episode_seed)
    rows: List[dict] = []
    winner = -1

    while game.turn_count < game.max_turns:
        game.turn_count += 1
        pre_state = game.get_state()
        pre_hand = pre_state.hand_sizes[0]
        player = game.current_player

        result: TurnResult = game.play_turn()

        if player == 0 and result.card_played is not None:
            post_hand = game.get_state().hand_sizes[0]
            next_state_vec = (
                encode_state(game.get_state()) if result.winner is None else None
            )
            inter_r = _intermediate_reward(result, pre_hand, post_hand)
            rows.append(
                {
                    "player": player,
                    "state": encode_state(pre_state),
                    "action_vec": encode_card(result.card_played),
                    "reward": inter_r,  # terminal reward added below
                    "next_state": next_state_vec,
                    "done": result.winner is not None,
                }
            )

        if result.winner is not None:
            winner = result.winner
            break

    # Backfilling terminal reward into all rows
    terminal_r = _terminal_reward(winner, player=0)
    for row in rows:
        row["reward"] += terminal_r

    return winner, rows


# ── Training Loop ─────────────────────────────────────────────────────────────


def train() -> None:
    """Running the full DQN training loop and saving the final checkpoint."""
    Paths.ensure_dirs()

    # Checking for pretrained weights to warm-start from
    warm_start = os.path.exists(PRETRAINED_PATH)
    phase2_start = PHASE2_START_WARM if warm_start else PHASE2_START_COLD
    epsilon_start = EPSILON_START_WARM if warm_start else EPSILON_START

    agent = RLAgent(
        name="RL",
        epsilon=epsilon_start,
        lr=LEARNING_RATE,
        gamma=GAMMA,
    )

    if warm_start:
        agent.load(PRETRAINED_PATH)
        print(f"  Warm-start: loaded pretrained weights from {PRETRAINED_PATH}")
    else:
        print("  Cold-start: training from random weights.")
        print("  Tip: run training/pretrain_rl.py first for better results.")

    # Recomputing epsilon decay based on actual start value
    eps_decay = (epsilon_start - EPSILON_END) / NUM_EPISODES
    agent.epsilon = epsilon_start

    buffer: Deque[dict] = deque(maxlen=BUFFER_SIZE)
    total_steps = 0
    wins = 0
    losses = 0
    draws = 0
    total_loss = 0.0
    loss_updates = 0

    print("\n" + "=" * 55)
    print("  DQN Training -- UNO RL Agent")
    print(f"  Episodes    : {NUM_EPISODES:,}")
    print(f"  Warm-start  : {warm_start}")
    print(f"  Epsilon     : {epsilon_start:.2f} -> {EPSILON_END:.2f}")
    print(f"  Phase 1 end : episode {phase2_start:,}  (vs Random)")
    print(f"  Phase 2 end : episode {PHASE3_START:,}  (vs RuleBad)")
    print(f"  Phase 3 end : episode {NUM_EPISODES:,}  (vs Rule)")
    print("=" * 55)

    for episode in range(NUM_EPISODES):
        opponent = _pick_opponent(episode, phase2_start)
        seed = episode

        winner, rows = _collect_episode(agent, opponent, seed)

        buffer.extend(rows)

        if winner == 0:
            wins += 1
        elif winner == -1:
            draws += 1
        else:
            losses += 1

        if len(buffer) >= BATCH_SIZE:
            batch = random.sample(buffer, BATCH_SIZE)
            loss = agent.train_step(batch)
            total_loss += loss
            loss_updates += 1
            total_steps += 1

            if total_steps % TARGET_SYNC_EVERY == 0:
                agent.sync_target_network()

        # Decaying epsilon
        agent.epsilon = max(EPSILON_END, agent.epsilon - eps_decay)

        if (episode + 1) % CHECKPOINT_EVERY == 0:
            agent.save(CHECKPOINT_PATH)

        if (episode + 1) % LOG_EVERY == 0:
            played = wins + losses + draws
            win_rate = 100 * wins / max(played, 1)
            avg_loss = total_loss / max(loss_updates, 1)
            phase = 1 if episode < phase2_start else 2 if episode < PHASE3_START else 3
            opponent_name = (
                "RandomAgent"
                if episode < phase2_start
                else "RuleBadAgent"
                if episode < PHASE3_START
                else "RuleAgent"
            )
            print(
                f"  ep {episode + 1:>6,}  "
                f"phase {phase} ({opponent_name:<12})  "
                f"eps={agent.epsilon:.3f}  "
                f"win%={win_rate:5.1f}  "
                f"buf={len(buffer):>6,}  "
                f"loss={avg_loss:.4f}"
            )
            wins = losses = draws = 0
            total_loss = 0.0
            loss_updates = 0

    agent.save(CHECKPOINT_PATH)
    print(f"\n  Training complete.  Weights saved to {CHECKPOINT_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    train()
