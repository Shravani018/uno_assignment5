"""
Supervised pretraining: teaching RLAgent to imitate RuleAgent.

ROLE IN SYSTEM
--------------
pretrain_rl.py runs BEFORE train_rl.py.  It generates games where
RuleAgent plays both sides, then trains the RLAgent network to predict
which card RuleAgent would play in each state.  The resulting weights
are saved to data/models/rl_agent_pretrained.pt.

train_rl.py then loads this checkpoint instead of starting from random
weights, so RL fine-tuning begins from a competent policy rather than
pure noise.  This is called warm-starting or behavioural cloning.

WHY THIS HELPS
--------------
Cold-start DQN on UNO fills the replay buffer with mostly losses before
the agent learns anything useful, which slows or prevents learning.
Warm-starting with RuleAgent behaviour means:
  - Phase 1 (vs Random) is skipped or shortened -- the agent already
    beats Random from episode 1.
  - The replay buffer fills with higher-quality experience immediately.
  - RL fine-tuning starts from a ~55% win rate and improves from there
    rather than climbing from ~30%.

ALGORITHM  (Behavioural Cloning)
---------------------------------
1. Run NUM_PRETRAIN_GAMES games of RuleAgent vs RuleAgent using
   run_self_play() -- these games already exist in the JSONL dataset
   if main.py has been run (rule-a_vs_rule-b.jsonl).  If not, we
   generate them here.
2. For each turn in the dataset:
       state_vec   = row["state"]          (51 floats)
       action_idx  = row["action_index"]   (int, index into valid_plays)
   We cannot recover the card object from action_index alone, but we
   CAN recover the card encoding because the dataset was built from
   TurnResult which contains pre_state.valid_plays.  Since we do not
   store valid_plays in the JSONL, we regenerate the card encoding by
   re-running the game with a fixed seed and capturing encode_card()
   for every played card.
3. Training objective: minimise MSE between the network's Q-value for
   the chosen card and a fixed high target (IMITATION_TARGET = 1.0),
   and between the Q-value for all unchosen cards and a fixed low
   target (IMITATION_LOW = -1.0).
   This is a ranking loss implemented as MSE for simplicity.  It pushes
   the network to score the RuleAgent card highest without requiring a
   softmax or cross-entropy head.

LOSS FUNCTION DETAIL
--------------------
For each turn we have:
    chosen card   -> target Q = +1.0
    all other valid cards -> target Q = -1.0

MSE loss = mean over all (input, target) pairs in the batch.
This is strictly supervised -- no Bellman equation, no rewards.

OUTPUT
------
Saves pretrained weights to data/models/rl_agent_pretrained.pt.
train_rl.py reads this path via PRETRAINED_PATH constant.

HOW TO RUN
----------
    python training/pretrain_rl.py

Then immediately run:
    python training/train_rl.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import random
from typing import List, Tuple

import torch

from core.game import UnoGame, TurnResult
from agents.rule_agent import RuleAgent
from agents.rl_agent import RLAgent
from training.feature_encoder import encode_state, encode_card
from config import Paths, TrainingConfig

# ── Hyperparameters ───────────────────────────────────────────────────────────

NUM_PRETRAIN_GAMES: int = 5_000  # games of Rule vs Rule to learn from
PRETRAIN_EPOCHS: int = 10  # passes over the collected dataset
PRETRAIN_LR: float = 5e-4  # lower than RL lr for stable imitation
BATCH_SIZE: int = TrainingConfig.BATCH_SIZE  # 64
IMITATION_TARGET: float = 1.0  # Q target for the chosen (RuleAgent) card
IMITATION_LOW: float = -1.0  # Q target for all unchosen cards
LOG_EVERY: int = 1  # log every epoch

OUTPUT_PATH: str = str(Paths.MODELS / "rl_agent_pretrained.pt")


# ── Data Collection ───────────────────────────────────────────────────────────
# In pretrain_rl.py, replace _collect_imitation_data() entirely:


def _collect_imitation_data(agent: "RLAgent") -> List[Tuple]:
    """Running Rule vs Rule games and collecting imitation samples.

    Uses agent._encode_state() to produce 54-float state vectors
    consistent with what the network expects at inference time.

    Args:
        agent: RLAgent instance used only for its _encode_state() method.

    Returning list of (state_vec, chosen_vec, unchosen_vecs) tuples.
    """
    rule_a = RuleAgent("Rule-A")
    rule_b = RuleAgent("Rule-B")
    samples = []

    agent = RLAgent(name="RL-pretrain", epsilon=0.0, lr=PRETRAIN_LR)

    print(f"  Collecting {NUM_PRETRAIN_GAMES:,} Rule vs Rule games...")

    for game_idx in range(NUM_PRETRAIN_GAMES):
        game = UnoGame(rule_a, rule_b, seed=game_idx)

        while game.turn_count < game.max_turns:
            game.turn_count += 1
            pre_state = game.get_state()
            result: TurnResult = game.play_turn()

            if result.card_played is None:
                continue

            state_vec = agent._encode_state(pre_state)  # CHANGED: 54 floats
            chosen_vec = encode_card(result.card_played)
            unchosen_vecs = [
                encode_card(c) for c in pre_state.valid_plays if c != result.card_played
            ]

            samples.append((state_vec, chosen_vec, unchosen_vecs))

            if result.winner is not None:
                break

        if (game_idx + 1) % 500 == 0:
            print(
                f"    [{game_idx + 1:,}/{NUM_PRETRAIN_GAMES:,}]  "
                f"samples so far: {len(samples):,}"
            )

    print(f"  Collected {len(samples):,} imitation samples total.\n")
    return samples


# ── Batch Builder ─────────────────────────────────────────────────────────────


def _build_batch(
    samples: List[Tuple],
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sampling a mini-batch of (input_vec, target_Q) pairs.

    For each sampled turn:
        - The chosen card gets target Q = IMITATION_TARGET  (+1.0)
        - Every unchosen card gets target Q = IMITATION_LOW (-1.0)

    Args:
        samples:    Full list of (state_vec, chosen_vec, unchosen_vecs).
        batch_size: Number of turns to sample (each turn may contribute
                    multiple input rows if there are many valid plays).
        device:     Torch device.

    Returning (inputs tensor, targets tensor) both of shape (N, 1)
    where N >= batch_size.
    """
    chosen_samples = random.sample(samples, min(batch_size, len(samples)))
    inputs_list = []
    targets_list = []

    for state_vec, chosen_vec, unchosen_vecs in chosen_samples:
        state_t = torch.tensor(state_vec, dtype=torch.float32)

        # Chosen card: high target
        inputs_list.append(
            torch.cat([state_t, torch.tensor(chosen_vec, dtype=torch.float32)])
        )
        targets_list.append(IMITATION_TARGET)

        # Unchosen cards: low target
        for uc_vec in unchosen_vecs:
            inputs_list.append(
                torch.cat([state_t, torch.tensor(uc_vec, dtype=torch.float32)])
            )
            targets_list.append(IMITATION_LOW)

    inputs = torch.stack(inputs_list).to(device)
    targets = torch.tensor(targets_list, dtype=torch.float32).to(device)
    return inputs, targets


# ── Pretraining Loop ──────────────────────────────────────────────────────────


def pretrain() -> None:
    """Running supervised imitation pretraining and saving the checkpoint."""
    Paths.ensure_dirs()

    # Initialising agent first so _encode_state() is available for data collection
    agent = RLAgent(name="RL-pretrain", epsilon=0.0, lr=PRETRAIN_LR)
    device = agent.device

    # pass agent into data collection
    samples = _collect_imitation_data(agent)

    """Running supervised imitation pretraining and saving the checkpoint."""
    Paths.ensure_dirs()

    print("=" * 55)
    print("  Behavioural Cloning Pretraining")
    print(f"  Games     : {NUM_PRETRAIN_GAMES:,}")
    print(f"  Epochs    : {PRETRAIN_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Output    : {OUTPUT_PATH}")
    print("=" * 55 + "\n")

    # Collecting imitation data from Rule vs Rule games
    samples = _collect_imitation_data(agent)

    # Initialising agent -- epsilon=0 during pretraining (no exploration needed)
    agent = RLAgent(name="RL-pretrain", epsilon=0.0, lr=PRETRAIN_LR)
    device = agent.device

    steps_per_epoch = max(len(samples) // BATCH_SIZE, 1)

    print(
        f"  Training for {PRETRAIN_EPOCHS} epochs ({steps_per_epoch} steps/epoch)...\n"
    )

    for epoch in range(PRETRAIN_EPOCHS):
        epoch_loss = 0.0

        agent.online_net.train()

        for _ in range(steps_per_epoch):
            inputs, targets = _build_batch(samples, BATCH_SIZE, device)

            # Forward pass
            q_values = agent.online_net(inputs).squeeze(1)
            loss = agent.loss_fn(q_values, targets)

            # Backward pass
            agent.optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.online_net.parameters(), max_norm=1.0)
            agent.optimiser.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch

        if (epoch + 1) % LOG_EVERY == 0:
            print(f"  Epoch {epoch + 1:>3}/{PRETRAIN_EPOCHS}  avg_loss={avg_loss:.4f}")

    # Syncing target network to match the pretrained online network
    agent.sync_target_network()

    # Saving pretrained weights
    agent.save(OUTPUT_PATH)
    print(f"\n  Pretraining complete.  Weights saved to {OUTPUT_PATH}")
    print("  Run training/train_rl.py next to start RL fine-tuning.")
    print("=" * 55)


if __name__ == "__main__":
    pretrain()
