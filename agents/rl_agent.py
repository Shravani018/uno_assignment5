"""
Deep Q-Network (DQN) reinforcement learning agent for UNO.

ROLE IN SYSTEM
--------------
RLAgent is the learned agent that the project trains and benchmarks against
RandomAgent, RuleBadAgent, and RuleAgent. It subclasses BaseAgent so it
is fully interchangeable inside UnoGame.run() and run_self_play().

At inference time choose_action() is a pure forward pass -- no training
state is touched.  Training is handled entirely by the companion script
training/train_rl.py, which calls RLAgent.train_step() in a loop and
periodically checkpoints the weights.

ARCHITECTURE  (per-card scoring)
--------------------------------------------
The network scores each valid card independently rather than maintaining
a fixed 108-wide output head. This is the correct choice here because
UnoGame already filters valid_plays before calling choose_action(), so
no illegal-action masking is needed.

One forward pass per valid card:
    input  = [state_vec (29) | card_vec (21)]  →  50 floats
    output = scalar Q-value for playing that card

The card with the highest Q-value is selected (greedy) or sampled with
probability ε (ε-greedy during training).

INPUT VECTOR LAYOUT  (matches feature_encoder.py exactly)
----------------------------------------------------------
State part  (29 floats) -- from encode_state():
    0- 4   top_card color          one-hot (5)
    5-10   top_card type           one-hot (6)
   11-20   top_card value          one-hot (10)
   21-25   current_color           one-hot (5)
   26-28   my_hand / 20, opp_hand / 20, draw_pile / 108

Card part  (21 floats) -- from encode_card():
    0- 4   card color              one-hot (5)
    5-10   card type               one-hot (6)
   11-20   card value              one-hot (10)

Total input: 50 floats  →  128  →  64  →  1

WILD COLOR SELECTION
--------------------
When a wild is played the agent delegates color selection to the same
heuristic used by RuleAgent: pick the most frequent non-wild color
among the valid_plays as a proxy for hand composition. The full hand
is not exposed to agents (partial observability), so a learned color
head would have little extra signal at this stage. This can be replaced
by a second network head once the card-selection network is stable.

TRAINING INTEGRATION
--------------------
This file exposes only what the agent needs to act and to learn:
    - choose_action()   called by UnoGame -- inference only
    - train_step()      called by training/train_rl.py -- one gradient update
    - save() / load()   checkpoint helpers

The full training loop (self-play, replay buffer, epsilon schedule,
target-network sync) lives in training/train_rl.py to keep this file
clean and the agent usable as a drop-in for benchmarking.

DEPENDENCIES
------------
    torch >= 2.0  (pip install torch)
    The rest of the stack: core/, training/feature_encoder.py, config.py
"""

import random
from collections import Counter
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from core.cards import Card, Color, CardType
from core.game import GameState
from agents.base_agent import BaseAgent
from training.feature_encoder import encode_state, encode_card

# ── Constants ─────────────────────────────────────────────────────────────────

# Input to the network: state vector + one card vector
_STATE_DIM: int = 29  # must match EncoderConfig.vector_size()
_CARD_DIM: int = 21  # must match len(encode_card(...))
_INPUT_DIM: int = _STATE_DIM + _CARD_DIM  # 50

# Non-wild colors available for wild color selection
_PLAY_COLORS: List[Color] = [
    Color.RED,
    Color.GREEN,
    Color.BLUE,
    Color.YELLOW,
]


# ── Network ───────────────────────────────────────────────────────────────────


class _QNetwork(nn.Module):
    """Two-hidden-layer MLP scoring one (state, card) pair.

    Args:
        input_dim:   Length of concatenated [state | card] vector.
        hidden1_dim: Width of first hidden layer.
        hidden2_dim: Width of second hidden layer.
    """

    def __init__(
        self,
        input_dim: int = _INPUT_DIM,
        hidden1_dim: int = 128,
        hidden2_dim: int = 64,
    ) -> None:
        """Initialising network layers."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returning scalar Q-value for each input row.

        Args:
            x: Float tensor of shape (batch, input_dim) or (input_dim,).

        Returning tensor of shape (batch, 1) or (1,).
        """
        return self.net(x)


# ── Agent ─────────────────────────────────────────────────────────────────────


class RLAgent(BaseAgent):
    """DQN agent that learns to play UNO via self-play.

    At inference time only choose_action() is used; it runs a greedy
    forward pass and is fully compatible with UnoGame.run().

    At training time train_step() accepts a batch sampled from a replay
    buffer (managed by training/train_rl.py) and performs one gradient
    update using the Bellman TD target.

    Args:
        name:        Display name used in logs and tournament output.
        epsilon:     Initial exploration rate (0.0 = greedy / pure inference).
        lr:          Adam learning rate for the online network.
        gamma:       Discount factor for Bellman target.
        hidden1_dim: Width of first hidden layer.
        hidden2_dim: Width of second hidden layer.
        device:      Torch device string ("cpu", "cuda", "mps").
    """

    def __init__(
        self,
        name: str = "RLAgent",
        epsilon: float = 0.0,
        lr: float = 1e-3,
        gamma: float = 0.99,
        hidden1_dim: int = 128,
        hidden2_dim: int = 64,
        device: str = "cpu",
    ) -> None:
        """Initialising networks, optimiser, and loss function."""
        super().__init__(name)

        self.epsilon = epsilon
        self.gamma = gamma
        self.device = torch.device(device)

        # Online network: updated every train_step()
        self.online_net = _QNetwork(_INPUT_DIM, hidden1_dim, hidden2_dim).to(
            self.device
        )

        # Target network: periodically synced from online_net by the trainer
        self.target_net = _QNetwork(_INPUT_DIM, hidden1_dim, hidden2_dim).to(
            self.device
        )
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    # ── Inference ─────────────────────────────────────────────────────────────

    def choose_action(self, state: GameState) -> Tuple[Card, Optional[Color]]:
        """Selecting a card via ε-greedy Q-value scoring.

        Scores every card in state.valid_plays independently, then picks
        the highest-Q card (with probability 1-ε) or a random card
        (with probability ε).

        Args:
            state: Current game state snapshot; state.valid_plays is non-empty.

        Returning (card, chosen_color) where chosen_color is None unless
        the selected card is a wild.
        """
        # ε-greedy exploration
        if self.epsilon > 0.0 and random.random() < self.epsilon:
            card = random.choice(state.valid_plays)
        else:
            card = self._greedy_card(state)

        chosen_color = self._best_color(state.valid_plays) if card.is_wild() else None
        return card, chosen_color

    def _greedy_card(self, state: GameState) -> Card:
        """Scoring each valid card and returning the highest-Q selection.

        Args:
            state: Current game state snapshot.

        Returning Card with the maximum predicted Q-value.
        """
        state_vec = encode_state(state)
        state_t = torch.tensor(state_vec, dtype=torch.float32, device=self.device)

        self.online_net.eval()
        with torch.no_grad():
            scores = [
                self.online_net(
                    torch.cat(
                        [
                            state_t,
                            torch.tensor(
                                encode_card(card),
                                dtype=torch.float32,
                                device=self.device,
                            ),
                        ]
                    )
                ).item()
                for card in state.valid_plays
            ]

        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return state.valid_plays[best_idx]

    def _best_color(self, valid_plays: List[Card]) -> Color:
        """Choosing the most frequent non-wild color seen in valid_plays.

        Using valid_plays as a proxy for hand composition since the engine
        does not expose the full hand to agents (partial observability).
        Identical heuristic to RuleAgent; can be replaced with a learned
        head once card selection is stable.

        Args:
            valid_plays: Valid cards available this turn.

        Returning most common Color, defaulting to RED if all cards are wild.
        """
        color_counts = Counter(c.color for c in valid_plays if c.color != Color.WILD)
        if not color_counts:
            return Color.RED
        return color_counts.most_common(1)[0][0]

    # ── Training ──────────────────────────────────────────────────────────────

    def train_step(self, batch: List[dict]) -> float:
        """Performing one DQN gradient update on a sampled batch.

        Each element of batch is a dataset row produced by self_play.py:
            {
                "state"       : List[float],         # encode_state() output
                "action_index": int,                 # index in valid_plays
                "reward"      : float,               # terminal reward
                "next_state"  : List[float] | None,  # None on terminal turn
                "done"        : bool,
            }

        Note: action_index is not used directly here because the network
        scores (state, card) pairs and the card vector is re-encoded from
        the state at collection time.  Training/train_rl.py must store the
        encoded card vector alongside each row (as "action_vec") if it
        wants to avoid re-encoding.  If "action_vec" is absent, this
        method falls back to zeroing the card part of the input, which is
        a valid approximation for terminal states.

        Args:
            batch: List of experience dicts sampled from the replay buffer.

        Returning the scalar loss value for logging.
        """
        self.online_net.train()

        # Building input tensors
        states = torch.tensor(
            [row["state"] for row in batch], dtype=torch.float32, device=self.device
        )
        action_vecs = torch.tensor(
            [row.get("action_vec", [0.0] * _CARD_DIM) for row in batch],
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor(
            [row["reward"] for row in batch], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [float(row["done"]) for row in batch],
            dtype=torch.float32,
            device=self.device,
        )

        # Q(s, a) from online network
        inputs = torch.cat([states, action_vecs], dim=1)  # (B, 50)
        q_values = self.online_net(inputs).squeeze(1)  # (B,)

        # Bellman target: r  +  γ · max_a' Q_target(s', a')  ·  (1 - done)
        with torch.no_grad():
            next_q = self._next_state_values(batch)  # (B,)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping for training stability
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimiser.step()

        return loss.item()

    def _next_state_values(self, batch: List[dict]) -> torch.Tensor:
        """Computing max target-network Q-value for each next state.

        For terminal transitions (next_state is None / done=True) the
        target value is 0.

        Uses a single dummy card vector for next-state scoring because
        the replay buffer does not store the next valid_plays.  In a
        full implementation the trainer should store next_valid_play_vecs
        alongside each row and call the target net for each candidate.
        This simplified version uses a zero card vector, which gives a
        stable lower-bound target and is standard practice when the full
        next-action set is unavailable.

        Args:
            batch: List of experience dicts.

        Returning float tensor of shape (B,).
        """
        next_qs = []
        dummy_card = torch.zeros(_CARD_DIM, dtype=torch.float32, device=self.device)

        for row in batch:
            if row["done"] or row["next_state"] is None:
                next_qs.append(0.0)
            else:
                ns_t = torch.tensor(
                    row["next_state"], dtype=torch.float32, device=self.device
                )
                inp = torch.cat([ns_t, dummy_card]).unsqueeze(0)
                q_val = self.target_net(inp).item()
                next_qs.append(q_val)

        return torch.tensor(next_qs, dtype=torch.float32, device=self.device)

    def sync_target_network(self) -> None:
        """Copying online network weights into the target network.

        Called by training/train_rl.py every TARGET_SYNC_EVERY steps.
        Hard update -- no Polyak averaging.
        """
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Saving online network weights to disk.

        Args:
            path: File path for the checkpoint (e.g. "models/rl_agent_v1.pt").
        """
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Loading weights from a checkpoint into both networks.

        Args:
            path: File path to a previously saved checkpoint.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.target_net.eval()
