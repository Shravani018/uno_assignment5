"""
Converting a GameState snapshot into a fixed-length numerical feature vector.

ROLE IN SYSTEM
--------------
feature_encoder.py is the bridge between the game engine and any
gradient-based model (RL, supervised). It must be:
  - Stateless: same input always produces the same output.
  - Framework-agnostic: returns a plain Python list, compatible with
    numpy, torch, and any other downstream consumer.
  - Consistent: vector length never changes at runtime (enforced by assert).

VECTOR LAYOUT  (total = EncoderConfig.vector_size() = 29 features)
-------------------------------------------------------------------
Indices   Width   Description
-------   -----   -----------
  0-  4     5     top_card color     one-hot (RED GREEN BLUE YELLOW WILD)
  5- 10     6     top_card type      one-hot (NUMBER SKIP REVERSE DRAW_TWO WILD WD4)
 11- 20    10     top_card value     one-hot (0-9); all zeros for non-number cards
 21- 25     5     current_color      one-hot (same 5-color order as above)
 26- 28     3     scalars            my_hand_size/20, opp_hand_size/20,
                                     draw_pile_size/108  -- normalised to [0, 1]

Total: 29 features

DESIGN DECISIONS
----------------
- One-hot encoding for all categoricals: no false ordinal relationships
  between colors or card types (e.g. RED=0 < BLUE=2 would be meaningless).
- Scalar normalisation by practical maximum keeps all values in [0, 1],
  which stabilises gradient-based optimisers.
- encode_card() is provided separately for agents that need to encode
  individual cards (e.g. action head in RL policy network).
"""
from typing import List, Optional
from core.cards import Card, Color, CardType
from core.game import GameState
from config import EncoderConfig, GameConfig


# Fixing iteration order so indices are stable across runs
_COLORS: List[Color] = [
    Color.RED,
    Color.GREEN,
    Color.BLUE,
    Color.YELLOW,
    Color.WILD,
]

_CARD_TYPES: List[CardType] = [
    CardType.NUMBER,
    CardType.SKIP,
    CardType.REVERSE,
    CardType.DRAW_TWO,
    CardType.WILD,
    CardType.WILD_DRAW_FOUR,
]

_MAX_HAND: float = 20.0
_MAX_DECK: float = float(GameConfig.DECK_SIZE)


def encode_state(state: GameState) -> List[float]:
    """Encoding a full GameState into a fixed-length feature vector.

    Args:
        state: Current game state snapshot.

    Returning a list of floats of length EncoderConfig.vector_size().
    Raises AssertionError if vector length does not match config.
    """
    vec: List[float] = []

    # Encoding top card: color, type, numeric value
    vec.extend(_one_hot_color(state.top_card.color))
    vec.extend(_one_hot_card_type(state.top_card.card_type))
    vec.extend(_one_hot_value(state.top_card.value))

    # Encoding active color separately (may differ from top card after wild)
    vec.extend(_one_hot_color(state.current_color))

    # Encoding normalised scalar features
    my_idx = state.current_player
    opp_idx = 1 - my_idx
    vec.append(min(state.hand_sizes[my_idx], _MAX_HAND) / _MAX_HAND)
    vec.append(min(state.hand_sizes[opp_idx], _MAX_HAND) / _MAX_HAND)
    vec.append(min(state.draw_pile_size, _MAX_DECK) / _MAX_DECK)

    assert len(vec) == EncoderConfig.vector_size(), (
        f"Vector length mismatch: got {len(vec)}, "
        f"expected {EncoderConfig.vector_size()}"
    )
    return vec


def encode_card(card: Card) -> List[float]:
    """Encoding a single card into a partial feature vector.

    Useful for encoding action candidates in a policy network.

    Args:
        card: Any Card object.

    Returning a list of floats of length
    NUM_COLORS + NUM_CARD_TYPES + NUM_VALUES = 21.
    """
    vec: List[float] = []
    vec.extend(_one_hot_color(card.color))
    vec.extend(_one_hot_card_type(card.card_type))
    vec.extend(_one_hot_value(card.value))
    return vec


def _one_hot_color(color: Color) -> List[float]:
    """Encoding a Color enum as a one-hot vector of length NUM_COLORS."""
    return [1.0 if c == color else 0.0 for c in _COLORS]


def _one_hot_card_type(card_type: CardType) -> List[float]:
    """Encoding a CardType enum as a one-hot vector of length NUM_CARD_TYPES."""
    return [1.0 if t == card_type else 0.0 for t in _CARD_TYPES]


def _one_hot_value(value: Optional[int]) -> List[float]:
    """Encoding a card's numeric value as a one-hot vector of length NUM_VALUES.

    Returning all zeros for non-number cards (value is None).
    """
    if value is None:
        return [0.0] * EncoderConfig.NUM_VALUES
    return [1.0 if i == value else 0.0 for i in range(EncoderConfig.NUM_VALUES)]