"""Enforcing UNO card validity and applying special card effects."""
from typing import List, Optional
from core.cards import Card, CardType, Color


def is_valid_play(card: Card, top_card: Card, current_color: Color) -> bool:
    """Checking whether a card can legally be played.

    Args:
        card: Card being considered for play.
        top_card: Current top card on the discard pile.
        current_color: Active color (may differ from top_card if wild was played).

    Returning True if the play is legal.
    """
    if card.is_wild():
        return True
    if card.color == current_color:
        return True
    if card.card_type == top_card.card_type:
        # Matching action type
        if card.card_type != CardType.NUMBER:
            return True
        # Matching number
        return card.value == top_card.value
    return False


def get_valid_plays(hand_cards: List[Card], top_card: Card, current_color: Color) -> List[Card]:
    """Filtering hand to return only legally playable cards.

    Args:
        hand_cards: All cards in the player's hand.
        top_card: Current top card on the discard pile.
        current_color: Active color in play.

    Returning list of valid Card objects.
    """
    return [c for c in hand_cards if is_valid_play(c, top_card, current_color)]


def apply_card_effect(card: Card, chosen_color: Optional[Color] = None) -> dict:
    """Deriving the effect a played card has on game state.

    Args:
        card: The card that was just played.
        chosen_color: Color selected when playing a wild card.

    Returning effect dict with keys:
        - new_color (Color): Active color after this play.
        - opponent_draws (int): Number of cards opponent must draw.
        - skip_opponent (bool): Whether opponent loses their turn.
    """
    effect = {
        "new_color": card.color if not card.is_wild() else chosen_color,
        "opponent_draws": 0,
        "skip_opponent": False,
    }
    if card.card_type == CardType.DRAW_TWO:
        effect["opponent_draws"] = 2
        effect["skip_opponent"] = True
    elif card.card_type == CardType.WILD_DRAW_FOUR:
        effect["opponent_draws"] = 4
        effect["skip_opponent"] = True
    elif card.card_type in (CardType.SKIP, CardType.REVERSE):
        # Reverse acts as skip in 2-player mode
        effect["skip_opponent"] = True
    return effect