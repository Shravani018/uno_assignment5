"""Defining UNO card structure and types."""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
class Color(str, Enum):
    """Representing valid card colors."""
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    WILD = "wild"
class CardType(str, Enum):
    """Representing all valid card types."""
    NUMBER = "number"
    SKIP = "skip"
    REVERSE = "reverse"
    DRAW_TWO = "draw_two"
    WILD = "wild"
    WILD_DRAW_FOUR = "wild_draw_four"
@dataclass(frozen=True)
class Card:
    """Representing a single UNO card.

    Args:
        color: Card color (wild for wild cards).
        card_type: Type of the card.
        value: Numeric value (0-9) for number cards, None for action/wild.
    """
    color: Color
    card_type: CardType
    value: Optional[int] = None

    def is_wild(self) -> bool:
        """Checking if card is a wild type."""
        return self.color == Color.WILD

    def is_action(self) -> bool:
        """Checking if card is a non-number action card."""
        return self.card_type in (
            CardType.SKIP,
            CardType.REVERSE,
            CardType.DRAW_TWO,
            CardType.WILD,
            CardType.WILD_DRAW_FOUR,
        )

    def point_value(self) -> int:
        """Returning point value used for hand scoring.
        Number cards = face value, action = 20, wilds = 50.
        """
        if self.card_type == CardType.NUMBER:
            return self.value
        if self.card_type in (CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO):
            return 20
        return 50

    def __str__(self) -> str:
        """Returning human-readable card string."""
        if self.card_type == CardType.NUMBER:
            return f"{self.color.value} {self.value}"
        if self.is_wild():
            return self.card_type.value.replace("_", " ")
        return f"{self.color.value} {self.card_type.value.replace('_', ' ')}"