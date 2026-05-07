"""Managing a player's hand of cards."""
from typing import List, Optional
from core.cards import Card


class Hand:
    """Representing and managing cards held by a player.

    Args:
        cards: Initial list of cards (default empty).
    """

    def __init__(self, cards: Optional[List[Card]] = None):
        """Initialising hand with optional starting cards."""
        self.cards: List[Card] = cards if cards is not None else []

    def add(self, card: Card) -> None:
        """Adding a single card to the hand."""
        self.cards.append(card)

    def add_many(self, cards: List[Card]) -> None:
        """Adding multiple cards to the hand."""
        self.cards.extend(cards)

    def remove(self, card: Card) -> None:
        """Removing a specific card from the hand."""
        self.cards.remove(card)

    def is_empty(self) -> bool:
        """Checking if hand has no cards remaining."""
        return len(self.cards) == 0

    def size(self) -> int:
        """Returning number of cards in hand."""
        return len(self.cards)

    def total_points(self) -> int:
        """Calculating total point value of cards remaining in hand."""
        return sum(card.point_value() for card in self.cards)

    def __repr__(self) -> str:
        """Returning indexed string of hand for CLI display."""
        return "\n".join(f"  [{i}] {card}" for i, card in enumerate(self.cards))