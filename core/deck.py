"""Managing UNO deck construction, shuffling, and drawing."""
import random
from typing import List, Optional
from core.cards import Card, Color, CardType
COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW]

def build_deck() -> List[Card]:
    """Building a standard 108-card UNO deck.

    Returning shuffled list of Card objects.
    """
    cards = []
    for color in COLORS:
        # Adding one 0 per color
        cards.append(Card(color, CardType.NUMBER, 0))
        # Adding two of each 1-9, skip, reverse, draw_two per color
        for value in range(1, 10):
            cards.extend([Card(color, CardType.NUMBER, value)] * 2)
        for action in (CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO):
            cards.extend([Card(color, action)] * 2)
    # Adding 4 wild and 4 wild draw four
    for _ in range(4):
        cards.append(Card(Color.WILD, CardType.WILD))
        cards.append(Card(Color.WILD, CardType.WILD_DRAW_FOUR))
    random.shuffle(cards)
    return cards


class Deck:
    """Handling draw pile and discard pile operations.

    Args:
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialising deck with optional seed."""
        if seed is not None:
            random.seed(seed)
        self.draw_pile: List[Card] = build_deck()
        self.discard_pile: List[Card] = []

    def draw(self) -> Card:
        """Drawing top card from draw pile, reshuffling discard if needed."""
        if not self.draw_pile:
            self._reshuffle()
        return self.draw_pile.pop()

    def discard(self, card: Card) -> None:
        """Placing card onto discard pile."""
        self.discard_pile.append(card)

    def top_card(self) -> Card:
        """Returning current top card of discard pile."""
        return self.discard_pile[-1]

    def setup_first_card(self) -> Card:
        """Drawing starting card, skipping wilds for initial discard."""
        card = self.draw()
        while card.is_wild():
            self.draw_pile.insert(0, card)
            card = self.draw()
        self.discard(card)
        return card

    def _reshuffle(self) -> None:
        """Reshuffling discard pile back into draw pile, keeping top card."""
        top = self.discard_pile.pop()
        self.draw_pile = self.discard_pile[:]
        random.shuffle(self.draw_pile)
        self.discard_pile = [top]

    @property
    def draw_pile_size(self) -> int:
        """Returning number of cards remaining in draw pile."""
        return len(self.draw_pile)