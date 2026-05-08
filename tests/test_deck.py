"""
Tests for core/deck.py

COVERING
--------
- Deck builds exactly 108 cards with correct composition
- Shuffling produces different order across seeds
- Drawing reduces pile by 1
- Discard and top_card work correctly
- Reshuffle triggers when draw pile exhausted
- setup_first_card always returns a colored number card (never wild,
  never action card -- per corrected UNO rule)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from collections import Counter
from core.deck import Deck, build_deck
from core.cards import Card, Color, CardType


class TestBuildDeck(unittest.TestCase):

    def setUp(self):
        """Building a fresh unshuffled deck for composition tests."""
        self.cards = build_deck()

    def test_total_cards(self):
        """Verifying deck contains exactly 108 cards."""
        self.assertEqual(len(self.cards), 108)

    def test_number_cards(self):
        """Verifying correct count of number cards per color.

        Each color: one 0, two each of 1-9 = 1 + 18 = 19 per color.
        4 colors × 19 = 76 total number cards.
        """
        numbers = [c for c in self.cards if c.card_type == CardType.NUMBER]
        self.assertEqual(len(numbers), 76)

    def test_action_cards_per_color(self):
        """Verifying 2 of each action card per color = 24 total action cards."""
        for color in [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW]:
            for action in [CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO]:
                count = sum(
                    1 for c in self.cards
                    if c.color == color and c.card_type == action
                )
                self.assertEqual(count, 2, f"Expected 2 {color} {action}, got {count}")

    def test_wild_cards(self):
        """Verifying 4 plain wilds and 4 wild draw fours."""
        wilds = [c for c in self.cards if c.card_type == CardType.WILD]
        wd4s = [c for c in self.cards if c.card_type == CardType.WILD_DRAW_FOUR]
        self.assertEqual(len(wilds), 4)
        self.assertEqual(len(wd4s), 4)

    def test_no_wild_has_color(self):
        """Verifying all wild cards carry Color.WILD, not a regular color."""
        for card in self.cards:
            if card.is_wild():
                self.assertEqual(card.color, Color.WILD)


class TestDeckOperations(unittest.TestCase):

    def test_draw_reduces_pile(self):
        """Verifying drawing one card reduces draw pile by 1."""
        d = Deck(seed=0)
        size_before = d.draw_pile_size
        d.draw()
        self.assertEqual(d.draw_pile_size, size_before - 1)

    def test_discard_and_top_card(self):
        """Verifying discarded card becomes the top card."""
        d = Deck(seed=0)
        card = d.draw()
        d.discard(card)
        self.assertEqual(d.top_card(), card)

    def test_different_seeds_differ(self):
        """Verifying different seeds produce different draw orders."""
        d0 = Deck(seed=0)
        d1 = Deck(seed=99)
        cards0 = [d0.draw() for _ in range(10)]
        cards1 = [d1.draw() for _ in range(10)]
        self.assertNotEqual(cards0, cards1)

    def test_reshuffle_on_empty(self):
        """Verifying reshuffle triggers when draw pile runs out."""
        d = Deck(seed=0)
        # Exhausting the draw pile
        while d.draw_pile_size > 0:
            card = d.draw()
            d.discard(card)
        # Draw pile is now empty -- next draw should trigger reshuffle
        drawn = d.draw()
        self.assertIsInstance(drawn, Card)

    def test_setup_first_card_is_number(self):
        """Verifying first card is always a colored number card."""
        for seed in range(50):
            d = Deck(seed=seed)
            first = d.setup_first_card()
            self.assertEqual(
                first.card_type, CardType.NUMBER,
                f"Seed {seed}: first card is {first}, expected a number card"
            )

    def test_setup_first_card_not_wild(self):
        """Verifying first card is never wild."""
        for seed in range(50):
            d = Deck(seed=seed)
            first = d.setup_first_card()
            self.assertFalse(
                first.is_wild(),
                f"Seed {seed}: first card {first} should not be wild"
            )

    def test_setup_first_card_not_action(self):
        """Verifying first card is never Skip, Reverse, or Draw Two."""
        action_types = {CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO}
        for seed in range(50):
            d = Deck(seed=seed)
            first = d.setup_first_card()
            self.assertNotIn(
                first.card_type, action_types,
                f"Seed {seed}: first card {first} must not be an action card"
            )

    def test_setup_first_card_goes_to_discard(self):
        """Verifying setup_first_card places the card on the discard pile."""
        d = Deck(seed=0)
        first = d.setup_first_card()
        self.assertEqual(d.top_card(), first)


if __name__ == "__main__":
    unittest.main(verbosity=2)