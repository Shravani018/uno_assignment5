"""
Tests for core/rules.py

COVERING
--------
- is_valid_play: color match, number match, type match, wild always valid
- is_valid_play: invalid plays correctly rejected
- get_valid_plays: correct filtering of a mixed hand
- apply_card_effect: correct effect for each card type
- apply_card_effect: wild color is set correctly
- 2-player rule: Reverse acts as Skip
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from core.cards import Card, Color, CardType
from core.rules import is_valid_play, get_valid_plays, apply_card_effect


# Convenience constructors
def num(color, value):
    return Card(color, CardType.NUMBER, value)

def skip(color):
    return Card(color, CardType.SKIP)

def reverse(color):
    return Card(color, CardType.REVERSE)

def draw_two(color):
    return Card(color, CardType.DRAW_TWO)

def wild():
    return Card(Color.WILD, CardType.WILD)

def wild_d4():
    return Card(Color.WILD, CardType.WILD_DRAW_FOUR)


class TestIsValidPlay(unittest.TestCase):

    def test_same_color_is_valid(self):
        """Verifying card with matching color can always be played."""
        top = num(Color.RED, 5)
        self.assertTrue(is_valid_play(num(Color.RED, 3), top, Color.RED))

    def test_same_number_is_valid(self):
        """Verifying card with matching number can be played regardless of color."""
        top = num(Color.RED, 7)
        self.assertTrue(is_valid_play(num(Color.BLUE, 7), top, Color.RED))

    def test_same_action_type_is_valid(self):
        """Verifying matching action type is a valid play."""
        top = skip(Color.RED)
        self.assertTrue(is_valid_play(skip(Color.BLUE), top, Color.RED))

    def test_wild_always_valid(self):
        """Verifying wild is always playable regardless of top card."""
        top = num(Color.GREEN, 3)
        self.assertTrue(is_valid_play(wild(), top, Color.GREEN))

    def test_wild_draw_four_always_valid(self):
        """Verifying Wild Draw 4 is always playable."""
        top = num(Color.YELLOW, 9)
        self.assertTrue(is_valid_play(wild_d4(), top, Color.YELLOW))

    def test_different_color_and_number_invalid(self):
        """Verifying card with no color or number match is rejected."""
        top = num(Color.RED, 5)
        self.assertFalse(is_valid_play(num(Color.BLUE, 3), top, Color.RED))

    def test_active_color_overrides_top_card_color(self):
        """Verifying active color (set by wild) takes priority over top card color."""
        # Wild was played last, active color is now BLUE
        top = wild()
        self.assertTrue(is_valid_play(num(Color.BLUE, 4), top, Color.BLUE))
        self.assertFalse(is_valid_play(num(Color.RED, 4), top, Color.BLUE))

    def test_draw_two_matches_draw_two(self):
        """Verifying Draw Two plays on another Draw Two regardless of color."""
        top = draw_two(Color.RED)
        self.assertTrue(is_valid_play(draw_two(Color.GREEN), top, Color.RED))

    def test_number_does_not_match_action(self):
        """Verifying a number card does not match an action type."""
        top = skip(Color.RED)
        self.assertFalse(is_valid_play(num(Color.BLUE, 5), top, Color.RED))


class TestGetValidPlays(unittest.TestCase):

    def test_returns_only_valid_cards(self):
        """Verifying get_valid_plays filters out all illegal cards."""
        top = num(Color.RED, 5)
        hand = [
            num(Color.RED, 3),    # valid: same color
            num(Color.BLUE, 5),   # valid: same number
            num(Color.GREEN, 7),  # invalid
            wild(),               # valid: always
        ]
        valid = get_valid_plays(hand, top, Color.RED)
        self.assertIn(hand[0], valid)
        self.assertIn(hand[1], valid)
        self.assertNotIn(hand[2], valid)
        self.assertIn(hand[3], valid)

    def test_empty_hand_returns_empty(self):
        """Verifying empty hand produces empty valid plays."""
        top = num(Color.RED, 5)
        self.assertEqual(get_valid_plays([], top, Color.RED), [])

    def test_no_valid_plays_returns_empty(self):
        """Verifying all-unplayable hand returns empty list."""
        top = num(Color.RED, 5)
        hand = [num(Color.BLUE, 3), num(Color.GREEN, 7)]
        self.assertEqual(get_valid_plays(hand, top, Color.RED), [])


class TestApplyCardEffect(unittest.TestCase):

    def test_number_card_sets_color(self):
        """Verifying number card sets new_color to card color."""
        effect = apply_card_effect(num(Color.GREEN, 4))
        self.assertEqual(effect["new_color"], Color.GREEN)
        self.assertEqual(effect["opponent_draws"], 0)
        self.assertFalse(effect["skip_opponent"])

    def test_skip_skips_opponent(self):
        """Verifying Skip sets skip_opponent=True."""
        effect = apply_card_effect(skip(Color.RED))
        self.assertTrue(effect["skip_opponent"])
        self.assertEqual(effect["opponent_draws"], 0)

    def test_reverse_acts_as_skip_in_2player(self):
        """Verifying Reverse sets skip_opponent=True (2-player rule)."""
        effect = apply_card_effect(reverse(Color.BLUE))
        self.assertTrue(effect["skip_opponent"])

    def test_draw_two_effect(self):
        """Verifying Draw Two forces opponent to draw 2 and lose turn."""
        effect = apply_card_effect(draw_two(Color.YELLOW))
        self.assertEqual(effect["opponent_draws"], 2)
        self.assertTrue(effect["skip_opponent"])

    def test_wild_sets_chosen_color(self):
        """Verifying Wild sets new_color to the chosen color."""
        effect = apply_card_effect(wild(), chosen_color=Color.BLUE)
        self.assertEqual(effect["new_color"], Color.BLUE)
        self.assertEqual(effect["opponent_draws"], 0)
        self.assertFalse(effect["skip_opponent"])

    def test_wild_draw_four_effect(self):
        """Verifying Wild Draw 4 forces opponent to draw 4, lose turn, sets color."""
        effect = apply_card_effect(wild_d4(), chosen_color=Color.GREEN)
        self.assertEqual(effect["new_color"], Color.GREEN)
        self.assertEqual(effect["opponent_draws"], 4)
        self.assertTrue(effect["skip_opponent"])


if __name__ == "__main__":
    unittest.main(verbosity=2)