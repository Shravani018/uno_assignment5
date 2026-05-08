"""
Tests for agents/random_agent.py and agents/rule_agent.py

COVERING
--------
RandomAgent:
  - Always returns a card from valid_plays
  - Returns a Color when playing a wild
  - Returns None color for non-wild cards
  - Produces varied choices over many trials (not stuck)

RuleAgent:
  - Always returns a card from valid_plays
  - Prefers action cards over number cards (normal mode)
  - Plays WD4/D2 immediately in threat mode (opponent <= 2 cards)
  - Conserves wilds when non-wilds are available
  - Uses WD4 before plain Wild when forced to play wild
  - Returns a Color when playing a wild
  - Returns None color for non-wild cards
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from core.cards import Card, Color, CardType
from core.game import GameState
from agents.random_agent import RandomAgent
from agents.rule_agent import RuleAgent


# ── Helpers ───────────────────────────────────────────────────────────────────

def num(color, value):
    return Card(color, CardType.NUMBER, value)

def skip(color):
    return Card(color, CardType.SKIP)

def draw_two(color):
    return Card(color, CardType.DRAW_TWO)

def wild():
    return Card(Color.WILD, CardType.WILD)

def wild_d4():
    return Card(Color.WILD, CardType.WILD_DRAW_FOUR)

def make_state(valid_plays, current_player=0, hand_sizes=None):
    """Building a minimal GameState for testing agents."""
    return GameState(
        current_player=current_player,
        top_card=num(Color.RED, 5),
        current_color=Color.RED,
        hand_sizes=hand_sizes or [7, 7],
        draw_pile_size=80,
        valid_plays=valid_plays,
    )


# ── RandomAgent Tests ─────────────────────────────────────────────────────────

class TestRandomAgent(unittest.TestCase):

    def setUp(self):
        """Initialising RandomAgent for all tests."""
        self.agent = RandomAgent("TestRandom")

    def test_returns_card_from_valid_plays(self):
        """Verifying returned card is always in valid_plays."""
        plays = [num(Color.RED, 3), num(Color.BLUE, 5), skip(Color.RED)]
        state = make_state(plays)
        card, _ = self.agent.choose_action(state)
        self.assertIn(card, plays)

    def test_returns_color_for_wild(self):
        """Verifying a Color is returned when a wild is played."""
        # Forcing wild selection with only a wild in valid_plays
        state = make_state([wild()])
        card, color = self.agent.choose_action(state)
        self.assertIsNotNone(color)
        self.assertIn(color, [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW])

    def test_returns_none_color_for_non_wild(self):
        """Verifying color is None when a non-wild card is played."""
        state = make_state([num(Color.RED, 3)])
        _, color = self.agent.choose_action(state)
        self.assertIsNone(color)

    def test_produces_varied_choices(self):
        """Verifying RandomAgent does not always pick the same card."""
        plays = [num(Color.RED, i) for i in range(5)]
        state = make_state(plays)
        choices = set()
        for _ in range(100):
            card, _ = self.agent.choose_action(state)
            choices.add(str(card))
        # Expecting at least 3 distinct cards chosen over 100 trials
        self.assertGreater(len(choices), 2)

    def test_single_valid_play(self):
        """Verifying single valid play is always returned."""
        plays = [num(Color.GREEN, 7)]
        state = make_state(plays)
        card, _ = self.agent.choose_action(state)
        self.assertEqual(card, plays[0])


# ── RuleAgent Tests ───────────────────────────────────────────────────────────

class TestRuleAgent(unittest.TestCase):

    def setUp(self):
        """Initialising RuleAgent for all tests."""
        self.agent = RuleAgent("TestRule")

    def test_returns_card_from_valid_plays(self):
        """Verifying returned card is always in valid_plays."""
        plays = [num(Color.RED, 3), skip(Color.RED), wild()]
        state = make_state(plays)
        card, _ = self.agent.choose_action(state)
        self.assertIn(card, plays)

    def test_prefers_skip_over_number(self):
        """Verifying Skip is chosen over a number card in normal mode."""
        plays = [num(Color.RED, 9), skip(Color.RED)]
        state = make_state(plays, hand_sizes=[7, 7])
        card, _ = self.agent.choose_action(state)
        self.assertEqual(card.card_type, CardType.SKIP)

    def test_prefers_draw_two_over_skip(self):
        """Verifying Draw Two is chosen over Skip."""
        plays = [skip(Color.RED), draw_two(Color.RED)]
        state = make_state(plays, hand_sizes=[7, 7])
        card, _ = self.agent.choose_action(state)
        self.assertEqual(card.card_type, CardType.DRAW_TWO)

    def test_conserves_wild_when_non_wild_available(self):
        """Verifying wild is not played when a non-wild option exists."""
        plays = [num(Color.RED, 3), wild()]
        state = make_state(plays, hand_sizes=[7, 7])
        card, _ = self.agent.choose_action(state)
        self.assertNotEqual(card.card_type, CardType.WILD)

    def test_plays_wild_when_only_option(self):
        """Verifying wild is played when no other option exists."""
        plays = [wild()]
        state = make_state(plays)
        card, _ = self.agent.choose_action(state)
        self.assertEqual(card.card_type, CardType.WILD)

    def test_prefers_wd4_over_wild(self):
        """Verifying WD4 is chosen over plain Wild when both available."""
        plays = [wild(), wild_d4()]
        state = make_state(plays)
        card, _ = self.agent.choose_action(state)
        self.assertEqual(card.card_type, CardType.WILD_DRAW_FOUR)

    def test_threat_mode_uses_wd4_immediately(self):
        """Verifying WD4 is played immediately when opponent has 2 cards."""
        plays = [num(Color.RED, 5), wild_d4()]
        state = make_state(plays, hand_sizes=[7, 2])  # opponent has 2
        card, _ = self.agent.choose_action(state)
        self.assertEqual(card.card_type, CardType.WILD_DRAW_FOUR)

    def test_threat_mode_uses_draw_two(self):
        """Verifying Draw Two is played immediately in threat mode."""
        plays = [num(Color.RED, 5), draw_two(Color.RED)]
        state = make_state(plays, hand_sizes=[7, 1])  # opponent has 1
        card, _ = self.agent.choose_action(state)
        self.assertEqual(card.card_type, CardType.DRAW_TWO)

    def test_returns_color_for_wild(self):
        """Verifying a Color is returned when a wild is played."""
        plays = [wild()]
        state = make_state(plays)
        card, color = self.agent.choose_action(state)
        self.assertIsNotNone(color)

    def test_returns_none_color_for_non_wild(self):
        """Verifying color is None when a non-wild is played."""
        plays = [num(Color.RED, 4)]
        state = make_state(plays)
        _, color = self.agent.choose_action(state)
        self.assertIsNone(color)

    def test_prefers_highest_number(self):
        """Verifying highest-value number card is preferred over lower."""
        plays = [num(Color.RED, 2), num(Color.RED, 8), num(Color.RED, 5)]
        state = make_state(plays, hand_sizes=[7, 7])
        card, _ = self.agent.choose_action(state)
        self.assertEqual(card.value, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)