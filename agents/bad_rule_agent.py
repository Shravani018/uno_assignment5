"""
Deliberately bad rule-based UNO agent for validation testing.

ROLE IN SYSTEM
--------------
RuleBadAgent exists purely as a validation tool. If RuleAgent (greedy)
cannot consistently beat RuleBadAgent by a large margin, something is
wrong with the greedy strategy, the engine, or both.

Expected outcome: RuleAgent should win ~65-75% against RuleBadAgent.
If the margin is smaller, the greedy heuristics need re-examination.

BAD STRATEGY -- ANTI-GREEDY DESIGN
------------------------------------
Every decision is the deliberate opposite of RuleAgent:

1. Card priority (reversed)
   Good agent  : WD4 > D2 > Skip > Reverse > highest number
   Bad agent   : lowest number > Reverse > Skip > D2 > WD4
   -- Wastes action cards. Hoards the most powerful cards uselessly.
   -- Plays lowest-value numbers, keeping high-value cards in hand.

2. Wild conservation (inverted)
   Good agent  : play non-wilds first, use wilds only as fallback
   Bad agent   : play wilds immediately whenever available
   -- Burns wild cards even when a matching color/number exists.

3. Wild color selection (worst color)
   Good agent  : picks most frequent color in hand
   Bad agent   : picks LEAST frequent color in hand
   -- Actively picks the color it cannot follow up on.

4. Threat mode (absent)
   Good agent  : plays strongest card when opponent has <= 2 cards
   Bad agent   : ignores opponent hand size entirely, no threat response

WHY THIS DESIGN IS USEFUL
--------------------------
A bad agent must still play legally (only valid_plays). The badness
comes entirely from poor decision-making within legal moves, not from
cheating or rule violations. This means:
  - Engine correctness is fully preserved
  - Any win gap between RuleAgent and RuleBadAgent is attributable
    purely to strategy quality
  - MCTS and RL agents that cannot beat RuleBadAgent reliably have
    a serious training or design problem
"""
from collections import Counter
from typing import List, Optional, Tuple
from core.cards import Card, Color, CardType
from core.game import GameState
from agents.base_agent import BaseAgent


WILD_TYPES = (CardType.WILD, CardType.WILD_DRAW_FOUR)

# Reversed priority: worst cards first, best cards last (hoarded uselessly)
BAD_PRIORITY = [
    CardType.NUMBER,       # play numbers first (backwards)
    CardType.REVERSE,
    CardType.SKIP,
    CardType.DRAW_TWO,
    CardType.WILD,         # waste wilds immediately when available
    CardType.WILD_DRAW_FOUR,
]


class RuleBadAgent(BaseAgent):
    """Playing with a deliberately anti-optimal strategy for validation.

    Every heuristic is the inverse of RuleAgent. Used to confirm that
    the greedy agent demonstrates measurably superior play.

    Args:
        name: Display name used in logs and tournament output.
    """

    def __init__(self, name: str = "RuleBadAgent"):
        """Initialising bad rule agent."""
        super().__init__(name)

    def choose_action(self, state: GameState) -> Tuple[Card, Optional[Color]]:
        """Selecting the worst legal card using anti-greedy priority.

        Args:
            state: Current game state snapshot.

        Returning (card, chosen_color).
        """
        card = self._select_worst_card(state.valid_plays)
        # Playing wilds immediately (bad) with worst possible color
        chosen_color = self._worst_color(state.valid_plays) if card.is_wild() else None
        return card, chosen_color

    def _select_worst_card(self, plays: List[Card]) -> Card:
        """Selecting card using reversed priority -- worst impact first.

        Prioritising number cards over action cards, lowest value numbers,
        and spending wilds immediately rather than conserving them.

        Args:
            plays: Valid cards available this turn.

        Returning chosen Card.
        """
        # Attempting to play wilds immediately (inverted conservation)
        wilds = [c for c in plays if c.card_type in WILD_TYPES]
        if wilds:
            # Playing plain Wild before WD4 (least impactful wild first)
            for card_type in (CardType.WILD, CardType.WILD_DRAW_FOUR):
                candidates = [c for c in wilds if c.card_type == card_type]
                if candidates:
                    return candidates[0]

        # Falling through to non-wilds with reversed priority
        non_wilds = [c for c in plays if c.card_type not in WILD_TYPES]
        for card_type in BAD_PRIORITY:
            candidates = [c for c in non_wilds if c.card_type == card_type]
            if candidates:
                # Picking lowest value for numbers (opposite of greedy)
                return self._lowest_value(candidates)

        return plays[0]

    def _lowest_value(self, cards: List[Card]) -> Card:
        """Returning card with the lowest point value from a list.

        Args:
            cards: Non-empty list of cards to compare.

        Returning Card with minimum point_value().
        """
        return min(cards, key=lambda c: c.point_value())

    def _worst_color(self, valid_plays: List[Card]) -> Color:
        """Choosing the LEAST frequent non-wild color in valid_plays.

        Opposite of RuleAgent -- picks the color least represented,
        actively making it harder to continue playing.

        Args:
            valid_plays: Valid cards available this turn.

        Returning least common Color, defaulting to RED.
        """
        color_counts = Counter(
            c.color for c in valid_plays if c.color != Color.WILD
        )
        if not color_counts:
            return Color.RED
        # Picking least frequent color (anti-greedy)
        return color_counts.most_common()[-1][0]