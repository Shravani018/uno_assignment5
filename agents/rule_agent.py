"""
Greedy heuristic rule-based UNO agent.

ROLE IN SYSTEM
--------------
RuleAgent is the primary baseline agent. It represents a competent human
player following conventional UNO strategy without any search or learning.
All stronger agents (MCTS, RL) must beat this agent to be considered useful.

ALGORITHM
---------
Given valid_plays from GameState, RuleAgent selects a card in two phases:

Phase 1 -- Threat detection
    If the opponent holds <= THREAT_THRESHOLD cards, enter threat mode.
    In threat mode, use the strongest card available immediately,
    ignoring the normal wild-conservation rule.
    Priority in threat mode: WD4 > D2 > Skip > Reverse > Wild > Number

Phase 2 -- Normal play
    Prefer non-wild cards to conserve wilds for when they are truly needed.
    Within non-wilds, apply priority: D2 > Skip > Reverse > Number (highest value).
    If no non-wild is available, play WD4 before plain Wild.

Wild color selection
    When a wild is played, choose the color most frequent among the
    non-wild cards in valid_plays as a proxy for the hand composition.
    Falls back to RED if the agent holds only wilds.

HEURISTIC RATIONALE
-------------------
- Playing high-value cards first maximises point penalty on opponent if
  they go out before us (standard UNO meta).
- Saving wilds preserves flexibility for turns where no match exists.
- Threat mode overrides conservation because blocking an imminent win
  takes priority over resource management.

LIMITATIONS (by design)
------------------------
- No lookahead: decisions are made without simulating future turns.
- Wild color uses valid_plays as a proxy, not the full hand -- the engine
  does not expose the full hand to agents to enforce partial observability.
  MCTS and RL agents can learn a better color policy from full game context.
"""
from collections import Counter
from typing import List, Optional, Tuple
from core.cards import Card, Color, CardType
from core.game import GameState
from agents.random_agent import BaseAgent
WILD_TYPES = (CardType.WILD, CardType.WILD_DRAW_FOUR)
THREAT_THRESHOLD = 2

# Descending priority for card selection; NUMBER handled separately by value
ACTION_PRIORITY = [
    CardType.WILD_DRAW_FOUR,
    CardType.DRAW_TWO,
    CardType.SKIP,
    CardType.REVERSE,
    CardType.WILD,
    CardType.NUMBER,
]


class RuleAgent(BaseAgent):
    """Selecting cards via greedy heuristics with threat-aware mode switching.

    Args:
        name: Display name used in logs and tournament output.
    """

    def __init__(self, name: str = "RuleAgent"):
        """Initialising rule-based agent."""
        super().__init__(name)

    def choose_action(self, state: GameState) -> Tuple[Card, Optional[Color]]:
        """Selecting best card per heuristic priority and threat level.

        Args:
            state: Current game state snapshot.

        Returning (card, chosen_color) where chosen_color is set only
        when card is a wild.
        """
        opponent_idx = 1 - state.current_player
        opponent_cards = state.hand_sizes[opponent_idx]
        card = self._select_card(state.valid_plays, opponent_cards)
        chosen_color = self._best_color(state.valid_plays) if card.is_wild() else None
        return card, chosen_color

    def _select_card(self, plays: List[Card], opponent_cards: int) -> Card:
        """Applying two-phase priority selection to choose a card.

        Args:
            plays: Valid cards available this turn.
            opponent_cards: Number of cards opponent currently holds.

        Returning chosen Card.
        """
        in_threat_mode = opponent_cards <= THREAT_THRESHOLD
        non_wilds = [c for c in plays if c.card_type not in WILD_TYPES]
        wilds = [c for c in plays if c.card_type in WILD_TYPES]

        # Phase 1: threat mode -- use strongest card immediately
        if in_threat_mode:
            for card_type in ACTION_PRIORITY:
                candidates = [c for c in plays if c.card_type == card_type]
                if candidates:
                    return self._highest_value(candidates)

        # Phase 2: normal mode -- exhaust non-wilds first
        if non_wilds:
            for card_type in ACTION_PRIORITY:
                candidates = [c for c in non_wilds if c.card_type == card_type]
                if candidates:
                    return self._highest_value(candidates)

        # Fallback: play wild, preferring WD4 over plain Wild
        for card_type in (CardType.WILD_DRAW_FOUR, CardType.WILD):
            candidates = [c for c in wilds if c.card_type == card_type]
            if candidates:
                return candidates[0]

        return plays[0]

    def _highest_value(self, cards: List[Card]) -> Card:
        """Returning card with the highest point value from a list.

        Args:
            cards: Non-empty list of cards to compare.

        Returning Card with maximum point_value().
        """
        return max(cards, key=lambda c: c.point_value())

    def _best_color(self, valid_plays: List[Card]) -> Color:
        """Choosing the most frequent non-wild color seen in valid_plays.

        Using valid_plays as a proxy for hand composition since the engine
        does not expose the full hand to agents (partial observability).

        Args:
            valid_plays: Valid cards available this turn.

        Returning most common Color, defaulting to RED if all cards are wild.
        """
        color_counts = Counter(
            c.color for c in valid_plays if c.color != Color.WILD
        )
        if not color_counts:
            return Color.RED
        return color_counts.most_common(1)[0][0]