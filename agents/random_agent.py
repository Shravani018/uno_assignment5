"""
Random agent -- plays any legal card uniformly at random.

ROLE IN SYSTEM
--------------
RandomAgent establishes the performance floor (lower bound) for all agents.
Any agent that cannot consistently beat RandomAgent is not learning anything
useful. It is also used to sanity-check the engine: win rates should converge
to ~50/50 in Random vs Random over enough games.

ALGORITHM
---------
1. Receive valid_plays from GameState (engine guarantees non-empty).
2. Sample one card uniformly at random using random.choice().
3. If card is wild: sample a color uniformly at random from the 4 non-wild
   colors (RED, GREEN, BLUE, YELLOW).
4. Return (card, chosen_color).

No heuristics, no lookahead, no memory -- purely stochastic.
This is intentional: a clean random baseline with zero bias.

WHY UNIFORM COLOR FOR WILDS
----------------------------
A smarter random agent might pick the color it holds most of. We deliberately
avoid this here -- RandomAgent must have zero strategic content so it serves
as a true lower bound. Color intelligence belongs in RuleAgent and above.
"""
import random
from typing import Tuple, Optional
from core.cards import Card, Color
from core.game import GameState
from agents.base_agent import BaseAgent

# Selecting only from playable colors when choosing wild color
_PLAY_COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW]


class RandomAgent(BaseAgent):
    """Selecting a legal card uniformly at random each turn.

    Args:
        name: Display name used in logs and tournament output.
    """

    def __init__(self, name: str = "RandomAgent"):
        """Initialising random agent."""
        super().__init__(name)

    def choose_action(self, state: GameState) -> Tuple[Card, Optional[Color]]:
        """Sampling a card and optional color uniformly at random.

        Args:
            state: Current game state snapshot containing valid_plays.

        Returning (card, chosen_color) where chosen_color is None unless
        the sampled card is a wild.
        """
        card = random.choice(state.valid_plays)
        # Choosing a random color only when a wild is played
        chosen_color = random.choice(_PLAY_COLORS) if card.is_wild() else None
        return card, chosen_color