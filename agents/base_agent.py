"""
Abstract base class for all UNO agents.

ROLE IN SYSTEM
--------------
Every agent (human, rule-based, MCTS, RL) must subclass BaseAgent and
implement choose_action(). The game engine calls only this method, keeping
all agents interchangeable inside UnoGame.run().

INTERFACE CONTRACT
------------------
Input  : GameState -- a frozen snapshot of the game at the current turn.
                      Contains top card, active color, hand sizes, draw
                      pile size, and the list of VALID plays only.
                      The engine guarantees valid_plays is non-empty before
                      calling choose_action.

Output : (Card, Optional[Color])
         - Card          : one card selected from state.valid_plays
         - Optional[Color]: chosen color when card is wild, else None

WHY THIS DESIGN
---------------
- Agents are stateless by default: all context is passed via GameState.
- Keeping hand management inside the engine (not the agent) prevents
  agents from reading the opponent's hand directly.
- Returning (card, color) as a single atomic tuple keeps the turn cycle
  in game.py clean -- one call, one complete decision.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from core.cards import Card, Color
from core.game import GameState
class BaseAgent(ABC):
    """Serving as the interface contract for all UNO agents.
    Args:
        name: Display name used in logs and tournament output.
    """

    def __init__(self, name: str):
        """Initialising agent with a display name."""
        self.name = name
    @abstractmethod
    def choose_action(self, state: GameState) -> Tuple[Card, Optional[Color]]:
        """Selecting a card to play and optionally a wild color.

        Args:
            state: Current game state snapshot. state.valid_plays is
                   guaranteed non-empty when this is called.

        Returning (card, chosen_color) where chosen_color is None
        unless card is a wild type.
        """
    def __repr__(self) -> str:
        """Returning agent identifier string."""
        return f"{self.__class__.__name__}(name={self.name!r})"