"""CLI-driven agent that prompts a human player for input each turn.

Flow
----
1. Print the current top card and active color.
2. Print the player's full hand with indices.
3. List only the valid plays (also with indices into valid_plays).
4. Read an integer index; re-prompt on invalid input.
5. If the chosen card is a wild, prompt for a color choice.
6. Return (card, color_or_None).

Input is always validated in a loop — no crashes on bad input.
"""
from typing import Optional, Tuple
from core.cards import Card, Color
from core.game import GameState
from agents.random_agent import BaseAgent
_COLOR_OPTIONS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW]
class HumanAgent(BaseAgent):
    """Interactive CLI agent that reads moves from standard input.

    Args:
        name: Player label shown in prompts (e.g. "Player 1").
    """

    def __init__(self, name: str = "Human") -> None:
        """Initialising with a display name."""
        super().__init__(name)

    def choose_action(self, state: GameState) -> Tuple[Card, Optional[Color]]:
        """Displaying the board state and collecting a validated move.

        Args:
            state: Current game snapshot; state.valid_plays is non-empty.

        Returns:
            (card, color) — color is None unless the card is a wild.
        """
        self._print_state(state)
        card = self._prompt_card(state.valid_plays)
        color = self._prompt_color() if card.is_wild() else None
        return card, color

    def _print_state(self, state: GameState) -> None:
        """Printing board context before asking for a move."""
        print(f"\n{'─' * 40}")
        print(f"  {self.name}'s turn  |  Turn #{state.current_player}")
        print(f"  Top card   : {state.top_card}")
        print(f"  Color      : {state.current_color.value.upper()}")
        print(f"  Opponent   : {state.hand_sizes[1 - state.current_player]} card(s)")
        print(f"  Draw pile  : {state.draw_pile_size} card(s)")
        print(f"{'─' * 40}")
        print("  Valid plays:")
        for i, card in enumerate(state.valid_plays):
            print(f"    [{i}] {card}")

    def _prompt_card(self, valid_plays: list) -> Card:
        """Prompting for a valid play index, looping until input is good.

        Args:
            valid_plays: Non-empty list of Card objects.

        Returns:
            The chosen Card.
        """
        while True:
            raw = input(f"\n  Enter index [0–{len(valid_plays) - 1}]: ").strip()
            if raw.isdigit():
                idx = int(raw)
                if 0 <= idx < len(valid_plays):
                    return valid_plays[idx]
            print(f"  ✗ Invalid — enter a number between 0 and {len(valid_plays) - 1}.")

    def _prompt_color(self) -> Color:
        """Prompting for a wild color choice, looping until input is good.

        Returns:
            The chosen Color (never Color.WILD).
        """
        print("\n  Choose a color:")
        for i, color in enumerate(_COLOR_OPTIONS):
            print(f"    [{i}] {color.value}")
        while True:
            raw = input(f"  Enter index [0–{len(_COLOR_OPTIONS) - 1}]: ").strip()
            if raw.isdigit():
                idx = int(raw)
                if 0 <= idx < len(_COLOR_OPTIONS):
                    return _COLOR_OPTIONS[idx]
            print(f"  ✗ Invalid — enter 0, 1, 2, or 3.")