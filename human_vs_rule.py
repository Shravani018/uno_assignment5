"""
Running a Human vs RuleAgent game in the terminal.

HOW TO RUN
----------
    python play_human.py

CONTROLS
--------
- Your full hand is shown every turn.
- Valid cards are marked with [*] and a play index.
- Invalid cards are marked with [ ] and no index.
- Enter the play index (shown in [*] rows) to play that card.
- Enter 'd' if you have no valid plays to draw a card.
- When playing a wild: enter r / g / b / y to choose a color.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import List, Optional
from core.cards import Card, Color
from core.game import UnoGame, GameState, TurnResult
from core.rules import get_valid_plays
from agents.rule_agent import RuleAgent
from agents.base_agent import BaseAgent


DIVIDER = "=" * 55
THIN = "-" * 55

COLOR_MAP = {
    "r": Color.RED,
    "g": Color.GREEN,
    "b": Color.BLUE,
    "y": Color.YELLOW,
}

# Color labels for display
COLOR_LABEL = {
    Color.RED: "RED",
    Color.GREEN: "GREEN",
    Color.BLUE: "BLUE",
    Color.YELLOW: "YELLOW",
    Color.WILD: "WILD",
}


def print_banner() -> None:
    """Printing the welcome banner."""
    print(f"\n{DIVIDER}")
    print("           UNO  --  You vs RuleAgent")
    print(DIVIDER)
    print("  You = Player 0  |  RuleAgent = Player 1")
    print("  First to empty their hand wins.")
    print(f"{DIVIDER}\n")


def print_game_header(state: GameState, game: UnoGame) -> None:
    """Printing current top card, active color and opponent card count.

    Args:
        state: Current game state snapshot.
        game: Running game instance (for accessing opponent hand size).
    """
    opp_count = state.hand_sizes[1]
    print(f"\n{DIVIDER}")
    print(f"  Turn {game.turn_count + 1}  --  YOUR TURN")
    print(THIN)
    print(f"  Top card    : {state.top_card}")
    print(f"  Active color: {COLOR_LABEL[state.current_color]}")
    print(f"  RuleAgent holds {opp_count} card(s)")
    print(THIN)


def print_full_hand(hand_cards: List[Card], valid_plays: List[Card]) -> dict:
    """Printing the player's full hand, marking valid cards with a play index.

    Valid cards show  [* N]  where N is the index to type.
    Invalid cards show  [   ]  with no selectable index.

    Args:
        hand_cards: All cards currently in the player's hand.
        valid_plays: Subset of hand_cards that can legally be played.

    Returning a dict mapping play_index -> Card for input validation.
    """
    print("  YOUR HAND:")
    print()
    play_index = 0
    index_map = {}

    for card in hand_cards:
        if card in valid_plays:
            print(f"    [* {play_index}]  {card}")
            index_map[play_index] = card
            play_index += 1
        else:
            print(f"    [   ]  {card}  (not playable)")

    print()
    if index_map:
        print(f"  Playable cards marked [* N] -- enter N to play.")
    else:
        print("  No valid plays -- enter 'd' to draw a card.")
    print(DIVIDER)

    return index_map


def prompt_card(index_map: dict) -> Optional[Card]:
    """Prompting the player to select a card index or draw.

    Args:
        index_map: Dict mapping play_index -> Card.

    Returning selected Card, or None if player chose to draw.
    """
    if not index_map:
        input("  Press Enter to draw a card... ")
        return None

    max_idx = max(index_map.keys())
    while True:
        raw = input(f"  Select [0-{max_idx}] or 'd' to draw: ").strip().lower()
        if raw == "d":
            return None
        try:
            idx = int(raw)
            if idx in index_map:
                return index_map[idx]
            print(f"  Invalid index. Enter a number between 0 and {max_idx}.")
        except ValueError:
            print("  Enter a number or 'd'.")


def prompt_color() -> Color:
    """Prompting the player to choose a color after playing a wild.

    Returning selected Color enum value.
    """
    print("  Choose a color:")
    print("    r = RED   g = GREEN   b = BLUE   y = YELLOW")
    while True:
        raw = input("  Color: ").strip().lower()
        if raw in COLOR_MAP:
            return COLOR_MAP[raw]
        print("  Invalid. Enter r, g, b, or y.")


def announce_rule_turn(result: TurnResult, cards_before: int, cards_after: int) -> None:
    """Printing a summary of what RuleAgent did on its turn.

    Args:
        result: TurnResult from the engine.
        cards_before: RuleAgent's hand size before the turn.
        cards_after: RuleAgent's hand size after the turn.
    """
    print(f"\n{THIN}")
    print("  RULEAGENT'S TURN")
    if result.card_played is None:
        print("  RuleAgent had no valid play -- drew a card.")
        print(f"  Hand size: {cards_before} → {cards_after}")
    else:
        color_str = ""
        if result.chosen_color:
            color_str = f"  (chose {COLOR_LABEL[result.chosen_color]})"
        print(f"  RuleAgent played: {result.card_played}{color_str}")
        print(f"  Hand size: {cards_before} → {cards_after}")
    print(THIN)


def announce_winner(winner: int, turn_count: int) -> None:
    """Printing the final game result.

    Args:
        winner: 0 = human, 1 = RuleAgent, -1 = timeout/draw.
        turn_count: Total turns played.
    """
    print(f"\n{DIVIDER}")
    if winner == -1:
        print("  DRAW -- maximum turns reached.")
    elif winner == 0:
        print("  YOU WIN! Well played.")
    else:
        print("  RULEAGENT WINS! Better luck next time.")
    print(f"  Game lasted {turn_count} turns.")
    print(f"{DIVIDER}\n")


def run_human_turn(game: UnoGame) -> Optional[int]:
    """Handling one full human turn: display, input, play, effect.

    Bypassing HumanAgent class so we can show the full hand inline.
    Directly calling game internals is acceptable here since this is
    a display layer, not a training component.

    Args:
        game: Running game instance.

    Returning winner index if game ends this turn, else None.
    """
    from core.rules import apply_card_effect

    state = game.get_state()
    hand = game.hands[0]

    print_game_header(state, game)
    index_map = print_full_hand(hand.cards, state.valid_plays)
    card = prompt_card(index_map)

    if card is None:
        # Drawing a card -- no valid play or player chose to draw
        drawn = game.deck.draw()
        hand.add(drawn)
        print(f"\n  You drew: {drawn}")
        game._log(state, action=None, drawn=drawn)
        game._switch_player()
        return None

    # Playing the selected card
    chosen_color = prompt_color() if card.is_wild() else None
    hand.remove(card)
    game.deck.discard(card)

    effect = apply_card_effect(card, chosen_color)
    game.current_color = effect["new_color"]
    game._log(state, action=card, drawn=None, chosen_color=chosen_color)

    color_msg = f" -- active color now {COLOR_LABEL[effect['new_color']]}" if card.is_wild() else ""
    print(f"\n  You played: {card}{color_msg}")

    if hand.is_empty():
        return 0

    opp_idx = 1
    if effect["opponent_draws"] > 0:
        drawn_cards = [game.deck.draw() for _ in range(effect["opponent_draws"])]
        game.hands[opp_idx].add_many(drawn_cards)
        print(f"  RuleAgent draws {effect['opponent_draws']} card(s).")

    if effect["skip_opponent"]:
        print("  RuleAgent's turn is skipped.")
    else:
        game._switch_player()

    return None


def run_rule_turn(game: UnoGame) -> Optional[int]:
    """Handling one full RuleAgent turn and printing a summary.

    Args:
        game: Running game instance.

    Returning winner index if game ends this turn, else None.
    """
    cards_before = game.hands[1].size()
    result: TurnResult = game.play_turn()
    cards_after = game.hands[1].size()
    announce_rule_turn(result, cards_before, cards_after)
    return result.winner


def main() -> None:
    """Running one Human vs RuleAgent game."""
    print_banner()

    rule = RuleAgent("RuleAgent")

    # Passing rule as both agents; human turns are handled manually below
    # so agent1 slot is never called for player 0
    game = UnoGame(rule, rule)

    # Reinitialising with correct agent assignments
    # Resetting so player 0 slot is handled by run_human_turn()
    game.agents = [None, rule]

    print(f"  Starting card : {game.deck.top_card()}")
    print(f"  Your hand size: {game.hands[0].size()} cards\n")

    while game.turn_count < game.max_turns:
        game.turn_count += 1

        if game.current_player == 0:
            winner = run_human_turn(game)
        else:
            winner = run_rule_turn(game)

        if winner is not None:
            announce_winner(winner, game.turn_count)
            return

    announce_winner(-1, game.turn_count)


if __name__ == "__main__":
    main()