"""
Managing UNO game state, turns, and game loop.

ROLE IN SYSTEM
--------------
UnoGame is the single source of truth for all game logic.
No other module should replicate turn progression, card effects,
or hand management. self_play.py and benchmark scripts call
play_turn() and read TurnResult -- they never touch private methods.

KEY DESIGN: TurnResult
----------------------
play_turn() returns a TurnResult dataclass instead of just Optional[int].
This gives self_play.py everything it needs to build a training sample
(pre-state, action taken, winner) without the collector re-running any logic.

Turn flow inside play_turn()
-----------------------------
1. Snapshot state via get_state() -- this is what the agent sees.
2. If no valid plays: draw one card, switch player, return TurnResult(action=None).
3. Ask agent for (card, chosen_color).
4. Remove card from hand, place on discard pile.
5. Apply card effect: update current_color, penalise opponent if needed.
6. Check win condition (hand empty).
7. Switch player unless effect skips opponent.
8. Return TurnResult with pre-state, card played, chosen_color, winner.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from core.cards import Card, Color
from core.deck import Deck
from core.hand import Hand
from core.rules import get_valid_plays, apply_card_effect
from config import GameConfig


@dataclass
class GameState:
    """Capturing a snapshot of the current game state for logging and agents.

    Args:
        current_player: Index of active player (0 or 1).
        top_card: Top card of discard pile.
        current_color: Active color in play.
        hand_sizes: Number of cards each player holds.
        draw_pile_size: Remaining cards in draw pile.
        valid_plays: Playable cards for the current player.
    """
    current_player: int
    top_card: Card
    current_color: Color
    hand_sizes: List[int]
    draw_pile_size: int
    valid_plays: List[Card]
    my_full_hand: List[Card] = field(default_factory=list)


@dataclass
class TurnResult:
    """Capturing everything that happened in a single turn.

    Used by self_play.py to build (state, action, reward) training rows
    without re-implementing any engine logic.

    Args:
        pre_state: GameState snapshot seen by the agent before acting.
        card_played: Card the agent played, or None if forced to draw.
        chosen_color: Color chosen for wild, else None.
        winner: Winning player index (0 or 1) if game ended, else None.
                -1 signals a timeout (max_turns exceeded).
    """
    pre_state: GameState
    card_played: Optional[Card]
    chosen_color: Optional[Color]
    winner: Optional[int]


class UnoGame:
    """Running a 2-player UNO game between two agents.

    Args:
        agent0: Agent for player 0.
        agent1: Agent for player 1.
        seed: Optional seed for reproducibility.
        max_turns: Turn cap preventing infinite games.
    """

    def __init__(self, agent0, agent1, seed: Optional[int] = None,
                 max_turns: int = GameConfig.MAX_TURNS):
        """Initialising game with two agents and a fresh shuffled deck."""
        self.agents = [agent0, agent1]
        self.max_turns = max_turns
        self.deck = Deck(seed=seed)
        self.hands = [
            Hand([self.deck.draw() for _ in range(GameConfig.HAND_SIZE)]),
            Hand([self.deck.draw() for _ in range(GameConfig.HAND_SIZE)]),
        ]
        self.deck.setup_first_card()
        self.current_color: Color = self.deck.top_card().color
        self.current_player: int = 0
        self.turn_count: int = 0
        self.game_log: List[dict] = []

    def get_state(self) -> GameState:
        """Building and returning the current game state snapshot."""
        top = self.deck.top_card()
        hand = self.hands[self.current_player]
        valid = get_valid_plays(hand.cards, top, self.current_color)
        return GameState(
            current_player=self.current_player,
            top_card=top,
            current_color=self.current_color,
            hand_sizes=[h.size() for h in self.hands],
            draw_pile_size=self.deck.draw_pile_size,
            valid_plays=valid,
            my_full_hand=list(hand.cards),
        )

    def play_turn(self) -> TurnResult:
        """Executing one full turn and returning a TurnResult.

        Handling draw-only turns (no valid plays) and normal play turns.
        Applying card effects, checking win condition, switching player.

        Returning TurnResult with pre_state, action taken, and winner if any.
        """
        state = self.get_state()
        agent = self.agents[self.current_player]
        hand = self.hands[self.current_player]
        opponent_idx = 1 - self.current_player

        # --- No valid plays: forced draw ---
        if not state.valid_plays:
            drawn = self.deck.draw()
            hand.add(drawn)
            self._log(state, action=None, drawn=drawn)
            self._switch_player()
            return TurnResult(pre_state=state, card_played=None,
                              chosen_color=None, winner=None)

        # --- Agent selects a card ---
        card, chosen_color = agent.choose_action(state)
        hand.remove(card)
        self.deck.discard(card)

        effect = apply_card_effect(card, chosen_color)
        self.current_color = effect["new_color"]
        self._log(state, action=card, drawn=None, chosen_color=chosen_color)

        # --- Win condition ---
        if hand.is_empty():
            return TurnResult(pre_state=state, card_played=card,
                              chosen_color=chosen_color, winner=self.current_player)

        # --- Apply draw penalty to opponent ---
        if effect["opponent_draws"] > 0:
            drawn_cards = [self.deck.draw() for _ in range(effect["opponent_draws"])]
            self.hands[opponent_idx].add_many(drawn_cards)

        # --- Switch player unless card effect skips opponent ---
        if not effect["skip_opponent"]:
            self._switch_player()

        return TurnResult(pre_state=state, card_played=card,
                          chosen_color=chosen_color, winner=None)

    def run(self) -> Tuple[int, List[dict]]:
        """Running game to completion.

        Returning (winner_index, game_log).
        Returning -1 as winner if max_turns exceeded.
        """
        while self.turn_count < self.max_turns:
            self.turn_count += 1
            result = self.play_turn()
            if result.winner is not None:
                return result.winner, self.game_log
        return -1, self.game_log

    def _switch_player(self) -> None:
        """Toggling active player between 0 and 1."""
        self.current_player = 1 - self.current_player

    def _log(self, state: GameState, action: Optional[Card],
             drawn: Optional[Card], chosen_color: Optional[Color] = None) -> None:
        """Appending turn data to game_log for dataset generation."""
        self.game_log.append({
            "turn": self.turn_count,
            "player": state.current_player,
            "top_card": str(state.top_card),
            "current_color": state.current_color.value,
            "hand_sizes": state.hand_sizes,
            "draw_pile_size": state.draw_pile_size,
            "action": str(action) if action else "draw",
            "chosen_color": chosen_color.value if chosen_color else None,
            "drew_card": str(drawn) if drawn else None,
        })