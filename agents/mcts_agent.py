"""
Information Set Monte Carlo Tree Search (ISMCTS) agent for UNO.

ROLE IN SYSTEM
--------------
MCTSAgent uses ISMCTS (UCB1 applied to information sets) to select cards.
At each decision it runs num_simulations iterations, each with a freshly
sampled opponent hand, building a single shared tree across all samples.

HANDLING PARTIAL OBSERVABILITY
-------------------------------
The agent knows its own full hand via state.my_full_hand. The opponent's hand
and the remaining draw pile are unknown. Unlike plain UCT (which samples once
and builds the whole tree on that single guess), ISMCTS re-samples the opponent
hand on every simulation so the shared tree covers many possible worlds.

SimulationState is a lightweight internal data container used only inside
this file. It reuses core.rules functions so its transitions are identical
to those in UnoGame, keeping results consistent with the real game engine.

ALGORITHM (ISMCTS)
------------------
repeat num_simulations:
    1. Sample  -- determinize: draw a fresh random opponent hand
    2. Select  -- walk shared tree with UCB1 on compatible children only
    3. Expand  -- add one new child for an untried action valid in this sample
    4. Rollout -- both players play randomly until terminal or depth limit
    5. Backprop -- propagate win/loss back to root;
                   update availability for all compatible children seen
Return the action of the child with the highest visit count.
"""
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional, Tuple

from agents.base_agent import BaseAgent
from core.cards import Card, CardType, Color
from core.deck import build_deck
from core.game import GameState
from core.rules import apply_card_effect, get_valid_plays

_PLAY_COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW]
_WILD_TYPES = (CardType.WILD, CardType.WILD_DRAW_FOUR)


# ---------------------------------------------------------------------------
# SimulationState
# ---------------------------------------------------------------------------

@dataclass
class SimulationState:
    """Lightweight mutable game state for MCTS internal rollouts.

    Convention: current_player 0 = MCTS agent, 1 = opponent.
    Always constructed so that it is the MCTS agent's turn (player 0) first,
    matching the fact that choose_action is only called on the agent's turn.
    """
    my_hand: List[Card]
    opp_hand: List[Card]
    draw_pile: List[Card]
    top_card: Card
    current_color: Color
    current_player: int

    def copy(self) -> "SimulationState":
        return SimulationState(
            my_hand=list(self.my_hand),
            opp_hand=list(self.opp_hand),
            draw_pile=list(self.draw_pile),
            top_card=self.top_card,
            current_color=self.current_color,
            current_player=self.current_player,
        )

    def current_hand(self) -> List[Card]:
        return self.my_hand if self.current_player == 0 else self.opp_hand

    def opponent_hand(self) -> List[Card]:
        return self.opp_hand if self.current_player == 0 else self.my_hand

    def get_valid_plays(self) -> List[Card]:
        return get_valid_plays(self.current_hand(), self.top_card, self.current_color)

    def step(self, card: Card, chosen_color: Optional[Color]) -> Optional[int]:
        """Apply one card play in-place. Return winner index (0 or 1) or None."""
        hand = self.current_hand()
        hand.remove(card)

        if not hand:
            return self.current_player

        effect = apply_card_effect(card, chosen_color)
        self.top_card = card
        self.current_color = effect["new_color"]

        opp = self.opponent_hand()
        for _ in range(effect["opponent_draws"]):
            if self.draw_pile:
                opp.append(self.draw_pile.pop())

        if not effect["skip_opponent"]:
            self.current_player = 1 - self.current_player

        return None

    def draw_step(self) -> None:
        """Force-draw one card for the current player, then switch."""
        if self.draw_pile:
            self.current_hand().append(self.draw_pile.pop())
        self.current_player = 1 - self.current_player


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_actions(
    sim: SimulationState,
) -> List[Tuple[Optional[Card], Optional[Color]]]:
    """Return deduplicated list of (card, color) actions for current player."""
    valid = sim.get_valid_plays()
    if not valid:
        return [(None, None)]
    seen: set = set()
    actions: List[Tuple[Optional[Card], Optional[Color]]] = []
    for card in valid:
        if card.is_wild():
            for color in _PLAY_COLORS:
                key = (card, color)
                if key not in seen:
                    seen.add(key)
                    actions.append(key)
        else:
            key = (card, None)
            if key not in seen:
                seen.add(key)
                actions.append(key)
    return actions


def _best_color(valid_plays: List[Card]) -> Color:
    """Most frequent non-wild color in valid_plays, fallback RED."""
    counts = Counter(c.color for c in valid_plays if c.color != Color.WILD)
    if not counts:
        return Color.RED
    return counts.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------

class MCTSNode:
    """A node in the ISMCTS shared tree.

    Unlike UCT nodes, this node does not store a concrete SimulationState.
    Instead, the sim is threaded through the tree during each iteration.
    `availability` counts how many times this node was reachable (its action
    was legal) across all determinizations — used as the UCB1 denominator.
    """

    __slots__ = (
        "parent", "action",
        "children", "wins", "visits", "availability",
    )

    def __init__(
        self,
        parent: Optional["MCTSNode"] = None,
        action: Optional[Tuple[Optional[Card], Optional[Color]]] = None,
    ) -> None:
        self.parent = parent
        self.action = action
        self.children: List["MCTSNode"] = []
        self.wins: float = 0.0
        self.visits: int = 0
        self.availability: int = 0

    def compatible_children(
        self, sim: SimulationState
    ) -> List["MCTSNode"]:
        """Children whose action is legal in the current determinization."""
        valid = set(_get_actions(sim))
        return [c for c in self.children if c.action in valid]

    def untried_actions_for(
        self, sim: SimulationState
    ) -> List[Tuple[Optional[Card], Optional[Color]]]:
        """Legal actions in sim that have no child node yet."""
        tried = {child.action for child in self.children}
        return [a for a in _get_actions(sim) if a not in tried]

    def is_leaf(self, sim: SimulationState) -> bool:
        return bool(self.untried_actions_for(sim)) or not self.compatible_children(sim)

    def ucb1(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        return self.wins / self.visits + c * math.sqrt(
            math.log(self.availability) / self.visits
        )

    def best_child(self, sim: SimulationState, c: float) -> "MCTSNode":
        compatible = self.compatible_children(sim)
        if sim.current_player == 0:
            return max(compatible, key=lambda n: n.ucb1(c))
        else:
            return min(compatible, key=lambda n: n.ucb1(c))


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _rollout(sim: SimulationState, rollout_depth: int) -> float:
    """Random playout from sim (modified in-place copy). Returns win prob for player 0."""
    for _ in range(rollout_depth):
        valid = sim.get_valid_plays()
        if not valid:
            sim.draw_step()
            continue
        card = random.choice(valid)
        color = random.choice(_PLAY_COLORS) if card.is_wild() else None
        winner = sim.step(card, color)
        if winner is not None:
            return 1.0 if winner == 0 else 0.0

    # Depth limit reached: estimate by hand-size difference
    my_size = len(sim.my_hand)
    opp_size = len(sim.opp_hand)
    score = 0.5 + 0.2 * (opp_size - my_size) / max(my_size + opp_size, 1)
    return max(0.1, min(0.9, score))


# ---------------------------------------------------------------------------
# MCTSAgent
# ---------------------------------------------------------------------------

class MCTSAgent(BaseAgent):
    """Selecting cards via ISMCTS (Information Set Monte Carlo Tree Search).

    Args:
        name: Display name used in logs and tournament output.
        num_simulations: Number of ISMCTS iterations per decision.
        c: UCB1 exploration constant (sqrt(2) is standard).
        rollout_depth: Max random-playout steps before using heuristic.
    """

    def __init__(
        self,
        name: str = "MCTSAgent",
        num_simulations: int = 100,
        c: float = 1.41,
        rollout_depth: int = 150,
    ) -> None:
        super().__init__(name)
        self.num_simulations = num_simulations
        self.c = c
        self.rollout_depth = rollout_depth

    def choose_action(self, state: GameState) -> Tuple[Card, Optional[Color]]:
        """Run ISMCTS and return the best (card, color) action.

        Args:
            state: Current game state. state.valid_plays is guaranteed non-empty.

        Returns (card, chosen_color) where chosen_color is set only for wilds.
        """
        root = MCTSNode()

        for _ in range(self.num_simulations):
            # 1. Sample a fresh determinization each iteration
            sim = self._build_sim_state(state)
            node = root
            terminal = False

            # 2. Selection
            while not node.is_leaf(sim):
                compatible = node.compatible_children(sim)
                for child in compatible:
                    child.availability += 1
                node = node.best_child(sim, self.c)
                card, color = node.action  # type: ignore[misc]
                if card is None:
                    sim.draw_step()
                else:
                    winner = sim.step(card, color)
                    if winner is not None:
                        result = 1.0 if winner == 0 else 0.0
                        self._backprop(node, result)
                        terminal = True
                        break

            if terminal:
                continue

            # 3. Expansion
            untried = node.untried_actions_for(sim)
            if untried:
                action = random.choice(untried)
                card, color = action
                winner = None
                if card is None:
                    sim.draw_step()
                else:
                    winner = sim.step(card, color)
                child = MCTSNode(parent=node, action=action)
                child.availability += 1
                node.children.append(child)
                node = child
                if winner is not None:
                    result = 1.0 if winner == 0 else 0.0
                    self._backprop(node, result)
                    continue

            # 4. Rollout (sim is already at the correct state after expansion)
            result = _rollout(sim, self.rollout_depth)

            # 5. Backpropagation
            self._backprop(node, result)

        if not root.children:
            card = state.valid_plays[0]
            color = _best_color(state.valid_plays) if card.is_wild() else None
            return card, color

        best = max(root.children, key=lambda n: n.visits)
        card, color = best.action  # type: ignore[misc]

        if card is not None and card.is_wild():
            color = _best_color(state.valid_plays)

        return card, color  # type: ignore[return-value]

    # ------------------------------------------------------------------

    def _build_sim_state(self, state: GameState) -> SimulationState:
        """Build a SimulationState by determinizing the unknown opponent hand."""
        mcts_player = state.current_player
        opp_player = 1 - mcts_player

        my_hand = list(state.my_full_hand)

        # Pool = all 108 cards minus our known hand and the top card
        pool = build_deck()
        known = list(my_hand) + [state.top_card]
        for card in known:
            try:
                pool.remove(card)
            except ValueError:
                pass  # card already gone (e.g. reshuffled discard)

        random.shuffle(pool)

        opp_size = state.hand_sizes[opp_player]
        draw_size = state.draw_pile_size

        opp_hand = pool[:opp_size]
        draw_pile = pool[opp_size: opp_size + draw_size]

        return SimulationState(
            my_hand=my_hand,
            opp_hand=opp_hand,
            draw_pile=draw_pile,
            top_card=state.top_card,
            current_color=state.current_color,
            current_player=0,
        )

    @staticmethod
    def _backprop(node: MCTSNode, result: float) -> None:
        """Walk from node to root, incrementing visits and wins."""
        current = node
        while current is not None:
            current.visits += 1
            current.wins += result
            current = current.parent
