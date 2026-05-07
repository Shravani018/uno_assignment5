"""
Automated agent-vs-agent self-play for data generation.

ROLE IN SYSTEM
--------------
self_play.py drives N games between any two BaseAgent subclasses and
persists the results in two formats:

  1. Raw logs  (data/logs/game_XXXXXX.json)
     Full turn-by-turn game record for debugging and replay.

  2. Processed dataset  (data/datasets/<agent0>_vs_<agent1>.jsonl)
     One JSON-lines record per turn, ready for RL or supervised training.

CRITICAL DESIGN RULE
--------------------
self_play.py NEVER reimplements game logic.
It calls game.play_turn() which returns a TurnResult (pre_state, card_played,
chosen_color, winner). The collector reads TurnResult and builds training rows.
All card effects, player switching, and hand management stay inside UnoGame.

DATASET ROW FORMAT
------------------
Each row is a dict:
    {
        "player"       : int,          # 0 or 1
        "state"        : List[float],  # encoded pre_state via encode_state()
        "action_index" : int,          # index of card_played in valid_plays
        "reward"       : float,        # +1 win / -1 loss / -0.5 draw (set post-game)
        "next_state"   : List[float] | None,  # None on terminal turn
        "done"         : bool,
    }

REWARD ASSIGNMENT
-----------------
Rewards are not known until the game ends. Every row is written with
reward=0.0 during the game. After game.run() returns a winner, the correct
terminal reward (+1/-1/-0.5) is backfilled into all rows before writing.
This avoids a second pass over the data.

COLLECTION LOOP (per game)
--------------------------
1. Instantiate UnoGame with both agents and optional seed.
2. Call game.play_turn() in a loop, collecting TurnResult each step.
3. Encode pre_state and next_state (get_state() after the turn).
4. Accumulate rows in memory with reward=0.0.
5. On game end: set correct reward on all rows, write to .jsonl.
6. Optionally write raw game_log to logs/.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.game import UnoGame, TurnResult
from agents.base_agent import BaseAgent
from training.feature_encoder import encode_state
from config import SelfPlayConfig, TrainingConfig, Paths


def run_self_play(
    agent0: BaseAgent,
    agent1: BaseAgent,
    num_games: int = SelfPlayConfig.NUM_GAMES,
    save_logs: bool = SelfPlayConfig.SAVE_LOGS,
    save_dataset: bool = SelfPlayConfig.SAVE_DATASET,
    base_seed: Optional[int] = SelfPlayConfig.BASE_SEED,
    log_every: int = SelfPlayConfig.LOG_EVERY,
) -> Dict[str, Any]:
    """Running N games between agent0 and agent1, persisting logs and dataset.

    Args:
        agent0: First player agent.
        agent1: Second player agent.
        num_games: Number of games to simulate.
        save_logs: Writing raw per-game JSON logs to data/logs/ when True.
        save_dataset: Writing processed JSONL dataset to data/datasets/ when True.
        base_seed: Starting seed; each game uses base_seed + game_idx.
                   Passing None disables seeding for full randomness.
        log_every: Printing progress every N games.

    Returning summary dict with wins, draws, avg_turns, elapsed_s.
    """
    Paths.ensure_dirs()
    wins = [0, 0]
    draws = 0
    total_turns = 0
    start = time.time()

    dataset_path = _dataset_path(agent0.name, agent1.name) if save_dataset else None
    dataset_file = open(dataset_path, "w", encoding="utf-8") if dataset_path else None

    try:
        for game_idx in range(num_games):
            seed = base_seed + game_idx if base_seed is not None else None
            game = UnoGame(agent0, agent1, seed=seed)

            winner, rows, turn_count = _collect_game(game)

            # Backfilling terminal reward into all rows now that winner is known
            reward0 = _terminal_reward(winner, player=0)
            reward1 = _terminal_reward(winner, player=1)
            for row in rows:
                row["reward"] = reward0 if row["player"] == 0 else reward1

            if winner == -1:
                draws += 1
            else:
                wins[winner] += 1
            total_turns += turn_count

            if save_logs:
                _write_log(game_idx, winner, game.game_log, agent0.name, agent1.name)

            if dataset_file:
                for row in rows:
                    dataset_file.write(json.dumps(row) + "\n")

            if (game_idx + 1) % log_every == 0:
                elapsed = time.time() - start
                print(
                    f"  [{game_idx + 1}/{num_games}]  "
                    f"W0={wins[0]}  W1={wins[1]}  D={draws}  "
                    f"avg_turns={total_turns / (game_idx + 1):.1f}  "
                    f"elapsed={elapsed:.1f}s"
                )
    finally:
        if dataset_file:
            dataset_file.close()

    summary = {
        "wins": wins,
        "draws": draws,
        "total_games": num_games,
        "avg_turns": total_turns / max(num_games, 1),
        "elapsed_s": round(time.time() - start, 2),
    }
    _print_summary(summary, agent0.name, agent1.name)
    return summary


def _collect_game(game: UnoGame) -> Tuple[int, List[dict], int]:
    """Running one game and collecting (state, action, reward) rows.

    Calling game.play_turn() only -- no engine logic reimplemented here.
    Rewards are left at 0.0; the caller backfills after winner is known.

    Args:
        game: Freshly initialised UnoGame instance.

    Returning (winner, dataset_rows, turn_count).
    Winner is -1 on timeout.
    """
    rows: List[dict] = []
    winner = -1

    while game.turn_count < game.max_turns:
        game.turn_count += 1

        # Snapshotting state before the turn (what the agent sees)
        pre_state_vec = encode_state(game.get_state())
        player = game.current_player

        result: TurnResult = game.play_turn()

        # Encoding state after the turn for the next_state field
        next_state_vec = encode_state(game.get_state()) if result.winner is None else None

        # Skipping draw-only turns (no card played = no training signal)
        if result.card_played is not None:
            action_idx = _action_index(result)
            rows.append({
                "player": player,
                "state": pre_state_vec,
                "action_index": action_idx,
                "reward": 0.0,          # backfilled by caller
                "next_state": next_state_vec,
                "done": result.winner is not None,
            })

        if result.winner is not None:
            winner = result.winner
            break

    return winner, rows, game.turn_count


def _action_index(result: TurnResult) -> int:
    """Finding the index of the played card in valid_plays from pre_state.

    Args:
        result: TurnResult containing pre_state and card_played.

    Returning index of card_played in pre_state.valid_plays, or 0 on miss.
    """
    try:
        return result.pre_state.valid_plays.index(result.card_played)
    except ValueError:
        return 0


def _write_log(game_idx: int, winner: int, raw_log: List[dict],
               name0: str, name1: str) -> None:
    """Writing raw game log as a formatted JSON file.

    Args:
        game_idx: Game number used in filename.
        winner: Winning player index or -1.
        raw_log: game.game_log list of turn dicts.
        name0: Agent 0 name.
        name1: Agent 1 name.
    """
    payload = {
        "game_index": game_idx,
        "winner": winner,
        "agents": [name0, name1],
        "turns": raw_log,
    }
    path = Paths.LOGS / f"game_{game_idx:06d}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _dataset_path(name0: str, name1: str) -> Path:
    """Building dataset file path from agent names."""
    tag = f"{name0}_vs_{name1}".replace(" ", "_").lower()
    return Paths.DATASETS / f"{tag}.jsonl"


def _terminal_reward(winner: int, player: int) -> float:
    """Returning terminal reward for a player given the game winner.

    Args:
        winner: Winning player index, or -1 for draw/timeout.
        player: Player index to compute reward for.

    Returning float reward value from TrainingConfig.
    """
    if winner == -1:
        return TrainingConfig.REWARD_DRAW
    if winner == player:
        return TrainingConfig.REWARD_WIN
    return TrainingConfig.REWARD_LOSE


def _print_summary(summary: Dict[str, Any], name0: str, name1: str) -> None:
    """Printing formatted self-play results to stdout."""
    w0, w1 = summary["wins"]
    total = summary["total_games"]
    print(
        f"\n{'─' * 50}\n"
        f"  Self-play complete  ({total} games)\n"
        f"  {name0:<22} wins: {w0:>5}  ({100 * w0 / total:5.1f}%)\n"
        f"  {name1:<22} wins: {w1:>5}  ({100 * w1 / total:5.1f}%)\n"
        f"  Draws / timeouts      : {summary['draws']:>5}\n"
        f"  Avg turns / game      : {summary['avg_turns']:>7.1f}\n"
        f"  Elapsed               : {summary['elapsed_s']:>7.1f}s\n"
        f"{'─' * 50}"
    )