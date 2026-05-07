"""
Central configuration for uno-agent-arena.

ROLE IN SYSTEM
--------------
Single source of truth for all constants: game rules, paths, self-play
settings, encoding dimensions, and training hyperparameters.
Importing from here means changing one value propagates everywhere.

SECTIONS
--------
GameConfig      -- fixed UNO rules (hand size, deck size, turn cap)
EncoderConfig   -- feature vector dimensions used by feature_encoder.py
SelfPlayConfig  -- defaults for run_self_play() in self_play.py
TrainingConfig  -- reward values and RL hyperparameters
Paths           -- all data directories as pathlib.Path objects
"""
from pathlib import Path


class GameConfig:
    """Storing fixed UNO game rule constants."""
    HAND_SIZE: int = 7
    DECK_SIZE: int = 108
    MAX_TURNS: int = 500
    NUM_PLAYERS: int = 2


class EncoderConfig:
    """Defining feature vector dimensions for state encoding."""
    NUM_COLORS: int = 5       # RED GREEN BLUE YELLOW WILD
    NUM_CARD_TYPES: int = 6   # NUMBER SKIP REVERSE DRAW_TWO WILD WILD_DRAW_FOUR
    NUM_VALUES: int = 10      # 0-9

    # Scalar features: my_hand / 20, opp_hand / 20, draw_pile / 108
    NUM_SCALARS: int = 3

    @classmethod
    def vector_size(cls) -> int:
        """Returning total feature vector length."""
        # top_card color + type + value + current_color + scalars
        return (
            cls.NUM_COLORS       # top card color
            + cls.NUM_CARD_TYPES # top card type
            + cls.NUM_VALUES     # top card value
            + cls.NUM_COLORS     # current color
            + cls.NUM_SCALARS    # hand sizes + draw pile
        )


class SelfPlayConfig:
    """Storing defaults for self-play data generation."""
    NUM_GAMES: int = 1_000
    SAVE_LOGS: bool = True
    SAVE_DATASET: bool = True
    BASE_SEED: int = 42
    LOG_EVERY: int = 100


class TrainingConfig:
    """Storing RL reward values and training hyperparameters."""
    REWARD_WIN: float = 1.0
    REWARD_LOSE: float = -1.0
    REWARD_DRAW: float = -0.5
    LEARNING_RATE: float = 1e-3
    BATCH_SIZE: int = 64
    GAMMA: float = 0.95        # discount factor


class Paths:
    """Defining all data directory paths relative to project root."""
    ROOT: Path = Path(__file__).parent
    DATA: Path = ROOT / "data"
    LOGS: Path = DATA / "logs"
    DATASETS: Path = DATA / "datasets"
    MODELS: Path = DATA / "models"

    @classmethod
    def ensure_dirs(cls) -> None:
        """Creating all data directories if they do not exist."""
        for d in (cls.LOGS, cls.DATASETS, cls.MODELS):
            d.mkdir(parents=True, exist_ok=True)