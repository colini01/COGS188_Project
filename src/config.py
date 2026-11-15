"""
config.py: the only place numbers live.
Every other file does `from config import ...` or `import config`.
To change a value: tell the group, edit the rules doc, then edit it here.
"""
import numpy as np
from pathlib import Path

# ---------- paths ----------
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RATES_PATH = PROCESSED_DIR / "rates.npy"
TRACT_TO_CELL_PATH = PROCESSED_DIR / "tract_to_cell.csv"

# ---------- world and time (section 3) ----------
GRID_ROWS = 20
GRID_COLS = 20
N_UNITS = 3
STEPS_PER_HOUR = 4                     # one step = 15 minutes
STEPS_PER_EPISODE = 96                 # one episode = one 24-hour day
N_TIME_BLOCKS = 4                      # 6-hour blocks: block = hour // 6
N_LEVELS = 3                           # neighbor labels: 0 low, 1 medium, 2 high

# action id -> (row change, col change)
ACTIONS = {
    0: (0, 0),     # stay
    1: (-1, 0),    # up
    2: (1, 0),     # down
    3: (0, -1),    # left
    4: (0, 1),     # right
}
N_ACTIONS = len(ACTIONS)

START_POSITIONS = [(4, 4), (4, 15), (15, 10)]   # unit 0, 1, 2

# fixed-route baseline rectangles (section 8): (row_min, row_max, col_min, col_max)
FIXED_ROUTES = [
    (2, 7, 2, 7),        # unit 0
    (2, 7, 12, 17),      # unit 1
    (12, 17, 7, 12),     # unit 2
]

# ---------- crime generation (section 5) ----------
DAILY_CRIMES = 150                     # expected true crimes per episode

# ---------- displacement (section 5) ----------
DISP_ON = False                        # turned on only in the displacement experiment
DISP_RADIUS = 2
DISP_DROP = 0.05
DISP_FLOOR = 0.5
DISP_RELAX = 0.02

# ---------- detection and reward (section 6) ----------
DETECT_SCALE = 2.0                     # P(d) = exp(-d / DETECT_SCALE)
MAX_DETECT_DIST = 6
REWARD_PER_CATCH = 1.0
FAIRNESS_PENALTY = 0.0                 # optional: 0.01 if we add it
FAIRNESS_WINDOW = 96

# ---------- memory and state (section 7) ----------
RECORD_DECAY = 0.999                   # per step, applied to the recorded-crime map
Q_SHAPE = (GRID_ROWS, GRID_COLS, N_TIME_BLOCKS,
           N_LEVELS, N_LEVELS, N_LEVELS, N_LEVELS, N_ACTIONS)   # (20,20,4,3,3,3,3,5)

# ---------- learning (section 9) ----------
ALPHA = 0.1
GAMMA = 0.95
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995                      # multiplied once per episode
UCB_C = 0.5
OPT_Q0 = 5.0

# ---------- dispatch with A* (section 10) ----------
DISPATCH_ON = False

# ---------- experiments (section 12) ----------
N_TRAIN_EPISODES = 2000
N_EVAL_EPISODES = 100
EVAL_SEED_OFFSET = 1000                # evaluation seed = run seed + 1000
TRAIN_SEEDS = [0, 1, 2, 3, 4]
TUNE_SEEDS = [100, 101, 102]
N_TUNE_EPISODES = 500

LEARNERS = ["qlearning", "sarsa", "mc"]
EXPLORATIONS = ["egreedy", "ucb", "optinit"]
BASELINES = ["random", "fixed", "greedy"]

TUNE_GRID = {
    "ALPHA": [0.05, 0.1, 0.2],
    "GAMMA": [0.9, 0.95, 0.99],
    "EPS_DECAY": [0.99, 0.995, 0.999],
}

# ---------- data pipeline (section 11) ----------
CENTER_ZIP = 92101
N_ZIPS = 40
MIN_ZIP_INCIDENTS = 500
BIN_METHOD = "rank"                   # "rank" works better thn "width"
TRIM_PCT = 2
N_YEARS = 3
DROP_MIDNIGHT = False
RATE_FLOOR_FRACTION = 0.01             # floor = 0.01 * mean cell-hour rate

# ---------- tiny test grid (section 11) ----------
TINY = {
    "rows": 5, "cols": 5, "n_units": 1, "steps": 24,
    "start": [(0, 0)], "hot_cell": (4, 4),
    "base_rate": 0.01, "hot_rate": 1.0,
}

# ---------- helpers ----------
def time_block(step):
    """6-hour block (0-3) for a step number 0-95."""
    return (step // STEPS_PER_HOUR) // 6

def hour_of(step):
    return step // STEPS_PER_HOUR

def detect_prob(d):
    """Probability a crime at Manhattan distance d from the nearest unit is detected."""
    return float(np.exp(-d / DETECT_SCALE)) if d <= MAX_DETECT_DIST else 0.0

# ---------- extra names the cleaning notebook uses ----------
GRID_SIZE = GRID_ROWS                  # the notebook calls the grid size GRID_SIZE
INTERVAL_MINUTES = 15
RAW_PATH = RAW_DIR / "CIBRS_Group_A_Public_Crime_Data_20261113.csv"
CENSUS_PATH = RAW_DIR / "2025_gaz_tracts_06.txt"