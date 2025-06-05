from pathlib import Path
import yaml

# Determine project root
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"

# Default configuration values
DEFAULT_CONFIG = {
    "data_dir": "data",
    "raw_data_dir": "data/raw",
    "processed_data_dir": "data/processed",
    "cache_dir": "data/cache",
    "drf_pattern": "*DRF",
    "parsed_race_data": "parsed_race_data_full.parquet",
    "current_race_info": "current_race_info.parquet",
    "workouts_long": "workouts_long_format.parquet",
    "past_starts_long": "past_starts_long_format.parquet",
    "bris_dict": "bris_dict.txt",
    "bris_spec_cache": "bris_spec.pkl",
    "workout_past_perf_columns": [],
}

# Load configuration from YAML if available
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        user_conf = yaml.safe_load(f) or {}
else:
    user_conf = {}

# Merge defaults with user overrides
CONF = {**DEFAULT_CONFIG, **user_conf}

# Data directories
DATA_DIR = BASE_DIR / CONF["data_dir"]
RAW_DATA_DIR = BASE_DIR / CONF["raw_data_dir"]
PROCESSED_DATA_DIR = BASE_DIR / CONF["processed_data_dir"]
CACHE_DIR = BASE_DIR / CONF["cache_dir"]

# File patterns
DRF_PATTERN = CONF["drf_pattern"]
CURRENT_DRF = None  # Will be set dynamically

# Output files
PARSED_RACE_DATA = PROCESSED_DATA_DIR / CONF["parsed_race_data"]
CURRENT_RACE_INFO = PROCESSED_DATA_DIR / CONF["current_race_info"]
WORKOUTS_LONG = PROCESSED_DATA_DIR / CONF["workouts_long"]
PAST_STARTS_LONG = PROCESSED_DATA_DIR / CONF["past_starts_long"]

# Bris Files
BRIS_DICT = RAW_DATA_DIR / CONF["bris_dict"]
BRIS_SPEC_CACHE = CACHE_DIR / CONF["bris_spec_cache"]

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Columns to drop (from current_race_info.txt)
WORKOUT_PAST_PERF_COLUMNS = CONF["workout_past_perf_columns"]
