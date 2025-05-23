from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
CACHE_DIR = DATA_DIR / 'cache'

# File patterns
DRF_PATTERN = "*DRF"
CURRENT_DRF = None # Will be set dynamically

# Output files
PARSED_RACE_DATA = PROCESSED_DATA_DIR / "parsed_race_data_full.parquet"
CURRENT_RACE_INFO = PROCESSED_DATA_DIR / "current_race_info.parquet"
WORKOUTS_LONG = PROCESSED_DATA_DIR / "workouts_long_format.parquet"
PAST_STARTS_LONG = PROCESSED_DATA_DIR / "past_starts_long_format.parquet"

# Bris Files
BRIS_DICT = RAW_DATA_DIR / "bris_dict.txt"
BRIS_SPEC_CACHE = CACHE_DIR / "bris_spec.pkl"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Columns to drop (from current_race_info.txt)
WORKOUT_PAST_PERF_COLUMNS = []