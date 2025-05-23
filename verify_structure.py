# verify_structure.py
from pathlib import Path
import sys

# Temporarily add config to path if running from root and config is not in python path yet
# This depends on where your config.settings is relative to this script
# If BASE_DIR in settings.py is defined as Path(__file__).resolve().parent.parent
# and settings.py is in config/, then this script in root needs to know where 'config' is.
# For simplicity, assuming config.settings can be imported if this script is in project root
# and the python path includes the project root.
# If you have PYTHONPATH issues, you might need:
# sys.path.append(str(Path(__file__).parent)) # Add project root to path if not already there

try:
    from config.settings import BASE_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR
except ImportError:
    print("❌ Could not import from config.settings.")
    print("Ensure verify_structure.py is in the project root and your PYTHONPATH is set up correctly,")
    print("or that config/settings.py defines BASE_DIR in a way that's accessible.")
    sys.exit(1)


def verify_project_structure():
    """Verify all files are in correct locations."""
    print("Checking project structure...")

    # Check directories exist
    for name, path in [
        ("Raw data", RAW_DATA_DIR),
        ("Processed data", PROCESSED_DATA_DIR),
        ("Cache", CACHE_DIR)
    ]:
        if path.exists() and path.is_dir():
            print(f"✅ {name} directory exists: {path}")
        else:
            print(f"❌ {name} directory missing or not a directory: {path}")

    # Check for parquet files in root (bad!)
    # Assuming BASE_DIR is the true project root from config.settings
    root_parquets = list(BASE_DIR.glob("*.parquet"))
    if root_parquets:
        print(f"\n❌ Found {len(root_parquets)} parquet files in project root ({BASE_DIR}) - please move to data/processed/")
        for f in root_parquets:
            print(f"   - {f.name}")
    else:
        print(f"\n✅ No parquet files in project root directory ({BASE_DIR}) (good!)")

    # Check for parquet files in correct location
    if PROCESSED_DATA_DIR.exists():
        processed_parquets = list(PROCESSED_DATA_DIR.glob("*.parquet"))
        if processed_parquets:
            print(f"\n✅ Found {len(processed_parquets)} parquet files in {PROCESSED_DATA_DIR}:")
            for f in processed_parquets:
                print(f"   - {f.name}")
        else:
            print(f"\n⚠️  No parquet files found in {PROCESSED_DATA_DIR} (This may be OK if pipeline hasn't run yet).")
    else:
        print(f"\n❌ Processed data directory does not exist: {PROCESSED_DATA_DIR}")


    # Check for DRF files
    if RAW_DATA_DIR.exists():
        drf_files = list(RAW_DATA_DIR.glob("*.DRF"))
        if drf_files:
            print(f"\n✅ Found {len(drf_files)} DRF files in {RAW_DATA_DIR}:")
            for f in drf_files:
                print(f"   - {f.name}")
        else:
            print(f"\n⚠️  No DRF files found in {RAW_DATA_DIR} (This may be OK if no raw data has been added).")
    else:
        print(f"\n❌ Raw data directory does not exist: {RAW_DATA_DIR}")


if __name__ == "__main__":
    verify_project_structure()