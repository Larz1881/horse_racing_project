import pandas as pd

# Check column names in each file
files_to_check = [
    'data/processed/current_race_info.parquet',
    'data/processed/past_starts_long_format.parquet', 
    'data/processed/workouts_long_format.parquet'
]

for file_path in files_to_check:
    df = pd.read_parquet(file_path)
    print(f"\n{'='*60}")
    print(f"Columns in {file_path.split('/')[-1]}:")
    print(f"{'='*60}")
    for i, col in enumerate(sorted(df.columns)):
        print(f"{i+1:3d}. {col}")
    print(f"\nShape: {df.shape}")