# Horse Racing Analysis Pipeline
A comprehensive data processing and analysis system for horse racing handicapping using Brisnet DRF (Daily Racing Form) data files. This project transforms raw racing data into actionable insights through automated parsing, feature engineering, and interactive visualizations.

🏇 ## Overview
This project provides a complete pipeline for processing Brisnet horse racing data, including:

* Automated parsing of DRF comma-delimited files
* Data transformation from wide to long format
* Feature engineering for handicapping metrics
* NLP-based trip note analysis
* Interactive dashboards for race analysis
* Pace and performance visualizations

✨ ## Features

**Data Processing**
* **Automated DRF Parsing:** Converts Brisnet's 1400+ column format into structured data
* **Smart Data Transformation:** Reshapes workout and past performance data for analysis
* **Feature Engineering:** Calculates key handicapping metrics (pace figures, speed ratings, etc.)

**Analysis Tools**
* **Trip Note Analysis:** NLP-powered categorization of race comments
* **Pace Analysis:** E1/E2 pace calculations and visualization
* **Performance Metrics:** Win percentages, ROI calculations, earnings analysis
* **Connection Analysis:** Trainer/jockey combinations and success rates

**Visualization**
* **Interactive Dashboards:** Plotly Dash-based web interface
* **Pace Ridgeline Plots:** Distribution visualizations for pace analysis
* **Performance Clustering:** ML-based horse performance grouping

📁 ## Project Structure

horse_racing_project/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── run_pipeline.py             # Main pipeline orchestrator
├── dev_workflow.md             # Development workflow guide
│
├── config/                     # Configuration files
│   └── settings.py            # Project settings and paths
│
├── data/
│   ├── raw/                   # Original DRF files and bris_dict.txt
│   ├── processed/             # Processed parquet files
│   └── cache/                 # Temporary cache files
│
├── src/
│   ├── parsers/               # DRF parsing modules
│   │   └── bris_spec.py      # Main Brisnet parser
│   ├── transformers/          # Data transformation modules
│   │   ├── current_race_info.py
│   │   ├── transform_workouts.py
│   │   ├── transform_past_starts.py
│   │   └── feature_engineering.py
│   ├── analysis/              # Analysis modules
│   │   ├── race_views.py     # Consolidated data views
│   │   ├── trip_pipeline.py  # NLP comment analysis
│   │   └── clustering.py     # Performance clustering
│   ├── visualization/         # Visualization modules
│   └── utils/                 # Utility functions
│       └── data_utils.py
│
├── dashboards/                # Web dashboards
│   ├── app_dash.py           # Main Dash application
│   └── assets/               # CSS/JS assets
│
├── notebooks/                 # Jupyter notebooks
│   └── data_exploration.ipynb
│
└── tests/                     # Test files (future)

🚀 ## Installation

**Prerequisites**
* Python 3.12.6 or higher
* Windows 10/11 (tested on Windows, should work on Mac/Linux)
* 4GB+ RAM recommended
* 1GB+ free disk space

**Setup Steps**
1.  **Clone or download the project**
    ```bash
    # If using git
    git clone <repository-url>
    cd horse_racing_project

    # Or simply extract the zip file
    ```
2.  **Create virtual environment**
    ```bash
    # Create virtual environment
    python -m venv venv

    # Activate it (Windows)
    venv\Scripts\activate

    # Activate it (Mac/Linux)
    source venv/bin/activate
    ```
3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt

    # For NLP features (optional)
    python -m spacy download en_core_web_sm
    ```
4.  **Set up data directories**
    ```bash
    # The pipeline will create these automatically, but you can create manually:
    mkdir -p data/raw data/processed data/cache
    ```

📊 ## Usage

**Quick Start**
1.  **Place your DRF file in the `data/raw/` directory**
    * Download from Brisnet website
    * File should have `.DRF` extension (e.g., `PIM0509.DRF`)
2.  **Run the complete pipeline**
    ```bash
    python run_pipeline.py
    ```
3.  **Launch the dashboard**
    ```bash
    python dashboards/app_dash.py
    ```
    Then open `http://localhost:8050` in your browser

**Individual Components**
Run specific parts of the pipeline:
```bash
# Just parse the DRF file
python -m src.parsers.bris_spec

# Transform to long format
python -m src.transformers.transform_workouts
python -m src.transformers.transform_past_starts

# Run feature engineering
python -m src.transformers.feature_engineering

# Analyze trip comments
python -m src.analysis.trip_pipeline

# Working with the Data
# Load processed data in Python
import pandas as pd

# Load current race information
current_df = pd.read_parquet('data/processed/current_race_info.parquet')

# Load past performances
past_df = pd.read_parquet('data/processed/past_starts_long_format.parquet')

# Load workouts
workouts_df = pd.read_parquet('data/processed/workouts_long_format.parquet')

🔄 ## Data Flow

Raw DRF File → bris_spec.py → Wide Format DataFrame
    * Parses 1400+ columns based on Brisnet specification
    * Handles data type conversion and validation
Wide Format → transformers/ → Multiple Specialized DataFrames
    * current_race_info.parquet: Today's race entries
    * workouts_long_format.parquet: 12 workouts per horse in long format
    * past_starts_long_format.parquet: 10 past races per horse in long format
Specialized DataFrames → feature_engineering.py → Enhanced Features
    * Pace calculations (E1, E2, turn time)
    * Performance metrics (win %, ROI)
    * Speed and pace figures
Enhanced Data → dashboards/ → Interactive Visualizations
⚙️ ## Configuration
Edit config/settings.py to customize:

# Data directories (usually no need to change)
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Add custom columns to drop from wide format
WORKOUT_PAST_PERF_COLUMNS = [
    # Add any additional columns you want to exclude
]

🔧 ## Development Workflow
See dev_workflow.md for detailed development practices. Quick reference:

# Daily workflow
1. cd horse_racing_project
2. venv\Scripts\activate
3. # Add new DRF to data/raw/
4. python run_pipeline.py
5. python dashboards/app_dash.py

📚 ## Module Documentation

Core Modules

src/parsers/bris_spec.py
Main parser for Brisnet DRF files. Handles the complex 1400+ column format.

src/analysis/race_views.py
Consolidated module providing different analytical views:

get_connections_view(): Trainer, jockey, owner information
get_pedigree_view(): Breeding and auction information
get_trainer_jockey_stats(): Performance statistics
src/analysis/trip_pipeline.py
NLP-based trip comment analysis:

Categorizes comments (trouble, equipment issues, etc.)
Assigns severity scores
Aggregates trip information by horse
Utility Functions

src/utils/data_utils.py
check_parquet_columns(): Verify parquet file structure
quick_eda(): Quick exploratory data analysis
verify_id_variables(): Check for required ID columns
🎯 ## Common Use Cases

Analyze a Specific Race
from src.analysis.race_views import RaceDataViews

viewer = RaceDataViews()
race_5_connections = viewer.get_connections_view(race_num=5)

Get Trip Note Summary
# After running trip_pipeline.py
# Assuming trip_pipeline.py saves its output to 'trip_comments_agg.csv'
# or returns a DataFrame that you can then save/use.
# Example:
# trip_summary_df = analyze_trip_comments_function() 
# print(trip_summary_df[trip_summary_df['race'] == 5])

# If it saves to a file:
# import pandas as pd
# trip_summary = pd.read_csv('data/processed/trip_comments_agg.csv') # Adjust path as needed
# print(trip_summary[trip_summary['race'] == 5])

Export Data for External Analysis
import pandas as pd
current_df = pd.read_parquet('data/processed/current_race_info.parquet')
current_df.to_excel('race_data.xlsx', index=False)

🐛 ## Troubleshooting

Common Issues

"No DRF files found"
Ensure your DRF file is in data/raw/
Check file extension is .DRF (uppercase)
"Module not found" errors
Make sure you're in the project root directory
Activate your virtual environment
Reinstall requirements: pip install -r requirements.txt
Memory errors with large files
Close other applications
Process races individually if needed (if your scripts support this)
Consider upgrading RAM for files over 10MB
Dash app won't start
Check if port 8050 is already in use
Try: python dashboards/app_dash.py --port 8051 (if your Dash app is set up to accept port arguments)
Logging
The pipeline creates timestamped log files (based on your run_pipeline.py setup):

pipeline_YYYYMMDD_HHMMSS.log Check these for detailed error messages.
🚧 ## Future Enhancements

[ ] Advanced NLP for trip notes using transformer models
[ ] Machine learning models for win probability
[ ] Real-time odds integration
[ ] Performance optimization for larger race cards
[ ] Mobile-responsive dashboard design
[ ] Export functionality for third-party handicapping software
[ ] Historical performance tracking database

🤝 ## Contributing
While this is a personal project, improvements are welcome:

Create a new branch for your feature
Write tests for new functionality
Ensure all existing tests pass
Update documentation as needed
📄 ## License
This project is for personal/educational use. Brisnet data is proprietary and subject to their terms of service.

🙏 ## Acknowledgments

Brisnet for providing comprehensive racing data
The Python data science community for excellent libraries
Online handicapping communities for domain knowledge