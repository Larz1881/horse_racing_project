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

**Advanced Fitness Metrics**
* **Recovery Rate Index:** Analyzes optimal days between races and current recovery status
* **Form Momentum Score:** Weights recent performances with exponential decay
* **Cardiovascular Fitness Proxy:** Uses pace sustainability and deceleration rates
* **Sectional Improvement Index:** Tracks improvement in 2f sectional times
* **Energy Distribution Profile:** Analyzes how horses distribute effort throughout races

**Workout Analysis**
* **Trainer Pattern Recognition:**
  * Identifies winning workout patterns by trainer
  * Tracks bullet work timing before wins
  * Analyzes optimal workout frequency for success
  * Categorizes patterns (maintenance, sharpening, building)

* **Workout Quality Scoring:**
  * Evaluates workout times relative to track standards
  * Scores bullet work frequency
  * Assesses workout spacing and consistency
  * Includes gate work and distance variety factors

* **Work-to-Race Translation:**
  * Correlates workout patterns with race success
  * Analyzes optimal days from last work
  * Tracks workout intensity impact on performance
  * Provides success rate statistics for different patterns

* **Trainer Intent Signals:**
  * Detects frequency changes (targeting vs maintenance)
  * Identifies distance progression patterns
  * Tracks equipment/surface experimentation
  * Provides confidence scores for intent detection

**Advanced Pace Projections**
* **Multi-Factor Pace Model:**
  * Combines BRIS run style with Quirin speed points
  * Incorporates historical pace figures (E1, E2, Late)
  * Adjusts for post position impact
  * Factors in recent form trends
  * Projects positions at each call with confidence levels

* **Dynamic Pace Pressure Calculator:**
  * Counts early speed horses and calculates pressure
  * Analyzes speed point distribution
  * Identifies pace setters, pressers, and closers
  * Classifies scenarios: Hot, Contested, Honest, Slow, Lone Speed
  * Projects likely fractional times

* **Energy Cost Modeling:**
  * Models energy expenditure by race phase
  * Calculates sustainability based on pace differentials
  * Assesses fade risk with specific fade points
  * Evaluates energy efficiency for running style
  * Identifies horses with optimal energy distribution

* **Pace Shape Classification:**
  * Categorizes races (Fast-Fast-Collapse, Wire-to-Wire, etc.)
  * Matches horses to pace shapes
  * Recommends betting strategies
  * Identifies likely beneficiaries

**Multi-Dimensional Class Assessment**
* **Earnings-Based Class Metrics:**
  * Earnings per start analysis (lifetime, current year, by surface)
  * Earnings velocity (improvement rate)
  * Percentile rankings within population
  * Consistency scoring (ITM% × earnings level)
  * Purse trend analysis

* **Race Classification Hierarchy:**
  * Proper hierarchy from Maiden Claiming to Grade 1
  * Dynamic class level calculation with purse adjustments
  * Class movement tracking (up/down/lateral)
  * Success pattern analysis at different levels
  * Class suitability scoring

* **Hidden Class Indicators:**
  * Sire stud fee analysis (log scale)
  * Auction price with depreciation modeling
  * BRIS pedigree ratings parsing
  * Quality of competition faced
  * Trainer/stable quality indicators

* **Competitive Level Index:**
  * Field quality assessment
  * Quality of horses beaten
  * Speed figures vs field average
  * Consistency against quality competition
  * Margin analysis (dominance indicators)

**Form Cycle Detection**
* **Beaten Lengths Trajectory Analysis:**
  * Converts beaten lengths to time using distance-specific factors
  * Tracks improvement trends with statistical significance
  * Identifies unlucky performances through trip comments
  * Calculates ground gained/lost metrics

* **Position Call Analytics:**
  * Early position vs finish correlation analysis
  * Optimal position curves by distance (sprint vs route)
  * Late position gain patterns and consistency
  * Traffic trouble indicators (wide trips, position losses)

* **Form Cycle Pattern Recognition:**
  * Bounce Detection: Identifies horses at risk after career-best efforts
  * Recovery Patterns: Tracks ability to rebound from poor races
  * Freshening Analysis: Success rates off layoffs
  * Peak Performance Prediction: Estimates races until peak form

* **Fractional Time Evolution:**
  * Tracks improvement in each race fraction
  * Pace sustainability analysis (early vs late splits)
  * Final time progression curves
  * Efficiency gain calculations

* **Field Size Adjusted Performance:**
  * Normalizes for horses beaten percentage
  * Adjusts for field quality (race type, purse, field size)
  * Performance vs expectations analysis
  * Consistency across different field sizes

**Form Cycle States:**
* **IMPROVING:** Clear upward trajectory
* **PEAKING:** At or near career best form
* **BOUNCING:** Risk of regression after peak
* **RECOVERING:** Rebounding from poor effort
* **DECLINING:** Downward trajectory
* **FRESHENING:** Returning from layoff
* **STABLE:** Consistent performer
* **ERRATIC:** Inconsistent pattern

**Integrated Analytics System**
* **Composite Fitness Score:**
  * Weights all component scores (fitness, workout, pace, class, form)
  * Time-to-peak predictions based on form state and momentum
  * Flags horses entering optimal form
  * Confidence scoring based on data completeness

* **Class-Adjusted Speed Figures:**
  * Adjusts raw speed figures by class level
  * Creates pound-for-pound ratings (performance relative to class)
  * Projects class movement success
  * Identifies horses that should handle class changes

* **Pace Impact Predictions:**
  * Individual advantages based on pace scenario
  * Energy reserve considerations
  * Race flow positioning
  * Tactical advantage assessment

* **Machine Learning Enhancements:**
  * Form Cycle Classifier: Predicts form states using RandomForest
  * Workout Clustering: Groups horses by training patterns
  * Class Trajectory Model: Predicts success in class moves
  * Pace Optimizer: Learns optimal positioning strategies
  * Performance Predictor: Integrated model for overall predictions

**Key Integrated Metrics:**
* **Final Rating:** Composite score combining all factors
* **Overall Rank:** Position within race
* **Key Angles:** Specific betting angles identified
* **Prediction Confidence:** Reliability of the assessment

**Interactive Dashboard**
* **Race Summary Tab:**
  * Complete race header with distance, surface, class, and purse
  * Summary table with all requested metrics (ML odds, days off, Prime Power, E1/E2/LP averages and bests)
  * Clean, sortable format for quick reference

* **Integrated Analytics Tab:**
  * Overall rankings visualization
  * Multi-factor radar chart showing all component scores
  * Key betting angles identification
  * Color-coded by performance levels

* **Fitness Analysis Tab:**
  * Fitness components heatmap
  * Energy distribution pie charts showing running style
  * Performance trends over recent races
  * Visual identification of improving/declining horses

* **Pace Projections Tab:**
  * Race pace scenario display with color coding
  * Projected running positions throughout race
  * Energy reserve vs efficiency scatter plot
  * Fade risk identification

* **Class Assessment Tab:**
  * Earnings vs Hidden vs Overall class comparison
  * Class movement distribution
  * Hidden class indicators (sire quality, pedigree, competition)
  * Visual identification of class droppers

* **Form Cycles Tab:**
  * Form cycle state visualization with intuitive colors
  * Form trajectory scatter plot
  * Position gains/losses bar chart
  * Easy identification of horses in optimal form windows

* **Workout Analysis Tab:**
  * Workout readiness scores with categories
  * Pattern analysis (bullets, frequency, fast works)
  * Trainer intent signals table
  * Clear visual of workout quality

**Dashboard Design Features:**
* Dark theme for easy viewing
* Color-coded metrics (green=good, red=bad)
* Interactive charts with hover details
* Responsive layout
* Single race filter for focused analysis

**Dashboard Usage:**
* Select a race from the dropdown
* Navigate through tabs to view different analyses
* Look for convergence of positive factors
* Identify horses with multiple green indicators
* Note any red flags in form cycles or pace scenarios

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
│   │   ├── sophisticated_workout_analysis.parquet  # Main workout analysis
│   │   ├── trainer_workout_patterns.csv           # Trainer-specific patterns
│   │   ├── workout_race_translation.csv           # Statistical correlations
│   │   ├── workout_analysis_summary.csv           # Quick reference summary
│   │   ├── advanced_pace_analysis.parquet         # Complete pace projections
│   │   ├── pace_pressure_summary.csv              # Race-by-race scenarios
│   │   ├── pace_shapes_summary.csv                # Pace shape classifications
│   │   ├── pace_analysis_summary.csv              # Quick reference summary
│   │   ├── multi_dimensional_class_assessment.parquet  # Complete class analysis
│   │   ├── class_movements_summary.csv            # Class movement tracking
│   │   ├── hidden_class_indicators.csv            # Hidden vs apparent class
│   │   ├── class_assessment_summary.csv           # Quick reference summary
│   │   ├── form_cycle_analysis.parquet            # Complete form cycle analysis
│   │   ├── beaten_lengths_trajectory.csv          # Detailed trajectory analysis
│   │   ├── form_cycle_patterns.csv                # Pattern detection results
│   │   ├── form_cycle_summary.csv                 # Quick reference summary
│   │   ├── integrated_analytics_report.parquet    # Complete integrated analysis
│   │   └── integrated_summary.csv                 # Quick reference summary
│   ├── models/                 # Saved ML models
│   │   ├── form_cycle_classifier.pkl
│   │   ├── workout_cluster_model.pkl
│   │   ├── class_trajectory_model.pkl
│   │   ├── pace_optimizer.pkl
│   │   └── performance_predictor.pkl
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
│   ├── app.py                # Main dashboard application
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
    python dashboards/app.py
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
5. python dashboards/app.py

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

### `dashboards/app.py`
Interactive Plotly Dash dashboard for exploring the processed racing data.
The app loads parquet files defined in `config/settings.py` and presents
analytics across multiple tabs:

* **Race Summary** – sortable table with averages and best figures
* **Integrated Analytics** – rankings bar chart, radar comparisons and key angles
* **Fitness Analysis** – heatmap of conditioning metrics and energy profiles
* **Pace Projections** – projected running positions and energy reserve scatter
* **Class Assessment** – earnings vs hidden class comparisons
* **Form Cycles** – state visualization and trajectory charts
* **Workout Analysis** – readiness scores and trainer patterns
* **Betting Intelligence** – value finder scatter plot highlighting overlays

Run the dashboard with:
```bash
python dashboards/app.py
```
By default it listens on port 8050.
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
Try: python dashboards/app.py --port 8051 (if your Dash app is set up to accept port arguments)
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