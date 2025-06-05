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
