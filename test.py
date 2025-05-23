import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Load the long-format DataFrame
df = pd.read_parquet('past_starts_long_format.parquet')

# Select only Race 7
df_race7 = df[df['race'] == 7].copy()

#print(df_race1.columns.tolist())


# Feature columns for clustering (as before)
features = ['pp_e1_pace', 'pp_e2_pace', 'pp_turn_time', 'pp_bris_late_pace']

# Context columns you want to see
context_cols = [
    'horse_name',
    'pp_race_date',
    'pp_track_condition',
    'pp_distance',
    'pp_distance_type',
    'pp_surface',
    'pp_num_entrants',
    'pp_odds',
    'morn_line_odds_if_available',
    'pp_purse',
    'pp_finish_pos',
    'pp_bris_speed_rating',
    'bris_run_style_designation',
    'quirin_style_speed_points'
]

# Prepare clustering as before
X = df_race7[features].dropna()
if len(X) >= 3:
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X)
    # Assign cluster only to those with complete pace data
    df_race7.loc[X.index, 'pace_cluster'] = kmeans.labels_

    # Build a contextual output table for inspection and decision-making
    result = df_race7.loc[X.index, context_cols + features + ['pace_cluster']]
    print(result.head()) 
    print(result.sort_values('pace_cluster').to_string(index=False))

    # Summary: average odds and finish by cluster
    print(result.groupby('pace_cluster')[['pp_odds', 'pp_finish_pos']].mean())
    # Count horses by run style and cluster
    print(result.groupby(['pace_cluster', 'bris_run_style_designation']).size())


    # Optional: Save to CSV for review
    result.to_csv('race1_clusters_with_context.csv', index=False)

    # (Optional) Visual: still show pairplot
    sns.pairplot(df_race7.loc[X.index], vars=features, hue='pace_cluster', palette='tab10')
    plt.suptitle('Pace Feature Clusters with Context (Race 1)', y=1.02)
    plt.show()
else:
    print("Not enough horses with complete pace data in Race 1 for clustering.")



