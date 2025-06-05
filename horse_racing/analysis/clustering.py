import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from pathlib import Path
import numpy as np
from config.settings import PAST_STARTS_LONG

def analyze_race_clustering(df_race, race_num, features, context_cols, min_horses=3):
    """
    Perform clustering analysis for a single race.
    
    Parameters:
    - df_race: DataFrame for the specific race
    - race_num: Race number
    - features: List of feature columns for clustering
    - context_cols: List of context columns to include in output
    - min_horses: Minimum number of horses needed for clustering
    
    Returns:
    - result_df: DataFrame with clustering results, or None if insufficient data
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING RACE {race_num}")
    print(f"{'='*50}")
    
    # Prepare clustering data
    X = df_race[features].dropna()
    
    if len(X) < min_horses:
        print(f"❌ Race {race_num}: Only {len(X)} horses with complete pace data. Minimum {min_horses} required for clustering.")
        return None
    
    print(f"✅ Race {race_num}: {len(X)} horses with complete pace data")
    
    # Perform K-means clustering with 3 clusters
    try:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
        
        # Assign cluster labels back to the race dataframe
        df_race_copy = df_race.copy()
        df_race_copy.loc[X.index, 'pace_cluster'] = kmeans.labels_
        
        # Build contextual output table
        result = df_race_copy.loc[X.index, context_cols + features + ['pace_cluster']].copy()
        
        # Add race number for identification
        result['race'] = race_num
        
        print(f"📊 Clustering Results for Race {race_num}:")
        print("-" * 30)
        
        # Show sample of results
        sample_cols = ['horse_name', 'morn_line_odds_if_available', 'pace_cluster'] + features[:2]  # Show first 2 features
        available_cols = [col for col in sample_cols if col in result.columns]
        print(result[available_cols].head(10).to_string(index=False))
        
        # Summary statistics by cluster
        if 'pp_odds' in result.columns and 'pp_finish_pos' in result.columns:
            cluster_summary = result.groupby('pace_cluster')[['pp_odds', 'pp_finish_pos']].agg({
                'pp_odds': ['mean', 'std', 'count'],
                'pp_finish_pos': ['mean', 'std']
            }).round(2)
            print(f"\n📈 Cluster Summary (Race {race_num}):")
            print(cluster_summary)
        
        # Count horses by run style and cluster if available
        if 'bris_run_style_designation' in result.columns:
            style_cluster_counts = result.groupby(['pace_cluster', 'bris_run_style_designation']).size()
            if not style_cluster_counts.empty:
                print(f"\n🏃 Run Style Distribution by Cluster (Race {race_num}):")
                print(style_cluster_counts)
        
        return result
        
    except Exception as e:
        print(f"❌ Error in clustering Race {race_num}: {e}")
        return None

def create_race_visualization(result_df, race_num, features, save_dir):
    """Create and save pairplot visualization for a race."""
    try:
        plt.figure(figsize=(12, 10))
        
        # Create pairplot
        available_features = [f for f in features if f in result_df.columns and result_df[f].notna().sum() > 0]
        if len(available_features) >= 2 and 'pace_cluster' in result_df.columns:
            pair_plot = sns.pairplot(
                result_df, 
                vars=available_features[:4],  # Limit to first 4 features to avoid overcrowding
                hue='pace_cluster', 
                palette='tab10',
                height=2.5
            )
            pair_plot.fig.suptitle(f'Pace Feature Clusters - Race {race_num}', y=1.02, size=16)
            
            # Save the plot
            plot_path = save_dir / f'race_{race_num}_pace_clusters.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"💾 Visualization saved: {plot_path}")
            plt.close()
        else:
            print(f"⚠️  Cannot create visualization for Race {race_num}: insufficient feature data")
            
    except Exception as e:
        print(f"❌ Error creating visualization for Race {race_num}: {e}")

def main():
    """Main function to run clustering analysis for all races."""
    print("🏇 HORSE RACING PACE CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Load the long-format DataFrame
    if not PAST_STARTS_LONG.exists():
        print(f"❌ Error: Past starts file not found at {PAST_STARTS_LONG}")
        return
    
    print(f"📂 Loading data from: {PAST_STARTS_LONG}")
    df = pd.read_parquet(PAST_STARTS_LONG)
    print(f"✅ Loaded {len(df)} past performance records")
    
    # Get all available races
    available_races = sorted(df['race'].dropna().unique())
    print(f"🏁 Found {len(available_races)} races: {available_races}")
    
    # Feature columns for clustering
    features = ['pp_e1_pace', 'pp_e2_pace', 'pp_turn_time', 'pp_bris_late_pace']
    
    # Context columns for analysis
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
    
    # Ensure only available columns are used
    available_context_cols = [col for col in context_cols if col in df.columns]
    available_features = [col for col in features if col in df.columns]
    
    if not available_features:
        print("❌ Error: No pace features found in the dataset")
        return
    
    print(f"📊 Using features: {available_features}")
    print(f"📋 Using context columns: {len(available_context_cols)} columns")
    
    # Create output directory for results
    output_dir = Path("clustering_results")
    output_dir.mkdir(exist_ok=True)
    
    # Store all results
    all_results = []
    successful_races = []
    failed_races = []
    
    # Analyze each race
    for race_num in available_races:
        df_race = df[df['race'] == race_num].copy()
        
        result = analyze_race_clustering(
            df_race, race_num, available_features, available_context_cols
        )
        
        if result is not None:
            all_results.append(result)
            successful_races.append(race_num)
            
            # Save individual race results
            race_output_path = output_dir / f'race_{race_num}_clusters_with_context.csv'
            result.to_csv(race_output_path, index=False)
            print(f"💾 Results saved: {race_output_path}")
            
            # Create visualization
            create_race_visualization(result, race_num, available_features, output_dir)
            
        else:
            failed_races.append(race_num)
    
    # Combine all results if we have any
    if all_results:
        print(f"\n🎯 OVERALL SUMMARY")
        print("=" * 40)
        print(f"✅ Successfully analyzed: {len(successful_races)} races")
        print(f"❌ Failed to analyze: {len(failed_races)} races")
        
        if failed_races:
            print(f"   Failed races: {failed_races}")
        
        # Combine all race results
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_output_path = output_dir / 'all_races_clusters_with_context.csv'
        combined_results.to_csv(combined_output_path, index=False)
        print(f"💾 Combined results saved: {combined_output_path}")
        
        # Overall statistics
        print(f"\n📈 COMBINED STATISTICS:")
        print(f"   Total horses analyzed: {len(combined_results)}")
        print(f"   Average horses per race: {len(combined_results) / len(successful_races):.1f}")
        
        # Cluster distribution across all races
        if 'pace_cluster' in combined_results.columns:
            cluster_dist = combined_results['pace_cluster'].value_counts().sort_index()
            print(f"   Cluster distribution: {dict(cluster_dist)}")
        
        # Average odds and finish position by cluster (across all races)
        if all(col in combined_results.columns for col in ['pace_cluster', 'pp_odds', 'pp_finish_pos']):
            overall_cluster_summary = combined_results.groupby('pace_cluster')[['pp_odds', 'pp_finish_pos']].agg({
                'pp_odds': ['mean', 'std', 'count'],
                'pp_finish_pos': ['mean', 'std']
            }).round(2)
            print(f"\n📊 Overall Cluster Performance:")
            print(overall_cluster_summary)
        
        print(f"\n🎉 Analysis complete! Check the '{output_dir}' directory for detailed results.")
        
    else:
        print("❌ No races could be analyzed successfully.")

if __name__ == "__main__":
    main()



