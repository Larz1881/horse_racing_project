import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
# Patch numpy to add bool8 alias (removed in numpy >=1.24)
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

# Cache data loading for performance
def load_data():
    return (
        pd.read_parquet("current_race_info.parquet", engine="pyarrow"),
        pd.read_parquet("past_starts_long_format.parquet", engine="pyarrow")
    )

# Main application

def main():
    st.set_page_config(page_title="Horse Handicapping Dashboard", layout="wide")
    st.sidebar.title("Filters")

    # Load data
    current_df, past_df = load_data()

    # Race filter
    races = sorted(current_df["race"].unique())
    selected_race = st.sidebar.selectbox("Select Race", races)

    # Filter for selected race
    race_current = current_df[current_df["race"] == selected_race]
    race_past    = past_df[past_df["race"] == selected_race]

    # your ID key for joining
    id_vars = ["track","race","post_position","horse_name"]

    # pick off just the two metric columns from the past‐starts data
    metrics_df = (
        race_past[id_vars + ["pp_avg_best2_bris_speed", "pp_combined_pace"]]
        .drop_duplicates(subset=id_vars)
    )

    # merge them into your current‐info DataFrame
    race_current = race_current.merge(
        metrics_df,
        on=id_vars,
        how="left"
    )

    st.title(f"Race {selected_race} Dashboard")

    # Race condition details
    if not race_current.empty:
        info = race_current.iloc[0]
        st.caption(f"Description: {info['race_conditions']}")
        st.caption(f"Conditions: {info['today_s_eqb_abbrev_race_conditions_character_17_17']}")
        st.caption(f"{info['furlongs']} {info['surface']}")
        st.caption(f"Purse: {info['purse']}")

    # Contender Summary
    st.subheader("Contender Summary")
    cs_cols = [
        "post_position", "horse_name", "morn_line_odds_if_available",
        "bris_prime_power_rating", "pp_avg_best2_bris_speed", "pp_combined_pace"
    ]
    cs = race_current[cs_cols].rename(columns={
        "post_position": "Post Position",
        "horse_name": "Horse",
        "morn_line_odds_if_available": "M/L Odds",
        "bris_prime_power_rating": "Prime Power",
        "pp_avg_best2_bris_speed": "Speed 2/3",
        "pp_combined_pace": "Combined"
    }).sort_values("Prime Power", ascending=False)
    st.dataframe(cs)

    # Connections
    st.subheader("Connections")
    conn_cols = [
        "program_number_if_available", "horse_name", "morn_line_odds_if_available",
        "today_s_owner", "auction_price", "breeder", "sire_stud_fee_current"
    ]
    conn = race_current[conn_cols].rename(columns={
        "program_number_if_available": "Number",
        "horse_name": "Name",
        "morn_line_odds_if_available": "M/L Odds",
        "today_s_owner": "Owner",
        "auction_price": "Auction Price",
        "breeder": "Breeder",
        "sire_stud_fee_current": "Stud Fee"
    }).sort_values("M/L Odds", ascending=True)
    st.dataframe(conn)

    # Connections (Cont'd)
    st.subheader("Connections (Cont'd)")
    conn2_cols = [
        "program_number_if_available", "horse_name", "morn_line_odds_if_available",
        "today_s_trainer", "today_s_jockey",
        "t_j_combo_starts_365d", "t_j_combo_wins_365d", "t_j_combo_2_roi_365d"
    ]
    conn2 = race_current[conn2_cols].rename(columns={
        "program_number_if_available": "Number",
        "horse_name": "Name",
        "morn_line_odds_if_available": "M/L Odds",
        "today_s_trainer": "Trainer",
        "today_s_jockey": "Jockey",
        "t_j_combo_starts_365d": "Starts",
        "t_j_combo_wins_365d": "Wins",
        "t_j_combo_2_roi_365d": "ROI"
    }).sort_values("M/L Odds", ascending=True)
    st.dataframe(conn2)

    # Pace Scenario
    st.subheader("Pace Scenario")
    # Prepare metrics from past data
    id_vars = ["track", "race", "post_position", "horse_name"]
    metrics = [
        "avg_best2_recent_pp_e1_pace", "avg_best2_recent_pp_turn_time",
        "avg_best2_recent_pp_e2_pace", "avg_best2_recent_pp_bris_late_pace",
        "avg_best2_recent_pp_combined_pace"
    ]
    # Extract one row per horse with metrics
    metrics_df = (
        race_past[id_vars + metrics]
        .drop_duplicates(subset=id_vars)
        .reset_index(drop=True)
    )
    ps = (
        race_current
        .merge(metrics_df, on=id_vars, how="left")
        [[
            "program_number_if_available", "horse_name", "morn_line_odds_if_available",
            "bris_run_style_designation", "quirin_style_speed_points",
            "avg_best2_recent_pp_e1_pace", "avg_best2_recent_pp_turn_time",
            "avg_best2_recent_pp_e2_pace", "avg_best2_recent_pp_bris_late_pace",
            "avg_best2_recent_pp_combined_pace"
        ]]
        .rename(columns={
            "program_number_if_available": "Number",
            "horse_name": "Name",
            "morn_line_odds_if_available": "M/L Odds",
            "bris_run_style_designation": "Run Style",
            "quirin_style_speed_points": "Speed Points",
            "avg_best2_recent_pp_e1_pace": "Avg E1",
            "avg_best2_recent_pp_turn_time": "Turn Time",
            "avg_best2_recent_pp_e2_pace": "Avg E2",
            "avg_best2_recent_pp_bris_late_pace": "Avg LP",
            "avg_best2_recent_pp_combined_pace": "Avg Combined"
        })
        .sort_values("M/L Odds", ascending=True)
    )
    st.dataframe(ps)

    # E2 Pace Ridgeline
    st.subheader("E2 Pace Ridgeline (Bokeh)")
    if not race_past.empty and "pp_e2_pace" in race_past.columns:
        import numpy as np
        from sklearn.neighbors import KernelDensity
        from bokeh.plotting import figure
        from bokeh.palettes import Viridis256

        # Prepare horses and KDE grid
        horses = race_past['horse_name'].unique().tolist()
        min_pace = race_past['pp_e2_pace'].min()
        max_pace = race_past['pp_e2_pace'].max()
        xs = np.linspace(min_pace, max_pace, 200)[:, None]

        # Create Bokeh figure
        p = figure(
            width=800,
            height=25 * len(horses),
            x_range=(20, 120),
            y_range=list(reversed(horses)),
            x_axis_label="E2 Pace (sec)",
            y_axis_label="Horse",
            title=f"Race {selected_race} – E2 Pace Ridgeline"
        )
        p.ygrid.grid_line_color = None
        p.xgrid.grid_line_alpha = 0.3
        p.yaxis.major_label_text_font_size = "10pt"

        # Plot one ridge per horse
        for i, name in enumerate(horses):
            data = race_past.loc[race_past['horse_name'] == name, 'pp_e2_pace'].dropna().values[:, None]
            if len(data) < 2:
                continue
            kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(data)
            dens = np.exp(kde.score_samples(xs))
            dens = dens / dens.max() * 0.8  # scale to fit row height

            x_coords = np.concatenate([xs[:,0], xs[::-1,0]])
            y_coords = np.concatenate([i + dens, i - dens[::-1]])

            # Color by median pace
            med = np.median(data)
            idx = int((med - min_pace) / (max_pace - min_pace) * (len(Viridis256)-1))
            color = Viridis256[idx]

            p.patch(x_coords, y_coords,
                    fill_color=color, fill_alpha=0.6,
                    line_color="black", line_width=0.5)

        st.bokeh_chart(p, use_container_width=True)

if __name__ == "__main__":
    main()
