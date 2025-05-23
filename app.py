import pandas as pd
import dash
from dash import dash_table, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Load data
df = pd.read_parquet("parsed_race_data_full.parquet")

# Column mappings
col_map = {
    "post_position": "Post",
    "horse_name": "Horse",
    "morn_line_odds_if_available": "M/L",
    "bris_run_style_designation": "Style",
    "quirin_style_speed_points": "Points",
    "of_days_since_last_race": "Layoff (Days)",
    "bris_prime_power_rating": "Prime Power",
    "speed_rating_1": "Speed 1",
    "speed_rating_2": "Speed 2",
    "speed_rating_3": "Speed 3",
    "speed_rating_4": "Speed 4",
    "speed_rating_5": "Speed 5"
}

display_cols = list(col_map.keys())

# Default text for missing columns
for col in display_cols:
    if col not in df.columns:
        df[col] = None

# Ensure race column is int for sorting and selection
df['race'] = df['race'].astype(int)

# Race selection options
race_options = [
    {"label": f"Race {r}", "value": r}
    for r in sorted(df["race"].unique())
]

# Dash app layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H2(id="race-title", style={"margin-top": "20px"}),
        html.Div(id="race-meta", style={"font-size": "1rem", "margin-bottom": "18px", "color": "gray"}),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id="race-select",
                    options=race_options,
                    value=race_options[0]['value'],
                    clearable=False,
                    style={"width": "200px"}
                ),
                width="auto",
            ),
        ], justify="start", style={"margin-bottom": "18px"}),
        dash_table.DataTable(
            id="race-table",
            columns=[{"name": col_map[c], "id": c, "presentation": "markdown"} for c in display_cols],
            style_cell={'textAlign': 'left', 'padding': '4px', "font-family": "monospace", "font-size": "1rem"},
            style_header={'fontWeight': 'bold', 'backgroundColor': '#f9f9f9', "font-size": "1rem"},
            style_table={"overflowX": "auto"},
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
            fill_width=False,
        )
    ],
    fluid=True
)

@app.callback(
    Output("race-title", "children"),
    Output("race-meta", "children"),
    Output("race-table", "data"),
    Input("race-select", "value"),
)
def update_dashboard(selected_race):
    # Filter for this race
    race_df = df[df["race"] == selected_race]
    if race_df.empty:
        return "Race", "No data", []

    # Title & subheadings (get meta for this race)
    first_row = race_df.iloc[0]
    meta_lines = []
    if "race_conditions" in race_df:
        meta_lines.append(f"<b>race_conditions:</b> {first_row.get('race_conditions', '')}")
    if "today_s_race_classification" in race_df and "purse" in race_df:
        meta_lines.append(f"<b>today_s_race_classification:</b> {first_row.get('today_s_race_classification', '')} &nbsp; <b>purse:</b> {first_row.get('purse', '')}")
    if "distance_in_yards" in race_df and "surface" in race_df:
        meta_lines.append(f"<b>distance_in_yards:</b> {first_row.get('distance_in_yards', '')} &nbsp; <b>surface:</b> {first_row.get('surface', '')}")
    meta_html = "<br>".join(meta_lines)

    # Race title (e.g., "Race 2")
    title = f"Race {selected_race}"

    # Table data (left justified by default in Dash)
    data = race_df[display_cols].to_dict("records")

    return title, meta_html, data

if __name__ == "__main__":
    app.run(debug=True)
