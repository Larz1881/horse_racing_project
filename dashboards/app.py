import pandas as pd
import dash
from dash import dash_table, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.analysis.race_views import RaceDataViews # Assuming race_views.py is in src/analysis/

# Initialize the data views
# Ensure your parquet files are correctly referenced by RaceDataViews
# For example, if CURRENT_RACE_INFO and PAST_STARTS_LONG are defined in config.settings
# and they point to the correct paths, this should work.
# Otherwise, you might need to pass paths explicitly:
# race_views = RaceDataViews(current_path=Path('path/to/current.parquet'), past_path=Path('path/to/past.parquet'))
race_views = RaceDataViews()

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Check if there are races to display
initial_race_numbers = race_views.get_all_race_numbers()
initial_value = initial_race_numbers[0] if initial_race_numbers else None

# Layout
app.layout = dbc.Container([
    html.H2(id="race-title", style={"margin-top": "20px"}),
    html.Div(id="race-meta", style={"font-size": "1rem", "margin-bottom": "18px", "color": "gray"}),

    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id="race-select",
                options=[{"label": f"Race {r}", "value": r}
                        for r in initial_race_numbers],
                value=initial_value, # Use the first race as default or None
                clearable=False,
                style={"width": "200px"},
                disabled=not initial_race_numbers # Disable if no races
            ),
            width="auto",
        ),
    ], justify="start", style={"margin-bottom": "18px"}),

    html.Div(id="race-tables")
])

@app.callback(
    [Output("race-title", "children"),
     Output("race-meta", "children"),
     Output("race-tables", "children")],
    Input("race-select", "value")
)
def update_dashboard(selected_race):
    if selected_race is None: # Handle case where no race is selected or available
        return "No Race Data Available", "", ""

    # Get race metadata
    meta = race_views.get_race_metadata(selected_race)

    # Title
    title = f"Race {selected_race}"

    # Meta information
    meta_lines = []
    if meta.get('conditions'):
        meta_lines.append(f"<b>Conditions:</b> {meta['conditions']}")
    if meta.get('classification') and meta.get('purse'):
        purse_value = meta['purse']
        try:
            purse_formatted = f"${int(purse_value):,}" if purse_value else "N/A"
        except ValueError:
            purse_formatted = str(purse_value) # Keep as string if not convertible to int
        meta_lines.append(f"<b>Class:</b> {meta['classification']} &nbsp; <b>Purse:</b> {purse_formatted}")
    if meta.get('distance') and meta.get('surface'):
        meta_lines.append(f"<b>Distance:</b> {meta['distance']} yards &nbsp; <b>Surface:</b> {meta['surface']}")
    meta_html = dcc.Markdown("<br>".join(meta_lines), dangerously_allow_html=True)

    # Tables
    tables_content = []

    # Contender Summary
    contender_df = race_views.get_contender_summary(selected_race)
    if not contender_df.empty:
        tables_content.append(html.H4("Contender Summary"))
        tables_content.append(create_data_table(contender_df, "contender-table"))
        tables_content.append(html.Br())

    # Connections
    connections_df = race_views.get_connections_view(selected_race)
    if not connections_df.empty:
        tables_content.append(html.H4("Connections"))
        tables_content.append(create_data_table(connections_df, "connections-table"))
        tables_content.append(html.Br())

    # Trainer/Jockey
    tj_df = race_views.get_trainer_jockey_view(selected_race)
    if not tj_df.empty:
        tables_content.append(html.H4("Trainer/Jockey Statistics"))
        tables_content.append(create_data_table(tj_df, "tj-table"))
        tables_content.append(html.Br())

    # Pace Scenario
    pace_df = race_views.get_pace_scenario(selected_race)
    if not pace_df.empty:
        tables_content.append(html.H4("Pace Scenario"))
        tables_content.append(create_data_table(pace_df, "pace-table"))

    return title, meta_html, html.Div(tables_content)

def create_data_table(df, table_id):
    """Create a formatted Dash DataTable."""
    # Convert all data to string to avoid issues with mixed types if any, though usually not needed
    # df = df.astype(str)
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": i, "id": i, "presentation": "markdown"} for i in df.columns],
        data=df.to_dict("records"),
        style_cell={
            'textAlign': 'left',
            'padding': '4px',
            'font-family': 'monospace', # Consider using a more standard web font like 'Arial' or 'sans-serif'
            'font-size': '0.9rem' # Adjusted for better readability
        },
        style_header={
            'fontWeight': 'bold',
            'backgroundColor': '#f9f9f9',
            'font-size': '0.9rem' # Adjusted for better readability
        },
        style_table={'overflowX': 'auto', "margin-bottom": "20px"}, # Added margin-bottom
        style_data={'whiteSpace': 'normal', 'height': 'auto'},
        fill_width=False
    )

if __name__ == "__main__":
    app.run(debug=True)
