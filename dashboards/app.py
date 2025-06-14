# dashboards/app_dash.py
"""
Comprehensive Horse Racing Analytics Dashboard
Visualizes all advanced handicapping metrics using Plotly Dash
"""

import dash
from dash import dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config.settings import PROCESSED_DATA_DIR, CURRENT_RACE_INFO, PAST_STARTS_LONG
from horse_racing.transformers.simple_pace import PaceAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.DARKLY])
app.title = "Horse Racing Analytics Dashboard"

# Define color schemes
COLORS = {
    'background': '#1e1e1e',
    'card_bg': '#2d2d2d',
    'text': '#ffffff',
    'primary': '#00d4ff',
    'secondary': '#ff6b6b',
    'success': '#51cf66',
    'warning': '#ffd93d',
    'danger': '#ff6b6b',
    'info': '#339af0'
}

# Form cycle state colors
FORM_STATE_COLORS = {
    'PEAKING': '#51cf66',
    'IMPROVING': '#74c0fc',
    'STABLE': '#868e96',
    'FRESHENING': '#ffd43b',
    'RECOVERING': '#fab005',
    'DECLINING': '#ff8787',
    'BOUNCING': '#ff6b6b',
    'ERRATIC': '#e599f7'
}


# Load all data
def load_all_data():
    """Load all analytical data files"""
    base_path = PROCESSED_DATA_DIR

    data = {}

    # Core data
    if CURRENT_RACE_INFO.exists():
        data['current'] = pd.read_parquet(CURRENT_RACE_INFO)
        logger.info("Loaded current race info")
    else:
        logger.warning("Current race info file not found")
        data['current'] = pd.DataFrame()

    if PAST_STARTS_LONG.exists():
        data['past'] = pd.read_parquet(PAST_STARTS_LONG)
        logger.info("Loaded past starts data")
    else:
        logger.warning("Past starts file not found")
        data['past'] = pd.DataFrame()

    # Component analyses
    files_to_load = {
        'fitness': 'advanced_fitness_metrics.parquet',
        'workout': 'sophisticated_workout_analysis.parquet',
        'pace': 'advanced_pace_analysis.parquet',
        'class': 'multi_dimensional_class_assessment.parquet',
        'form': 'form_cycle_analysis.parquet',
        'integrated': 'integrated_analytics_report.parquet'
    }

    for name, filename in files_to_load.items():
        filepath = base_path / filename
        if filepath.exists():
            data[name] = pd.read_parquet(filepath)
            logger.info(f"Loaded {name} data")
        else:
            logger.warning(f"{name} data not found")
            data[name] = pd.DataFrame()

    return data


# Load data on startup
DATA = load_all_data()

# Initialize simple pace analyzer for additional dashboard tab
analyzer = PaceAnalyzer()
analyzer.load_data()
analyzer.validate_columns()


# Layout for the Simple Pace tab
def get_simple_pace_layout():
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Simple Pace Analysis", className="text-center mb-4"),
                    html.Hr()
                ], width=12)
            ]),

            html.Hr(),

            dbc.Tabs([
                dbc.Tab(label="Pace Results", tab_id="results-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H3("Pace Analysis Results", className="mb-3"),
                            html.Div(id="pace-table-container")
                        ], width=12)
                    ]),

                    html.Hr()
                ]),

                dbc.Tab(label="Pace Visualizations", tab_id="viz-tab", children=[
                    dbc.Row([
                        dbc.Col([
                            html.H3("Visualizations", className="mb-3"),
                            dcc.Graph(id="pace-visualization")
                        ], width=12)
                    ])
                ])
            ], id="main-tabs", active_tab="results-tab"),

            html.Hr(),

            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.H5("Debug: If you can see this, scrolling works!", className="text-warning"),
                    html.P("This is a test section at the bottom of the page.", className="text-muted"),
                    html.Div(style={'height': '200px', 'backgroundColor': 'rgba(255,255,255,0.1)',
                                    'border': '1px dashed white'})
                ], width=12)
            ])

        ], fluid=True)
    ], style={
        'minHeight': '100vh',
        'paddingBottom': '100px',
        'overflow': 'visible',
        'position': 'relative'
    })


# Layout for the Pace Clustering tab
def get_clustering_layout():
    return html.Div([
        dbc.Container([
            html.H1("Pace Clustering", className="text-center mb-4"),
            html.Div(id='clustering-content')
        ], fluid=True)
    ])

# Get unique races for dropdown
if not DATA['current'].empty and 'race' in DATA['current']:
    RACE_OPTIONS = [
        {'label': f'Race {r}', 'value': r}
        for r in sorted(DATA['current']['race'].dropna().unique())
    ]
else:
    RACE_OPTIONS = []

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("Horse Racing Analytics Dashboard",
                style={'textAlign': 'center', 'color': COLORS['primary']}),
        html.Hr(style={'borderColor': COLORS['primary']})
    ], style={'padding': '20px', 'backgroundColor': COLORS['card_bg']}),

    # Race Selector and Scratch Selector
    html.Div([
        html.Div([
            html.Label("Select Race:", style={'color': COLORS['text'], 'marginRight': '10px'}),
            dcc.Dropdown(
                id='race-selector',
                options=RACE_OPTIONS,
                value=RACE_OPTIONS[0]['value'] if RACE_OPTIONS else None,
                className="dropdown-options",
                style={
                    'width': '200px',
                    'display': 'inline-block',
                    'backgroundColor': COLORS['card_bg'],
                    'color': COLORS['text']
                }
            )
        ], style={'display': 'inline-block', 'marginRight': '40px'}),

        html.Div([
            html.Label("Scratch Horses:", style={'color': COLORS['text'], 'marginRight': '10px'}),
            dcc.Dropdown(
                id='scratch-selector',
                options=[],
                value=[],
                multi=True,
                className="dropdown-options",
                style={
                    'width': '200px',
                    'display': 'inline-block',
                    'backgroundColor': COLORS['card_bg'],
                    'color': COLORS['text']
                }
            )
        ], style={'display': 'inline-block'})
    ], style={'padding': '20px', 'backgroundColor': COLORS['background']}),

    # Race Header Info
    html.Div(id='race-header', style={'padding': '20px', 'backgroundColor': COLORS['card_bg']}),

    # Main Content Tabs
    dcc.Tabs([
        dcc.Tab(
            label='Race Summary',
            children=[
                html.Div(id='race-summary-content', style={'padding': '20px'})
            ],
            style={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text']},
            selected_style={
                'backgroundColor': COLORS['primary'],
                'color': '#000000',
                'fontWeight': 'bold'
            }
        ),
        dcc.Tab(
            label='Simple Pace',
            children=[
                html.Div(
                    id='simple-pace-content',
                    children=get_simple_pace_layout(),
                    style={'padding': '20px'}
                )
            ],
            style={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text']},
            selected_style={
                'backgroundColor': COLORS['primary'],
                'color': '#000000',
                'fontWeight': 'bold'
            }
        ),
        dcc.Tab(
            label='Workout Analysis',
            children=[
                html.Div(id='workout-content', style={'padding': '20px'})
            ],
            style={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text']},
            selected_style={
                'backgroundColor': COLORS['primary'],
                'color': '#000000',
                'fontWeight': 'bold'
            }
        ),
        dcc.Tab(
            label='Pace Clustering',
            children=[
                html.Div(id='pace-clustering-content', children=get_clustering_layout(), style={'padding': '20px'})
            ],
            style={'backgroundColor': COLORS['card_bg'], 'color': COLORS['text']},
            selected_style={
                'backgroundColor': COLORS['primary'],
                'color': '#000000',
                'fontWeight': 'bold'
            }
        )
    ], style={'backgroundColor': COLORS['background']})

], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh'})


# Callbacks
@app.callback(
    Output('race-header', 'children'),
    Input('race-selector', 'value')
)
def update_race_header(selected_race):
    """Update race header information"""
    if not selected_race:
        return html.Div("No race selected")

    race_data = DATA['current'][DATA['current']['race'] == selected_race].iloc[0]

    # Extract race details
    distance = race_data.get('distance_in_yards', 0)
    furlongs = distance / 220 if distance > 0 else 0
    surface = race_data.get('surface', 'D')
    surface_text = 'Dirt' if surface == 'D' else 'Turf' if surface == 'T' else surface

    race_type = race_data.get('race_type', '')
    classification = race_data.get('today_s_race_classification', '')
    purse = race_data.get('purse', 0)

    field_size = race_data.get('field_size', None)

    stats_bar = html.Div([
        html.Span(f"Horses: {field_size}", style={'marginRight': '20px'}),
        html.Span(f"Distance: {furlongs:.1f}f", style={'marginRight': '20px'}),
        html.Span(f"Surface: {surface_text}", style={'marginRight': '20px'}),
        html.Span(f"Purse: ${purse:,}")
    ], style={'color': COLORS['text'], 'fontSize': '18px'})

    header = html.Div([
        html.H2(f"Race {selected_race}", style={'color': COLORS['primary']}),
        stats_bar,
        html.P(f"Classification: {classification} | Type: {race_type}",
               style={'color': COLORS['text'], 'fontSize': '16px', 'marginTop': '5px'})
    ])

    return header


@app.callback(
    [Output('scratch-selector', 'options'),
     Output('scratch-selector', 'value')],
    Input('race-selector', 'value')
)
def update_scratch_selector(selected_race):
    """Update scratch selector options based on selected race"""
    if not selected_race or DATA['current'].empty:
        return [], []

    horses = DATA['current'][DATA['current']['race'] == selected_race]['horse_name'].tolist()
    options = [{'label': h, 'value': h} for h in horses]
    return options, []


@app.callback(
    Output('race-summary-content', 'children'),
    [Input('race-selector', 'value'),
     Input('scratch-selector', 'value')]
)
def update_race_summary(selected_race, scratched_horses):
    """Update race summary and connections tables"""
    if not selected_race:
        return html.Div("No race selected")

    scratches = scratched_horses or []

    # Get race data for all horses in the selected race excluding scratches
    race_horses = DATA['current'][
        (DATA['current']['race'] == selected_race) &
        (~DATA['current']['horse_name'].isin(scratches))
    ].copy()

    # --- Create Race Summary Table ---

    # Merge with past performance data for averages
    past_data = DATA['past'][
        (DATA['past']['race'] == selected_race) &
        (~DATA['past']['horse_name'].isin(scratches))
    ].copy()

    # Calculate average pace figures by horse
    pace_avgs = past_data.groupby('horse_name').agg({
        'pp_e1_pace': ['mean', 'max'],
        'pp_e2_pace': ['mean', 'max'],
        'pp_bris_late_pace': ['mean', 'max'],
        'pp_bris_speed_rating': ['mean', 'max']
    }).reset_index()

    # Flatten column names
    pace_avgs.columns = ['horse_name', 'avg_e1', 'best_e1', 'avg_e2', 'best_e2',
                         'avg_lp', 'best_lp', 'avg_speed', 'best_speed']

    # Merge with current data
    summary_df = race_horses.merge(pace_avgs, on='horse_name', how='left')

    # Select and round columns for display
    summary_display_columns = [
        'post_position', 'horse_name', 'morn_line_odds_if_available',
        'of_days_since_last_race', 'bris_prime_power_rating',
        'avg_e1', 'avg_e2', 'avg_lp', 'avg_speed',
        'best_e1', 'best_e2', 'best_lp', 'best_speed'
    ]
    numeric_cols = [
        'avg_e1', 'avg_e2', 'avg_lp', 'avg_speed',
        'best_e1', 'best_e2', 'best_lp', 'best_speed'
    ]
    for col in numeric_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(1)

    summary_df = summary_df.sort_values('post_position')

    # Prepare styles for heatmap
    style_data_conditional = [
        {'if': {'row_index': 'odd'}, 'backgroundColor': COLORS['background']}
    ]
    heatmap_cols = [
        'bris_prime_power_rating', 'avg_e1', 'avg_e2', 'avg_lp', 'avg_speed',
        'best_e1', 'best_e2', 'best_lp', 'best_speed'
    ]

    for col in heatmap_cols:
        if col in summary_df.columns:
            col_data = summary_df[col].dropna()
            if col_data.empty or col_data.min() == col_data.max():
                continue
            min_val, max_val = col_data.min(), col_data.max()
            for i in range(10):
                lower_bound = min_val + (i / 10.0) * (max_val - min_val)
                hue = (i / 9.0) * 120
                color = f'hsl({hue}, 80%, 35%)'
                style_data_conditional.append({
                    'if': {'filter_query': f'{{{col}}} >= {lower_bound}', 'column_id': col},
                    'backgroundColor': color, 'color': 'white'
                })

    summary_table = dash_table.DataTable(
        data=summary_df[summary_display_columns].to_dict('records'),
        columns=[
            {'name': 'Post', 'id': 'post_position'},
            {'name': 'Horse', 'id': 'horse_name'},
            {'name': 'ML Odds', 'id': 'morn_line_odds_if_available'},
            {'name': 'Days Off', 'id': 'of_days_since_last_race'},
            {'name': 'Prime Power', 'id': 'bris_prime_power_rating'},
            {'name': 'Avg E1', 'id': 'avg_e1'},
            {'name': 'Avg E2', 'id': 'avg_e2'},
            {'name': 'Avg LP', 'id': 'avg_lp'},
            {'name': 'Avg Speed', 'id': 'avg_speed'},
            {'name': 'Best E1', 'id': 'best_e1'},
            {'name': 'Best E2', 'id': 'best_e2'},
            {'name': 'Best LP', 'id': 'best_lp'},
            {'name': 'Best Speed', 'id': 'best_speed'}
        ],
        sort_action="native",
        style_cell={'textAlign': 'center', 'backgroundColor': COLORS['card_bg'], 'color': COLORS['text']},
        style_header={'backgroundColor': COLORS['primary'], 'color': COLORS['background'], 'fontWeight': 'bold'},
        style_data_conditional=style_data_conditional
    )

    # --- Create Connections Table ---
    connections_cols_map = {
        'post_position': 'PP',
        'horse_name': 'Horse',
        'morn_line_odds_if_available': 'ML Odds',
        'main_track_only_ae_indicator': 'MTO/AE',
        'today_s_trainer': 'Trainer',
        'today_s_jockey': 'Jockey',
        'today_s_owner': 'Owner',
        'auction_price': 'Auction Price',
        'breeder': 'Breeder',
        'stat_country_abrv_where_bred': 'Bred',
        'sire_stud_fee_current': 'Sire Fee',
        'sire': 'Sire',
        'sire_s_sire': "Sire's Sire",
        'dam_s_sire': "Dam's Sire"
    }

    # Use only columns that exist in the dataframe
    connections_display_cols = [col for col in connections_cols_map.keys() if col in race_horses.columns]
    connections_df = race_horses[connections_display_cols].sort_values('post_position')

    connections_table = dash_table.DataTable(
        data=connections_df.to_dict('records'),
        columns=[{'name': v, 'id': k} for k, v in connections_cols_map.items() if k in connections_display_cols],
        sort_action="native",
        style_cell={
            'textAlign': 'left',
            'backgroundColor': COLORS['card_bg'],
            'color': COLORS['text'],
            'padding': '8px',
            'whiteSpace': 'normal',
            'height': 'auto',
            'minWidth': '100px'
        },
        style_header={
            'backgroundColor': COLORS['primary'],
            'color': COLORS['background'],
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': COLORS['background']}
        ],
        style_table={'overflowX': 'auto'}
    )

    # --- Combine Tables into a single Div ---
    return html.Div([
        html.H3("Race Summary", style={'color': COLORS['primary']}),
        summary_table,
        html.Br(),
        html.H3("Connections", style={'color': COLORS['primary'], 'marginTop': '40px'}),
        connections_table
    ])


@app.callback(
    Output('workout-content', 'children'),
    [Input('race-selector', 'value'),
     Input('scratch-selector', 'value')]
)
def update_workout_view(selected_race, scratched_horses):
    """Update workout analysis view"""
    if not selected_race or DATA['workout'].empty:
        return html.Div("No workout data available")

    scratches = scratched_horses or []

    race_data = DATA['workout'][
        (DATA['workout']['race'] == selected_race) &
        (~DATA['workout']['horse_name'].isin(scratches))
    ].copy()

    charts = []

    # 1. Workout Quality Scores
    fig_quality = go.Figure()

    race_data_sorted = race_data.sort_values('workout_readiness_score', ascending=False)

    # Color by readiness category
    colors = []
    for score in race_data_sorted['workout_readiness_score']:
        if score >= 80:
            colors.append(COLORS['success'])
        elif score >= 60:
            colors.append(COLORS['info'])
        elif score >= 40:
            colors.append(COLORS['warning'])
        else:
            colors.append(COLORS['danger'])

    fig_quality.add_trace(go.Bar(
        x=race_data_sorted['horse_name'],
        y=race_data_sorted['workout_readiness_score'],
        marker_color=colors,
        text=race_data_sorted['workout_readiness_category'],
        textposition='outside'
    ))

    fig_quality.update_layout(
        title="Workout Readiness Scores",
        xaxis_title="Horse",
        yaxis_title="Readiness Score",
        template='plotly_dark',
        height=400
    )

    charts.append(dcc.Graph(figure=fig_quality))

    # 2. Workout Pattern Analysis
    if all(col in race_data.columns for col in ['bullet_work_count', 'works_last_14d', 'pct_fast_works']):
        fig_patterns = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Bullet Works', 'Recent Activity', 'Fast Work %']
        )

        # Bullet works
        fig_patterns.add_trace(
            go.Bar(
                x=race_data['horse_name'],
                y=race_data['bullet_work_count'],
                marker_color=COLORS['primary']
            ),
            row=1, col=1
        )

        # Recent works
        fig_patterns.add_trace(
            go.Bar(
                x=race_data['horse_name'],
                y=race_data['works_last_14d'],
                marker_color=COLORS['secondary']
            ),
            row=1, col=2
        )

        # Fast work percentage
        fig_patterns.add_trace(
            go.Bar(
                x=race_data['horse_name'],
                y=race_data['pct_fast_works'],
                marker_color=COLORS['success']
            ),
            row=1, col=3
        )

        fig_patterns.update_layout(
            title="Workout Pattern Analysis",
            template='plotly_dark',
            height=400,
            showlegend=False
        )

        charts.append(dcc.Graph(figure=fig_patterns))

    return html.Div(charts)


# ------- Simple Pace Callbacks -------

@app.callback(
    [Output('pace-table-container', 'children'),
     Output('pace-visualization', 'figure')],
    [Input('race-selector', 'value'),
     Input('scratch-selector', 'value')],
    prevent_initial_call=False
)
def update_display(selected_race, scratched_horses):
    """Update table and visualization based on selected race"""
    if selected_race is None or analyzer.current_race_df is None:
        return html.Div("No race selected"), go.Figure()

    # Calculate pace metrics
    results_df = analyzer.calculate_pace_metrics()

    # Filter for selected race
    race_df = results_df[results_df['race'] == selected_race].copy()

    scratches = scratched_horses or []
    if scratches:
        race_df = race_df[~race_df['horse_name'].isin(scratches)]

    if race_df.empty:
        return html.Div("No data for selected race"), go.Figure()

    # Round numeric columns for display
    numeric_cols = ['e1_consistency', 'lp_consistency', 'average_e1', 'first_call_mean',
                    'average_lp', 'pace_vol', 'e1_trend', 'lp_trend',
                    'latest_energy_dist', 'latest_accel_pct', 'latest_finish_pct']

    for col in numeric_cols:
        if col in race_df.columns:
            race_df[col] = race_df[col].round(2)

    # Create display dataframe with key columns
    display_columns = [
        'post_position', 'horse_name', 'bris_run_style_designation',
        'quirin_style_speed_points', 'e1_consistency', 'lp_consistency',
        'average_e1', 'first_call_mean', 'average_lp', 'lead_abundance_flag',
        'pace_vol', 'e1_trend', 'lp_trend', 'latest_energy_dist',
        'latest_accel_pct', 'latest_finish_pct'
    ]

    display_columns = [col for col in display_columns if col in race_df.columns]

    display_df = race_df[display_columns].copy()

    column_rename = {
        'post_position': 'PP',
        'horse_name': 'Horse',
        'bris_run_style_designation': 'Style',
        'quirin_style_speed_points': 'SP',
        'e1_consistency': 'E1 SD',
        'lp_consistency': 'LP SD',
        'average_e1': 'Avg E1',
        'first_call_mean': 'Top3 E1',
        'average_lp': 'Avg LP',
        'lead_abundance_flag': 'Lead Count',
        'pace_vol': 'Pace Vol',
        'e1_trend': 'E1 Trend',
        'lp_trend': 'LP Trend',
        'latest_energy_dist': 'Energy Dist',
        'latest_accel_pct': 'Accel %',
        'latest_finish_pct': 'Finish %'
    }

    display_df = display_df.rename(columns=column_rename)

    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=[{"name": col, "id": col} for col in display_df.columns],
        style_cell={
            'textAlign': 'left',
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            'fontSize': '12px',
            'padding': '5px'
        },
        style_header={
            'backgroundColor': 'rgb(50, 50, 50)',
            'fontWeight': 'bold',
            'fontSize': '13px'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(40, 40, 40)'
            },
            {
                'if': {
                    'filter_query': '{Style} = "E" || {Style} = "E/P"',
                    'column_id': 'Style'
                },
                'backgroundColor': 'rgb(70, 30, 30)',
                'color': 'white',
            },
            {
                'if': {
                    'filter_query': '{SP} >= 6',
                    'column_id': 'SP'
                },
                'color': 'rgb(255, 200, 100)',
                'fontWeight': 'bold'
            },
            {
                'if': {
                    'filter_query': '{E1 Trend} < 0',
                    'column_id': 'E1 Trend'
                },
                'color': 'rgb(100, 255, 100)'
            },
            {
                'if': {
                    'filter_query': '{E1 SD} < 2',
                    'column_id': 'E1 SD'
                },
                'color': 'rgb(100, 255, 100)'
            }
        ],
        sort_action="native",
        page_size=20
    )

    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=(
            'E1 Pace Analysis', 'Late Pace Analysis',
            'Energy Distribution Profile', 'Pace Trends'
        ),
        vertical_spacing=0.1
    )

    race_df = race_df.sort_values('post_position')

    fig.add_trace(
        go.Bar(
            x=race_df['post_position'],
            y=race_df['average_e1'],
            name='Avg E1',
            text=race_df['horse_name'],
            textposition='outside',
            marker_color='lightblue'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=race_df['post_position'],
            y=race_df['e1_consistency'],
            name='E1 Consistency',
            mode='lines+markers',
            line=dict(color='red', width=2),
            yaxis='y2'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=race_df['post_position'],
            y=race_df['average_lp'],
            name='Avg LP',
            text=race_df['horse_name'],
            textposition='outside',
            marker_color='lightgreen'
        ),
        row=2, col=1
    )

    if 'latest_energy_dist' in race_df.columns:
        energy_dist_size = race_df['latest_energy_dist'].fillna(0) * 20
        energy_dist_size = energy_dist_size.clip(lower=5)

        fig.add_trace(
            go.Scatter(
                x=race_df['latest_accel_pct'].fillna(0),
                y=race_df['latest_finish_pct'].fillna(0),
                mode='markers+text',
                text=race_df['horse_name'],
                textposition='top center',
                marker=dict(
                    size=energy_dist_size,
                    color=race_df['quirin_style_speed_points'].fillna(0),
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Speed Points")
                ),
                name='Energy Profile'
            ),
            row=3, col=1
        )

    if 'e1_trend' in race_df.columns and 'lp_trend' in race_df.columns:
        valid_trends = race_df[['e1_trend', 'lp_trend', 'horse_name', 'post_position']].dropna()

        if len(valid_trends) > 0:
            fig.add_trace(
                go.Scatter(
                    x=valid_trends['e1_trend'],
                    y=valid_trends['lp_trend'],
                    mode='markers+text',
                    text=valid_trends['horse_name'],
                    textposition='top center',
                    marker=dict(
                        size=12,
                        color=valid_trends['post_position'],
                        colorscale='Rainbow'
                    ),
                    name='Pace Trends'
                ),
                row=4, col=1
            )

    fig.update_layout(
        title=f"Race {selected_race} - Comprehensive Pace Analysis",
        template="plotly_dark",
        height=1600,
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    fig.update_xaxes(title_text="Post Position", row=1, col=1)
    fig.update_xaxes(title_text="Post Position", row=2, col=1)
    fig.update_xaxes(title_text="Acceleration %", row=3, col=1)
    fig.update_xaxes(title_text="E1 Trend", row=4, col=1)

    fig.update_yaxes(title_text="E1 Pace", row=1, col=1)
    fig.update_yaxes(title_text="Late Pace", row=2, col=1)
    fig.update_yaxes(title_text="Finish %", row=3, col=1)
    fig.update_yaxes(title_text="LP Trend", row=4, col=1)

    lead_count = race_df['lead_abundance_flag'].iloc[0] if 'lead_abundance_flag' in race_df.columns else 0
    pace_vol = race_df['pace_vol'].iloc[0] if 'pace_vol' in race_df.columns else 'N/A'

    summary_div = html.Div([
        html.H5("Race Summary:", className="mt-3"),
        html.P(f"Lead Abundance: {lead_count} horses with E/E-P style and 6+ speed points"),
        html.P(f"Pace Volatility: {pace_vol}")
    ])

    return [summary_div, table], fig


# ------- Pace Clustering Callback -------

@app.callback(
    Output('clustering-content', 'children'),
    [Input('race-selector', 'value'),
     Input('scratch-selector', 'value')]
)
def update_clustering_view(selected_race, scratched_horses):
    """Create clustering visualizations for the selected race."""
    if not selected_race or DATA['past'].empty:
        return html.Div("No data available")

    scratches = scratched_horses or []

    race_df = DATA['past'][
        (DATA['past']['race'] == selected_race) &
        (~DATA['past']['horse_name'].isin(scratches))
    ].copy()

    features = ['pp_e1_pace', 'pp_e2_pace', 'pp_turn_time', 'pp_bris_late_pace']
    features = [f for f in features if f in race_df.columns]

    X = race_df[features].dropna()
    if len(X) < 3:
        return html.Div("Not enough data for clustering")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    df_cluster = X.copy()
    df_cluster['cluster'] = labels
    df_cluster['horse_name'] = race_df.loc[X.index, 'horse_name']

    pair_fig = px.scatter_matrix(
        df_cluster,
        dimensions=features,
        color='cluster',
        hover_name='horse_name',
        title='Pair Plot - Pace Features'
    )
    pair_fig.update_layout(template='plotly_dark', height=600)

    pca = PCA(n_components=2)
    comps = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame({
        'PC1': comps[:, 0],
        'PC2': comps[:, 1],
        'cluster': labels,
        'horse_name': race_df.loc[X.index, 'horse_name']
    })

    pca_fig = px.scatter(
        pca_df, x='PC1', y='PC2', color='cluster',
        hover_name='horse_name',
        title='PCA Cluster Visualization'
    )
    pca_fig.update_layout(template='plotly_dark', height=600)

    cluster_means = df_cluster.groupby('cluster')[features].mean()
    radar_fig = go.Figure()
    for cluster_id, row in cluster_means.iterrows():
        radar_fig.add_trace(go.Scatterpolar(
            r=row.values,
            theta=features,
            fill='toself',
            name=f'Cluster {cluster_id}'
        ))
    radar_fig.update_layout(
        template='plotly_dark',
        polar=dict(radialaxis=dict(visible=True)),
        title='Cluster Profiles',
        height=600
    )

    par_fig = px.parallel_coordinates(
        df_cluster,
        dimensions=features,
        color='cluster',
        color_continuous_scale=px.colors.qualitative.Dark24,
        title='Parallel Coordinates'
    )
    par_fig.update_layout(template='plotly_dark', height=600)

    return html.Div([
        dcc.Graph(figure=pair_fig),
        dcc.Graph(figure=pca_fig),
        dcc.Graph(figure=radar_fig),
        dcc.Graph(figure=par_fig)
    ])


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
