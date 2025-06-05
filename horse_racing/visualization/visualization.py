import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Viridis256
from bokeh.transform import linear_cmap

# 1) Load the data
df = pd.read_parquet("past_starts_long_format.parquet", engine="pyarrow")

# 2) Get all races
races = sorted(df['race'].dropna().unique())

# 3) Set up one big HTML (alternatively, loop output_file per race)
output_file("ridge_e2_pace_all_races.html")

# 4) Loop over each race
for race in races:
    sub = df[df['race'] == race]
    # Pick horses with at least one past-start (or threshold)
    horses = sub['horse_name'].unique().tolist()
    if not horses:
        continue

    # Create a figure sized to the number of contenders
    p = figure(
        width=800, 
        height=25 * len(horses),
        title=f"Race {race}: Ridgeline of E2 Pace by Horse",
        x_axis_label="E2 Pace (sec)",
        y_range=list(reversed(horses)),
        toolbar_location=None
    )
    
    # Precompute shared grid
    xs = np.linspace(sub['pp_e2_pace'].min(), sub['pp_e2_pace'].max(), 200)[:, None]
    
    for i, name in enumerate(horses):
        d = sub.loc[sub['horse_name'] == name, 'pp_e2_pace'].dropna().values[:, None]
        if len(d) < 2:
            continue
        
        kde = KernelDensity(kernel='gaussian', bandwidth=1.0).fit(d)
        dens = np.exp(kde.score_samples(xs))
        dens = dens / dens.max() * 0.8
        
        x_coords = np.concatenate([xs[:,0], xs[::-1,0]])
        y_coords = np.concatenate([i + dens, i - dens[::-1]])
        
        # Color by median pace
        med = np.median(d)
        color_idx = int((med - sub['pp_e2_pace'].min()) /
                        (sub['pp_e2_pace'].max() - sub['pp_e2_pace'].min()) * 255)
        color = Viridis256[color_idx]
        
        p.patch(x_coords, y_coords,
                fill_color=color, fill_alpha=0.6,
                line_color="black", line_width=0.5)

    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_alpha = 0.3
    p.yaxis.major_label_text_font_size = "9pt"
    
    # Render this race's plot
    show(p)
