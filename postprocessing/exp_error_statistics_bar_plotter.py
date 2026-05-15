#%%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

def plot_error_comparison(
    data,
    systems=None,
    states=None,
    units=None,
    figsize=None,
    colors=None,
    title="Error metric comparison",
):
    """
    Plots grouped bar charts comparing error metrics (RMSE, STD, Max Error)
    across multiple states (position, yaw, hitch angle) for two systems.

    Parameters
    ----------
    data : pd.DataFrame or str
        DataFrame or path to a CSV file with columns:
            system  : system name (e.g. 'EKF', 'DL-VIO')
            metition: numeric value for the position state
            yawric  : one of 'RMSE', 'STD', 'Max Error'
            pos     : numeric value for the yaw state
            hitch   : numeric value for the hitch angle state
        Example row: ['EKF', 'RMSE', 269.64, 1.98, 0.39]

    systems : list of str, optional
        The two system names to compare, in display order.
        Defaults to the first two unique values in data['system'].

    states : list of str, optional
        Column names for each state. Defaults to ['position', 'yaw', 'hitch'].

    units : dict, optional
        Maps state column name -> unit label string.
        Defaults to {'position': 'm', 'yaw': 'deg', 'hitch': 'deg'}.

    figsize : tuple, optional
        Figure size. Defaults to (12, 4).

    colors : list of two str, optional
        Bar colors for [system_1, system_2].
        Defaults to ['#3266ad', '#c0553e'].

    title : str, optional
        Overall figure title.

    Returns
    -------
    fig, axes : matplotlib Figure and array of Axes
    """
    if isinstance(data, str):
        data = pd.read_csv(data)

    states = states or ['position', 'yaw', 'hitch']
    units  = units  or {'position': '[m]', 'yaw': '[deg]', 'hitch': '[deg]'}
    colors = colors or ["#ff0000", "#000000"]
    figsize = figsize or (12, 4)

    systems = systems or data['system'].unique()[:2].tolist()
    metrics = ['RMSE', 'STD', 'Max Error']

    state_labels = {
        'position': 'Position',
        'yaw':      'Yaw',
        'hitch':    'Hitch angle',
    }

    fig, axes = plt.subplots(1, len(states), figsize=figsize)
    # fig.suptitle(title, fontsize=18, y=1.02)

    x = np.arange(len(metrics))
    bar_w = 0.35

    for ax, state in zip(axes, states):
        for i, (sys, color) in enumerate(zip(systems, colors)):
            row = data[(data['system'] == sys)]
            vals = [
                row.loc[row['metric'] == m, state].values[0]
                for m in metrics
            ]
            offset = (i - 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, label=sys,
                          color=color, alpha=0.88, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f'{v:.2f}',
                    ha='center', va='bottom', fontsize=10, color=color
                )

        unit = units.get(state, '')
        ax.set_title(f'{state_labels.get(state, state)}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=14)
        ax.set_ylabel(f'{unit}', fontsize=14)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.grid(axis='y', alpha=0.25, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(-0.6, len(metrics) - 0.4)

    axes[0].legend(fontsize=14, framealpha=0.4)
    fig.supxlabel('Error metrics', fontsize=14, y=-0.02)
    fig.tight_layout()
    return fig, axes


# ── Example usage with your data ──────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.DataFrame([
        {"system": "EKF",    "metric": "RMSE",      "position": 150.99, "yaw": 2.63, "hitch": 0.44},
        {"system": "EKF",    "metric": "STD",       "position": 94.90, "yaw": 2.44, "hitch": 0.26},
        {"system": "EKF",    "metric": "Max Error",  "position": 440.37, "yaw": 2.54, "hitch": 0.92},
        {"system": "DL-VIO", "metric": "RMSE",      "position":  76.03, "yaw": 1.80, "hitch": 0.15},
        {"system": "DL-VIO", "metric": "STD",       "position":  59.77, "yaw": 1.53, "hitch": 0.13},
        {"system": "DL-VIO", "metric": "Max Error",  "position": 240.68, "yaw": 4.49, "hitch": 1.43},
    ])

    fig, axes = plot_error_comparison(df, title="Error Statistics")
    plt.show()