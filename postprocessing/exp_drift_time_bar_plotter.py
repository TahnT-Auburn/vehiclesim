#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['axes.labelsize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14

def plot_drift_times(
    data,
    systems=None,
    thresholds=None,
    threshold_labels=None,
    figsize=(10, 4),
    colors=None,
    title="Time before position drift threshold",
    orientation='v',  # 'v' for vertical, 'h' for horizontal
):
    if isinstance(data, str):
        data = pd.read_csv(data)

    thresholds       = thresholds       or ['1m_error', '5m_error', '10m_error']
    threshold_labels = threshold_labels or ['1m', '5m', '10m']
    colors           = colors           or ["#ff0000", '#000000']
    systems          = systems          or data['system'].unique().tolist()

    x     = np.arange(len(thresholds))
    bar_w = 0.35 / max(len(systems) / 2, 1)

    fig, ax = plt.subplots(figsize=figsize)

    for i, (sys, color) in enumerate(zip(systems, colors)):
        row    = data[data['system'] == sys].iloc[0]
        vals   = [row[t] for t in thresholds]
        offset = (i - (len(systems) - 1) / 2) * bar_w

        if orientation == 'h':
            bars = ax.barh(x + offset, vals, bar_w, label=sys,
                           color=color, alpha=0.88, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(
                    v + ax.get_xlim()[1] * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:.2f}s',
                    ha='left', va='center', fontsize=14, color=color
                )
        else:
            bars = ax.bar(x + offset, vals, bar_w, label=sys,
                          color=color, alpha=0.88, zorder=3)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.01,
                    f'{v:.2f}s',
                    ha='center', va='bottom', fontsize=14, color=color
                )

    if orientation == 'h':
        ax.set_yticks(x)
        ax.set_yticklabels(threshold_labels, fontsize=14)
        ax.set_xlabel('Time [s]', fontsize=14)
        ax.set_ylabel('Absolute Position Error', fontsize=14)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.set_ylim(-0.6, len(thresholds) - 0.4)
        ax.invert_yaxis()  # largest threshold at bottom, matching table order
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(threshold_labels, fontsize=14)
        ax.set_ylabel('Time [s]', fontsize=14)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        ax.set_xlim(-0.6, len(thresholds) - 0.4)

    # ax.set_title(title, fontsize=14)
    ax.grid(axis='x' if orientation == 'h' else 'y', alpha=0.25, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=14, framealpha=0.4)

    fig.tight_layout()
    return fig, ax


# ── Example usage with your data ──────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.DataFrame([
        {"system": "EKF",    "1m_error": 6.63, "5m_error": 16.85, "10m_error": 25.05},
        {"system": "DL-VIO", "1m_error": 19.38, "5m_error": 47.25, "10m_error": 78.38},
    ])

    fig, ax = plot_drift_times(df, title="Drift Times", orientation='h')
    plt.show()