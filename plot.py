# plot.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Plotly-Resamplerはオプション
try:
    from plotly_resampler import FigureResampler
    _HAS_RESAMPLER = True
except ImportError:
    _HAS_RESAMPLER = False


def create_figure(df):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Heart Rate Time Series", "Heart Rate Distribution")
    )
    fig.add_trace(
        go.Scatter(x=df["time"], y=df["heart_rate"], mode="lines", name="HR"),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=df["heart_rate"], nbinsx=50, name="Distribution"),
        row=2, col=1
    )
    fig.update_layout(
        height=800,
        template="plotly_white",
        xaxis=dict(rangeslider=dict(visible=True))
    )
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Heart Rate (bpm)", row=1, col=1)
    fig.update_xaxes(title_text="Heart Rate (bpm)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    return fig

def create_resampled_figure(df, default_n_shown_samples=1000, height=800):
    if not _HAS_RESAMPLER:
        raise ImportError("plotly_resampler がインストールされていません。pip install plotly-resampler してください。")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.2,
        row_heights=[1.0, 1.0],
        subplot_titles=("Heart Rate Time Series", "Heart Rate Distribution")
    )
    fr = FigureResampler(fig, default_n_shown_samples=default_n_shown_samples)
    fr.add_trace(
        go.Scatter(x=df["time"], y=df["heart_rate"], mode="lines", name="HR"),
        row=1, col=1
    )
    fr.add_trace(
        go.Histogram(x=df["heart_rate"], nbinsx=50, name="Distribution"),
        row=2, col=1
    )
    # レイアウト設定: X軸固定、Y軸のみリスケール
    fr.update_layout(
        title="Heart Rate Over Time (Resampled)",
        xaxis_title="Time",
        yaxis_title="Heart Rate (bpm)",
        template="plotly_white",
        height=height,
    )
    return fr


if __name__ == "__main__":
    import glob
    import numpy as np
    import pandas as pd
    from datetime import datetime

    files = sorted(glob.glob("./data/rr_np/*.npz"))
    if not files:
        print("No .npz files found in ./data/rr_np")
    else:
        fp = files[-1]
        print(f"Loading: {fp}")
        data = np.load(fp)

        hrs = data['hr']
        ts = data['ts']
        times = [datetime.fromtimestamp(t) for t in ts.tolist()]
        df = pd.DataFrame({"time": times, "heart_rate": hrs.tolist()})
        fig = create_resampled_figure(df)
        fig.show()
