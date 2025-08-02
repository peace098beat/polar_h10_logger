import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# Plotly-Resamplerはオプション
try:
    from plotly_resampler import FigureResampler
    _HAS_RESAMPLER = True
except ImportError:
    _HAS_RESAMPLER = False


def plot_hr(df, default_n_shown_samples=1000, height=800):
    if not _HAS_RESAMPLER:
        raise ImportError("plotly_resampler がインストールされていません。pip install plotly-resampler を実行してください。")
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.2,
        row_heights=[1.0, 1.0],
        subplot_titles=("Heart Rate Time Series", "Heart Rate Distribution")
    )
    fr = FigureResampler(fig, default_n_shown_samples=default_n_shown_samples)
    fr.add_trace(
        go.Scatter(x=df["time"], y=df["heart_rate"], mode="lines", name="HR"), row=1, col=1
    )
    fr.add_trace(
        go.Histogram(x=df["heart_rate"], nbinsx=50, name="Distribution"), row=2, col=1
    )
    fr.update_layout(
        title="Heart Rate Over Time (Resampled)",
        xaxis_title="Time",
        yaxis_title="Heart Rate (bpm)",
        template="plotly_white",
        height=height,
    )
    return fr


def plot_rr(df, default_n_shown_samples=1000, height=800):
    if not _HAS_RESAMPLER:
        raise ImportError("plotly_resampler がインストールされていません。pip install plotly-resampler を実行してください。")
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.2,
        row_heights=[1.0, 1.0],
        subplot_titles=("RR Time Series", "RR Distribution")
    )
    fr = FigureResampler(fig, default_n_shown_samples=default_n_shown_samples)
    fr.add_trace(
        go.Scatter(x=df["time"], y=df["rr"], mode="lines", name="RR"), row=1, col=1
    )
    fr.add_trace(
        go.Histogram(x=df["rr"], nbinsx=50, name="Distribution"), row=2, col=1
    )
    fr.update_layout(
        title="RR Over Time (Resampled)",
        xaxis_title="Time",
        yaxis_title="RR (ms)",
        template="plotly_white",
        height=height,
    )
    return fr


def plot_rrdiff(df, default_n_shown_samples=1000, height=1000):
    # RR diff のクリーニング
    df = df[df["rr_diff"].between(-50, 50)]
    sr_rr_diff = df.set_index("time")["rr_diff"]

    # RMSSD 計算 (30s, 5min, 15min)
    rmssds = {
        "30s": sr_rr_diff.rolling("30s", center=True).std(),
        "5min": sr_rr_diff.rolling("5min", center=True).std(),
        "15min": sr_rr_diff.rolling("15min", center=True).std(),
    }

    # 5分ウィンドウでFFTスペクトルを計算し重ねてプロット
    spectra = []
    for label, group in sr_rr_diff.resample("5T"):
        arr = group.dropna().values
        if arr.size <= 1:
            continue
        dt = group.index.to_series().diff().median().total_seconds()
        y = arr - np.mean(arr)
        yf = np.fft.rfft(y)
        xf = np.fft.rfftfreq(len(y), d=dt)
        power = np.abs(yf) ** 2
        spectra.append((label, xf, power))

    if not _HAS_RESAMPLER:
        raise ImportError("plotly_resampler がインストールされていません。pip install plotly-resampler を実行してください。")

    titles = [
        "RR Difference Time Series", "RR Difference Distribution",
        "RMSSD (overlaid 30s, 5min, 15min)", "RMSSD Distribution",
        "FFT Spectra (5min-resampled)", ""
    ]
    fig = make_subplots(
        rows=3, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.2,
        row_heights=[1.0, 1.0, 1.0],
        column_widths=[0.75, 0.25],
        subplot_titles=titles
    )
    fr = FigureResampler(fig, default_n_shown_samples=default_n_shown_samples)

    # 1行目: RR diff
    fr.add_trace(
        go.Scatter(x=df["time"], y=df["rr_diff"], mode="lines", name="RR Diff"), row=1, col=1
    )
    fr.add_trace(
        go.Histogram(x=df["rr_diff"], nbinsx=50, name="Dist"), row=1, col=2
    )

    # 2行目: RMSSD overlay and overlay histograms
    for key, series in rmssds.items():
        fr.add_trace(
            go.Scatter(x=series.index, y=series.values, mode="lines", name=f"RMSSD {key}"), row=2, col=1
        )
        fr.add_trace(
            go.Histogram(x=series.values, nbinsx=50, name=f"RMSSD {key} Dist", opacity=0.5), row=2, col=2
        )

    # 3行目: FFTスペクトル
    for label, xf, power in spectra:
        fr.add_trace(
            go.Scatter(x=xf, y=power, mode="lines", name=str(label)), row=3, col=1
        )
    # LF/HF帯域線
    for freq in [0.04, 0.15, 0.40]:
        fig.add_shape(
            type="line", x0=freq, x1=freq,
            y0=0, y1=1, xref="x5", yref="paper",
            line=dict(color="black", dash="dash")
        )

    fr.update_layout(
        title="RR Diff Analysis (Resampled)",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        template="plotly_white",
        height=height,
    )
    return fr