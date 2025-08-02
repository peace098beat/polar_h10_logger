from typing import Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import stft

# Define HRV frequency bands (Hz)
LF_BAND = (0.04, 0.15)
HF_BAND = (0.15, 0.40)


def plot_timeseries(
    rows: tuple[tuple[pd.Series, ...], ...] | tuple[pd.Series, ...],
    subplot_titles: list[str] | None = "Time Series",
    height: int = 600
) -> go.Figure:
    """
    Generic time series plot for multiple rows.
    rows: single tuple of Series or nested tuple per row.
    subplot_titles: optional list of titles per row.
    """
    # Normalize input
    if isinstance(rows, pd.Series):
        rows = ((rows,),)
    elif isinstance(rows, tuple) and rows and isinstance(rows[0], pd.Series):
        rows = (rows,)
    n_rows = len(rows)
    titles = subplot_titles or [None] * n_rows
    fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=titles,
        shared_xaxes=(n_rows > 1),
        vertical_spacing=0.1
    )
    for i, series_list in enumerate(rows, start=1):
        if isinstance(series_list, pd.Series):
            series_list = (series_list,)
        for sr in series_list:
            fig.add_trace(
                go.Scatter(x=sr.index, y=sr.values, mode='lines', name=sr.name or ''),
                row=i, col=1
            )
    fig.update_layout(
        template='plotly_white',
        height=height,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    fig.update_xaxes(title_text='Time')
    return fig


def plot_histogram(
    items: tuple[pd.Series | np.ndarray, ...] | pd.Series | np.ndarray,
    title: str | None = "Histogram",
    bins: int = 50,
    height: int = 400
) -> go.Figure:
    """
    Generic overlaid histogram plot in a single row.
    items: single Series/ndarray or tuple of them. Each histogram is overlaid.
    title: optional title for the histogram.
    """
    # Normalize to list
    if isinstance(items, (pd.Series, np.ndarray)):
        data_list = [items]
    else:
        data_list = list(items)
    # Create figure
    fig = go.Figure()
    for arr in data_list:
        vals = arr.values if isinstance(arr, pd.Series) else arr
        name = arr.name if isinstance(arr, pd.Series) else ''
        fig.add_trace(
            go.Histogram(x=vals, nbinsx=bins, name=name, opacity=0.5)
        )
    fig.update_layout(
        barmode='overlay',
        title=title,
        xaxis_title=title or '',
        yaxis_title='Count',
        template='plotly_white',
        height=height,
        legend_title_text='Series'
    )
    return fig


def plot_fftspectrum(
    freqs: np.ndarray,
    power: np.ndarray,
    title: str | None = "Spectrum",
    height: int = 400
) -> go.Figure:
    """
    FFT power spectrum plot.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=power, mode='lines', name='Spectrum'))
    for freq in (*LF_BAND, *HF_BAND):
        fig.add_vline(x=freq, line=dict(color='black', dash='dash'))
    fig.update_layout(
        title=title or 'FFT Spectrum',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power',
        template='plotly_white',
        height=height
    )
    return fig



def stft(series, window):
    spectra = []
    for label, grp in series.resample(window):
        vals = grp.dropna().values
        if len(vals) < 2:
            continue
        dt = grp.index.to_series().diff().median().total_seconds()
        yf = np.fft.rfft(vals - np.mean(vals))
        xf = np.fft.rfftfreq(len(vals), d=dt)
        power = np.abs(yf) ** 2
        spectra.append((label, xf, power))
    return spectra

def plot_stft(
    series: pd.Series,
    window: str,
    title: str | None = 'STFT Spectrums',
    height: int = 400
) -> go.Figure:
    """
    Short-Time Fourier Transform spectrogram using pandas resample.
    series: pandas Series indexed by datetime
    window: resample frequency string (e.g., '5T', '30S')
    """
    spectra = stft(series, window)

    fig = go.Figure()
    for label, xf, power in spectra:
        fig.add_trace(go.Scatter(x=xf, y=power, mode='lines', name=str(label)))
    for freq in (*LF_BAND, *HF_BAND):
        fig.add_vline(x=freq, line=dict(color='black', dash='dash'))
    fig.update_layout(
        title=title,
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power',
        template='plotly_white',
        height=height
    )
    return fig



def plot_spectrogram(
    series: pd.Series,
    window: str = '5T',
    title: str | None = 'Spectrogram',
    height: int = 400
) -> go.Figure:
    """
    Spectrogram heatmap from STFT.
    """
    def stft(series, window):
        spectra = []
        times = []
        # fsと最大長を計算
        N_max = 0

        for label, grp in series.resample(window):
            N = len(grp.dropna().values)
            if N > N_max:
                N_max = N

        for label, grp in series.resample(window):
            vals = grp.dropna().values
            if len(vals) < 2:
                continue
            # 0fill
            vals = np.pad(vals, (0, N_max - len(vals)), mode='constant')
            dt = grp.index.to_series().diff().median().total_seconds()
            yf = np.fft.rfft(vals - np.mean(vals))
            xf = np.fft.rfftfreq(len(vals), d=dt)
            power = np.abs(yf) ** 2
            spectra.append((label, xf, power))
            times.append(label)
        return spectra, times, xf

    spectra, times, xf = stft(series, window)
    times_ary = np.array(times)
    power_ary = np.array([power for _, _, power in spectra]).T

    fig = go.Figure(
        go.Heatmap(x=times_ary, y=xf, z=power_ary, colorscale='Viridis')
    )
    for freq in (*LF_BAND, *HF_BAND):
        fig.add_shape(
            type='line',
            x0=times_ary.min(),
            x1=times_ary.max(),
            y0=freq,
            y1=freq,
            line=dict(dash='dash', color='white')
        )
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        template='plotly_white',
        height=height
    )
    return fig
