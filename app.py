import streamlit as st
from streamlit_autorefresh import st_autorefresh
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from plot import plot_timeseries, plot_histogram, plot_spectrogram, plot_stft
import os
import json

st.set_page_config(layout="wide", page_title="Heart Rate Monitor")

# --- Sidebar configuration ---
st.sidebar.title("Settings")
interval_sec = st.sidebar.slider(
    "Auto-refresh interval (sec)", min_value=5, max_value=600, value=60, step=5
)
time_window = st.sidebar.selectbox(
    "Data window (hours)", options=[1, 3, 6, 12, 24], index=1
)

# Trigger auto-refresh
st_autorefresh(interval=interval_sec * 1000, limit=None, key="auto_refresh")

# --- Data loading ---
@st.cache_data
def load_data(hours: int = 3) -> tuple[pd.Series, pd.Series]:
    cutoff = datetime.now() - timedelta(hours=hours)
    data_dir = Path("./data/rr_np")
    buf_hr, buf_rr = [], []
    for fp in sorted(data_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime):
        mtime = datetime.fromtimestamp(fp.stat().st_mtime)
        if mtime < cutoff:
            continue
        with np.load(fp) as data:
            times = [datetime.fromtimestamp(ts) for ts in data['ts']]

            # 1. HR
            buf_hr.append(pd.Series(data['hr'], index=times, name='heart_rate'))

            # 2. RR
            rr_vals = data['rr']
            time_rr = pd.date_range(start=times[0], end=times[-1], periods=len(rr_vals))
            buf_rr.append(pd.Series(rr_vals, index=time_rr, name='rr'))

    # Concatenate series
    if buf_hr:
        sr_hr = pd.concat(buf_hr)
    else:
        sr_hr = pd.Series(dtype=float, name='heart_rate')
    if buf_rr:
        sr_rr = pd.concat(buf_rr)
    else:
        sr_rr = pd.Series(dtype=float, name='rr')
    return sr_hr, sr_rr

def preprocess_hr(sr_hr: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        'hr': sr_hr.values,
        'hr_sd_30s': sr_hr.rolling('30s', center=True).std().values,
        'hr_sd_5min': sr_hr.rolling('5min', center=True).std().values,
        'hr_sd_15min': sr_hr.rolling('15min', center=True).std().values,
        'hr_mean_5min': sr_hr.rolling('5min', center=True).mean().values,  # Add rolling mean for 5 minutes
    }, index=sr_hr.index)

# Preprocessing: apply median filter and compute rr_diff and RMSSD
def preprocess_rr(sr_rr: pd.Series) -> pd.DataFrame:
    # Outlier
    rr_med = sr_rr.rolling('120s', center=True).median()
    # Outlier filter based on median based on median
    mask = (sr_rr >= rr_med - 200) & (sr_rr <= rr_med + 200)
    # mask部分は, np.nan
    sr = sr_rr.where(mask)

    def rms(x):
        return np.sqrt(np.mean(x**2))

    sdnn_30s = sr.rolling('30s', center=True).std()
    sdnn_5min = sr.rolling('5min', center=True).std()
    sdnn_15min = sr.rolling('15min', center=True).std()

    # Compute differences and RMSSD
    rr_diff = sr.diff()
    # rr_diff > +-250はnp.nan
    rr_diff = rr_diff.where((rr_diff > -100) & (rr_diff < 100))

    rmssd_30s = rr_diff.rolling('30s', center=True, min_periods=1).std()
    rmssd_5min = rr_diff.rolling('5min', center=True, min_periods=1).std()
    rmssd_15min = rr_diff.rolling('15min', center=True, min_periods=1).std()

    # Compute heart rate (bpm)
    hr_values = 60000 / sr.values  # bpm from RR intervals (ms)

    # Assemble DataFrame with computed metrics
    return pd.DataFrame({
        'rr': sr.values,
        'sdnn_30s': sdnn_30s.values,
        'sdnn_5min': sdnn_5min.values,
        'sdnn_15min': sdnn_15min.values,
        'rr_med': rr_med.values,
        'rr_diff': rr_diff.values,
        'rmssd_30s': rmssd_30s.values,
        'rmssd_5min': rmssd_5min.values,
        'rmssd_15min': rmssd_15min.values,
        'hr': hr_values
    }, index=sr.index)

# Load and preprocess data
sr_hr, sr_rr = load_data(hours=time_window)
df_rr = preprocess_rr(sr_rr)

df_hr = preprocess_hr(sr_hr)

# --- Main view with tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Heart Rate", "RR Intervals", "RR Diff Analysis", "Data View"])

with tab1:
    st.header("Heart Rate Time Series")
    st.plotly_chart(plot_timeseries((df_rr["hr"], sr_hr,)), height=1200)
    st.plotly_chart(plot_histogram((df_rr["hr"], sr_hr), "Heart Rate", height=300))

    st.plotly_chart(plot_timeseries((df_hr["hr_sd_30s"], df_hr["hr_sd_5min"], df_hr["hr_sd_15min"]),
))
    st.plotly_chart(plot_histogram((df_hr["hr_sd_30s"], df_hr["hr_sd_5min"], df_hr["hr_sd_15min"]),
))

with tab2:
    st.header("RR Interval Time Series")
    st.plotly_chart(
        plot_timeseries(
        (
            (df_rr["rr"], df_rr["rr_med"]),
            (df_rr["sdnn_30s"], df_rr["sdnn_5min"], df_rr["sdnn_15min"]),
        )),
        height=400)

    st.plotly_chart(plot_histogram(
        (df_rr["sdnn_30s"], df_rr["sdnn_5min"], df_rr["sdnn_15min"]),"SDNN", height=300))

    # st.plotly_chart(plot_stft(df_rr["rr"], window="5min"))
    # st.plotly_chart(plot_spectrogram(df_rr["rr"], window="5min"))


with tab3:
    st.header("RR Difference Analysis")
    st.plotly_chart(plot_timeseries(
        (
            (df_rr["rr_diff"],),
            (df_rr["rmssd_30s"], df_rr["rmssd_5min"], df_rr["rmssd_15min"]),
        )
    ))
    st.plotly_chart(
        plot_histogram(
            (df_rr["rmssd_30s"], df_rr["rmssd_5min"], df_rr["rmssd_15min"]),
            ),
    )

    st.plotly_chart(plot_stft(df_rr["rr_diff"], window="5min"))
    st.plotly_chart(plot_spectrogram(df_rr["rr_diff"], window="5min"), title="RR Difference Spectrogram")

with tab4:
    st.header("Data View")

    st.subheader("df_rr Data (TSV Format)")
    st.text_area("df_rr (TSV)", df_rr.to_csv(index=True, sep='\t'), height=300)

    st.subheader("df_hr Data (TSV Format)")
    st.text_area("df_hr (TSV)", df_hr.to_csv(index=True, sep='\t'), height=300)

    st.subheader("df_hr 5-min Rolling Mean (TSV Format)")
    st.text_area("df_hr Rolling Mean (TSV)", df_hr.resample("5min").mean().to_csv(index=True, sep='\t'), height=300)

# --- Annotation Section ---
annotations = []  # List to store annotations
predefined_tags = ["Exercise", "Resting", "Working", "Eating", "Sleeping"]  # Predefined tags

# Path to the annotations file
ANNOTATIONS_FILE = "data/annot/annot.json"

# Ensure the directory exists
os.makedirs(os.path.dirname(ANNOTATIONS_FILE), exist_ok=True)

# Load existing annotations if the file exists
if os.path.exists(ANNOTATIONS_FILE):
    with open(ANNOTATIONS_FILE, "r") as f:
        annotations = json.load(f)
else:
    annotations = []

# Ensure annotations is global and persists
st.session_state.setdefault("annotations", annotations)
annotations = st.session_state["annotations"]

with st.sidebar:
    st.header("Add Annotation")

    # Section to add new tags
    new_tag = st.text_input("Add a new tag")
    if st.button("Add Tag"):
        if new_tag:
            if new_tag not in predefined_tags:
                predefined_tags.append(new_tag)
                st.success(f"Tag '{new_tag}' added successfully!")
            else:
                st.warning(f"Tag '{new_tag}' already exists.")
        else:
            st.error("Tag cannot be empty.")

    with st.form("annotation_form"):
        annotation_date = st.date_input("Date")
        annotation_time = st.time_input("Time")
        annotation_posture = st.selectbox("Posture", ["Standing", "Sitting", "Lying Down"])
        annotation_situation = st.multiselect("Situation (Tags)", options=predefined_tags, default=[])
        submit_annotation = st.form_submit_button("Add Annotation")

    if submit_annotation:
        try:
            annotation_datetime = datetime.combine(annotation_date, annotation_time)
            annotations.append({
                "Time": annotation_datetime,
                "Posture": annotation_posture,
                "Situation": ", ".join(annotation_situation)
            })
            st.success("Annotation added successfully!")
        except Exception as e:
            st.error(f"Error adding annotation: {e}")

# Display annotations
    st.subheader("Annotations")
    if annotations:
        try:
            annotations_df = pd.DataFrame(annotations)

            # Ensure the 'Time' column is in datetime format
            annotations_df["Time"] = pd.to_datetime(annotations_df["Time"])

            # Find the annotation closest to the current time
            current_time = datetime.now()
            annotations_df["Time_Diff"] = (annotations_df["Time"] - current_time).abs()
            closest_annotation_idx = annotations_df["Time_Diff"].idxmin()
            annotations_df = annotations_df.drop(columns=["Time_Diff"])

            # Highlight the closest annotation
            def highlight_closest(row):
                return ["background-color: yellow"] * len(row) if row.name == closest_annotation_idx else [""] * len(row)

            st.dataframe(annotations_df.style.apply(highlight_closest, axis=1))

            # テキストエリアでJSON形式で表示
            annotations_json = json.dumps(annotations, indent=4, default=str)
            st.text_area("Annotations (JSON Format)", annotations_json, height=300)
        except Exception as e:
            st.error(f"Error displaying annotations: {e}")
    else:
        st.write("No annotations available.")

    # Save annotations to JSON
    if st.button("Save Annotations to JSON"):
        try:
            serialized_annotations = [
                {
                    "Time": annotation["Time"].isoformat() if isinstance(annotation["Time"], datetime) else annotation["Time"],
                    "Posture": annotation["Posture"],
                    "Situation": annotation["Situation"]
                }
                for annotation in annotations
            ]

            with open(ANNOTATIONS_FILE, "w") as f:
                json.dump(serialized_annotations, f, indent=4)
            st.success(f"Annotations saved to {ANNOTATIONS_FILE}")
        except Exception as e:
            st.error(f"Error saving annotations: {e}")
