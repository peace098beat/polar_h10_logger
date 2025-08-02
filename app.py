import streamlit as st
from streamlit_autorefresh import st_autorefresh
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from plot import plot_hr, plot_rr, plot_rrdiff

st.set_page_config(layout="wide")
st.title("Heart Rate Monitor")

interval_sec = st.number_input("再描画間隔（秒）", min_value=1, max_value=3600, value=60, step=1)

plot_hr_placeholder = st.empty()
plot_rr_placeholder = st.empty()
plot_rrdiff_placeholder = st.empty()

def load_data(with_rr=False):
    data_dir = Path("./data/rr_np")
    cutoff = datetime.now() - timedelta(hours=3)
    buf_hr, buf_rr = [], []
    for fp in sorted(data_dir.glob("*.npz"), key=lambda p: p.stat().st_mtime):
        if datetime.fromtimestamp(fp.stat().st_mtime) > cutoff:
            continue
        with np.load(fp) as data:
            # Heart rate data
            times = pd.to_datetime(data["ts"], unit="s")
            df_hr_single = pd.DataFrame({"time": times, "heart_rate": data["hr"]})
            buf_hr.append(df_hr_single)

            if with_rr:
                # RR interval data
                rr_vals = data["rr"]
                n_rr = len(rr_vals)
                # Use positional indexing on DatetimeIndex
                start_time = times[0]
                end_time = times[-1]
                time_rr = pd.date_range(start=start_time, end=end_time, periods=n_rr)
                df_rr_single = pd.DataFrame({"time": time_rr, "rr": rr_vals})
                # Rolling median filter over 30-second window
                df_rr_single.set_index("time", inplace=True)
                med = df_rr_single["rr"].rolling("120s", center=True).median()
                mask = (df_rr_single["rr"] >= (med - 200)) & (df_rr_single["rr"] <= (med + 200))
                df_rr_filtered = df_rr_single[mask].reset_index()
                buf_rr.append(df_rr_filtered)

    df_hr = pd.concat(buf_hr, ignore_index=True) if buf_hr else pd.DataFrame(columns=["time", "heart_rate"])
    df_rr = pd.concat(buf_rr, ignore_index=True) if buf_rr else pd.DataFrame(columns=["time", "rr"])

    # Prepare RR difference
    if not df_rr.empty:
        df_rr = df_rr.sort_values("time").reset_index(drop=True)
        rr_diff = df_rr["rr"].diff().iloc[1:]
        df_rrdiff = pd.DataFrame({"time": df_rr["time"].iloc[1:], "rr_diff": rr_diff.values})
    else:
        df_rrdiff = pd.DataFrame(columns=["time", "rr_diff"])

    return df_hr, df_rr, df_rrdiff

def update_plot():
    df_hr, df_rr, df_rrdiff = load_data(with_rr=True)
    plot_hr_placeholder.plotly_chart(plot_hr(df_hr), use_container_width=True)
    plot_rr_placeholder.plotly_chart(plot_rr(df_rr), use_container_width=True)
    plot_rrdiff_placeholder.plotly_chart(plot_rrdiff(df_rrdiff), use_container_width=True)

# Initial plot and auto-refresh
update_plot()
st_autorefresh(interval=interval_sec * 1000, limit=None, key="plot_only_refresh")
