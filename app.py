# app.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ページ設定をワイドレイアウトに
st.set_page_config(layout="wide")

# 自動リフレッシュ（5000msごと）
st_autorefresh(interval=5000, limit=None, key="auto_refresh")

st.title("Heart Rate Monitor")

# セッションステートでロード済みファイルとデータフレームを保持
if 'loaded_files' not in st.session_state:
    st.session_state.loaded_files = []
    # 初期空DF
    st.session_state.df = pd.DataFrame(columns=["time", "heart_rate"])

# 新規ファイルの検出
all_files = sorted(glob.glob("./data/rr_np/*.npz"))
new_files = [f for f in all_files if f not in st.session_state.loaded_files]
print(f"new_files: {new_files}")

# 新しいファイルのみ読み込み
for fp in new_files:
    print(f"load {fp}")
    data = np.load(fp)
    hrs = data['hr']
    ts = data['ts']
    times = [datetime.fromtimestamp(t) for t in ts.tolist()]
    df_new = pd.DataFrame({"time": times, "heart_rate": hrs.tolist()})
    st.session_state.df = pd.concat([st.session_state.df, df_new], ignore_index=True)

# 読み込んだファイルをリストに追加
st.session_state.loaded_files.extend(new_files)

# 表示用DF
df = st.session_state.df

# 2行1列サブプロット
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

st.plotly_chart(fig, use_container_width=True)
