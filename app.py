# app.py
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from plot import create_resampled_figure

# ページ設定をワイドレイアウトに
st.set_page_config(layout="wide")

# 自動リフレッシュ（5000msごと）
st_autorefresh(interval=5000, limit=None, key="auto_refresh")

st.title("Heart Rate Monitor")

# セッションステートでロード済みファイルとデータフレームを保持
if 'loaded_files' not in st.session_state:
    st.session_state.loaded_files = []
    st.session_state.df = pd.DataFrame(columns=["time", "heart_rate"])

# NPZファイル一覧を取得し、新規ファイルのみ読み込み
all_files = sorted(glob.glob("./data/rr_np/*.npz"))
new_files = [fp for fp in all_files if fp not in st.session_state.loaded_files]
for fp in new_files:
    with np.load(fp) as data:
        hrs = data["hr"]
        ts = data["ts"]
    times = [datetime.fromtimestamp(t) for t in ts.tolist()]
    df_new = pd.DataFrame({"time": times, "heart_rate": hrs.tolist()})
    # 空のDataFrameには直接代入、既存データがあればconcat
    if st.session_state.df.empty:
        st.session_state.df = df_new.copy()
    else:
        st.session_state.df = pd.concat([st.session_state.df, df_new], ignore_index=True)

# 読み込んだファイルリストを更新
st.session_state.loaded_files.extend(new_files)

# データフレーム取得
df = st.session_state.df

# 現在時刻を表示
st.markdown(
    f"<p style='text-align:right; font-size:16px;'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True
)

# 最新のHRを大きく表示
if not df.empty:
    latest_hr = df["heart_rate"].iloc[-1]
    st.markdown(
        f"<h1 style='text-align:center; font-size:72px;'>{latest_hr:.1f} bpm</h1>",
        unsafe_allow_html=True
    )


# 描画
fig = create_resampled_figure(df)
st.plotly_chart(fig, use_container_width=True)
