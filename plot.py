# plot2_subplots.py
import glob
import numpy as np
from datetime import datetime
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# npzファイルをソートして取得
files = sorted(glob.glob("./data/rr_np/*.npz"))

timestamps = []
heart_rates = []

# データを読み込んでリストに追加
for fp in files:
    with np.load(fp) as data:
        hrs = data["hr"]
        ts = data["ts"]
    heart_rates.extend(hrs.tolist())
    timestamps.extend([datetime.fromtimestamp(t) for t in ts.tolist()])

# DataFrameに変換
df = pd.DataFrame({
    "time": timestamps,
    "heart_rate": heart_rates
})

# 2行1列のサブプロットを作成
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.1,
    row_heights=[0.7, 0.3],
    subplot_titles=("Heart Rate Time Series", "Heart Rate Distribution")
)

# 時系列グラフ
fig.add_trace(
    go.Scatter(x=df["time"], y=df["heart_rate"], mode="lines", name="HR"),
    row=1, col=1
)

# ヒストグラム
fig.add_trace(
    go.Histogram(x=df["heart_rate"], nbinsx=50, name="HR Distribution"),
    row=2, col=1
)

# レイアウト調整
fig.update_layout(
    height=800,
    template="plotly_white"
)

# 軸ラベル設定
fig.update_xaxes(title_text="Time", row=1, col=1)
fig.update_yaxes(title_text="Heart Rate (bpm)", row=1, col=1)
fig.update_xaxes(title_text="Heart Rate (bpm)", row=2, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)

# 範囲スライダーを上段に追加
fig.update_layout(xaxis=dict(rangeslider=dict(visible=True)))

fig.show()
