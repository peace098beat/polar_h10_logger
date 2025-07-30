import json
import os
import glob
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

from plotly.subplots import make_subplots




def load_one_data(p_json):
    with open(p_json, "r") as f:
        data = json.load(f)

    rr = data["rr_intervals"]
    hr = data["hr_values"]
    start_time = datetime.strptime(data["start_time"], "%Y-%m-%d %H:%M:%S")
    end_time = datetime.strptime(data["end_time"], "%Y-%m-%d %H:%M:%S")
    elapsed_time = (end_time - start_time).total_seconds()

    rr_t = pd.date_range(start=start_time, periods=len(rr), end=end_time)
    hr_t = pd.date_range(start=start_time, periods=len(hr), end=end_time)

    sr_rr = pd.Series(rr, index=rr_t)
    sr_rr_diff = sr_rr.diff().iloc[1:]  # 1st value is NaN
    sr_hr_polar = pd.Series(hr, index=hr_t) # from Polar H10



    return sr_rr, sr_hr_polar, sr_rr_diff


def load_datas(data_dir="./data/rr_intervals"):
    data_files = glob.glob(os.path.join(data_dir, "*.json"))

    rr_list = []
    polar_hr_list = []
    rr_diff_list = []
    for data_file in data_files:
        sr_rr, sr_hr_polar, sr_rr_diff = load_one_data(data_file)
        rr_list.append(sr_rr)
        polar_hr_list.append(sr_hr_polar)
        rr_diff_list.append(sr_rr_diff)
    sr_rr = pd.concat(rr_list, axis=0)
    sr_hr_polar = pd.concat(polar_hr_list, axis=0)
    sr_rr_diff = pd.concat(rr_diff_list, axis=0)


    sr_rr.sort_index(inplace=True)
    sr_hr_polar.sort_index(inplace=True)
    sr_rr_diff.sort_index(inplace=True)
    assert sr_rr.index.is_monotonic_increasing
    assert sr_hr_polar.index.is_monotonic_increasing
    assert sr_rr_diff.index.is_monotonic_increasing
    return sr_rr, sr_hr_polar, sr_rr_diff



def sdnn(sr):
    return np.std(sr)

def rmssd(sr):
    diff_rr = np.diff(sr)
    return np.sqrt(np.mean(diff_rr ** 2))

def rms(sr):
    return np.sqrt(np.mean(sr ** 2))


# def plot_data(sr_rr, sr_hr_polar):

#     sr_rr_rmssd = sr_rr.rolling(window="30s").apply(rmssd)
#     sr_rr_sdnn = sr_rr.rolling(window="30s").apply(sdnn)

#     print(sr_rr)
#     # グラフ作成
#     fig = go.Figure()

#     fig.add_trace(go.Scatter(x=sr_rr.index, y=sr_rr.values, mode='lines+markers', name='RR [ms]'))
#     fig.add_trace(go.Scatter(x=sr_hr_polar.index, y=sr_hr_polar.values, mode='lines+markers', name='HR [bpm]', yaxis='y2'))
#     fig.add_trace(go.Scatter(x=sr_rr_rmssd.index, y=sr_rr_rmssd.values, mode='lines+markers', name='RR-RMSSD [ms]'))
#     fig.add_trace(go.Scatter(x=sr_rr_sdnn.index, y=sr_rr_sdnn.values, mode='lines+markers', name='RR-SDNN [ms]'))

#     # レイアウト調整
#     fig.update_layout(
#         title=f"Polar H10 HR & RR Visualization<br>{sr_rr.index[0]} ~ {sr_rr.index[-1]}",
#         xaxis_title="Time [sec]",
#         yaxis=dict(title="RR Interval [ms]", side="left"),
#         yaxis2=dict(title="Heart Rate [bpm]", overlaying="y", side="right"),
#         legend=dict(x=0.01, y=0.99),
#         template="plotly_white",
#         height=500
#     )

#     fig.show()
#     fig.write_html("latest.html", auto_open=True)



def plot_data_2(sr_rr, sr_hr_polar, sr_rr_diff, win_hrv="5min"):
    # [TODO] plotly-resamplerに変更する
    # ローリングHRV計算
    sr_rr_rmssd = sr_rr_diff.rolling(window=win_hrv).apply(rms, raw=False)
    sr_rr_sdnn = sr_rr.rolling(window=win_hrv).apply(sdnn, raw=False)

    # サブプロット作成（2行, 1列）
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=("RR / HR", f"Diff RR", f"RMSSD / SDNN ({win_hrv})"),
                        vertical_spacing=0.1,
                        specs=[[{"secondary_y": True}], [{}], [{"secondary_y": True}]])

    # 1段目: RR / HR
    # height 100px
    fig.add_trace(go.Scatter(x=sr_rr.index, y=sr_rr.values, mode='lines+markers',
                             name='RR [ms]'), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(x=sr_hr_polar.index, y=sr_hr_polar.values, mode='lines+markers',
                             name='HR [bpm]'), row=1, col=1, secondary_y=True)

    # 2段目: Diff RR
    fig.add_trace(go.Scatter(x=sr_rr_diff.index, y=sr_rr_diff.values, mode='lines+markers',
                             name='Diff RR [ms]'), row=2, col=1)

    # 3段目: RMSSD / SDNN / Diff RR
    # y軸は0から50msまで
    fig.add_trace(go.Scatter(x=sr_rr_rmssd.index, y=sr_rr_rmssd.values, mode='lines',
                             name='RMSSD [ms]', yaxis='y2'), row=3, col=1, secondary_y=False)

    # y軸は0から50msまで
    fig.add_trace(go.Scatter(x=sr_rr_sdnn.index, y=sr_rr_sdnn.values, mode='lines',
                             name='SDNN [ms]', yaxis='y2'), row=3, col=1, secondary_y=True)

    # レイアウト調整
    fig.update_layout(
        height=700,
        title_text="Polar H10 HR & HRV Visualization",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99)
    )
    fig.update_yaxes(title_text="RR [ms]", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="HR [bpm]", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Diff RR [ms]", row=2, col=1)
    fig.update_yaxes(title_text="RMSSD [ms]", row=3, col=1, secondary_y=False, range=[0, 60])
    fig.update_yaxes(title_text="SDNN [ms]", row=3, col=1, secondary_y=True, range=[0, 100])

    # fig.show()
    fig.write_html("latest.html", auto_open=False)



import time

def watch_and_plot(data_dir="./data/rr_intervals", interval_sec=2):
    last_files = set()
    while True:
        # ファイルの変化を監視
        data_files = set(glob.glob(os.path.join(data_dir, "*.json")))
        if data_files != last_files:
            print("[INFO] データ更新を検知、再描画します")
            sr_rr, sr_hr_polar, sr_rr_diff = load_datas(data_dir)
            plot_data_2(sr_rr, sr_hr_polar, sr_rr_diff)
            last_files = data_files
        else:
            print("[INFO] 新しいデータなし、スリープ中")

        time.sleep(interval_sec)

if __name__ == "__main__":
    watch_and_plot()

