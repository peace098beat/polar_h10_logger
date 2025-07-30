import asyncio
from datetime import datetime
from bleak import BleakClient
import numpy as np
import json

RR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
DEVICE_ADDRESS = "ABDA0013-523B-24DC-53F2-E4F934D7BE94"

rr_intervals = []
hr_values = []

def handle_rr_data(_, data):
    flags = data[0]
    rr_present = (flags & 0x10) != 0
    i = 1

    if flags & 0x01:
        hr = int.from_bytes(data[i:i+2], byteorder='little')
        i += 2
    else:
        hr = data[i]
        i += 1

    hr_values.append(hr)
    rr_from_hr = 60 / hr * 1000
    print(f"[HR]: {hr} bpm, {rr_from_hr:.1f} ms")

    while rr_present and i + 1 < len(data):
        rr = int.from_bytes(data[i:i+2], byteorder='little') / 1024.0 * 1000
        rr_intervals.append(rr)
        print(f"[RR]: {rr:.1f} ms")
        i += 2

def calc_rmssd(rrs):
    diff_rr = np.diff(rrs)
    return np.sqrt(np.mean(diff_rr ** 2)) if len(diff_rr) > 0 else 0

def calc_sdnn(rrs):
    return np.std(rrs) if len(rrs) > 0 else 0

async def run_once(t_interval=60*3):
    t_interval = int(t_interval)
    global rr_intervals, hr_values
    rr_intervals, hr_values = [], []

    try:
        async with BleakClient(DEVICE_ADDRESS) as client:
            print("[INFO] 接続中...")
            _ = client.services
            await client.start_notify(RR_UUID, handle_rr_data)
            print(f"[INFO] {t_interval}秒取得開始")
            start_time = datetime.now()
            await asyncio.sleep(t_interval)
            await client.stop_notify(RR_UUID)
            end_time = datetime.now()
            print(f"[INFO] {t_interval}秒取得完了 / データ数: {len(rr_intervals)}")

        savedata_dict = {
            "rr_intervals": rr_intervals,
            "hr_values": hr_values,
            "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time": (end_time - start_time).total_seconds(),
            "hrv": {
                "rmssd": calc_rmssd(rr_intervals),
                "sdnn": calc_sdnn(rr_intervals),
            }
        }

        filename = f"./data/rr_intervals/{start_time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, "w") as f:
            json.dump(savedata_dict, f, indent=2)

    except Exception as e:
        err_filename = f"./data/error/{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(err_filename, "w") as f:
            f.write(f"[ERROR] {e}")
        print(f"[ERROR] {e}")

async def main():
    while True:
        await run_once()
        # await asyncio.sleep(1)  # 連続実行の場合の待機（任意）

asyncio.run(main())
