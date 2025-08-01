import asyncio
from datetime import datetime
from bleak import BleakClient
import numpy as np
import json
from datetime import timedelta

from led import set_led, set_led_color, turn_off

RR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"
DEVICE_ADDRESS = "ABDA0013-523B-24DC-53F2-E4F934D7BE94"

# SWITCHBOT
TOKEN = "f26eaa7007230625ffc25ce78cae8444b62036f9bdd9fb43caa31ae873af6021f3cc0a81a16dac9c13247cc03f686915"
DEVICE_ID = "6055F92A78EE"

from collections import deque

hr_value_buffer = deque(maxlen=100)

rr_intervals = []
hr_values = []


def plot_led(v, v_min, v_max):
    level = (v - v_min) / (v_max - v_min)
    set_led(DEVICE_ID, TOKEN, level=level)


def handle_rr_data(_, data):
    flags = data[0]
    rr_present = (flags & 0x10) != 0
    i = 1

    if flags & 0x01:
        hr = int.from_bytes(data[i : i + 2], byteorder="little")
        i += 2
    else:
        hr = data[i]
        i += 1

    if hr == 0:
        print("[WARN] HR=0 のデータをスキップ")
        return

    hr_values.append(hr)
    print(f"[INFO] HR: {hr:.1f}bpm")

    while rr_present and i + 1 < len(data):
        rr = int.from_bytes(data[i : i + 2], byteorder="little") / 1024.0 * 1000
        rr_intervals.append(rr)
        i += 2

    # -- LED --
    hr_value_buffer.append(hr)
    if len(hr_value_buffer) <= 10:
        v = len(hr_value_buffer)
        plot_led(v, 1, 10)
    else:
        plot_led(hr, 70, 90)  # 青:70, 水色:85, 赤:100

    # -- save data --
    # [TODO] SQLiteに保存するのが早い


async def run_once(t_interval=60, timeout_sec=10, check_interval=1):
    print(f"[INFO] データ取得開始: {t_interval}秒")
    t_interval = int(t_interval)
    global rr_intervals, hr_values
    rr_intervals, hr_values = [], []

    start_time = datetime.now()
    end_time = None
    elapsed = 0

    try:
        async with BleakClient(DEVICE_ADDRESS) as client:
            print("[INFO] 接続中...")
            await asyncio.wait_for(client.connect(), timeout=timeout_sec)
            _ = client.services
            await asyncio.wait_for(
                client.start_notify(RR_UUID, handle_rr_data), timeout=timeout_sec
            )

            print(f"[INFO] {t_interval}秒取得開始")
            start_time = datetime.now()

            while elapsed < t_interval:
                if not client.is_connected:
                    raise ConnectionError("BT接続が途中で切断されました")
                await asyncio.sleep(check_interval)
                elapsed += check_interval

            await asyncio.wait_for(client.stop_notify(RR_UUID), timeout=timeout_sec)
            end_time = datetime.now()
            print(f"[INFO] {t_interval}秒取得完了 / データ数: {len(rr_intervals)}")

    except asyncio.TimeoutError:
        print(f"[ERROR] タイムアウト発生（{timeout_sec}秒以内に応答なし）")
        err_filename = (
            f"./data/error/{datetime.now().strftime('%Y%m%d_%H%M%S')}_timeout.txt"
        )
        with open(err_filename, "w") as f:
            f.write("[ERROR] TimeoutError: 接続または通知処理がタイムアウトしました。")

        turn_off(DEVICE_ID, TOKEN)

    except Exception as e:
        print(f"[ERROR] {e}")
        err_filename = f"./data/error/{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(err_filename, "w") as f:
            f.write(f"[ERROR] {e}")

        turn_off(DEVICE_ID, TOKEN)

    except bleak.exc.BleakDeviceNotFoundError as e:
        print(f"[ERROR] {e}")
        err_filename = f"./data/error/{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(err_filename, "w") as f:
            f.write(f"[ERROR] {e}")

        turn_off(DEVICE_ID, TOKEN)

        await asyncio.sleep(3)

    finally:
        if rr_intervals or hr_values:
            if end_time is None:
                end_time = start_time + timedelta(seconds=elapsed)
            is_partial = elapsed >= t_interval

            savedata_dict = {
                "rr_intervals": rr_intervals,
                "hr_values": hr_values,
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_time": (end_time - start_time).total_seconds(),
                "is_partial": is_partial,
            }

            filename = (
                f"./data/rr_intervals/{start_time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(filename, "w") as f:
                json.dump(savedata_dict, f, indent=2)
            print(f"[INFO] データを保存しました: {filename}")


async def main():
    while True:
        await run_once()


asyncio.run(main())
