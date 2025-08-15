import asyncio
from datetime import datetime
from bleak import BleakClient
import numpy as np
import yaml

from data_saver import AsyncDataSaver

RR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# 設定ファイルからロード
with open("config.yml", "r") as f:
    cfg = yaml.safe_load(f)

POLAR_BT_ADDRESS = cfg["POLAR_BT_ADDRESS"]

data_saver = AsyncDataSaver()

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

    print(f"[INFO] HR: {hr:.1f}bpm")
    rr_list = []
    while rr_present and i + 1 < len(data):
        rr = int.from_bytes(data[i : i + 2], byteorder="little") / 1024.0 * 1000
        rr_list.append(rr)
        i += 2

    # -- save --
    time_now = datetime.now()
    
    ts = time_now.timestamp()
    print(f"[INFO][handle_rr_data] Time: {ts},{time_now}")

    asyncio.create_task(data_saver.queue.put({
        "hr": np.array([hr]),
        "rr": np.array(rr_list),
        "ts": np.array([ts])
    }))


async def run_once(t_interval=60, timeout_sec=10):
    print(f"[INFO] データ取得開始: {t_interval}秒")

    try:
        async with BleakClient(POLAR_BT_ADDRESS) as client:
            await asyncio.wait_for(client.connect(), timeout=timeout_sec)
            _ = client.services
            await asyncio.wait_for(
                client.start_notify(RR_UUID, handle_rr_data), timeout=timeout_sec
            )
            await asyncio.sleep(int(t_interval))
            await asyncio.wait_for(client.stop_notify(RR_UUID), timeout=timeout_sec)

    except Exception as e:
        await asyncio.sleep(3)


async def main():
    await data_saver.start()
    try:
        while True:
            await run_once()
    finally:
        await data_saver.stop()



asyncio.run(main())
