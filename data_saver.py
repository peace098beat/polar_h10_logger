# data_saver.py
import asyncio
import numpy as np
from datetime import datetime
from pathlib import Path

class AsyncDataSaver:
    def __init__(self, save_dir="./data/rr_np"):
        self.queue = asyncio.Queue()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(self._run())

    async def stop(self):
        await self.queue.put(None)
        if self.task:
            await self.task

    async def _run(self):
        while True:
            item = await self.queue.get()
            if item is None:
                break
            await self._save(item)

    async def _save(self, data):
        key = datetime.now().strftime("%Y%m%d_%H%M")
        path = self.save_dir / f"{key}.npz"

        if path.exists():
            existing = np.load(path)
            hr = np.concatenate([existing["hr"], data["hr"]])
            rr = np.concatenate([existing["rr"], data["rr"]]) # タイムスタンプの仕様を決めてないので適当
            ts = np.concatenate([existing["ts"], data["ts"]])
        else:
            hr = data["hr"]
            rr = data["rr"]
            ts = data["ts"]

        np.savez_compressed(path, hr=hr, rr=rr, ts=ts)
        print(f"[DataSaver] Saved to: {path}")
