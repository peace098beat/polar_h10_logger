
from bleak import BleakScanner
import asyncio

async def list_devices():
    print("[INFO] スキャン中...")
    devices = await BleakScanner.discover()
    for i, device in enumerate(devices):
        print(f"{i}: {device.name} - {device.address}")

async def main():
    await list_devices()
    # await run_once(...) など続けてもOK

if __name__ == "__main__":
    asyncio.run(main())
