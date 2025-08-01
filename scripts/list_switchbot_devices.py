import requests
import yaml
def list_devices(token: str):
    url = "https://api.switch-bot.com/v1.0/devices"
    headers = {
        "Authorization": token
    }

    try:
        response = requests.get(url, headers=headers, timeout=5, verify=False)
        response.raise_for_status()
        data = response.json()

        print("【SwitchBot デバイス一覧】")
        for d in data.get("body", {}).get("deviceList", []):
            print(f"- {d['deviceName']} ({d['deviceType']}) : {d['deviceId']}")
        for d in data.get("body", {}).get("infraredRemoteList", []):
            print(f"- {d['deviceName']} (IR) : {d['deviceId']}")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] {e}")

# 使用例
if __name__ == "__main__":

    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    token = config.get("SWBOT_TOKEN")
    list_devices(token)