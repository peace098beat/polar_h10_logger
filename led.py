import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_device_list(token: str):
    """Utility function to get the list of devices"""
    url = "https://api.switch-bot.com/v1.0/devices"
    headers = {"Authorization": token}
    response = requests.get(url, headers=headers, verify=False)
    response.raise_for_status()
    return response.json()


def set_led_color(device_id: str, token: str, rgb: tuple[int, int, int]):
    url = f"https://api.switch-bot.com/v1.0/devices/{device_id}/commands"
    headers = {"Authorization": token, "Content-Type": "application/json; charset=utf8"}
    r, g, b = rgb
    payload = {
        "command": "setColor",
        "parameter": f"{r}:{g}:{b}",
        "commandType": "command",
    }
    response = requests.post(url, headers=headers, json=payload, verify=False)
    response.raise_for_status()
    return response.status_code, response.text


def turn_off(device_id: str, token: str):
    url = f"https://api.switch-bot.com/v1.0/devices/{device_id}/commands"
    headers = {"Authorization": token, "Content-Type": "application/json; charset=utf8"}
    payload = {"command": "turnOff", "parameter": "default", "commandType": "command"}
    response = requests.post(url, headers=headers, json=payload, verify=False)
    response.raise_for_status()
    return response.status_code, response.text


def set_led_blue_to_red(device_id: str, token: str, level: float):
    """level=0.0で青、1.0で赤"""
    level = max(0.0, min(1.0, level))  # clamp
    r = int(255 * level)
    g = 0
    b = int(255 * (1 - level))
    return set_led_color(device_id, token, (r, g, b))


def set_led(device_id: str, token: str, level: float):
    """0.0=青, 0.5=水色, 1.0=赤"""
    level = max(0.0, min(1.0, level))  # clamp

    if level <= 0.5:
        # 青(0,0,255) → 水色(0,255,255)
        ratio = level / 0.5
        r = 0
        g = int(255 * ratio)
        b = 255
    else:
        # 水色(0,255,255) → 赤(255,0,0)
        ratio = (level - 0.5) / 0.5
        r = int(255 * ratio)
        g = int(255 * (1 - ratio))
        b = int(255 * (1 - ratio))

    return set_led_color(device_id, token, (r, g, b))


if __name__ == "__main__":
    TOKEN = "f26eaa7007230625ffc25ce78cae8444b62036f9bdd9fb43caa31ae873af6021f3cc0a81a16dac9c13247cc03f686915"
    DEVICE_ID = "6055F92A78EE"

    # 例: 紫（中間色）
    # print(set_led_blue_to_red(DEVICE_ID, TOKEN, level=0.0))

    # set_led_color(DEVICE_ID, TOKEN, (255, 255, 255)) # 白
    set_led_color(DEVICE_ID, TOKEN, (1, 1, 1))  # 黒
    turn_off(DEVICE_ID, TOKEN)
