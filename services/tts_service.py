# -*- coding:utf-8 -*-
import websocket
import datetime
import hashlib
import base64
import hmac
import json
from urllib.parse import urlencode
import ssl
from wsgiref.handlers import format_date_time
from datetime import datetime
from time import mktime
import os
import time
from pathlib import Path

from config import Config


class XF_TTS_Worker:
    def __init__(self, APPID, APIKey, APISecret):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = "ws-api.xfyun.cn"

    def create_url(self):
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = "host: " + self.host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET /v2/tts HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        auth_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(auth_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {"authorization": authorization, "date": date, "host": self.host}
        return url + '?' + urlencode(v)

    def generate(self, text, output_path):
        """同步生成音频文件"""
        self.text = text
        self.output_path = output_path

        # 清除旧文件
        if os.path.exists(output_path):
            os.remove(output_path)

        def on_message(ws, message):
            try:
                message = json.loads(message)
                code = message["code"]
                if code != 0:
                    print(f"TTS Error: {message['message']}")
                    return

                audio = message["data"]["audio"]
                audio = base64.b64decode(audio)
                status = message["data"]["status"]

                with open(self.output_path, 'ab') as f:
                    f.write(audio)

                if status == 2: # 合成结束
                    ws.close()
            except Exception as e:
                print("receive msg,but parse exception:", e)

        def on_error(ws, error):
            print("### TTS WebSocket Error:", error)

        def on_close(ws, close_status_code, close_msg):
            pass

        def on_open(ws):
            d = {
                "common": {"app_id": self.APPID},
                "business": {
                    "aue": "lame", # 使用 lame 生成 mp3
                    "sfl": 1,
                    "auf": "audio/L16;rate=16000",
                    "vcn": "x4_yezi", # 波普先生推荐发音人：叶子（活力）
                    "tte": "utf8"
                },
                "data": {
                    "status": 2,
                    "text": str(base64.b64encode(self.text.encode('utf-8')), "utf-8")
                }
            }
            ws.send(json.dumps(d))

        ws_url = self.create_url()
        ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.on_open = on_open
        # 运行直到合成结束
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return os.path.exists(output_path)


class TTSService:
    """Project-compatible wrapper around the Xunfei TTS worker."""

    def __init__(self):
        self.worker = XF_TTS_Worker(Config.XF_APPID, Config.XF_API_KEY, Config.XF_API_SECRET)
        self.config = {
            "voice": Config.TTS_DEFAULT_VOICE or "x4_yezi",
            "provider": "xf",
        }

    def generate_speech(self, text, output_path):
        if not text or not text.strip():
            return False
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        return self.worker.generate(text.strip(), str(output_file))

    def synthesize(self, text, voice_type=None):
        del voice_type
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = Config.TTS_OUTPUT_DIR / f"tts_{ts}.mp3"
        return str(output_path) if self.generate_speech(text, output_path) else None

    def set_config(self, config):
        self.config.update(config or {})
        return True

    def get_status(self):
        return {"provider": "xf", "status": "ready", "config": dict(self.config)}
