#!/usr/bin/env python3
import json
import os
import rclpy
import glob
import logging
import requests
import pyaudio
import wave
from rclpy.node import Node
from std_msgs.msg import String
from datetime import datetime
from config import Config

logger = logging.getLogger("ASR")

# 路径对齐
SAVE_DIR = Config.ASR_SAVE_DIR
MAX_FILES = 50

class ASRService:
    """听觉识别服务 (含云端/本地)"""
    def __init__(self):
        logger.info("🎙️ [INIT] 正在初始化 ASR 服务...")
        self.api_config = {
            "baidu": {
                "app_id": Config.BAIDU_ASR_APP_ID,
                "api_key": Config.BAIDU_ASR_API_KEY,
                "secret_key": Config.BAIDU_ASR_SECRET_KEY
            },
            "tencent": {
                "app_id": Config.TENCENT_ASR_APP_ID,
                "secret_id": Config.TENCENT_ASR_SECRET_ID,
                "secret_key": Config.TENCENT_ASR_SECRET_KEY
            }
        }
        self.recognition_mode = "cloud"  # cloud or local
        self.is_listening = False
        self.audio_buffer = []
        
        # 初始化音频采集
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
    
    def start_listening(self):
        """开始语音监听"""
        logger.info("🎧 [ASR] 开始语音监听")
        self.is_listening = True
        self.audio_buffer = []
        
        # 开始音频采集
        try:
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=1024
            )
            logger.info("✅ [ASR] 音频采集已启动")
        except Exception as e:
            logger.error("❌ [ASR] 音频采集启动失败: %s", e)
        
        return True
    
    def stop_listening(self):
        """停止监听并返回识别结果"""
        logger.info("🛑 [ASR] 停止语音监听")
        self.is_listening = False
        
        # 停止音频采集
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        # 保存音频文件
        audio_file = os.path.join(SAVE_DIR, f"audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
        self._save_audio(audio_file)
        
        # 识别语音
        text = self._recognize_speech(audio_file)
        
        # 清理
        self.cleanup()
        
        logger.info("🎤 [ASR] 识别结果: %s", text)
        return text
    
    def _save_audio(self, output_file):
        """保存音频到文件"""
        try:
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(self.pyaudio_instance.get_sample_size(pyaudio.paInt16))
                wf.setframerate(16000)
                wf.writeframes(b''.join(self.audio_buffer))
            logger.info("💾 [ASR] 音频已保存: %s", output_file)
        except Exception as e:
            logger.error("❌ [ASR] 保存音频失败: %s", e)
    
    def _recognize_speech(self, audio_file):
        """识别语音"""
        if self.recognition_mode == "cloud":
            return self._recognize_cloud(audio_file)
        else:
            return self._recognize_local(audio_file)
    
    def _recognize_cloud(self, audio_file):
        """使用云API识别语音"""
        try:
            # 这里使用百度语音API作为示例
            # 实际项目中需要根据百度语音API文档实现具体调用
            logger.info("☁️ [ASR] 使用云API识别语音")
            # 模拟识别结果
            return "这是云API识别的结果"
        except Exception as e:
            logger.error("❌ [ASR] 云API识别失败: %s", e)
            # 切换到本地模型
            self.switch_recognition_mode("local")
            return self._recognize_local(audio_file)
    
    def _recognize_local(self, audio_file):
        """使用本地模型识别语音"""
        try:
            logger.info("🖥️ [ASR] 使用本地模型识别语音")
            # 这里可以集成本地ASR模型，如VOSK等
            # 模拟识别结果
            return "这是本地模型识别的结果"
        except Exception as e:
            logger.error("❌ [ASR] 本地模型识别失败: %s", e)
            return "识别失败，请重试"
    
    def set_api_config(self, config):
        """设置云API配置"""
        try:
            self.api_config.update(config)
            logger.info("🔧 [ASR] API配置已更新")
            return True
        except Exception as e:
            logger.error("❌ [ASR] API配置更新失败: %s", e)
            return False
    
    def switch_recognition_mode(self, mode):
        """切换识别模式"""
        if mode in ["cloud", "local"]:
            self.recognition_mode = mode
            logger.info("🔄 [ASR] 识别模式已切换到: %s", mode)
            return True
        else:
            logger.warning("⚠️ [ASR] 无效的识别模式: %s", mode)
            return False
    
    def is_listening_status(self):
        """获取监听状态"""
        return self.is_listening
    
    def process_audio(self, audio_data):
        """处理音频数据"""
        if self.is_listening:
            self.audio_buffer.append(audio_data)
    
    @staticmethod
    def extract_text(msg_data: str):
        """解析讯飞 AIUI 协议中的文本"""
        try:
            data = json.loads(msg_data)
            content = data.get("content", {}).get("result", {})
            if "intent" in content:
                return content["intent"].get("text")
            
            cbm_meta = content.get("cbm_meta", {})
            if "text" in cbm_meta:
                sub_text = json.loads(cbm_meta.get("text", "{}"))
                key = next(iter(sub_text)) if sub_text else None
                if key:
                    return json.loads(content.get(key, {}).get("text", "{}")).get('query')
            return None
        except Exception:
            return None
    
    def cleanup(self):
        """清理临时文件"""
        files = sorted(glob.glob(os.path.join(SAVE_DIR, "*.wav")), key=os.path.getmtime)
        while len(files) > MAX_FILES: 
            os.remove(files.pop(0))
        
        files = sorted(glob.glob(os.path.join(SAVE_DIR, "*.txt")), key=os.path.getmtime)
        while len(files) > MAX_FILES: 
            os.remove(files.pop(0))

class ASRMonitor(Node):
    def __init__(self):
        super().__init__('asr_monitor')
        self.subscription = self.create_subscription(String, '/xunfei/aiui_msg', self.msgs_callback, 10)
        self.text_publisher = self.create_publisher(String, '/asr/user_input', 10)
        
        self.last_processed_text = None
        self.last_processed_time = None
        self.asr_service = ASRService()

        os.makedirs(SAVE_DIR, exist_ok=True)
        logger.info("🎙️ [ASR] 节点启动，结果保存路径: %s", SAVE_DIR)

    def msgs_callback(self, msg: String):
        try:
            text = ASRService.extract_text(msg.data)
            
            if text and self.is_new_input(text):
                logger.info("🎧 [ASR] 拾音成功: %s", text)
                
                # 保存文本
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                with open(os.path.join(SAVE_DIR, f"asr_{ts}.txt"), 'w', encoding='utf-8') as f:
                    f.write(text)

                # 发布给大脑
                self.text_publisher.publish(String(data=text))
                self.last_processed_text = text
                self.last_processed_time = self.get_clock().now().nanoseconds
                self.asr_service.cleanup()
        except Exception as e:
            logger.error("❌ [ASR] 解析错误: %s", e)

    def is_new_input(self, text):
        if len(text) < 2: return False
        now = self.get_clock().now().nanoseconds
        if text == self.last_processed_text and (now - (self.last_processed_time or 0)) < 3e9:
            return False
        return True

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ASRMonitor())
    rclpy.shutdown()

if __name__ == '__main__':
    main()

