#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import os
import json
import threading
from datetime import datetime
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

# 路径对齐
from qwen_vl import QwenVLModel
from tts_ws import XF_TTS_Worker

class StreamingPopProcessor(Node):
    def __init__(self):
        super().__init__('streaming_pop_processor')
        
        # 1. 初始化模型与TTS
        self.model = QwenVLModel()
        # ！！！请务必填入真实的讯飞凭据！！！
        self.tts = XF_TTS_Worker(APPID=os.getenv("XF_APPID", ""), APIKey=os.getenv("XF_API_KEY", ""), APISecret=os.getenv("XF_API_SECRET", ""))
        
        # 2. 路径配置
        self.image_dir = os.path.join(os.getcwd(), "rviz_captured_images")
        # 移除固定的 latest.jpg，每次生成带时间戳的文件名
        self.latest_image_path = None
        # 核心修复：确保 audio_dir 是绝对路径
        self.audio_dir = os.path.abspath(os.path.join(os.getcwd(), "data", "audio_out"))
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        # 3. 初始化图像捕捉相关组件
        self.bridge = CvBridge()
        self.image_subscription = None
        self.current_cv_image = None
        self.image_capture_enabled = False
        self.save_count = 0
        
        # 4. 记忆管理
        self.chat_history = [] # 用于保存上下文
        self.max_history = 10  # 最多保留5轮对话(10条记录)

        # 5. ROS 通信
        self.sub = self.create_subscription(String, '/asr/user_input', self.on_input, 10)
        self.tts_pub = self.create_publisher(String, '/xunfei/tts_play', 10)  # 修复这里！！！
        
        # 6. 其他初始化
        self.sentence_seps = ["。", "！", "？", "\n", "；", "!", "?", "..."]
        
        # 7. 启动图像捕捉（直接开启，不询问）
        self.start_image_capture()
        
        self.get_logger().info(f"✅ [波普先生] 语音存放路径: {self.audio_dir}")
        self.get_logger().info(f"✅ [波普先生] 图像存放路径: {self.image_dir}")
        self.get_logger().info("✅ [波普先生] 已上线，记忆系统已启动。")
        self.get_logger().info("✅ [波普先生] 图像捕捉已直接启动，无需询问。")

    def start_image_capture(self):
        """启动图像捕捉订阅"""
        if self.image_subscription is None:
            self.image_subscription = self.create_subscription(
                Image,
                '/camera/color/image_raw',
                self.image_callback,
                10
            )
            self.image_capture_enabled = True
            self.get_logger().info("📸 图像捕捉已启动")

    def stop_image_capture(self):
        """停止图像捕捉订阅"""
        if self.image_subscription is not None:
            self.destroy_subscription(self.image_subscription)
            self.image_subscription = None
            self.image_capture_enabled = False
            self.current_cv_image = None
            self.get_logger().info("📸 图像捕捉已停止")

    def image_callback(self, msg):
        """接收并缓存图像，不直接保存"""
        try:
            # 将ROS Image消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            # 转换RGB到BGR用于保存
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.current_cv_image = cv_image_bgr
            self.save_count += 1    
        except Exception as e:
            self.get_logger().error(f'❌ 图像转换失败: {str(e)}')

    def capture_current_image(self):
        """捕获当前帧的图像并保存为带时间戳的文件"""
        if self.current_cv_image is not None:
            try:
                # 生成带时间戳的文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒级时间戳
                filename = f"capture_{timestamp}.jpg"
                self.latest_image_path = os.path.join(self.image_dir, filename)
                
                # 保存图像
                cv2.imwrite(self.latest_image_path, self.current_cv_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.get_logger().info(f"📸 已捕获并保存图像: {self.latest_image_path}")
                return True
            except Exception as e:
                self.get_logger().error(f"❌ 图像捕获失败: {e}")
                return False
        else:
            self.get_logger().warning("⚠️ 当前无可用图像")
            return False

    def get_latest_image_with_timestamp(self):
        """获取最新保存的带时间戳的图像"""
        try:
            # 查找最新的图像文件
            image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')]
            if not image_files:
                return None, None
            
            # 按修改时间排序，获取最新的
            image_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.image_dir, x)), reverse=True)
            latest_file = os.path.join(self.image_dir, image_files[0])
            
            # 读取图像数据
            with open(latest_file, "rb") as f:
                image_data = f.read()
            
            return image_data, latest_file
        except Exception as e:
            self.get_logger().error(f"❌ 获取历史图像失败: {e}")
            return None, None

    def on_input(self, msg):
        user_text = msg.data.strip()
        if not user_text: return
        
        print(f"\n🎧 [ASR 输入]: {user_text}")
        
        # 直接捕获当前图像（不询问）
        image_data = None
        capture_image = False
        
        if self.capture_current_image():
            # 捕获成功，读取图像数据
            try:
                with open(self.latest_image_path, "rb") as f:
                    image_data = f.read()
                capture_image = True
                self.get_logger().info(f"✅ 已捕获当前图像: {self.latest_image_path}")
            except Exception as e:
                self.get_logger().error(f"❌ 读取图像文件失败: {e}")
        else:
            # 如果当前没有图像，尝试使用历史图像（最新带时间戳的）
            image_data, image_path = self.get_latest_image_with_timestamp()
            if image_data and image_path:
                capture_image = True
                self.get_logger().info(f"✅ 使用历史图像: {image_path}")

        current_sentence = ""
        full_reply = ""
        print("🤖 [波普先生回复]: ", end="", flush=True)

        # --- 带上下文的流式调用 ---
        if capture_image and image_data:
            self.get_logger().info("🖼️ 正在发送图像和文本给大模型...")
            generator = self.model.identify_product_stream(image_data, user_text, history=self.chat_history)
        else:
            self.get_logger().info("📝 无可用图像，仅发送文本给大模型...")
            generator = self.model.identify_product_stream(None, user_text, history=self.chat_history)

        for chunk in generator:
            print(chunk, end="", flush=True)
            current_sentence += chunk
            full_reply += chunk

            

        # --- 更新记忆 ---
        self.chat_history.append({"role": "user", "content": user_text})
        self.chat_history.append({"role": "assistant", "content": full_reply})
        # 保持记忆长度
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def run_tts_and_play(self, text):
        """合成 MP3 并按照机器人规范发布指令"""
        try:
            # 1. 生成文件名
            ts = datetime.now().strftime("%H%M%S_%f")
            audio_path = os.path.join(self.audio_dir, f"pop_{ts}.mp3")
            # 2. TTS 合成文件
            self.tts.generate(text, audio_path)
            # 3. 检查文件是否生成成功
            if os.path.exists(audio_path):
                # 4. 按照机器人文档要求的格式构建 JSON
                # 统一使用 append 模式实现流式队列播放
                play_cmd = {
                    "cmd": "append",
                    "file": audio_path
                }
                msg = String()
                msg.data = json.dumps(play_cmd)
                self.tts_pub.publish(msg)
                # self.get_logger().info(f"🔊 已发送播放指令: {audio_path}")
            else:
                self.get_logger().error(f"❌ TTS文件生成失败: {audio_path}")

        except Exception as e:
            self.get_logger().error(f"❌ 语音播报逻辑异常: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = StreamingPopProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 确保停止图像捕捉
        if node.image_capture_enabled:
            node.stop_image_capture()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
