#!/usr/bin/env python3
from datetime import datetime
import glob
import json
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

MAX_FILES = 100
SAVE_DIR = 'asr_text_data'

class ASRMonitor(Node):
    def __init__(self):
        super().__init__('asr_monitor')
        # 订阅语音识别结果
        self.subscription = self.create_subscription(
            String,
            '/xunfei/aiui_msg',
            self.msgs_callback,
            10
        )
        
        # 发布识别文本给 Coze 处理器
        self.text_publisher = self.create_publisher(
            String,
            '/asr/user_text',
            10
        )
        
        # 防重复机制
        self.last_processed_text = None
        self.last_processed_time = None
        
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.get_logger().info(f"已订阅话题: /xunfei/aiui_msg")
        self.get_logger().info(f"文本将保存到目录: {SAVE_DIR}")
        self.get_logger().info(f"将发布文本到: /asr/user_text")

    def msgs_callback(self, msg: String):
        json_data = None
        try:
            json_data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warn("Received invalid JSON. Skipping.")
            return

        # 清理旧文件
        self.cleanup_old_files()
        
        # 解析并处理JSON数据
        extracted_text = self.try_parse_and_extract_text(json_data)
        
        if extracted_text and self.should_process_text(extracted_text):
            # 保存文本到文件
            self.save_text_to_file(extracted_text)
            
            # 发布文本到 Coze 处理器
            text_msg = String()
            text_msg.data = extracted_text
            self.text_publisher.publish(text_msg)
            
            # 更新最后处理状态
            self.last_processed_text = extracted_text
            self.last_processed_time = self.get_clock().now().nanoseconds
            
            # 打印到控制台以便调试
            print("=" * 50)
            print(f"识别结果: {extracted_text}")
            print("=" * 50)

    def should_process_text(self, text):
        """
        判断是否应该处理这个文本
        """
        # 过滤掉机器人的回答
        robot_responses = ['很抱歉', '无法回答', '对不起', '需要更多', '详细信息', '。']
        if any(robot_response in text for robot_response in robot_responses):
            self.get_logger().info(f"过滤机器人回答: {text}")
            return False
            
        # 防重复：相同文本在3秒内不重复处理
        current_time = self.get_clock().now().nanoseconds
        if (self.last_processed_text == text and 
            self.last_processed_time and 
            current_time - self.last_processed_time < 3e9):  # 3秒内
            self.get_logger().info(f"忽略重复文本: {text}")
            return False
            
        return True

    def try_parse_and_extract_text(self, json_data):
        """
        基于原始代码的解析逻辑提取文本
        """
        if not json_data:
            return None
            
        if "content" not in json_data or "result" not in json_data.get("content"):
            return None

        result = json_data.get("content").get("result")
        if not result:
            return None

        if "cbm_meta" not in result:
            return None

        if not result.get("cbm_meta"):
            return None
            
        cbm_meta = result.get("cbm_meta") 
        if "text" not in cbm_meta:
            return None

        text_data = json.loads(cbm_meta.get("text"))
        if not text_data:
            return None
            
        key = next(iter(text_data))
        if key not in result:
            return None
            
        result_text = result.get(key).get("text") 
        
        try:
            # 尝试解析为JSON
            res_data = json.loads(result_text)
            
            # 如果是JSON，提取query字段（这是用户的实际问题）
            if isinstance(res_data, dict) and 'query' in res_data:
                extracted_text = res_data['query']
                self.get_logger().info(f"提取用户问题: {extracted_text}")
                return extracted_text
            else:
                # 如果是其他JSON结构，不处理
                return None
                
        except json.JSONDecodeError:
            # 如果不是JSON，不处理（避免处理机器人回答）
            return None

    def save_text_to_file(self, text):
        """
        将文本保存到文件
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"asr_result_{timestamp}.txt"
            filepath = os.path.join(SAVE_DIR, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            self.get_logger().info(f"已保存识别文本到: {filepath}")
            
            # 同时保存一个最新的文件，方便Coze访问
            latest_file = os.path.join(SAVE_DIR, "latest_asr_result.txt")
            with open(latest_file, 'w', encoding='utf-8') as f:
                f.write(text)
                
        except Exception as e:
            self.get_logger().error(f"保存文本文件失败: {e}")

    def cleanup_old_files(self):
        """
        清理旧文件，只保留最新的MAX_FILES个文件
        """
        try:
            files = sorted(
                glob.glob(os.path.join(SAVE_DIR, 'asr_result_*.txt')),
                key=os.path.getmtime
            )
            if len(files) > MAX_FILES:
                to_delete = files[:len(files) - MAX_FILES]
                for f in to_delete:
                    try:
                        os.remove(f)
                        self.get_logger().info(f"已删除旧文件: {f}")
                    except Exception as e:
                        self.get_logger().warn(f"删除文件失败 {f}: {e}")
        except Exception as e:
            self.get_logger().error(f"清理旧文件时出错: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ASRMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__': 
    main()
