#!/usr/bin/env python3
import rclpy
import cv2
import os
import logging
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from datetime import datetime
from config import Config

logger = logging.getLogger("Vision")

class ImageService:
    def __init__(self):
        self.bridge = CvBridge()
        self.save_dir = Config.VISION_SAVE_DIR

    def save_image(self, msg, filename="latest.jpg"):
        """将 ROS Image 消息转换为 OpenCV 格式并保存"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            save_path = os.path.join(self.save_dir, filename)
            cv2.imwrite(save_path, cv_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return save_path
        except Exception as e:
            logger.error("❌ [Vision] ImageService 保存失败: %s", e)
            return None

class RvizImageCaptureNode(Node):
    def __init__(self):
        super().__init__('rviz_image_capture_node')
        self.image_service = ImageService()

        # 订阅虚拟摄像头话题
        self.subscription = self.create_subscription(
            Image,
            'virtual_camera/image_raw',
            self.image_callback,
            10)

        logger.info("🎯 [Vision] RViz2 图像捕捉节点已启动")
        self.save_count = 0

    def image_callback(self, msg):
        """保存接收到的图像"""
        save_path = self.image_service.save_image(msg)
        if save_path:
            self.save_count += 1
            if self.save_count % 10 == 0:
                logger.info("📸 [Vision] 已保存 %d 帧图像 (文件: %s)", self.save_count, save_path)

def main():
    rclpy.init()
    node = RvizImageCaptureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.info("🛑 [Vision] 图像捕捉节点已关闭")
        logger.info("📸 [Vision] 总计保存 %d 帧图像", node.save_count)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
