#!/usr/bin/env python3
"""
PLY 文件可视化工具, 写为 ROS2 节点形式，以便自动更新展示新的点云。
功能：
1. 显示点云和夹爪姿态
2. 通过话题触发重新加载可视化（收到消息即刷新点云）

实现思路（参考 ROS1 遗留代码 visual_grasp_from_file01.py）：
- 不使用阻塞式 vis.run()，而是在主线程手写轮询循环：
    poll_events() + update_renderer()，每帧检查队列是否有新的刷新请求。
- 所有 GUI 操作（clear_geometries / add_geometry）都在主线程执行，天然线程安全。
- 话题回调只把文件名放入线程安全队列（发布端 fire-and-forget，无需等待响应）；
  因 ROS2 需要 spin，把 rclpy.spin 放后台线程。

使用方法：
1. 启动节点：python3 visualize_node.py
2. 触发刷新：ros2 topic pub --once /reload_visualization std_msgs/msg/Empty
"""

import os
import sys
import time
import threading
from queue import Queue

import open3d as o3d
import rclpy
from rclpy.node import Node

from std_msgs.msg import Empty


class VisualizeNode(Node):
    def __init__(self, vis_dir):
        super().__init__('visualize_node')

        self.get_logger().info("正在初始化节点...")
        self.point_cloud_filename = os.path.join(vis_dir, "latest_point_cloud.ply")
        self.gripper_filename = os.path.join(vis_dir, "latest_grippers.ply")

        # ---- 刷新请求队列：话题回调线程 put，GUI 主线程 get ----
        self.queue = Queue()
        self.queue.put((self.point_cloud_filename, self.gripper_filename))

        # ---- 初始化可视化窗口 (GUI 必须在主线程) ----
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="GSNet Grasp Visualization", width=1280, height=720)

        # ---- 订阅刷新话题：收到消息即重新加载点云展示 ----
        self.reload_visualization_sub = self.create_subscription(
            Empty, '/reload_visualization', self.reload_visualization_cb, 10)

        self.get_logger().info(
            "可视化节点就绪。触发刷新: ros2 topic pub --once /reload_visualization std_msgs/msg/Empty")


    # ====================================================================
    #  话题回调：运行在后台 spin 线程，只把文件名放入队列（线程安全）
    # ====================================================================
    def reload_visualization_cb(self, msg):
        self.get_logger().info("收到刷新消息，正在刷新点云...")
        self.queue.put((self.point_cloud_filename, self.gripper_filename))

    # ====================================================================
    #  主循环：GUI 主线程轮询队列并刷新
    # ====================================================================
    def run(self):
        while rclpy.ok():
            if not self.queue.empty():
                point_cloud_file, gripper_file = self.queue.get()

                # ---- 读取文件 ----
                geometries = []
                if os.path.exists(point_cloud_file):
                    geometries.append(o3d.io.read_point_cloud(point_cloud_file))
                else:
                    self.get_logger().warn(f"[WARN] 场景点云文件不存在: {point_cloud_file}")

                if os.path.exists(gripper_file):
                    geometries.append(o3d.io.read_triangle_mesh(gripper_file))
                else:
                    self.get_logger().warn(f"[WARN] 夹爪网格文件不存在: {gripper_file}")

                # ---- 清除旧几何体并加入新的（均在主线程，线程安全） ----
                if geometries:
                    self.vis.clear_geometries()
                    for geo in geometries:
                        self.vis.add_geometry(geo)

                    # 加载时设置初始相机视角
                    view_control = self.vis.get_view_control()
                    view_control.set_front([0, 0.7, -1])   # 沿 -Z 看（深度方向向下）
                    view_control.set_up([0, -1, 0])        # RealSense 相机 Y 朝下
                    view_control.set_zoom(0.7)             # 显示全场景

                    self.get_logger().info("点云可视化已刷新。")

            # ---- 驱动 GUI 事件与渲染 ----
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.05) # 影响点云画面的最大帧率

        self.vis.destroy_window()


def main(args=None):
    rclpy.init(args=args)

    VIS_DIR = '/media/gao/2/Grasp/GSNet/detect_ros/vis'
    node = VisualizeNode(VIS_DIR)

    # ROS 自旋放到后台守护线程，保证话题回调可被触发；GUI 轮询循环留在主线程
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        print("[INFO] visualize_node 已退出。")


if __name__ == '__main__':
    print("=" * 50)
    print("  PLY 可视化工具 (ROS2 节点版)")
    print("=" * 50)
    main()
