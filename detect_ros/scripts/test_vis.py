#!/usr/bin/env python3
"""
PLY 文件可视化工具 (基于 ROS 消息触发重载)
"""

import os
from queue import Queue
import open3d as o3d
import rospy
from std_msgs.msg import Empty  # 或保留 Int8


class VisualPointCloudNode:
    def __init__(self):
        # 队列用于跨线程传递重载信号
        self.queue = Queue()

        # 固定路径配置
        self.vis_dir = '/media/gao/2/Grasp/GSNet/detect_ros/vis'
        self.pc_path = os.path.join(self.vis_dir, "latest_point_cloud.ply")
        self.gripper_path = os.path.join(self.vis_dir, "latest_grippers.ply")

        # 初始化 Open3D 窗口
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="GSNet Grasp Visualization", width=1280, height=720)

        # 初始化 ROS 节点与订阅
        rospy.init_node('grasp_visualizer_node', anonymous=True)
        # 订阅触发话题，这里使用 Empty 消息类型（任何发送都会触发刷新）
        rospy.Subscriber('/grasp_trigger', Empty, self.trigger_callback, queue_size=1)

        # 启动时默认先加载一次当前文件
        self.queue.put(True)
        rospy.loginfo("可视化节点就绪，等待 '/grasp_trigger' 话题触发...")

    def trigger_callback(self, msg):
        rospy.loginfo("收到重载信号，准备更新点云...")
        self.queue.put(True)

    def reload_and_render(self):
        # 检查文件是否存在
        if not os.path.exists(self.pc_path):
            rospy.logwarn(f"点云文件不存在: {self.pc_path}")
            return

        # 读取点云
        cloud = o3d.io.read_point_cloud(self.pc_path)

        # 清理旧几何体并添加新数据
        self.vis.clear_geometries()
        self.vis.add_geometry(cloud)

        # 加载夹爪（如果存在）
        if os.path.exists(self.gripper_path):
            gripper_mesh = o3d.io.read_triangle_mesh(self.gripper_path)
            self.vis.add_geometry(gripper_mesh)

        # 设置与你原始代码一致的俯视视角
        ctr = self.vis.get_view_control()
        ctr.set_front([0, 0.7, -1])  # 相机朝向：沿 -Z 看 (深度方向向下)
        ctr.set_up([0, -1, 0])      # 上方向：-Y (RealSense 坐标系 Y 朝下)
        ctr.set_zoom(0.7)

        rospy.loginfo("点云与夹爪已重新加载！")

    def run(self):
        while not rospy.is_shutdown():
            if not self.queue.empty():
                self.queue.get()
                self.reload_and_render()

            # 驱动窗口响应交互
            self.vis.poll_events()
            self.vis.update_renderer()
            rospy.sleep(0.05)

        self.vis.destroy_window()


if __name__ == '__main__':
    node = VisualPointCloudNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass