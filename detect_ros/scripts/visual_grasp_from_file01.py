#!/home/baxter/hd1/miniconda3/envs/economic/bin/python

""" Visualize the point cloud and gripper data.
    Author: Mingyu-Liao
"""

import open3d as o3d
import os
import rospy
import threading
from std_msgs.msg import Int8
from queue import Queue

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

class VisualPointCloudNode:
    def __init__(self):
        self.queue = Queue()  # Initialize a queue to hold point cloud and gripper filenames
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='ROS Point Cloud Grasp Viewer', width=800, height=600)
        rospy.init_node('visual_point_cloud01', anonymous=True)  # Initialize the ROS node
        rospy.Subscriber('/grasp_trigger01', Int8, self.visual_start_callback, queue_size=1)  # Subscribe to 'visual_start' topic
        rospy.loginfo("visual_point_cloud node is ready, waiting for 'visual_start' topic...")

    def visual_start_callback(self, msg):
        # Construct the file paths for the point cloud and gripper data
        point_cloud_filename = os.path.join(ROOT_DIR, 'config', 'vis', "latest_point_cloud.ply")
        gripper_filename = os.path.join(ROOT_DIR, 'config', 'vis', "latest_grippers.ply")
        self.queue.put((point_cloud_filename, gripper_filename))  # Add filenames to the queue

    def run_visualizer(self):
        while not rospy.is_shutdown():  # Run until ROS is shut down
            if not self.queue.empty():  # Check if there are items in the queue
                point_cloud_file, gripper_file = self.queue.get()  # Get the filenames from the queue
                cloud = o3d.io.read_point_cloud(point_cloud_file)  # Read the point cloud file
                grippers = o3d.io.read_triangle_mesh(gripper_file)  # Read the gripper file
                self.vis.clear_geometries()  # Clear previous geometries from the visualizer
                self.vis.add_geometry(cloud)  # Add the point cloud geometry
                self.vis.add_geometry(grippers)  # Add the gripper geometry
                
                # Get view control and set camera parameters
                view_control = self.vis.get_view_control()
                view_control.set_front([0, 0, -1])  # Look straight down so the table plane is parallel to the view
                view_control.set_lookat([0, 0, 0.3])  # Set the target position for the camera to look at
                view_control.set_up([0, -1, 0])  # Set the upward direction of the camera
                view_control.set_zoom(0.3)  # Set zoom level of the camera

                # # 获取视图控制对象
                # view_control = self.vis.get_view_control()

                # # 设置摄像机的前视方向为平行于桌面方向，例如从 x 轴看向 y 轴方向
                # view_control.set_front([0, 1, 0])  # 摄像机从 x 轴方向看向目标

                # # 设置摄像机的目标点
                # view_control.set_lookat([0, -1, 0])  # 观察的中心在原点

                # # 设置上方向垂直于桌面，即 z 轴方向
                # view_control.set_up([0, 0, -1])  # 保持 z 轴向上

                # # 调整缩放级别
                # view_control.set_zoom(0.5)

                
                rospy.loginfo("Visualizing point cloud and grippers...")  # Log visualization status

            self.vis.poll_events()  # Poll for events (like window close)
            self.vis.update_renderer()  # Update the renderer to reflect changes
            rospy.sleep(0.1)  # Sleep briefly to reduce CPU usage

    def close(self):
        self.vis.destroy_window()  # Close the visualizer window

if __name__ == '__main__':
    node = VisualPointCloudNode()  # Create an instance of VisualPointCloudNode
    try:
        node.run_visualizer()  # Start visualizing point clouds
    except rospy.ROSInterruptException:
        node.close()  # Ensure window is closed on ROS interrupt
