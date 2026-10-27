#!/home/gao/miniconda3/envs/GSNet/bin/python

# 针对透明物体的抓取姿态检测

""" Grasp detection based on ROS2 (Python).
    针对透明物体的抓取姿态检测，配合 depth_completion 节点使用。
    订阅 depth_completion 节点发布的补全后 RGBD 图像，
    当收到同一时间戳的 RGB + 深度图像对后自动运行 GSNet 推理，发布抓取姿态。
    (无独立触发服务，深度补全完成即自动检测)
    检测完成后发布 /reload_visualization (std_msgs/msg/Empty)，
    通知 visualize_node 节点重新加载点云和抓取姿态可视化。
"""

import os
import sys
import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Int8, Float32, Empty

from geometry_msgs.msg import PoseStamped, Point, Quaternion
from sensor_msgs.msg import Image


import torch
from graspnetAPI import GraspGroup
# ======================== 路径修正：指向 GSNet 根目录 ========================
GSNET_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, GSNET_ROOT)
sys.path.append(os.path.join(GSNET_ROOT, 'utils'))


from models.graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from dataset.graspnet_dataset import minkowski_collate_fn
from transformations import quaternion_from_matrix


class GraspDetection(Node):
    def __init__(self, checkpoint_path, grasp_poses_num=1,
                 num_point=20000, collision_thresh=0.01, voxel_size=0.005):
        super().__init__('grasp_detection')

        # ---- 参数 ----
        self.checkpoint_path = checkpoint_path
        self.grasp_poses_num = grasp_poses_num
        self.num_point = num_point
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size

        # ---- 可视化文件保存路径 ----
        vis_dir = os.path.join(GSNET_ROOT, 'detect_ros', 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        self.point_cloud_filename = os.path.join(vis_dir, "latest_point_cloud.ply")
        self.gripper_filename = os.path.join(vis_dir, "latest_grippers.ply")

        # ---- 图像缓存 ----
        self.img_alignedDepth = None
        self.img_rgb = None
        # 各自图像的时间戳 (depth_completion 节点发布同一帧 RGB/深度时使用相同时间戳)
        self.rgb_stamp = None
        self.depth_stamp = None
        # 已处理过的时间戳 (避免同一帧重复检测)
        self.last_processed_stamp = None
        # 检测进行中标志 (防止并发推理)
        self._detecting = False

        # ---- 可重入回调组: 允许 RGB/深度图像回调并发处理 ----
        self.cb_group = ReentrantCallbackGroup()

        # ---- 加载模型（耗时操作，在订阅前完成） ----
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self._load_net()

        # ======================== 订阅 (触发入口: 收到补全后的 RGBD 图像即自动检测) ========================
        self.completed_depth_topic = '/depth_completion/aligned_depth_to_color/image_raw'
        self.completed_rgb_topic = '/depth_completion/color/image_raw'
        self.create_subscription(
            Image, self.completed_depth_topic,
            self._cb_depth, 10, callback_group=self.cb_group)
        self.create_subscription(
            Image, self.completed_rgb_topic,
            self._cb_rgb, 10, callback_group=self.cb_group)

        # ======================== 发布 ========================
        # 向grasp_policy发布最优抓取姿态 (PoseStamped)
        self.grasp_pose_pub = self.create_publisher(PoseStamped, '/grasp_pose', 1)
        # 向grasp_policy发布最优抓取宽度 (Float32, 单位: 米)
        self.grasp_width_pub = self.create_publisher(Float32, '/grasp_width', 1)

        # 向visualize_node发布空消息，用于通知可视化节点刷新点云和抓取姿态可视化
        self.reload_visualization_pub = self.create_publisher(Empty, '/reload_visualization', 10)

        self.get_logger().info("======Transparent Object Grasp Detection Node Initialized ======")
        self.get_logger().info(f"  Device      : {self.device}")
        self.get_logger().info(f"  Checkpoint  : {self.checkpoint_path}")
        self.get_logger().info(f"  Num points  : {self.num_point}")
        self.get_logger().info("触发方式: 收到depth_completion节点补全后的 RGBD 图像后自动检测")
        self.get_logger().info("触发命令: ros2 service call /depth_completion std_srvs/srv/Trigger")
        self.get_logger().info(f"订阅补全后深度: {self.completed_depth_topic}")
        self.get_logger().info(f"订阅补全后RGB : {self.completed_rgb_topic}")
        self.get_logger().info("===============================================")

    # ====================================================================
    #  图像转换：代替 cv_bridge，手动将 sensor_msgs/Image 转为 numpy
    # ====================================================================
    @staticmethod
    def _imgmsg_to_cv2(msg, encoding):
        """手动将 sensor_msgs/Image 转换为 numpy 数组，代替 cv_bridge"""
        h, w = msg.height, msg.width
        if encoding == "16UC1":
            dtype = np.uint16
            channels = 1
        elif encoding in ("bgr8", "rgb8"):
            dtype = np.uint8
            channels = 3
        elif encoding == "8UC1":
            dtype = np.uint8
            channels = 1
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")

        if channels > 1:
            img = np.frombuffer(msg.data, dtype=dtype).reshape(h, w, channels)
        else:
            img = np.frombuffer(msg.data, dtype=dtype).reshape(h, w)
        return img.copy()

    # ====================================================================
    #  回调：缓存图像
    # ====================================================================
    def _cb_depth(self, msg):
        """缓存补全后的深度图 (16UC1, 单位: mm)，RGB/深度到齐后自动触发检测"""
        try:
            self.img_alignedDepth = self._imgmsg_to_cv2(msg, "16UC1")
            self.depth_stamp = msg.header.stamp
            self._maybe_run_detection()
        except Exception as e:
            self.get_logger().error(f"Depth convert error: {e}")

    def _cb_rgb(self, msg):
        """缓存补全后的 RGB 图 (rgb8 -> 归一化到 0~1, RGB 格式)，RGB/深度到齐后自动触发检测"""
        try:
            img = self._imgmsg_to_cv2(msg, "rgb8")
            self.img_rgb = img.astype(np.float32) / 255.0
            self.rgb_stamp = msg.header.stamp
            self._maybe_run_detection()
        except Exception as e:
            self.get_logger().error(f"RGB convert error: {e}")

    # ====================================================================
    #  模型加载
    # ====================================================================
    def _load_net(self):
        self.get_logger().info("Loading GSNet model...")
        net = GraspNet(seed_feat_dim=512, is_training=False)
        net.to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        net.load_state_dict(checkpoint['model_state_dict'])
        self.get_logger().info(f"Loaded checkpoint (epoch {checkpoint['epoch']})")
        net.eval()
        return net

    # ====================================================================
    #  自动触发: 收到同一时间戳的补全后 RGBD 图像对后进行检测
    # ====================================================================
    def _maybe_run_detection(self):
        """
        检查 RGB 和深度图是否已配对 (同一时间戳) 且未处理过，
        满足条件则自动运行一次抓取位姿检测。
        depth_completion 节点发布同一帧 RGB/深度时使用相同时间戳。
        """
        if self._detecting:
            return
        if self.img_rgb is None or self.img_alignedDepth is None:
            return
        if self.rgb_stamp is None or self.depth_stamp is None:
            return
        if self.rgb_stamp != self.depth_stamp:
            # RGB 和深度尚未配对成同一帧
            return
        if self.rgb_stamp == self.last_processed_stamp:
            # 该帧已处理过
            return

        self._detecting = True
        try:
            self.get_logger().info("收到补全后的 RGBD 图像，自动开始抓取位姿检测...")
            self._run_detection()
        except Exception as e:
            self.get_logger().error(f"抓取位姿检测失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._detecting = False

    # ====================================================================
    #  数据获取与预处理
    # ====================================================================
    def _get_and_process_data(self):
        """
        获取 RGBD 图像 → 生成点云 → 采样 → 构造网络输入字典
        返回: (data_input_dict, cloud_o3d) 或 (None, None)
        """
        if self.img_rgb is None or self.img_alignedDepth is None:
            self.get_logger().warn("未收到 RGB 或深度图像，本次检测取消！")
            return None, None

        color = self.img_rgb.copy()
        depth = self.img_alignedDepth.copy()
        # 订阅的是 rgb8 编码图像，已经是 RGB 格式，无需再转换

        # ---- RealSense D455 内参 (640x480) ----
        # 如果相机分辨率改变，需要同步更新这里的内参和宽高
        intrinsic = np.array([[384.7211, 0.0, 312.9769], [0.0, 384.7211, 245.0151], [0.0, 0.0, 1.0]]) # D455 640x480
        factor_depth = 1000.0   # 深度值除以该因子得到米
        # width, height = 640.0, 480.0
        # fx, fy = 384.7211, 384.7211
        # cx, cy = 312.9769, 245.0151

        camera = CameraInfo(
                    width=640.0, height=480.0,
                    fx=intrinsic[0][0], fy=intrinsic[1][1],
                    cx=intrinsic[0][2], cy=intrinsic[1][2],
                    scale=factor_depth
                    )
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # ---- 工作空间遮罩 (裁剪无效区域) ----
        workspace_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=bool)
        # 根据实际场景调整 ROI 范围
        workspace_mask[30:460, 50:480] = True
        mask = workspace_mask & (depth > 0)

        cloud_masked = cloud[mask]
        color_masked = color[mask]

        if len(cloud_masked) == 0:
            self.get_logger().warn("遮罩区域内无有效点云，跳过。")
            return None, None

        # ---- 点云采样 (模型固定接收 num_point 个点) ----
        n = len(cloud_masked)
        if n >= self.num_point:
            idxs = np.random.choice(n, self.num_point, replace=False)
        else:
            idxs1 = np.arange(n)
            idxs2 = np.random.choice(n, self.num_point - n, replace=True)
            idxs = np.concatenate([idxs1, idxs2])

        cloud_sampled = cloud_masked[idxs].astype(np.float32)

        # ---- 构造网络输入字典 (保持 numpy 类型，minkowski_collate_fn 内部会转 tensor) ----
        data_input = {
            'point_clouds': cloud_sampled,
            'coors': cloud_sampled / self.voxel_size,
            'feats': np.ones_like(cloud_sampled),
        }

        # ---- 构建 Open3D 点云用于可视化和保存 ----
        cloud_o3d = o3d.geometry.PointCloud()
        cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

        return data_input, cloud_o3d

    # ====================================================================
    #  GSNet 推理
    # ====================================================================
    def _run_inference(self, data_input):
        """运行 GSNet 前向推理，返回 GraspGroup"""
        # 必须经过 minkowski_collate_fn 处理，将 numpy 数据转为稀疏张量 batch
        batch_data = minkowski_collate_fn([data_input])
        for key in batch_data:
            if isinstance(batch_data[key], list):
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                batch_data[key] = batch_data[key].to(self.device)

        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = pred_decode(end_points)

        preds = grasp_preds[0].detach().cpu().numpy()
        return GraspGroup(preds)

    # ====================================================================
    #  碰撞检测与后处理
    # ====================================================================
    def _postprocess(self, gg, cloud_np):
        """NMS → 碰撞过滤 → 按分数排序"""
        gg.nms()

        if self.collision_thresh > 0:
            collision_detector = ModelFreeCollisionDetector(cloud_np, voxel_size=0.01)
            collision_mask = collision_detector.detect(
                gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
            gg = gg[~collision_mask]

        gg.sort_by_score()
        return gg

    # 旋转矩阵转换为四元数
    def quaternion_from_rotation(self, rot):
        """
        return:
            四元数
        """
        M = np.identity(4)
        M[:3, :3] = rot
        # return transformations.quaternion_from_matrix(self.rotate_mat)
        return quaternion_from_matrix(M)

    # ====================================================================
    #  发布抓取姿态 (PoseStamped + 宽度)
    # ====================================================================
    def _publish_grasp_pose(self, gg):
        """发布得分最高的抓取姿态"""
        if len(gg) == 0:
            self.get_logger().warn("无有效抓取姿态可发布。")
            return

        top = gg[0]
        center = top.translation
        rot = top.rotation_matrix

        # 旋转矩阵 → 四元数
        qx, qy, qz, qw = self.quaternion_from_rotation(rot)

        # PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'camera_color_optical_frame'
        pose_msg.pose.position = Point(x=float(center[0]), y=float(center[1]), z=float(center[2]))
        pose_msg.pose.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
        self.grasp_pose_pub.publish(pose_msg)

        # 抓取宽度
        width_msg = Float32()
        width_msg.data = float(top.width)
        self.grasp_width_pub.publish(width_msg)

        self.get_logger().info(
            f"Best grasp: pos=[{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}] "
            f"width={top.width:.4f} score={top.score:.4f}")
        

    # ====================================================================
    #  保存抓取结果到文件 (供外部可视化节点读取)
    # ====================================================================
    def _save_results(self, gg, cloud_o3d):
        gg_top = gg[:self.grasp_poses_num]
        grippers = gg_top.to_open3d_geometry_list()

        o3d.io.write_point_cloud(self.point_cloud_filename, cloud_o3d)
        self.get_logger().info(f"Point cloud saved → {self.point_cloud_filename}")

        combined_grippers = o3d.geometry.TriangleMesh()
        for g in grippers:
            combined_grippers += g
        o3d.io.write_triangle_mesh(self.gripper_filename, combined_grippers)
        self.get_logger().info(f"Grippers saved → {self.gripper_filename}")

    # ====================================================================
    #  抓取位姿检测主流程
    # ====================================================================
    def _run_detection(self):
        """数据预处理 → GSNet 推理 → 后处理 → 发布抓取姿态 → 保存可视化"""
        # 1. 获取并预处理数据
        data_input, cloud_o3d = self._get_and_process_data()
        if data_input is None:
            self.get_logger().error("数据获取失败（未收到 RGB 或深度图像）")
            return

        # 2. GSNet 推理
        self.get_logger().info("Running GSNet inference...")
        gg = self._run_inference(data_input)
        self.get_logger().info(f"Raw grasps: {len(gg)}")

        # 3. 后处理 (NMS + 碰撞过滤 + 排序)
        gg = self._postprocess(gg, np.asarray(cloud_o3d.points))
        self.get_logger().info(f"Valid grasps after filtering: {len(gg)}")

        if len(gg) == 0:
            self.get_logger().warn("未检测到有效抓取姿态")
            return

        # 4. 发布最优抓取姿态 (供运动节点订阅)
        self._publish_grasp_pose(gg)

        # 5. 发布可视化 + 保存文件
        self._save_results(gg, cloud_o3d)

        # 6. 输出关键结果
        top = gg[0]
        cx, cy, cz = top.translation
        rot = top.rotation_matrix
        # 旋转矩阵 → 四元数
        qx, qy, qz, qw = self.quaternion_from_rotation(rot)

        result_str = (
            f"score={top.score:.4f}, "
            f"pos=[{cx:.4f}, {cy:.4f}, {cz:.4f}], "
            f"ori=[{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}], "
            f"width={top.width:.4f}, "
            f"total_valid={len(gg)}"
        )

        self.get_logger().info(f"检测完成: {result_str}")
        # 发布刷新消息，通知可视化节点重新加载点云和抓取姿态可视化
        # 话题为 fire-and-forget，publish() 立即返回。
        self.reload_visualization_pub.publish(Empty())

        self.get_logger().info("已发布 /reload_visualization 消息，刷新点云和抓取姿态。")
        self.get_logger().info("等待补全后的 RGBD 图像...")
        self.get_logger().info("=" * 50)


def main(args=None):
    rclpy.init(args=args)

    # ---- 配置路径 ----
    checkpoint_path = os.path.join(
        GSNET_ROOT,
        'np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar')
    grasp_poses_num = 5

    node = None
    try:
        node = GraspDetection(
            checkpoint_path=checkpoint_path,
            grasp_poses_num=grasp_poses_num,
            num_point=20000,
            collision_thresh=0,
            voxel_size=0.005,
        )
        # 多线程执行器: 检测耗时较长，阻塞在某个图像回调中运行推理时，
        # 另一个图像回调 (可重入回调组) 仍可正常处理和缓存图像
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
