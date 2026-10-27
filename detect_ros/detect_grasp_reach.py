#!/home/gao/miniconda3/envs/GSNet/bin/python

""" Grasp detection based on ROS2 (Python).
    订阅 RealSense D455 的 RGBD 图像，运行 GSNet 推理，发布抓取姿态。
    新增一个功能，发布话题通知 visualize_node 节点重新加载点云和抓取姿态可视化
    话题名称: /reload_visualization (std_msgs/msg/Empty)
"""

import os
import sys
import struct
import numpy as np
import open3d as o3d
import cv2

import rclpy
from rclpy.node import Node
import tf2_ros
from std_msgs.msg import Int8, Float32, Empty
from std_srvs.srv import Trigger

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
from transformations import quaternion_from_matrix, quaternion_matrix


class GraspDetection(Node):
    def __init__(self, checkpoint_path, grasp_poses_num=1,
                 num_point=20000, collision_thresh=0.01, voxel_size=0.005,
                 topdown_cone_deg=30.0, side_reject_cone_deg=70.0):
        super().__init__('grasp_detection_reach')

        # ---- 参数 ----
        self.checkpoint_path = checkpoint_path
        self.grasp_poses_num = grasp_poses_num
        self.num_point = num_point
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size

        # ---- 可达性预筛参数 (基于机械臂基座方位的接近方向几何过滤) ----
        # topdown_cone_deg    : 接近方向与竖直向下夹角小于此值 => 视为顶视抓取, 直接保留
        # side_reject_cone_deg: 侧向抓取时, 接近方向水平分量与"物体->基座"方向夹角
        #                       小于此值 => 判定为从远离基座一侧反向夹取, 予以剔除
        self.topdown_cone_deg = topdown_cone_deg
        self.side_reject_cone_deg = side_reject_cone_deg
        self._cam2base_cache = None  # 静态手眼标定, 缓存 (R_base_camera, t_base_camera)

        # ---- TF2 (查询 link_base <- camera_color_optical_frame) ----
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ---- 可视化文件保存路径 ----
        vis_dir = os.path.join(GSNET_ROOT, 'detect_ros', 'vis')
        os.makedirs(vis_dir, exist_ok=True)
        self.point_cloud_filename = os.path.join(vis_dir, "latest_point_cloud.ply")
        self.gripper_filename = os.path.join(vis_dir, "latest_grippers.ply")

        # ---- 图像缓存 ----
        self.img_alignedDepth = None
        self.img_rgb = None

        # ---- 加载模型（耗时操作，在订阅前完成） ----
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self._load_net()

        # ======================== 服务 (主触发入口) ========================
        # 客户端调用此服务即可触发检测，并在响应中直接获得结果摘要
        self.detect_srv = self.create_service(Trigger, '/detect_grasp', self._service_cb)

        # ======================== 订阅 (仅用于图像缓存) ========================
        self.create_subscription(
            Image, "/camera/camera/aligned_depth_to_color/image_raw",
            self._cb_depth, 10)
        self.create_subscription(
            Image, "/camera/camera/color/image_raw",
            self._cb_rgb, 10)

        # ======================== 发布 ========================
        # 向grasp_policy发布最优抓取姿态 (PoseStamped)
        self.grasp_pose_pub = self.create_publisher(PoseStamped, '/grasp_pose', 1)
        # 向grasp_policy发布最优抓取宽度 (Float32, 单位: 米)
        self.grasp_width_pub = self.create_publisher(Float32, '/grasp_width', 1)

        # 向visualize_node发布空消息，用于通知可视化节点刷新点云和抓取姿态可视化
        self.reload_visualization_pub = self.create_publisher(Empty, '/reload_visualization', 10)

        self.get_logger().info("====== Grasp Detection Node Initialized ======")
        self.get_logger().info(f"  Device      : {self.device}")
        self.get_logger().info(f"  Checkpoint  : {self.checkpoint_path}")
        self.get_logger().info(f"  Num points  : {self.num_point}")
        self.get_logger().info("检测服务: ros2 service call /detect_grasp std_srvs/srv/Trigger")
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
        """缓存对齐后的深度图 (16UC1, 单位: mm)"""
        try:
            self.img_alignedDepth = self._imgmsg_to_cv2(msg, "16UC1")
        except Exception as e:
            self.get_logger().error(f"Depth convert error: {e}")

    def _cb_rgb(self, msg):
        """缓存 RGB 图 (归一化到 0~1, BGR 格式)"""
        try:
            img = self._imgmsg_to_cv2(msg, "rgb8")
            self.img_rgb = img.astype(np.float32) / 255.0
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
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

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
        """NMS → 碰撞过滤 → 可达性预筛 → 按分数排序"""
        gg.nms()

        if self.collision_thresh > 0:
            collision_detector = ModelFreeCollisionDetector(cloud_np, voxel_size=0.01)
            collision_mask = collision_detector.detect(
                gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
            gg = gg[~collision_mask]

        # 可达性预筛: 剔除从远离基座一侧反向夹取的姿态 (易触发 xArm7 奇异点/超限位)
        gg = self._filter_by_reachability(gg)

        gg.sort_by_score()
        return gg

    # ====================================================================
    #  可达性预筛 (基于基座方位的接近方向几何过滤)
    # ====================================================================
    def _get_camera_to_base(self):
        """
        查询手眼标定得到的静态 TF link_base ← camera_color_optical_frame
        返回 (R_base_camera 3x3, t_base_camera 3x1)。

        手眼标定为静态变换, 运行期不变, 首次成功后缓存复用。
        查询失败返回 (None, None), 调用方据此跳过预筛。
        
        输出：
        R_base_camera: 相机坐标系到基座坐标系的旋转矩阵 (3x3)
        t_base_camera: 相机坐标系到基座坐标系的平移向量 (3)
        """
        if self._cam2base_cache is not None:
            return self._cam2base_cache
        try:
            trans = self._tf_buffer.lookup_transform(
                'link_base', 'camera_color_optical_frame',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().warn(
                f'TF 查询失败 (link_base <- camera_color_optical_frame): {e}')
            return None, None

        q = trans.transform.rotation
        R = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        t = np.array([trans.transform.translation.x,
                      trans.transform.translation.y,
                      trans.transform.translation.z], dtype=np.float64)
        self._cam2base_cache = (R, t)
        self.get_logger().info('已缓存手眼标定 TF (link_base <- camera_color_optical_frame)。')
        return R, t

    def _filter_by_reachability(self, gg):
        """基于机械臂基座方位过滤抓取姿态。

        graspnet 约定: rotation_matrix[:, 0] (X 轴) 为接近方向, 由夹爪本体指向物体。
        将该方向与抓取中心变换到 link_base 坐标系后:
          1) 顶视抓取 (接近方向竖直向下, 落在 topdown_cone_deg 锥内) => 保留;
          2) 侧向抓取 => 仅保留"从基座一侧接近"的姿态, 即接近方向的水平分量背离基座;
             若水平分量指向基座 (从物体远端反向夹取), 则机械臂需绕到对面/过度伸展,
             极易逼近奇异点, 予以剔除。
        若全部被剔除则回退使用原始候选, 避免无结果。
        """
        n = len(gg)
        if n == 0:
            return gg

        R_bc, t_bc = self._get_camera_to_base() # 获取相机坐标系到基座坐标系的转换矩阵和向量
        if R_bc is None:
            self.get_logger().warn('无法获取基座 TF, 跳过可达性预筛 (按原分数排序)。')
            return gg

        cos_topdown = np.cos(np.radians(self.topdown_cone_deg))
        cos_reject = np.cos(np.radians(self.side_reject_cone_deg))
        down_base = np.array([0.0, 0.0, -1.0])

        keep = np.zeros(n, dtype=bool)
        n_topdown, n_side, n_reject = 0, 0, 0
        for i in range(n):
            g = gg[i]
            # 把抓取位姿从相机坐标系转换到基座坐标系下
            a_base = R_bc @ np.asarray(g.rotation_matrix, dtype=np.float64)[:, 0]
            c_base = R_bc @ np.asarray(g.translation, dtype=np.float64).reshape(3) + t_bc

            # 1) 顶视抓取: 接近方向竖直向下
            if float(np.dot(a_base, down_base)) >= cos_topdown:
                keep[i] = True
                n_topdown += 1
                continue

            # 2) 侧向抓取: 仅保留从基座一侧接近的姿态
            to_base = np.array([-c_base[0], -c_base[1], 0.0])  # 物体指向基座的水平方向
            a_h = np.array([a_base[0], a_base[1], 0.0])         # 接近方向水平分量
            nb, na = np.linalg.norm(to_base), np.linalg.norm(a_h)
            if nb < 1e-6 or na < 1e-6:
                keep[i] = True  # 退化情形 (接近正上方/正对基座), 保守保留
                n_side += 1
                continue
            if float(np.dot(a_h / na, to_base / nb)) > cos_reject:
                keep[i] = False  # 从远离基座一侧反向夹取 => 剔除
                n_reject += 1
            else:
                keep[i] = True
                n_side += 1

        kept = int(keep.sum())
        self.get_logger().info(
            f'可达性预筛: 保留 {kept}/{n} (顶视 {n_topdown}, 侧向可达 {n_side}, '
            f'剔除远端反向 {n_reject})。')
        if kept == 0:
            self.get_logger().warn('可达性预筛剔除了全部姿态, 回退使用原始候选 (避免无结果)。')
            return gg
        return gg[keep]

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
    #  Service 回调：触发检测并将结果直接返回给调用方
    # ====================================================================
    def _service_cb(self, request, response):
        """  /detect_grasp 服务处理函数
             调用方式 (命令行):
               ros2 service call /detect_grasp std_srvs/srv/Trigger
             调用方式 (Python):
               from std_srvs.srv import Trigger
               client = node.create_client(Trigger, '/detect_grasp')
               result = client.call(Trigger.Request())
               print(result.success, result.message)
        """
        self.get_logger().info("Detection triggered via service call!")

        # 1. 获取并预处理数据
        data_input, cloud_o3d = self._get_and_process_data()
        if data_input is None:
            response.success = False
            response.message = "数据获取失败（未收到 RGB 或深度图像）"
            self.get_logger().error(response.message)
            return response

        # 2. GSNet 推理
        self.get_logger().info("Running GSNet inference...")
        gg = self._run_inference(data_input)
        self.get_logger().info(f"Raw grasps: {len(gg)}")

        # 3. 后处理 (NMS + 碰撞过滤 + 排序)
        gg = self._postprocess(gg, np.asarray(cloud_o3d.points))
        self.get_logger().info(f"Valid grasps after filtering: {len(gg)}")

        if len(gg) == 0:
            response.success = False
            response.message = "未检测到有效抓取姿态"
            self.get_logger().warn(response.message)
            return response

        # 4. 发布最优抓取姿态 (供运动节点订阅)
        self._publish_grasp_pose(gg)

        # 5. 发布可视化 + 保存文件
        self._save_results(gg, cloud_o3d)

        # 6. 将关键结果写入 service 响应，调用方无需订阅话题即可获取
        top = gg[0]
        cx, cy, cz = top.translation
        rot = top.rotation_matrix
        # 旋转矩阵 → 四元数
        qx, qy, qz, qw = self.quaternion_from_rotation(rot)
        
        response.success = True
        response.message = (
            f"score={top.score:.4f}, "
            f"pos=[{cx:.4f}, {cy:.4f}, {cz:.4f}], "
            f"ori=[{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}], "
            f"width={top.width:.4f}, "
            f"total_valid={len(gg)}"
        )

        self.get_logger().info(f"检测完成: {response.message}")
        # 发布刷新消息，通知可视化节点重新加载点云和抓取姿态可视化
        # 话题为 fire-and-forget，publish() 立即返回，无同步服务调用的死锁风险。
        self.reload_visualization_pub.publish(Empty())

        self.get_logger().info("已发布 /reload_visualization 消息，刷新点云和抓取姿态。")
        self.get_logger().info("等待服务调用")
        self.get_logger().info("=" * 50)
        return response


def main(args=None):
    rclpy.init(args=args)

    # ---- 配置路径 ----
    checkpoint_path = os.path.join(
        GSNET_ROOT,
        'np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar')
    grasp_poses_num = 50

    node = None
    try:
        node = GraspDetection(
            checkpoint_path=checkpoint_path,
            grasp_poses_num=grasp_poses_num,
            num_point=20000,
            collision_thresh=0,
            voxel_size=0.005,
            topdown_cone_deg=60.0,     # 顶视抓取判定锥角(半角), 越大越宽松
            side_reject_cone_deg=60.0, # 侧向"反向夹取"判定锥角(半角), 越大剔除越激进
        )
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
