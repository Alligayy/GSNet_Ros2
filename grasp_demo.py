import os
import sys
import argparse
import numpy as np
import torch
import scipy.io as scio
from PIL import Image
import open3d as o3d
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

# 从项目代码中导入核心模块
from models.graspnet import GraspNet, pred_decode
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from dataset.graspnet_dataset import minkowski_collate_fn


parser = argparse.ArgumentParser()

parser.add_argument('--dump_dir', help='Dump dir to save outputs', default='/logs')
parser.add_argument('--seed_feat_dim', default=512, type=int, help='Point wise feature dim')
parser.add_argument('--camera', default='kinect', help='Camera split [realsense/kinect]')
parser.add_argument('--num_point', type=int, default=15000, help='Point Number [default: 15000]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during inference [default: 1]')
parser.add_argument('--voxel_size', type=float, default=0.005, help='Voxel Size for sparse convolution')
parser.add_argument('--collision_thresh', type=float, default=-1,
                    help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size_cd', type=float, default=0.01, help='Voxel Size for collision detection')
parser.add_argument('--infer', action='store_true', default=False)
parser.add_argument('--vis', action='store_true', default=False)
parser.add_argument('--scene', type=str, default='0188')
parser.add_argument('--index', type=str, default='0000')
cfgs = parser.parse_args()

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
# if not os.path.exists(cfgs.dump_dir):
#     os.mkdir(cfgs.dump_dir)

def data_process(color_path, depth_path, meta_path, mask_path):
    """
    输入：
      color_path: RGB 图像路径
      depth_path: 深度图像路径
      meta_path: 相机内参矩阵路径
      mask_path: 工作空间遮罩路径

    返回：
      data_input: 输入数据字典
      cloud: 输入点云
    
    """
    # ------------------------------------------------------------------
    # 1. 读取 RGB-D 数据并转换为 3D 点云
    # ------------------------------------------------------------------
    print("-> 正在处理 RGB-D 输入数据...")
    # 读取相机内参矩阵及深度图像转换因子
    meta = scio.loadmat(meta_path)
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # 读取图像
    color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    depth = np.array(Image.open(depth_path))
    workspace_mask = np.array(Image.open(mask_path))

    # 构建相机对象并生成点云
    camera = CameraInfo(
        width=1280.0, height=720.0,
        fx=intrinsic[0][0], fy=intrinsic[1][1],
        cx=intrinsic[0][2], cy=intrinsic[1][2],
        scale=factor_depth
    )
    cloud = create_point_cloud_from_depth_image(depth, camera)

    # 遮罩过滤：仅保留 Mask 范围内且深度值有效的点
    depth_mask = (depth > 0)
    mask = (workspace_mask & depth_mask)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # 点云采样 (模型固定接收 20000 个点)
    num_points = 20000
    if len(cloud_masked) >= num_points:
        idxs = np.random.choice(len(cloud_masked), num_points, replace=False)
    else:
        idxs = np.random.choice(len(cloud_masked), num_points, replace=True)

    cloud_sampled = cloud_masked[idxs]
    #color_sampled = color_masked[idxs]

    # cloud = o3d.geometry.PointCloud()
    # cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    # cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32)) 

    # 转为网络输入格式 [Batch_size=1, N, 3]
    #cloud_tensor = torch.from_numpy(cloud_sampled[np.newaxis, ...].astype(np.float32)).to(device)
    ret_dict = {
            'point_clouds': cloud_sampled.astype(np.float32),
            'coors': cloud_sampled.astype(np.float32) / cfgs.voxel_size,
            'feats': np.ones_like(cloud_sampled).astype(np.float32),
            }
    return ret_dict, cloud_masked, color_masked


def run_gsnet_inference(data_input, checkpoint_path):

    batch_data = minkowski_collate_fn([data_input])
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    for key in batch_data:
        if 'list' in key:
            for i in range(len(batch_data[key])):
                for j in range(len(batch_data[key][i])):
                    batch_data[key][i][j] = batch_data[key][i][j].to(device)
        else:
            batch_data[key] = batch_data[key].to(device)
    # ------------------------------------------------------------------
    # 1. 初始化模型并加载权重
    # ------------------------------------------------------------------
    print("-> 正在初始化 GSNet 模型...")
    net = GraspNet(seed_feat_dim=512, is_training=False)
    net.to(device)

    # 加载权重
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)" % (checkpoint_path, start_epoch))
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval() # 推理模式

    # ------------------------------------------------------------------
    # 2. 前向推理 (Predict Grasps)
    # ------------------------------------------------------------------
    print("-> 执行网络推理...")
    with torch.no_grad():
        end_points = net(batch_data)
        # 将网络输出特征解码为标准的抓取位姿表示
        grasp_preds = pred_decode(end_points)

    preds = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(preds) # 转换为 GraspNet API 格式方便后续操控或可视化

    # ------------------------------------------------------------------
    # 4. 后处理：无模型碰撞检测 (Collision Detection)
    # ------------------------------------------------------------------
    print("-> 正在执行碰撞检测与抓取框过滤...")
    cloud_sampled = data_input['point_clouds']
    collision_detector = ModelFreeCollisionDetector(cloud_sampled, voxel_size=0.01)
    collision_mask = collision_detector.detect(gg, approach_dist=0.05, collision_thresh=0.01)
    gg = gg[~collision_mask]
    
    # 移除碰撞框
    valid_preds = preds[~collision_mask]

    # 按置信度得分 (Score) 进行降序排序
    valid_preds = valid_preds[np.argsort(-valid_preds[:, 0])]

    print(f"-> 推理完成！共生成 {len(preds)} 个抓取姿态，过滤碰撞后剩余 {len(valid_preds)} 个有效姿态。")

    return gg


if __name__ == '__main__':
    # 路径配置（这里直接使用代码自带的 example_data 目录数据）
    
    DATA_DIR = 'doc/example_data'
    COLOR_PATH = os.path.join(DATA_DIR, 'color.png')
    DEPTH_PATH = os.path.join(DATA_DIR, 'depth.png')
    META_PATH = os.path.join(DATA_DIR, 'meta.mat')
    MASK_PATH = os.path.join(DATA_DIR, 'workspace_mask.png')
    
    # 模型权重路径
    CHECKPOINT_PATH = os.path.join(ROOT_DIR,'np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar'
 )
    # 数据处理
    ret_dict, cloud_masked, color_masked= data_process(COLOR_PATH, DEPTH_PATH, META_PATH, MASK_PATH)

    # 执行推理
    gg = run_gsnet_inference(ret_dict, CHECKPOINT_PATH)

    # 取前 50 个最高得分的抓取姿态
    gg_top50 = gg[:50]
    gg_0 = gg[0]
    
    # 可视化点云与抓取框
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_masked)
    pcd.colors = o3d.utility.Vector3dVector(color_masked)
    
    # 获取抓取框 3D 网格模型
    grippers = gg_top50.to_open3d_geometry_list()
    
    # ------------------------------------------------------------------
    # 保存点云为 PLY 文件
    # ------------------------------------------------------------------
    save_dir = os.path.join(ROOT_DIR, 'pointcloud')
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存场景点云
    scene_ply_path = os.path.join(save_dir, 'scene_pointcloud.ply')
    o3d.io.write_point_cloud(scene_ply_path, pcd)
    print(f"-> 场景点云已保存至: {scene_ply_path}")
    
    # 合并所有夹爪网格并保存为单个 PLY
    merged_gripper = o3d.geometry.TriangleMesh()
    for g in grippers:
        merged_gripper += g
    gripper_ply_path = os.path.join(save_dir, 'grasp_grippers.ply')
    o3d.io.write_triangle_mesh(gripper_ply_path, merged_gripper)
    print(f"-> 抓取姿态网格已保存至: {gripper_ply_path}")
    
    o3d.visualization.draw_geometries([pcd, *grippers])