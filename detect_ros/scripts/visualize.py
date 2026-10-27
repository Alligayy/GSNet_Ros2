#!/usr/bin/env python3
"""
PLY 文件可视化工具
"""

import os
import sys
import open3d as o3d


def main():
    # 默认文件路径
    #VIS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vis')
    VIS_DIR = '/media/gao/2/Grasp/GSNet/detect_ros/vis'
    # 点云文件路径
    point_cloud_filename = os.path.join(VIS_DIR,  "latest_point_cloud.ply")
    gripper_filename = os.path.join(VIS_DIR, "latest_grippers.ply")

    geometries = []
    # ---- 加载场景点云 ----
    if os.path.exists(point_cloud_filename):
        point_cloud = o3d.io.read_point_cloud(point_cloud_filename)
        geometries.append(point_cloud)
    else:
        print(f"[ERROR] 场景点云文件不存在")
        sys.exit(1)

    # ---- 加载夹爪网格 ----
    if os.path.exists(gripper_filename):
        gripper_mesh = o3d.io.read_triangle_mesh(gripper_filename)
        geometries.append(gripper_mesh)
    else:
        print(f"[WARN] 夹爪网格文件不存在")

    # ---- 坐标轴 ----
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    # geometries.append(axis)

    # ---- 可视化 (初始视角：从上方俯视场景) ----
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="GSNet Grasp Visualization", width=1280, height=720)

    for geo in geometries:
        vis.add_geometry(geo)

    # 设置相机视角：从正上方往下看场景
    ctr = vis.get_view_control()
    ctr.set_front([0, 0.7, -1])   # 相机朝向：沿 -Z 看 (深度方向向下)
    ctr.set_up([0, -1, 0])      # 上方向：-Y (RealSense 相机坐标系 Y 朝下)
    ctr.set_zoom(0.7)           # 缩小视角以显示全场景

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    print("=" * 50)
    print("  PLY 可视化工具")
    print("=" * 50)

    main()
