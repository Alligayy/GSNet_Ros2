# 查看vis文件夹下的点云

import os   
import open3d as o3d



if __name__ == "__main__":

    # 根目录
    root_dir = '/media/gao/2/Grasp/GSNet/detect_ros/vis'

    # 点云文件路径
    point_cloud_filename = os.path.join(root_dir,  "latest_point_cloud.ply")
    gripper_filename = os.path.join(root_dir, "latest_grippers.ply")

    # 加载点云文件
    gg = o3d.io.read_triangle_mesh(gripper_filename)
    cloud_masked = o3d.io.read_point_cloud(point_cloud_filename)

    o3d.visualization.draw_geometries([cloud_masked, gg])


    




