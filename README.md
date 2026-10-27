# GSNet_Ros2

基于 [GSNet (Graspness Supervision-based Sub-Tensor Grasp Generation)](https://github.com/graspnet/GSNet) 的 **ROS2 机械臂抓取项目**。

在原始 GSNet 抓取检测网络的基础上，接入了 RealSense D455 相机与 **xArm-7 机械臂**，构建了从**相机采集 → 点云推理 → 抓取姿态发布 → 实时可视化 → 真机抓取**的完整流水线。

> 原 GSNet 项目（Graspness 监督训练与 GraspNet-1Billion 评测）的完整说明保留在 [doc/GSNet_original_README.md](doc/GSNet_original_README.md)。

## 配套仓库：xarm7_grasp

本仓库只负责**抓取感知（检测端）**，策略执行与真机控制位于配套仓库 **[xarm7_grasp](https://github.com/Alligayy/xarm7_grasp)**（位于 `grasp_ws/src/xarm7_grasp`）。两仓库通过 ROS2 话题解耦协作：

```
GSNet_Ros2 (检测端)                       xarm7_grasp (执行端)
RealSense D455 ─► GSNet 推理节点 ─► /grasp_pose ──┬─► 抓取策略节点 grasp_policy_xarm.py
                          /grasp_width ────────────┘        │ xarm_planner 规划 + 轨迹执行
              ▲                        手眼标定 TF           ▼
              └── /reload_visualization ─► visualize_node   xArm-7 + 夹爪 (仿真/真机)
```

`xarm7_grasp` 提供：

- **抓取策略执行**：订阅本仓库发布的 `/grasp_pose`、`/grasp_width`，经可达性几何预筛与手眼标定 TF 变换到基座系后，通过 `xarm_planner` 服务完成位姿规划与轨迹执行，支持仿真（fake）与真机两种模式；
- **手眼标定**：easy_handeye2（ArUco）与自研 ChArUco 检测节点，标定板生成工具；
- **运动控制**：关节/笛卡尔运动、回 home、夹爪开合等测试脚本，以及 launch 启动文件（`grasp_simulation.launch.py` / `grasp_real.launch.py` / 一键启动 `xarm7_grasp_all.launch.xml`）。

两仓库需置于同一 ROS2 工作空间体系下使用（本仓库可独立运行，也可由 `grasp_ws` 内的 launch 统一拉起检测 + 执行全链路），具体部署与运行步骤见 [xarm7_grasp 的 README](https://github.com/Alligayy/xarm7_grasp)。

## 主要新增内容

### 1. ROS2 实时抓取检测流水线（`detect_ros/`）

以 ROS2 节点形式封装 GSNet 推理，订阅 RealSense D455 的 RGBD 图像，运行检测后发布抓取姿态：

| 节点脚本 | 功能 |
|---|---|
| [detect_ros/detect_grasp.py](detect_ros/detect_grasp.py) | 基础检测节点：订阅 RGBD 图像，运行 GSNet 推理，发布 `PoseStamped` 抓取姿态（`/grasp_pose`、`/grasp_width`，供 xarm7_grasp 执行端订阅）；检测完成后发布 `/reload_visualization` 话题通知可视化节点刷新 |
| [detect_ros/detect_grasp_reach.py](detect_ros/detect_grasp_reach.py) | 带可达性分析的检测节点 |
| [detect_ros/detect_grasp_trans.py](detect_ros/detect_grasp_trans.py) | 针对**透明物体**的检测节点：配合 depth_completion 节点，收到同时间戳的补全深度 + RGB 图像对后自动推理 |
| [detect_ros/scripts/visualize_node.py](detect_ros/scripts/visualize_node.py) | Open3D 可视化节点：主线程轮询渲染 + 话题触发重载点云/夹爪，非阻塞实时刷新 |

可视化输出（点云与夹爪姿态）保存在 `detect_ros/vis/`，可直接用 Open3D 或 RViz 查看。

### 2. 抓取执行接口与离线脚本

- 检测节点发布的 `/grasp_pose`、`/grasp_width` 话题即与 [xarm7_grasp](https://github.com/Alligayy/xarm7_grasp) 执行端的对接接口，抓取规划与真机控制由该仓库的 `grasp_policy_xarm.py` 完成；
- [grasp_real.py](grasp_real.py)：早期单机版真机抓取脚本，加载预训练权重对相机采集场景推理，经碰撞检测与 NMS 筛选后直接将抓取姿态变换到机器人基坐标系执行（机器人接口封装于 `UR_Robot`，该文件未随仓库分发，现已被 ROS2 节点化方案取代，仅作参考）；
- 离线演示见 [grasp_demo.py](grasp_demo.py)。

### 3. 其他改动

- 数据集支持 realsense 相机采集数据（`dataset/graspnet_dataset.py`、`utils/data_utils.py`）；
- 预训练权重与 GSPNet 结构适配真机部署（`models/`）；
- 依赖源码内置：`MinkowskiEngine/`、`graspnetAPI/`、`knn/`、`pointnet2/` 均随仓库携带，无需另行克隆。

## 目录结构

```
GSNet_Ros2/
├── detect_ros/          # ROS2 实时检测与可视化节点（本项目核心新增）
├── grasp_real.py        # 早期单机版抓取脚本（已被 ROS2 方案取代）
├── grasp_demo.py        # 离线抓取演示
├── models/              # GSNet 网络与损失
├── dataset/             # 数据加载、graspness 标签生成
├── utils/               # 碰撞检测、坐标变换、数据工具
├── train.py / test.py   # 原 GSNet 训练与评测入口
├── MinkowskiEngine/     # 内置依赖（稀疏卷积）
├── graspnetAPI/         # 内置依赖（抓取格式/NMS/可视化）
├── knn/ pointnet2/      # 内置 CUDA 算子
└── doc/                 # 文档与原项目 README
```

## 环境安装

依赖：Python 3.10、PyTorch（含 CUDA）、ROS2、Open3D、OpenCV、MinkowskiEngine。

```bash
# 1. Python 依赖
pip install -r requirements.txt

# 2. 编译内置算子
cd pointnet2 && python setup.py install && cd ..
cd knn && python setup.py install && cd ..

# 3. MinkowskiEngine（源码已内置）
cd MinkowskiEngine && python setup.py install && cd ..

# 4. graspnetAPI（源码已内置）
cd graspnetAPI && pip install . && cd ..
```

预训练权重 `np15000_graspness1e-1_bs4_lr1e-3_viewres_dataaug_fps_14D_epoch10.tar`（约 176MB，超出 GitHub 单文件限制未随仓库分发）请参考原项目说明获取，或自行训练。

## 使用

### 训练 / 测试（GraspNet-1Billion）

与原版一致，参见 [command_train.sh](command_train.sh) 与 [command_test.sh](command_test.sh)，以及 [doc/GSNet_original_README.md](doc/GSNet_original_README.md) 中的 Training and Testing 一节。

### ROS2 实时检测 + 抓取执行

```bash
# 终端 1：启动 RealSense 相机驱动节点
ros2 launch realsense2_camera rs_launch.py

# 终端 2：启动抓取检测节点（本仓库，脚本直接运行，无需 ros2 run）
python detect_ros/detect_grasp.py

# 终端 3：启动抓取策略执行节点（xarm7_grasp，订阅 /grasp_pose 执行抓取）
ros2 launch xarm7_grasp grasp_simulation.launch.py   # 仿真模式
ros2 launch xarm7_grasp grasp_real.launch.py         # 真机模式

# 终端 4：启动可视化节点（可选）
python detect_ros/scripts/visualize_node.py
```

检测节点订阅相机 RGBD 话题，发布抓取姿态话题与 `/reload_visualization` 刷新通知；完整参数见各脚本头部说明，执行端配置与话题约定见 [xarm7_grasp](https://github.com/Alligayy/xarm7_grasp)。

### 单机离线抓取（早期方案，不依赖 ROS2）

```bash
python grasp_real.py --checkpoint_path <权重文件> --camera realsense
```

## 致谢

- [xarm7_grasp](https://github.com/Alligayy/xarm7_grasp)：本项目的配套执行端（xArm-7 抓取策略、手眼标定、运动控制）
- [GSNet](https://github.com/graspnet/GSNet) / [graspness_implementation](https://github.com/rhett-chen/graspness_implementation)：抓取检测网络与 Graspness 监督训练
- [graspnet-baseline](https://github.com/graspnet/graspnet-baseline)：基础代码框架
- [graspnetAPI](https://github.com/graspnet/graspnetAPI)：抓取格式、NMS 与评测工具
- [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)：稀疏张量卷积引擎
