#!/usr/bin/env python3
"""
主系统模块 - KLT框架任务区域集成版
四类对象覆盖系统的主要逻辑和程序入口
集成激光雷达实时避障功能
新增：真实Franka机械臂抓取功能 - 使用test.py验证参数
优化：精确角度控制和释放距离调整
"""

import psutil
import torch
import numpy as np
import time
import random
import gc
import math
import rospy
from typing import List, Dict

def print_memory_usage(stage_name: str = ""):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"内存: {memory_mb:.1f}MB - {stage_name}")

# 先初始化SimulationApp，再导入Isaac Sim模块
print("启动KLT框架任务区域版四类对象覆盖算法机器人系统...")
print(f"对象类型:")
print(f"  1. 障碍物(红色) - 路径规划避障")
print(f"  2. 清扫目标(黄色) - 触碰消失") 
print(f"  3. 抓取物体(绿色) - 运送到KLT框子")
print(f"  4. KLT框子任务区域 - 物体投放区域 (具有物理碰撞)")
print(f"  5. 激光雷达实时避障 - 分层避障策略")
print(f"  6. 真实Franka机械臂抓取 - 精确关节控制")

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "enable_livestream": False, 
    "enable_cameras": True,
    "enable_rtx": True,
    "physics_dt": 1.0/60.0,
    "rendering_dt": 1.0/30.0,
})

# 初始化ROS节点 - 新增
print("初始化ROS节点...")
rospy.init_node('isaac_sim_lidar_robot', anonymous=True)

# 添加这段：检查并启用ROS Bridge
print("检查ROS Bridge扩展状态...")
import omni.kit.app
import carb

# 获取扩展管理器
ext_manager = omni.kit.app.get_app().get_extension_manager()

# 检查ROS Bridge是否启用
if not ext_manager.is_extension_enabled("omni.isaac.ros_bridge"):
    print("启用ROS Bridge扩展...")
    ext_manager.set_extension_enabled_immediate("omni.isaac.ros_bridge", True)
    print("ROS Bridge扩展已启用")
else:
    print("ROS Bridge扩展已经启用")

# 等待扩展完全加载
simulation_app.update()
simulation_app.update()
simulation_app.update()

# 检查roscore连接
print("检查roscore连接...")
try:
    import rosgraph
    if not rosgraph.is_master_online():
        carb.log_warn("roscore未运行，请先运行: roscore")
        print("警告: roscore未运行，请在另一个终端先运行: roscore")
    else:
        print("roscore连接正常")
except ImportError:
    print("警告: 无法导入rosgraph，请确保ROS环境正确设置")

# 现在安全导入Isaac Sim核心模块
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, DynamicSphere
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdLux, UsdPhysics, Gf, UsdGeom, Usd, UsdShade, Sdf
import isaacsim.core.utils.prims as prim_utils

# 导入本地模块
from data_structures import *
from visualizer import RealMovementVisualizer
from path_planner import FourObjectPathPlanner
from robot_controller import FixedRobotController
from lidar_avoidance import LidarAvoidanceController, DynamicPathReplanner  # 新增导入

class FourObjectCoverageSystem:
    """四类对象覆盖系统 - KLT框子任务区域版 + 激光雷达避障 + 真实Franka抓取"""
    
    def __init__(self):
        self.world = None
        self.mobile_base = None
        self.robot_controller = None
        self.path_planner = None
        self.visualizer = None
        
        # 新增：激光雷达避障系统
        self.lidar_avoidance = None
        self.dynamic_replanner = None
        
        self.scene_objects = []
        self.coverage_path = []
        self.coverage_stats = {
            'swept_objects': 0,
            'grasped_objects': 0,
            'delivered_objects': 0,
            'total_coverage_points': 0,
            'obstacles_avoided': 0,  # 新增：避障统计
            'emergency_stops': 0     # 新增：紧急停车统计
        }
        
        # KLT框子配置
        self.klt_box_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/NVIDIA/Assets/ArchVis/Lobby/My_asset/T/small_KLT.usd"
        self.klt_approach_distance = 0.8 # 优化：减小距离，确保释放在框子内
        
        # 真实Franka机械臂抓取配置 - 使用test.py验证参数
        self.arm_poses = {
            "home": [0.0, 0.524, 0.0, -0.785, 0.0, 1.571, 0.785],        # 30°, -170°, 90°, 45°
            "grasp_approach": [0.0, 1.676, 0.0, -0.646, 0.0, 2.234, 0.785],  # 96°, -37°, 128°, 45°
            "grasp_lift": [0.0, 1.047, 0.0, -0.646, 0.0, 2.234, 0.785],     # 60°, -37°, 128°, 45°
            "carry": [0.0, 1.047, 0.0, -0.646, 0.0, 2.234, 0.785]           # 60°, -37°, 128°, 45°（同lift）
        }
        
        # 抓取参数 - 使用test.py验证的精确参数
        self.grasp_trigger_distance = 1.5      # 触发抓取的距离 - 修正：小于INTERACTION_DISTANCE
        self.grasp_approach_distance = 0.295    # 抓取接近距离 - test.py验证参数
        self.arm_stabilization_time = 2.0      # 机械臂稳定时间 - test.py验证参数
        self.gripper_stabilization_time = 1.0  # 夹爪稳定时间 - test.py验证参数
        
        # 夹爪位置 - test.py验证参数
        self.gripper_open_pos = 0.05     # 张开位置
        self.gripper_close_pos = 0.025   # 闭合位置
        
        # 优化：角度和距离控制参数
        self.angle_tolerance = 0.05      # 角度容差，从原来的0.2减小到0.05，更精确
        self.position_tolerance = 0.15   # 位置容差，从原来的0.2减小到0.15
        self.angular_kp = 5.0            # 角速度控制增益，从原来的4.0增加到5.0
        self.max_angular_vel = 2.0       # 最大角速度，从原来的1.5增加到2.0
        
        # 抓取状态
        self.carrying_object = None
        self.return_position = None
        
        print("真实Franka机械臂抓取系统初始化完成 - 使用test.py验证参数")
        print(f"抓取触发距离: {self.grasp_trigger_distance}m")
        print(f"抓取接近距离: {self.grasp_approach_distance}m")
        print(f"机械臂稳定时间: {self.arm_stabilization_time}s")
        print(f"夹爪稳定时间: {self.gripper_stabilization_time}s")
        print(f"夹爪张开/闭合位置: {self.gripper_open_pos}m / {self.gripper_close_pos}m")
        print(f"优化：KLT接近距离调整为: {self.klt_approach_distance}m")
        print(f"优化：角度容差调整为: {self.angle_tolerance}rad")
        print(f"优化：角速度增益调整为: {self.angular_kp}")
        print(f"优化：最大角速度调整为: {self.max_angular_vel}rad/s")
    
    def initialize_system(self):
        """初始化系统"""
        print("初始化KLT框子任务区域版四类对象覆盖系统...")
        
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0/60.0,
            rendering_dt=1.0/30.0
        )
        self.world.scene.clear()
        
        # 物理设置
        physics_context = self.world.get_physics_context()
        physics_context.set_gravity(-9.81)
        physics_context.enable_gpu_dynamics(True)
        
        # 创建地面
        ground = FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.5]),
            scale=np.array([50.0, 50.0, 1.0]),
            color=np.array([0.4, 0.4, 0.4])
        )
        self.world.scene.add(ground)
        
        self._setup_lighting()
        self._initialize_components()
        self._create_four_type_environment()
        
        print(f"KLT框子任务区域系统初始化完成")
        print(f"激光雷达避障系统已集成")
        print(f"真实Franka机械臂抓取系统已集成")
        return True
    
    def _setup_lighting(self):
        """设置照明"""
        main_light = prim_utils.create_prim("/World/MainLight", "DistantLight")
        distant_light = UsdLux.DistantLight(main_light)
        distant_light.CreateIntensityAttr(4000)
        distant_light.CreateColorAttr((1.0, 1.0, 0.95))
        
        env_light = prim_utils.create_prim("/World/EnvLight", "DomeLight")
        dome_light = UsdLux.DomeLight(env_light)
        dome_light.CreateIntensityAttr(800)
        dome_light.CreateColorAttr((0.8, 0.9, 1.0))
        print("照明设置完成")
    
    def _initialize_components(self):
        """初始化组件"""
        self.path_planner = FourObjectPathPlanner()
        self.visualizer = RealMovementVisualizer(self.world)
        
        # 新增：初始化激光雷达避障系统
        self.lidar_avoidance = LidarAvoidanceController()
        self.dynamic_replanner = DynamicPathReplanner(self.path_planner, self.lidar_avoidance)
        
        print("组件初始化完成（含超丝滑覆盖可视化 + 激光雷达避障 + Franka抓取）")
    
    def _create_four_type_environment(self):
        """创建四类对象环境 - KLT框子任务区域版"""
        print("创建四类对象环境（KLT框子任务区域）...")
        
        # 1. 障碍物 (红色) - 路径规划避障
        obstacles_config = [
            {"pos": [1.2, 0.8, 0.15], "size": [0.6, 0.4, 1.8], "color": [0.8, 0.2, 0.2], "shape": "box"},
            {"pos": [0.5, -1.5, 0.2], "size": [1.2, 0.3, 1.8], "color": [0.8, 0.1, 0.1], "shape": "box"},
            {"pos": [-1.0, 1.2, 0.25], "size": [0.7], "color": [0.7, 0.2, 0.2], "shape": "sphere"},
        ]
        
        # 2. 清扫目标 (黄色) - 触碰消失
        sweep_config = [
            {"pos": [0.8, 0.2, 0.05], "size": [0.1], "color": [1.0, 1.0, 0.2], "shape": "sphere"},
            {"pos": [1.5, 1.5, 0.05], "size": [0.1], "color": [0.9, 0.9, 0.1], "shape": "sphere"},
            {"pos": [-0.8, -0.8, 0.05], "size": [0.1], "color": [1.0, 0.8, 0.0], "shape": "sphere"},
            {"pos": [2.0, 0.5, 0.05], "size": [0.1], "color": [0.8, 0.8, 0.2], "shape": "sphere"},
        ]
        
        # 3. 抓取物体 (绿色) - 运送到KLT框子 - 使用test.py验证的尺寸
        grasp_config = [
            {"pos": [1.5, 0.0, 0.08], "size": [0.05, 0.05, 0.05], "color": [0.2, 0.8, 0.2], "shape": "box"},
            {"pos": [-2, -5, 0.08], "size": [0.05, 0.05, 0.05], "color": [0.1, 0.9, 0.1], "shape": "box"},
            {"pos": [0.3, 1.8, 0.08], "size": [0.05, 0.05, 0.05], "color": [0.0, 0.8, 0.0], "shape": "box"},
        ]
        
        # 4. KLT框子任务区域 - 使用自定义USD资产，放大尺寸
        klt_box_config = [
            {"pos": [5, 0, 0.1], "size": [1.6, 0.8, 0.3], "usd_path": self.klt_box_usd_path, "shape": "usd_asset"},
        ]
        
        # 创建所有对象
        all_configs = [
            (obstacles_config, ObjectType.OBSTACLE),
            (sweep_config, ObjectType.SWEEP),
            (grasp_config, ObjectType.GRASP),
            (klt_box_config, ObjectType.TASK)
        ]
        
        for config_list, obj_type in all_configs:
            for i, config in enumerate(config_list):
                self._create_scene_object(config, obj_type, i)
        
        # 5. 添加外围围墙
        self._create_boundary_walls()
        
        print(f"四类对象环境创建完成:")
        print(f"  障碍物: {len(obstacles_config)}个")
        print(f"  清扫目标: {len(sweep_config)}个") 
        print(f"  抓取物体: {len(grasp_config)}个 (使用test.py验证尺寸)")
        print(f"  KLT框子任务区域: {len(klt_box_config)}个")
        print(f"  外围围墙: 已创建")
        print(f"  激光雷达实时避障: 已启用")
        print(f"  真实Franka机械臂抓取: 已配置")
    
    def _create_scene_object(self, config: Dict, obj_type: ObjectType, index: int):
        """创建场景对象 - 支持USD资产加载"""
        name = f"{obj_type.value}_{index}"
        prim_path = f"/World/{name}"
        
        # KLT框子任务区域使用USD资产
        if obj_type == ObjectType.TASK and config.get("shape") == "usd_asset":
            isaac_obj = self._create_klt_box_task_area(config, prim_path, name)
            collision_boundary = CollisionBoundary(
                center=np.array(config["pos"]),
                shape_type="box",
                dimensions=np.array(config["size"])
            )
        elif config["shape"] == "sphere":
            isaac_obj = DynamicSphere(
                prim_path=prim_path,
                name=name,
                position=np.array(config["pos"]),
                radius=config["size"][0],
                color=np.array(config["color"])
            )
            collision_boundary = CollisionBoundary(
                center=np.array(config["pos"]),
                shape_type="sphere",
                dimensions=np.array(config["size"])
            )
        else:  # box
            if obj_type in [ObjectType.OBSTACLE, ObjectType.TASK]:
                isaac_obj = FixedCuboid(
                    prim_path=prim_path,
                    name=name,
                    position=np.array(config["pos"]),
                    scale=np.array(config["size"]),
                    color=np.array(config["color"])
                )
            else:
                isaac_obj = DynamicCuboid(
                    prim_path=prim_path,
                    name=name,
                    position=np.array(config["pos"]),
                    scale=np.array(config["size"]),
                    color=np.array(config["color"])
                )
            
            collision_boundary = CollisionBoundary(
                center=np.array(config["pos"]),
                shape_type="box",
                dimensions=np.array(config["size"])
            )
        
        # 对于USD资产，不添加到Isaac场景中，直接管理
        if obj_type == ObjectType.TASK and config.get("shape") == "usd_asset":
            isaac_obj = self._create_klt_box_task_area(config, prim_path, name)
        elif isaac_obj:
            self.world.scene.add(isaac_obj)
        
        # 创建场景对象
        scene_obj = SceneObject(
            name=name,
            object_type=obj_type,
            position=np.array(config["pos"]),
            collision_boundary=collision_boundary,
            isaac_object=isaac_obj,
            color=np.array([0.3, 0.5, 0.8]) if obj_type == ObjectType.TASK else np.array(config["color"]),
            original_position=np.array(config["pos"])
        )
        
        self.scene_objects.append(scene_obj)
        self.path_planner.add_scene_object(scene_obj)
        
        print(f"创建{obj_type.value}对象: {name}")
    
    def _create_boundary_walls(self):
        """创建外围围墙 - 确保远离所有任务区域"""
        print("创建外围围墙...")
        
        # 分析当前所有对象的分布范围
        all_positions = []
        for obj in self.scene_objects:
            all_positions.append(obj.position[:2])
        
        if all_positions:
            positions_array = np.array(all_positions)
            min_x, min_y = positions_array.min(axis=0)
            max_x, max_y = positions_array.max(axis=0)
            
            # 额外扩展围墙范围，确保充足的安全距离
            wall_margin = 3.0
            wall_x_range = max(abs(min_x), abs(max_x)) + wall_margin
            wall_y_range = max(abs(min_y), abs(max_y)) + wall_margin
            
            print(f"  对象分布范围: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")
            print(f"  围墙安全距离: {wall_margin}m")
            print(f"  围墙范围: X[±{wall_x_range:.1f}], Y[±{wall_y_range:.1f}]")
        else:
            # 默认围墙范围
            wall_x_range = 10.0
            wall_y_range = 10.0
            print(f"  使用默认围墙范围: ±{wall_x_range}m")
        
        # 围墙配置参数
        wall_thickness = 0.5
        wall_height = 2.0
        wall_color = [0.6, 0.6, 0.6]
        
        # 创建四面围墙
        walls_config = [
            {"name": "north_wall", "pos": [0.0, wall_y_range + wall_thickness/2, wall_height/2], "size": [wall_x_range*2 + wall_thickness*2, wall_thickness, wall_height], "color": wall_color},
            {"name": "south_wall", "pos": [0.0, -(wall_y_range + wall_thickness/2), wall_height/2], "size": [wall_x_range*2 + wall_thickness*2, wall_thickness, wall_height], "color": wall_color},
            {"name": "east_wall", "pos": [wall_x_range + wall_thickness/2, 0.0, wall_height/2], "size": [wall_thickness, wall_y_range*2, wall_height], "color": wall_color},
            {"name": "west_wall", "pos": [-(wall_x_range + wall_thickness/2), 0.0, wall_height/2], "size": [wall_thickness, wall_y_range*2, wall_height], "color": wall_color}
        ]
        
        # 创建围墙对象
        for i, wall_config in enumerate(walls_config):
            wall_name = wall_config["name"]
            prim_path = f"/World/BoundaryWalls/{wall_name}"
            
            wall_obj = FixedCuboid(
                prim_path=prim_path,
                name=wall_name,
                position=np.array(wall_config["pos"]),
                scale=np.array(wall_config["size"]),
                color=np.array(wall_config["color"])
            )
            
            self.world.scene.add(wall_obj)
            
            collision_boundary = CollisionBoundary(
                center=np.array(wall_config["pos"]),
                shape_type="box",
                dimensions=np.array(wall_config["size"])
            )
            
            scene_obj = SceneObject(
                name=wall_name,
                object_type=ObjectType.OBSTACLE,
                position=np.array(wall_config["pos"]),
                collision_boundary=collision_boundary,
                isaac_object=wall_obj,
                color=np.array(wall_config["color"]),
                original_position=np.array(wall_config["pos"])
            )
            
            self.scene_objects.append(scene_obj)
            self.path_planner.add_scene_object(scene_obj)
            
            print(f"    创建围墙: {wall_name}")
        
        print(f"  外围围墙创建完成: 4面围墙，已添加到激光雷达避障系统")
    
    def _create_klt_box_task_area(self, config: Dict, prim_path: str, name: str):
        """创建KLT框子任务区域"""
        try:
            stage = self.world.stage
            klt_prim = stage.DefinePrim(prim_path, "Xform")
            
            references = klt_prim.GetReferences()
            references.AddReference(config["usd_path"])
            
            xform = UsdGeom.Xformable(klt_prim)
            xform.ClearXformOpOrder()
            
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(float(config["pos"][0]), float(config["pos"][1]), 0.0))  # 强制Z=0，贴地面
            
            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3d(2.0, 2.0, 2.0))
            
            # 添加物理属性 - 修复浮空问题，启用动态重力
            if not klt_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(klt_prim)
                rigid_body_api.CreateRigidBodyEnabledAttr().Set(False)  # 改回静态，防止浮空
            
            if not klt_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(klt_prim)
                collision_api.CreateCollisionEnabledAttr().Set(True)
            
            print(f"  创建KLT框子任务区域: {name} (强制贴地面)")
            
            class KLTBoxReference:
                def __init__(self, prim_path, name, position):
                    self.prim_path = prim_path
                    self.name = name
                    self._position = position
                
                def set_visibility(self, visible):
                    pass
                
                def set_world_pose(self, position, orientation):
                    pass
            
            return KLTBoxReference(prim_path, name, np.array(config["pos"]))
            
        except Exception as e:
            print(f"创建KLT框子失败: {e}")
            return FixedCuboid(
                prim_path=prim_path,
                name=name,
                position=np.array(config["pos"]),
                scale=np.array(config["size"]),
                color=np.array([0.3, 0.5, 0.8])
            )
    
    def initialize_robot(self):
        """初始化机器人"""
        print("初始化机器人...")
        
        robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm4.usd"
        print(f"使用机器人资产: {robot_usd_path}")
        
        self.mobile_base = WheeledRobot(
            prim_path="/World/create3_robot",
            name="create3_robot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=robot_usd_path,
            position=np.array([0.0, 0.0, 0.0])
        )
        
        self.world.scene.add(self.mobile_base)
        print("机器人添加到场景完成")
        
        return True
    
    def setup_post_load(self):
        """后加载设置"""
        print("后加载设置...")
        
        # 重置世界
        self.world.reset()
        print("世界重置完成")
        
        # 稳定物理
        print("物理稳定中...")
        for i in range(120):
            self.world.step(render=False)
            if i % 30 == 0:
                print(f"  稳定进度: {i+1}/120")
        
        # 获取机器人对象
        print("获取机器人对象...")
        self.mobile_base = None
        for retry in range(5):
            self.mobile_base = self.world.scene.get_object("create3_robot")
            if self.mobile_base is not None:
                print(f"机器人对象获取成功 (尝试 {retry+1}/5)")
                break
            else:
                for _ in range(30):
                    self.world.step(render=False)
        
        # 验证机器人对象
        controller = self.mobile_base.get_articulation_controller()
        print(f"机器人验证通过，DOF数量: {len(self.mobile_base.dof_names)}")
        
        # 修正物理层次结构
        self._fix_robot_physics_conservative()
        
        # 设置控制增益
        self._setup_robust_control_gains()
        
        # 移动机械臂到home位置
        self._move_arm_to_pose("home")
        
        # 初始化控制器
        self.robot_controller = FixedRobotController(self.mobile_base, self.world)
        
        # 设置覆盖可视化器引用
        self.robot_controller.set_coverage_visualizer(self.visualizer)
        
        # 新增：设置激光雷达避障系统
        self.robot_controller.set_lidar_avoidance(self.lidar_avoidance, self.dynamic_replanner)
        
        print("机器人控制器与超丝滑覆盖可视化器连接完成")
        print("激光雷达避障系统集成完成")
        print("真实Franka机械臂抓取系统已就绪")
        
        # 验证机器人状态
        pos, yaw = self.robot_controller._get_robot_pose()
        print(f"机器人状态验证: 位置[{pos[0]:.3f}, {pos[1]:.3f}], 朝向{np.degrees(yaw):.1f}°")
        
        print("后加载设置完成")
        return True
    
    def _fix_robot_physics_conservative(self):
        """保守的机器人物理层次结构修正"""
        print("保守修正机器人物理层次结构...")
        
        robot_prim = self.world.stage.GetPrimAtPath("/World/create3_robot")
        
        problem_paths = [
            "/World/create3_robot/create_3/left_wheel/visuals_01",
            "/World/create3_robot/create_3/right_wheel/visuals_01"
        ]
        
        for wheel_path in problem_paths:
            wheel_prim = self.world.stage.GetPrimAtPath(wheel_path)
            if wheel_prim and wheel_prim.IsValid():
                if wheel_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rigid_body = UsdPhysics.RigidBodyAPI(wheel_prim)
                    rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                    print(f"  修正轮子视觉物理: {wheel_path.split('/')[-1]}")
                
                if wheel_prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision = UsdPhysics.CollisionAPI(wheel_prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
        
        print("保守物理层次结构修正完成")
    
    def _setup_robust_control_gains(self):
        """设置鲁棒的控制增益"""
        print("设置控制增益...")
        
        articulation_controller = self.mobile_base.get_articulation_controller()
        num_dofs = len(self.mobile_base.dof_names)
        kp = torch.zeros(num_dofs, dtype=torch.float32)
        kd = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 轮子控制
        wheel_names = ["left_wheel_joint", "right_wheel_joint"]
        for wheel_name in wheel_names:
            if wheel_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(wheel_name)
                kp[idx] = 0.0
                kd[idx] = 500.0
        
        # 机械臂控制
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        for joint_name in arm_joint_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 800.0
                kd[idx] = 40.0
        
        # 夹爪控制
        gripper_names = ["panda_finger_joint1", "panda_finger_joint2"]
        for joint_name in gripper_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 1e5
                kd[idx] = 1e3
        
        articulation_controller.set_gains(kps=kp, kds=kd)
        print("控制增益设置完成")
    
    def _move_arm_to_pose(self, pose_name: str):
        """移动机械臂到指定姿态 - 使用test.py验证参数"""
        print(f"机械臂移动到: {pose_name}")
        
        target_positions = self.arm_poses[pose_name]
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        # 获取当前关节位置，保持夹爪状态
        current_joint_positions = self.mobile_base.get_joint_positions()
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 保持当前所有关节位置
        for i in range(num_dofs):
            joint_positions[i] = float(current_joint_positions[i])
        
        # 只设置机械臂关节位置，保持夹爪位置不变
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        set_joints = 0
        
        for i, joint_name in enumerate(arm_joint_names):
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                if i < len(target_positions):
                    joint_positions[idx] = target_positions[i]
                    set_joints += 1
        
        # 应用动作
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        # 等待稳定 - 使用test.py验证的时间
        stabilization_steps = int(self.arm_stabilization_time * 60)  # 60 FPS
        for _ in range(stabilization_steps):
            self.world.step(render=True)
        
        print(f"  机械臂移动完成: {pose_name} (设置{set_joints}个关节，稳定{self.arm_stabilization_time}s)")
    
    def plan_coverage_mission(self):
        """规划覆盖任务"""
        print("规划四类对象覆盖任务...")
        
        current_pos, _ = self.robot_controller._get_robot_pose()
        print(f"机器人当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
        
        # 生成覆盖路径
        self.coverage_path = self.path_planner.generate_coverage_path(current_pos)
        
        # 设置完整路径可视化
        self.visualizer.setup_complete_path_visualization(self.coverage_path)
        
        # 新增：设置路径到机器人控制器用于动态重规划
        self.robot_controller.set_coverage_path(self.coverage_path)
        
        print(f"KLT框子任务区域覆盖规划完成: {len(self.coverage_path)}个覆盖点")
        print("已集成超丝滑实时覆盖区域可视化系统")
        print("激光雷达实时避障系统已启用")
        print("真实Franka机械臂抓取系统已准备就绪")
        print_memory_usage("覆盖规划完成")
        
        return True
    
    def execute_four_object_coverage(self):
        """执行四类对象覆盖 - 集成激光雷达避障和真实抓取"""
        print("\n开始执行四类对象覆盖（KLT框子任务区域版 + 激光雷达避障 + 真实Franka抓取）...")
        print_memory_usage("任务开始")
        
        # 展示路径预览
        print("展示路径预览...")
        for step in range(60):
            self.world.step(render=True)
            time.sleep(0.03)
        
        print("开始执行路径...")
        
        successful_points = 0
        step_counter = 0
        
        for i, point in enumerate(self.coverage_path):
            print(f"\n=== 导航到点 {i+1}/{len(self.coverage_path)} ===")
            
            # 检查激光雷达避障状态
            if self.lidar_avoidance:
                status = self.lidar_avoidance.get_avoidance_status()
                if status['mode'] != 'NORMAL':
                    print(f"  激光雷达状态: {status['mode']}")
                    if status['mode'] == 'DANGER':
                        self.coverage_stats['emergency_stops'] += 1
            
            # 鲁棒移动到目标点 - 内置激光雷达避障
            success = self.robot_controller.move_to_position_robust(point.position, point.orientation)
            
            if success:
                successful_points += 1
                print(f"  点 {i+1} 导航成功")
            else:
                print(f"  点 {i+1} 导航失败，继续下一个点")
            
            # 获取当前位置
            current_pos, _ = self.robot_controller._get_robot_pose()
            
            # 检查四类对象交互 - 集成真实抓取
            self._check_four_object_interactions(current_pos)
            
            # 短暂停顿
            for _ in range(5):
                self.world.step(render=True)
            
            time.sleep(0.1)
        
        print(f"\n四类对象覆盖执行完成!")
        print(f"成功到达点数: {successful_points}/{len(self.coverage_path)}")
        self._show_four_object_results()
    
    def _check_four_object_interactions(self, robot_pos: np.ndarray):
        """检查四类对象交互 - 集成真实抓取"""
        for scene_obj in self.scene_objects:
            if not scene_obj.is_active:
                continue
            
            # 检查碰撞
            if self.path_planner.check_collision_with_object(robot_pos, scene_obj):
                self._handle_object_interaction(scene_obj, robot_pos)
    
    def _handle_object_interaction(self, scene_obj: SceneObject, robot_pos: np.ndarray):
        """处理对象交互 - 集成真实抓取"""
        print(f"    交互检测: {scene_obj.name} ({scene_obj.object_type.value})")
        
        if scene_obj.object_type == ObjectType.SWEEP:
            self._handle_sweep_object(scene_obj)
        elif scene_obj.object_type == ObjectType.GRASP:
            self._handle_grasp_object_with_real_arm(scene_obj, robot_pos)
    
    def _handle_sweep_object(self, sweep_obj: SceneObject):
        """处理清扫对象"""
        print(f"      清扫目标消失: {sweep_obj.name}")
        
        # 隐藏对象
        sweep_obj.isaac_object.set_visibility(False)
        sweep_obj.isaac_object.set_world_pose(
            np.array([100.0, 100.0, -5.0]), 
            np.array([0, 0, 0, 1])
        )
        
        # 标记为非活跃
        sweep_obj.is_active = False
        self.coverage_stats['swept_objects'] += 1
        
        print(f"      清扫完成，总清扫数: {self.coverage_stats['swept_objects']}")
    
    def _handle_grasp_object_with_real_arm(self, grasp_obj: SceneObject, robot_pos: np.ndarray):
        """处理抓取对象 - 真实Franka机械臂抓取 - 使用test.py验证参数"""
        print(f"      检测到抓取对象: {grasp_obj.name}")
        
        # 如果已经在运送其他物体，跳过
        if self.carrying_object is not None:
            print(f"      已在运送物体，跳过")
            return
        
        # 检查是否在抓取触发距离内
        distance = np.linalg.norm(robot_pos[:2] - grasp_obj.position[:2])
        print(f"      当前距离抓取对象: {distance:.3f}m (触发阈值: {self.grasp_trigger_distance}m)")
        
        if distance > self.grasp_trigger_distance:
            print(f"      距离超出触发范围，跳过抓取")
            return
        
        print(f"      触发抓取任务! 距离: {distance:.3f}m")
        
        # 记录返回位置
        self.return_position = robot_pos.copy()
        
        # 执行完整抓取序列 - 使用test.py验证的流程
        self._execute_full_grasp_sequence_with_test_params(grasp_obj)
        
        # 运送到KLT框子
        self._deliver_to_klt_box_with_real_arm(grasp_obj)
        
        # 返回继续覆盖
        self._return_to_coverage()
    
    def _execute_full_grasp_sequence_with_test_params(self, grasp_obj: SceneObject):
        """执行完整抓取序列 - 使用test.py验证参数和流程"""
        print(f"        执行真实Franka机械臂抓取序列 (test.py验证参数)...")
        
        # 步骤1: 定位到抓取对象正前方 - 精确对准
        self._position_for_grasp_precise(grasp_obj)
        
        # 步骤2: 机械臂执行抓取动作 - test.py流程
        #print("        2.1 闭合夹爪抓取")
        #self._control_gripper_with_test_params("close")

        self._control_gripper_with_test_params("open")
        print("        2.2 机械臂移动到抓取姿态")
        self._move_arm_to_pose("grasp_approach")
        self._control_gripper_with_test_params("close")
        print("        2.3 抬起物体")
        self._move_arm_to_pose("grasp_lift")
        
        # 标记为运送中 - 不消失物体，通过真实抓取移动
        self.carrying_object = grasp_obj
        grasp_obj.is_active = False
        self.coverage_stats['grasped_objects'] += 1
        
        print(f"        Franka机械臂抓取完成 - 物体保持可见状态")
    
    def _position_for_grasp_precise(self, grasp_obj: SceneObject):
        """精确定位到抓取对象 - 简化为直接朝向移动"""
        print(f"          直接朝向抓取对象移动...")
        
        object_pos = grasp_obj.position
        current_pos, current_yaw = self.robot_controller._get_robot_pose()
        
        # 计算从当前位置到对象的方向
        direction_to_object = object_pos[:2] - current_pos[:2]
        distance_to_object = np.linalg.norm(direction_to_object)
        
        print(f"          对象位置: [{object_pos[0]:.3f}, {object_pos[1]:.3f}]")
        print(f"          当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
        print(f"          当前距离: {distance_to_object:.3f}m")
        
        # 计算目标角度 - 面向抓取对象
        target_yaw = np.arctan2(direction_to_object[1], direction_to_object[0])
        print(f"          目标角度: {np.degrees(target_yaw):.1f}°")
        
        # 先精确角度调整
        self._precise_angle_adjustment(target_yaw)
        
        # 计算停止位置 - 距离对象0.37米
        if distance_to_object > self.grasp_approach_distance:
            direction_norm = direction_to_object / distance_to_object
            approach_pos = object_pos[:2] - direction_norm * self.grasp_approach_distance
            approach_pos_3d = np.array([approach_pos[0], approach_pos[1], 0.0])
            
            print(f"          接近位置: [{approach_pos_3d[0]:.3f}, {approach_pos_3d[1]:.3f}]")
            
            # 移动到接近位置
            success = self.robot_controller.move_to_position_robust(approach_pos_3d, target_yaw)
        else:
            print(f"          已在接近距离内，无需移动")
            success = True
        
        print(f"          直接抓取定位完成")
    
    def _precise_angle_adjustment(self, target_yaw: float):
        """精确角度调整 - 优化角度控制"""
        print(f"            执行精确角度调整到: {np.degrees(target_yaw):.1f}°")
        
        max_angle_steps = 250
        
        for step in range(max_angle_steps):
            current_pos, current_yaw = self.robot_controller._get_robot_pose()
            
            # 计算角度误差
            angle_error = self._normalize_angle(target_yaw - current_yaw)
            
            # 优化：使用更小的角度容差
            if abs(angle_error) < self.angle_tolerance:
                print(f"            角度调整完成! 误差: {np.degrees(angle_error):.2f}°")
                break
            
            # 优化：使用更大的角速度增益和最大角速度
            angular_vel = np.clip(self.angular_kp * angle_error, 
                                -self.max_angular_vel, self.max_angular_vel)
            
            # 发送角度调整指令
            self._send_velocity_command(0.0, angular_vel)
            self.world.step(render=True)
            
            if step % 50 == 0:
                print(f"              角度调整中... 误差: {np.degrees(angle_error):.2f}°")
        
        # 停止旋转
        self._send_velocity_command(0.0, 0.0)
        
        # 最终验证角度
        final_pos, final_yaw = self.robot_controller._get_robot_pose()
        final_angle_error = self._normalize_angle(target_yaw - final_yaw)
        print(f"            最终角度误差: {np.degrees(final_angle_error):.2f}°")
    
    def _send_velocity_command(self, linear_vel: float, angular_vel: float):
        """发送速度指令到轮子"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        # 轮子参数
        wheel_radius = 0.036
        wheel_base = 0.235
        
        # 计算轮子速度
        left_wheel_vel = (linear_vel - angular_vel * wheel_base / 2.0) / wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * wheel_base / 2.0) / wheel_radius
        
        # 构建关节速度
        num_dofs = len(self.mobile_base.dof_names)
        joint_velocities = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 设置轮子速度
        if "left_wheel_joint" in self.mobile_base.dof_names:
            left_idx = self.mobile_base.dof_names.index("left_wheel_joint")
            joint_velocities[left_idx] = float(left_wheel_vel)
        
        if "right_wheel_joint" in self.mobile_base.dof_names:
            right_idx = self.mobile_base.dof_names.index("right_wheel_joint")
            joint_velocities[right_idx] = float(right_wheel_vel)
        
        # 应用控制指令
        action = ArticulationAction(joint_velocities=joint_velocities)
        articulation_controller.apply_action(action)
    
    def _normalize_angle(self, angle):
        """角度归一化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _control_gripper_with_test_params(self, action: str):
        """控制夹爪 - 使用test.py验证参数"""
        print(f"    执行夹爪动作: {action} (test.py参数)")
        
        gripper_names = ["panda_finger_joint1", "panda_finger_joint2"]
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        # 设置夹爪位置 - 使用test.py验证参数
        if action == "close":
            gripper_pos = self.gripper_close_pos
        else:  # "open"
            gripper_pos = self.gripper_open_pos
        
        # 获取当前关节位置，避免其他关节移动
        current_joint_positions = self.mobile_base.get_joint_positions()
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 保持当前所有关节位置 - 确保类型转换
        for i in range(num_dofs):
            joint_positions[i] = float(current_joint_positions[i])
        
        # 只修改夹爪关节位置
        for joint_name in gripper_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                joint_positions[idx] = gripper_pos
        
        action_obj = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action_obj)
        
        # 夹爪稳定时间 - 使用test.py验证参数
        stabilization_steps = int(self.gripper_stabilization_time * 60)  # 60 FPS
        for _ in range(stabilization_steps):
            self.world.step(render=True)
        
        print(f"    夹爪动作完成: {action} -> {gripper_pos}m (稳定{self.gripper_stabilization_time}s)")
    
    def _ensure_gripper_closed_tight(self):
        """确保夹爪紧密关闭 - 用于运输过程中防止松动"""
        print(f"          检查夹爪状态，确保紧密关闭...")
        
        gripper_names = ["panda_finger_joint1", "panda_finger_joint2"]
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        # 获取当前关节位置
        current_joint_positions = self.mobile_base.get_joint_positions()
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 保持当前所有关节位置
        for i in range(num_dofs):
            joint_positions[i] = float(current_joint_positions[i])
        
        # 强制设置夹爪为关闭位置
        for joint_name in gripper_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                joint_positions[idx] = self.gripper_close_pos
        
        # 应用关节控制
        action_obj = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action_obj)
        
        # 短暂稳定
        for _ in range(30):  # 0.5秒稳定时间
            self.world.step(render=True)
        
        print(f"          夹爪状态检查完成，确保紧密关闭到 {self.gripper_close_pos}m")
    
    def _deliver_to_klt_box_with_real_arm(self, grasp_obj: SceneObject):
        """运送到KLT框子 - 真实机械臂版本，加强夹爪稳定性"""
        print(f"        运送到KLT框子...")
        
        # 找到KLT框子任务区域
        klt_boxes = [obj for obj in self.scene_objects if obj.object_type == ObjectType.TASK and obj.is_active]
        if not klt_boxes:
            print(f"        没有找到KLT框子")
            return
            
        klt_box = klt_boxes[0]
        klt_position = klt_box.position.copy()
        
        # **运输前，确保夹爪紧密关闭**
        print(f"        运输前检查，确保夹爪紧密关闭...")
        self._ensure_gripper_closed_tight()
        
        # 计算最优接近位置 - 优化：使用更近的距离
        current_pos, _ = self.robot_controller._get_robot_pose()
        
        # 定义KLT框子四周的接近点 - 优化：使用更近的距离
        approach_offsets = [
            [self.klt_approach_distance, 0.0],   # 右侧
            [-self.klt_approach_distance, 0.0],  # 左侧  
            [0.0, self.klt_approach_distance],   # 前面
            [0.0, -self.klt_approach_distance],  # 后面
            [self.klt_approach_distance * 0.7, self.klt_approach_distance * 0.7],   # 右前
            [-self.klt_approach_distance * 0.7, self.klt_approach_distance * 0.7],  # 左前
            [self.klt_approach_distance * 0.7, -self.klt_approach_distance * 0.7],  # 右后
            [-self.klt_approach_distance * 0.7, -self.klt_approach_distance * 0.7], # 左后
        ]
        
        # 选择最近的安全接近点
        approach_candidates = []
        for offset in approach_offsets:
            candidate_pos = klt_position.copy()
            candidate_pos[0] += offset[0]
            candidate_pos[1] += offset[1] 
            candidate_pos[2] = 0.0
            
            # 检查候选点是否与障碍物冲突
            is_safe = True
            for scene_obj in self.scene_objects:
                if scene_obj.object_type == ObjectType.OBSTACLE and scene_obj.is_active:
                    if self.path_planner.check_collision_with_object(candidate_pos, scene_obj):
                        is_safe = False
                        break
            
            if is_safe:
                distance = np.linalg.norm(current_pos[:2] - candidate_pos[:2])
                approach_candidates.append((candidate_pos, distance, offset))
        
        # 如果没有安全的接近点，使用最远接近点
        if not approach_candidates:
            print(f"        警告: 所有接近点都有障碍物冲突，使用最远接近点")
            for offset in approach_offsets:
                candidate_pos = klt_position.copy()
                candidate_pos[0] += offset[0]
                candidate_pos[1] += offset[1] 
                candidate_pos[2] = 0.0
                distance = np.linalg.norm(current_pos[:2] - candidate_pos[:2])
                approach_candidates.append((candidate_pos, distance, offset))
        
        # 选择距离最近的安全接近点
        approach_candidates.sort(key=lambda x: x[1])
        approach_pos, min_distance, chosen_offset = approach_candidates[0]
        
        print(f"        智能选择接近方向: 偏移[{chosen_offset[0]:.1f}, {chosen_offset[1]:.1f}]")
        print(f"        导航到KLT框子接近位置: [{approach_pos[0]:.3f}, {approach_pos[1]:.3f}]")
        
        # **导航到接近位置（集成激光雷达避障）**
        success = self.robot_controller.move_to_position_robust(approach_pos)
        
        # **导航过程中检查夹爪状态**
        print(f"        导航完成，检查夹爪状态...")
        self._ensure_gripper_closed_tight()
        
        if success:
            # 检查是否足够接近KLT框子
            current_pos, _ = self.robot_controller._get_robot_pose()
            distance_to_klt = np.linalg.norm(current_pos[:2] - klt_position[:2])
            
            print(f"        当前到KLT框子距离: {distance_to_klt:.3f}m")
            
            # 优化：使用更宽松的距离检查，因为接近距离已经减小
            if distance_to_klt <= self.klt_approach_distance + 0.3:
                # **到达后再次确保夹爪关闭**
                print(f"        到达KLT框子，最后检查夹爪状态...")
                self._ensure_gripper_closed_tight()
                
                # 足够接近，可以投放
                self._perform_release_with_real_arm_and_physics(grasp_obj, klt_box)
                print(f"        运送到KLT框子完成")
            else:
                print(f"        距离KLT框子太远: {distance_to_klt:.3f}m > {self.klt_approach_distance + 0.3:.3f}m")
        else:
            print(f"        导航到KLT框子失败")
    
    def _perform_release_with_real_arm_and_physics(self, grasp_obj: SceneObject, klt_box: SceneObject):
        """执行真实机械臂释放到KLT框子 - 物体自然掉落"""
        print(f"          执行真实Franka机械臂释放...")
        
        # 机械臂移动到释放姿态并张开夹爪
        self._move_arm_to_pose("carry")  # 保持抬起状态
        self._control_gripper_with_test_params("open")    # 张开夹爪，物体自然掉落
        
        # 观察物体掉落过程
        for _ in range(30):
            self.world.step(render=True)
            time.sleep(0.02)
        
        print(f"          物体自然掉落完成")
        
        # 机械臂回到home姿态
        self._move_arm_to_pose("home")
        
        # 更新统计
        self.carrying_object = None
        self.coverage_stats['delivered_objects'] += 1
        
        print(f"          真实Franka机械臂释放完成")
    
    def _return_to_coverage(self):
        """返回覆盖位置"""
        if self.return_position is not None:
            print(f"        返回覆盖位置: [{self.return_position[0]:.3f}, {self.return_position[1]:.3f}]")
            self.robot_controller.move_to_position_robust(self.return_position)
            self.return_position = None
            print(f"        返回完成，继续覆盖")
    
    def _show_four_object_results(self):
        """显示四类对象结果 - 集成激光雷达避障统计"""
        coverage_marks = len(self.visualizer.fluent_coverage_visualizer.coverage_marks)
        
        # 统计各类对象
        obstacle_count = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.OBSTACLE])
        sweep_total = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.SWEEP])
        grasp_total = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.GRASP])
        klt_count = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.TASK])
        
        print(f"\n=== 四类对象覆盖结果（KLT框子任务区域版 + 激光雷达避障 + 真实Franka抓取） ===")
        print(f"覆盖路径点数: {len(self.coverage_path)}")
        print(f"超丝滑覆盖标记区域: {coverage_marks}个")
        print(f"")
        print(f"环境对象统计:")
        print(f"  障碍物: {obstacle_count}个 (避障)")
        print(f"  清扫目标: {sweep_total}个")
        print(f"  抓取物体: {grasp_total}个 (test.py验证尺寸)") 
        print(f"  KLT框子任务区域: {klt_count}个")
        print(f"")
        print(f"KLT框子配置:")
        print(f"  USD资产: {self.klt_box_usd_path}")
        print(f"  接近距离阈值: {self.klt_approach_distance}m (优化)")
        print(f"")
        print(f"真实Franka机械臂抓取配置 (test.py验证参数):")
        print(f"  抓取触发距离: {self.grasp_trigger_distance}m")
        print(f"  抓取接近距离: {self.grasp_approach_distance}m")
        print(f"  机械臂稳定时间: {self.arm_stabilization_time}s")
        print(f"  夹爪稳定时间: {self.gripper_stabilization_time}s")
        print(f"  夹爪张开/闭合位置: {self.gripper_open_pos}m / {self.gripper_close_pos}m")
        print(f"")
        print(f"优化参数:")
        print(f"  角度容差: {self.angle_tolerance}rad (更精确)")
        print(f"  位置容差: {self.position_tolerance}m")
        print(f"  角速度增益: {self.angular_kp} (提高)")
        print(f"  最大角速度: {self.max_angular_vel}rad/s (提高)")
        print(f"")
        print(f"任务执行统计:")
        print(f"  清扫完成: {self.coverage_stats['swept_objects']}/{sweep_total}")
        print(f"  抓取完成: {self.coverage_stats['grasped_objects']}/{grasp_total}")
        print(f"  投放到KLT框子: {self.coverage_stats['delivered_objects']}/{grasp_total}")
        
        # 新增：激光雷达避障统计
        if self.lidar_avoidance:
            avoidance_status = self.lidar_avoidance.get_avoidance_status()
            print(f"")
            print(f"激光雷达避障统计:")
            print(f"  紧急停车次数: {self.coverage_stats['emergency_stops']}")
            print(f"  当前避障模式: {avoidance_status['mode']}")
            print(f"  激光数据状态: {'正常' if avoidance_status['scan_available'] else '异常'}")
            print(f"  数据延迟: {avoidance_status['data_age']:.2f}秒")
        
        print(f"")
        sweep_rate = (self.coverage_stats['swept_objects'] / sweep_total * 100) if sweep_total > 0 else 0
        grasp_rate = (self.coverage_stats['delivered_objects'] / grasp_total * 100) if grasp_total > 0 else 0
        print(f"任务完成率:")
        print(f"  清扫任务: {sweep_rate:.1f}%")
        print(f"  KLT框子投放任务: {grasp_rate:.1f}%")
        print("KLT框子四类对象覆盖任务完成（集成激光雷达实时避障系统 + 真实Franka机械臂抓取）!")
    
    def run_demo(self):
        """运行演示"""
        print("\n" + "="*80)
        print("四类对象真实移动覆盖算法机器人系统 - KLT框子任务区域版 + 激光雷达避障 + 真实Franka抓取")
        print("障碍物避障 | 清扫目标消失 | 抓取运送 | KLT框子投放")
        print("优化路径规划 | 距离判断到达 | 智能弓字形避障")
        print("集成超丝滑实时覆盖区域可视化 | KLT框子USD资产")
        print("激光雷达分层避障 | 紧急避障 | 动态路径重规划")
        print("真实Franka机械臂抓取 | 精确关节控制 | 稳定化处理")
        print("优化特性: 精确角度控制 | 减小接近距离 | 提高控制精度")
        print("="*80)
        
        pos, yaw = self.robot_controller._get_robot_pose()
        print(f"机器人初始位置: [{pos[0]:.3f}, {pos[1]:.3f}], 朝向: {np.degrees(yaw):.1f}°")
        
        # 展示激光雷达状态
        if self.lidar_avoidance:
            status = self.lidar_avoidance.get_avoidance_status()
            print(f"激光雷达状态: {status}")
        
        # 展示机械臂配置
        print(f"Franka机械臂配置: {len(self.arm_poses)}个预设姿态 (test.py验证)")
        print(f"优化配置: 角度容差{self.angle_tolerance}rad, KLT接近距离{self.klt_approach_distance}m")
        
        self.plan_coverage_mission()
        
        self.execute_four_object_coverage()
        
        self._move_arm_to_pose("home")
        
        print("\nKLT框子四类对象覆盖系统演示完成!")
        print("激光雷达实时避障系统运行正常!")
        print("真实Franka机械臂抓取系统运行完美!")
        print("优化功能: 精确角度控制和距离优化已生效!")
    
    def cleanup(self):
        """清理系统"""
        print("清理系统资源...")
        print_memory_usage("清理前")
        
        if self.visualizer:
            self.visualizer.cleanup()
        
        # 新增：清理ROS资源
        if self.lidar_avoidance:
            # LaserScan订阅器会自动清理
            pass
        
        # 关闭ROS节点
        rospy.signal_shutdown("系统关闭")
        
        # 清理内存
        for _ in range(5):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.world:
            self.world.stop()
        
        print_memory_usage("清理后")
        print("系统资源清理完成")

def main():
    """主函数"""
    print("KLT框子任务区域版机器人覆盖系统 + 激光雷达实时避障 + 真实Franka机械臂抓取")
    print(f"机器人半径: {ROBOT_RADIUS}m")
    print(f"安全边距: {SAFETY_MARGIN}m") 
    print(f"交互距离: {INTERACTION_DISTANCE}m")
    print("优化特性: 高效弓字形避障路径，KLT框子物理碰撞")
    print("激光雷达避障特性:")
    print(f"  最近检测距离: 0.6m")
    print(f"  危险距离阈值: 0.7m") 
    print(f"  警告距离阈值: 1.0m")
    print(f"  安全距离阈值: 1.5m")
    print(f"  分层避障策略: 危险停止，警告减速，安全正常")
    print("")
    print("KLT框子特性:")
    print("  USD资产加载")
    print("  具有物理碰撞")
    print("  距离判断到达")
    print("  框内物体掉落投放")
    print("")
    print("真实Franka机械臂抓取特性 (test.py验证参数):")
    print("  7-DOF精确关节控制")
    print("  真实物理夹爪操作")
    print("  机械臂稳定化处理")
    print("  完整抓取-运送-释放流程")
    print("")
    print("优化特性:")
    print("  精确角度控制 - 角度容差0.05rad")
    print("  增强角速度控制 - 角速度增益5.0, 最大角速度2.0rad/s")
    print("  KLT接近距离优化 - 从1.2m减至1.0m")
    print("  位置容差优化 - 从0.2m减至0.15m")
    print("")
    print("超丝滑实时可视化特性:")
    print(f"  精细网格: {FINE_GRID_SIZE}m")
    print(f"  覆盖标记半径: {COVERAGE_MARK_RADIUS}m")
    print(f"  距离阈值: {MARK_DISTANCE_THRESHOLD}m")
    print(f"  更新频率: {COVERAGE_UPDATE_FREQUENCY}")
    print(f"  渐变颜色: 15档灰度系统")
    print(f"  实时跟踪: 超丝滑标记机器人移动轨迹")
    
    system = FourObjectCoverageSystem()
    
    print("\n=== 系统初始化阶段 ===")
    system.initialize_system()
    print("系统初始化完成")
    
    print("\n=== 机器人初始化阶段 ===")
    system.initialize_robot()
    print("机器人初始化完成")
    
    print("\n=== 后加载设置阶段 ===")
    system.setup_post_load()
    print("后加载设置完成")
    
    # 额外稳定，确保所有组件就绪
    print("\n=== 最终系统稳定阶段 ===")
    for i in range(60):
        system.world.step(render=False)
        if i % 20 == 0:
            print(f"  稳定进度: {i+1}/60")
    print("系统最终稳定完成")
    
    # 运行演示
    print("\n=== 演示执行阶段 ===")
    system.run_demo()
    
    # 保持运行观察效果
    print("\n=== 效果观察阶段 ===")
    for i in range(200):
        system.world.step(render=True)
        time.sleep(0.05)
        if i % 50 == 0:
            print(f"  观察进度: {i+1}/200")
    
    print("\n=== 系统清理阶段 ===")
    system.cleanup()
    print("程序结束")

if __name__ == "__main__":
    main()