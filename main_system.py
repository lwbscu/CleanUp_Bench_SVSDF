#!/usr/bin/env python3
"""
主系统模块 - KLT框子任务区域集成版
四类对象覆盖系统的主要逻辑和程序入口
"""

import psutil
import torch
import numpy as np
import time
import random
import gc
from typing import List, Dict

def print_memory_usage(stage_name: str = ""):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"内存: {memory_mb:.1f}MB - {stage_name}")

# 先初始化SimulationApp，再导入Isaac Sim模块
print("启动KLT框子任务区域版四类对象覆盖算法机器人系统...")
print(f"对象类型:")
print(f"  1. 障碍物(红色) - 路径规划避障")
print(f"  2. 清扫目标(黄色) - 触碰消失") 
print(f"  3. 抓取物体(绿色) - 运送到KLT框子")
print(f"  4. KLT框子任务区域 - 物体投放区域 (具有物理碰撞)")

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "enable_livestream": False, 
    "enable_cameras": True,
    "enable_rtx": True,
    "physics_dt": 1.0/60.0,
    "rendering_dt": 1.0/30.0,
})

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

class FourObjectCoverageSystem:
    """四类对象覆盖系统 - KLT框子任务区域版"""
    
    def __init__(self):
        self.world = None
        self.mobile_base = None
        self.robot_controller = None
        self.path_planner = None
        self.visualizer = None
        
        self.scene_objects = []
        self.coverage_path = []
        self.coverage_stats = {
            'swept_objects': 0,
            'grasped_objects': 0,
            'delivered_objects': 0,
            'total_coverage_points': 0
        }
        
        # KLT框子配置
        self.klt_box_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/NVIDIA/Assets/ArchVis/Lobby/My_asset/T/small_KLT.usd"
        self.klt_approach_distance = 1.2  # 机器人到达KLT框子的距离阈值 - 框子变大后调整距离
        
        # 机械臂配置
        self.arm_poses = {
            "home": [0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.785],
            "pickup": [0.0, 0.3, 0.0, -1.5, 0.0, 2.2, 0.785],
            "carry": [0.0, -0.4, 0.0, -2.0, 0.0, 1.8, 0.785]
        }
        
        # 抓取状态
        self.carrying_object = None
        self.return_position = None
    
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
        print(f"KLT框子资产: {self.klt_box_usd_path}")
        print(f"到达距离阈值: {self.klt_approach_distance}m")
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
        self.visualizer = RealMovementVisualizer(self.world)  # 已集成超丝滑覆盖可视化
        print("组件初始化完成（含超丝滑实时覆盖可视化）")
    
    def _create_four_type_environment(self):
        """创建四类对象环境 - KLT框子任务区域版"""
        print("创建四类对象环境（KLT框子任务区域）...")
        
        # 1. 障碍物 (红色) - 路径规划避障
        obstacles_config = [
            {"pos": [1.2, 0.8, 0.15], "size": [0.6, 0.4, 0.3], "color": [0.8, 0.2, 0.2], "shape": "box"},
            {"pos": [0.5, -1.5, 0.2], "size": [1.2, 0.3, 0.4], "color": [0.8, 0.1, 0.1], "shape": "box"},
            {"pos": [-1.0, 1.2, 0.25], "size": [0.5], "color": [0.7, 0.2, 0.2], "shape": "sphere"},
        ]
        
        # 2. 清扫目标 (黄色) - 触碰消失
        sweep_config = [
            {"pos": [0.8, 0.2, 0.05], "size": [0.1], "color": [1.0, 1.0, 0.2], "shape": "sphere"},
            {"pos": [1.5, 1.5, 0.05], "size": [0.1], "color": [0.9, 0.9, 0.1], "shape": "sphere"},
            {"pos": [-0.8, -0.8, 0.05], "size": [0.1], "color": [1.0, 0.8, 0.0], "shape": "sphere"},
            {"pos": [2.0, 0.5, 0.05], "size": [0.1], "color": [0.8, 0.8, 0.2], "shape": "sphere"},
        ]
        
        # 3. 抓取物体 (绿色) - 运送到KLT框子
        grasp_config = [
            {"pos": [1.8, -1.2, 0.08], "size": [0.16, 0.16, 0.16], "color": [0.2, 0.8, 0.2], "shape": "box"},
            {"pos": [-1.5, 0.5, 0.08], "size": [0.16, 0.16, 0.16], "color": [0.1, 0.9, 0.1], "shape": "box"},
            {"pos": [0.3, 1.8, 0.08], "size": [0.16, 0.16, 0.16], "color": [0.0, 0.8, 0.0], "shape": "box"},
        ]
        
        # 4. KLT框子任务区域 - 使用自定义USD资产，放大尺寸
        klt_box_config = [
            {"pos": [5, 0, 0.15], "size": [1.6, 0.8, 0.3], "usd_path": self.klt_box_usd_path, "shape": "usd_asset"},  # 尺寸放大2倍
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
        
        print(f"四类对象环境创建完成:")
        print(f"  障碍物: {len(obstacles_config)}个")
        print(f"  清扫目标: {len(sweep_config)}个") 
        print(f"  抓取物体: {len(grasp_config)}个")
        print(f"  KLT框子任务区域: {len(klt_box_config)}个")
    
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
            # KLT框子保持物理属性，支持碰撞掉落
        elif isaac_obj:
            # 只有标准Isaac对象才添加到场景
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
    
    def _create_klt_box_task_area(self, config: Dict, prim_path: str, name: str):
        """创建KLT框子任务区域"""
        try:
            # 创建基础prim并添加USD reference
            stage = self.world.stage
            klt_prim = stage.DefinePrim(prim_path, "Xform")
            
            # 添加USD引用
            references = klt_prim.GetReferences()
            references.AddReference(config["usd_path"])
            
            # 设置位置和缩放
            xform = UsdGeom.Xformable(klt_prim)
            xform.ClearXformOpOrder()
            
            # 位置变换
            translate_op = xform.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(float(config["pos"][0]), float(config["pos"][1]), float(config["pos"][2])))
            
            # 适当缩放KLT框子 - 放大2倍
            scale_op = xform.AddScaleOp()
            scale_op.Set(Gf.Vec3d(2.0, 2.0, 2.0))  # 放大2倍，根据需要调整
            
            print(f"  创建KLT框子任务区域: {name}")
            print(f"  USD资产: {config['usd_path']}")
            print(f"  位置: [{config['pos'][0]:.3f}, {config['pos'][1]:.3f}, {config['pos'][2]:.3f}]")
            
            # 返回一个简单的对象引用
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
            # 创建备用立方体
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
        
        robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
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
        
        # 修复物理层次结构
        self._fix_robot_physics_conservative()
        
        # 设置控制增益
        self._setup_robust_control_gains()
        
        # 移动机械臂到home位置
        self._move_arm_to_pose("home")
        
        # 初始化控制器
        self.robot_controller = FixedRobotController(self.mobile_base, self.world)
        
        # 关键：设置覆盖可视化器引用
        self.robot_controller.set_coverage_visualizer(self.visualizer)
        print("机器人控制器与超丝滑覆盖可视化器连接完成")
        
        # 验证机器人状态
        pos, yaw = self.robot_controller._get_robot_pose()
        print(f"机器人状态验证: 位置[{pos[0]:.3f}, {pos[1]:.3f}], 朝向{np.degrees(yaw):.1f}°")
        
        print("后加载设置完成")
        return True
    
    def _fix_robot_physics_conservative(self):
        """保守的机器人物理层次结构修复"""
        print("保守修复机器人物理层次结构...")
        
        robot_prim = self.world.stage.GetPrimAtPath("/World/create3_robot")
        
        # 只修复明确的问题轮子
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
                    print(f"  修复轮子视觉物理: {wheel_path.split('/')[-1]}")
                
                if wheel_prim.HasAPI(UsdPhysics.CollisionAPI):
                    collision = UsdPhysics.CollisionAPI(wheel_prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
        
        print("保守物理层次结构修复完成")
    
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
        """移动机械臂到指定姿态"""
        print(f"移动机械臂到: {pose_name}")
        
        target_positions = self.arm_poses[pose_name]
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 设置机械臂关节位置
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
        
        # 等待机械臂移动到位
        for _ in range(30):
            self.world.step(render=False)
        
        print(f"  机械臂移动完成: {pose_name} (设置{set_joints}个关节)")
    
    def plan_coverage_mission(self):
        """规划覆盖任务"""
        print("规划四类对象覆盖任务...")
        
        current_pos, _ = self.robot_controller._get_robot_pose()
        print(f"机器人当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
        
        # 生成覆盖路径
        self.coverage_path = self.path_planner.generate_coverage_path(current_pos)
        
        # 设置完整路径可视化
        self.visualizer.setup_complete_path_visualization(self.coverage_path)
        
        print(f"KLT框子任务区域覆盖规划完成: {len(self.coverage_path)}个覆盖点")
        print("已集成超丝滑实时覆盖区域可视化系统")
        print_memory_usage("覆盖规划完成")
        
        return True
    
    def execute_four_object_coverage(self):
        """执行四类对象覆盖 - KLT框子版本"""
        print("\n开始执行四类对象覆盖（KLT框子任务区域版）...")
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
            
            # 鲁棒移动到目标点 - 内置超丝滑覆盖标记
            success = self.robot_controller.move_to_position_robust(point.position, point.orientation)
            
            if success:
                successful_points += 1
                print(f"  点 {i+1} 导航成功")
            else:
                print(f"  点 {i+1} 导航失败，继续下一个点")
            
            # 获取当前位置
            current_pos, _ = self.robot_controller._get_robot_pose()
            
            # 检查四类对象交互
            self._check_four_object_interactions(current_pos)
            
            # 短暂停顿
            for _ in range(5):
                self.world.step(render=True)
            
            time.sleep(0.1)
        
        print(f"\n四类对象覆盖执行完成!")
        print(f"成功到达点数: {successful_points}/{len(self.coverage_path)}")
        self._show_four_object_results()
    
    def _check_four_object_interactions(self, robot_pos: np.ndarray):
        """检查四类对象交互"""
        for scene_obj in self.scene_objects:
            if not scene_obj.is_active:
                continue
            
            # 检查碰撞
            if self.path_planner.check_collision_with_object(robot_pos, scene_obj):
                self._handle_object_interaction(scene_obj, robot_pos)
    
    def _handle_object_interaction(self, scene_obj: SceneObject, robot_pos: np.ndarray):
        """处理对象交互"""
        print(f"    交互检测: {scene_obj.name} ({scene_obj.object_type.value})")
        
        if scene_obj.object_type == ObjectType.SWEEP:
            self._handle_sweep_object(scene_obj)
        elif scene_obj.object_type == ObjectType.GRASP:
            self._handle_grasp_object(scene_obj, robot_pos)
    
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
    
    def _handle_grasp_object(self, grasp_obj: SceneObject, robot_pos: np.ndarray):
        """处理抓取对象 - KLT框子版本"""
        print(f"      抓取物体: {grasp_obj.name}")
        
        # 如果已经在运送其他物体，跳过
        if self.carrying_object is not None:
            print(f"      已在运送物体，跳过")
            return
        
        # 记录返回位置
        self.return_position = robot_pos.copy()
        
        # 执行抓取
        self._perform_grasp_sequence(grasp_obj)
        
        # 运送到KLT框子
        self._deliver_to_klt_box(grasp_obj)
        
        # 返回继续覆盖
        self._return_to_coverage()
    
    def _perform_grasp_sequence(self, grasp_obj: SceneObject):
        """执行抓取序列"""
        print(f"        执行抓取动作...")
        
        # 机械臂抓取动作
        self._move_arm_to_pose("pickup")
        self._control_gripper("close")
        self._move_arm_to_pose("carry")
        
        # 隐藏原物体
        grasp_obj.isaac_object.set_visibility(False)
        grasp_obj.isaac_object.set_world_pose(
            np.array([100.0, 100.0, -5.0]), 
            np.array([0, 0, 0, 1])
        )
        
        # 标记为运送中
        self.carrying_object = grasp_obj
        grasp_obj.is_active = False
        self.coverage_stats['grasped_objects'] += 1
        
        print(f"        抓取完成")
    
    def _deliver_to_klt_box(self, grasp_obj: SceneObject):
        """运送到KLT框子 - 距离判断版本"""
        print(f"        运送到KLT框子...")
        
        # 找到KLT框子任务区域
        klt_boxes = [obj for obj in self.scene_objects if obj.object_type == ObjectType.TASK and obj.is_active]
        if not klt_boxes:
            print(f"        没有找到KLT框子")
            return
            
        klt_box = klt_boxes[0]  # 使用第一个KLT框子
        klt_position = klt_box.position.copy()
        
        # 计算最优接近位置 - 从KLT框子四周最近的方向接近
        current_pos, _ = self.robot_controller._get_robot_pose()
        
        # 定义KLT框子四周的接近点（相对于框子中心的偏移）
        approach_offsets = [
            [self.klt_approach_distance, 0.0],   # 右侧
            [-self.klt_approach_distance, 0.0],  # 左侧  
            [0.0, self.klt_approach_distance],   # 前面
            [0.0, -self.klt_approach_distance],  # 后面
            # 对角线接近点，距离稍远一些
            [self.klt_approach_distance * 0.7, self.klt_approach_distance * 0.7],   # 右前
            [-self.klt_approach_distance * 0.7, self.klt_approach_distance * 0.7],  # 左前
            [self.klt_approach_distance * 0.7, -self.klt_approach_distance * 0.7],  # 右后
            [-self.klt_approach_distance * 0.7, -self.klt_approach_distance * 0.7], # 左后
        ]
        
        # 计算所有可能接近点的位置，并检查是否安全
        approach_candidates = []
        for offset in approach_offsets:
            candidate_pos = klt_position.copy()
            candidate_pos[0] += offset[0]
            candidate_pos[1] += offset[1] 
            candidate_pos[2] = 0.0  # 地面高度
            
            # 检查候选点是否与障碍物冲突
            is_safe = True
            for scene_obj in self.scene_objects:
                if scene_obj.object_type == ObjectType.OBSTACLE and scene_obj.is_active:
                    if self.path_planner.check_collision_with_object(candidate_pos, scene_obj):
                        is_safe = False
                        break
            
            if is_safe:
                # 计算从当前位置到候选点的距离
                distance = np.linalg.norm(current_pos[:2] - candidate_pos[:2])
                approach_candidates.append((candidate_pos, distance, offset))
        
        # 如果没有安全的接近点，使用最远的点
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
        approach_candidates.sort(key=lambda x: x[1])  # 按距离排序
        approach_pos, min_distance, chosen_offset = approach_candidates[0]
        
        print(f"        智能选择接近方向: 偏移[{chosen_offset[0]:.1f}, {chosen_offset[1]:.1f}]")
        print(f"        从当前位置到接近点距离: {min_distance:.3f}m")
        print(f"        安全接近点数量: {len(approach_candidates)}")
        
        print(f"        导航到KLT框子接近位置: [{approach_pos[0]:.3f}, {approach_pos[1]:.3f}]")
        print(f"        KLT框子位置: [{klt_position[0]:.3f}, {klt_position[1]:.3f}]")
        print(f"        接近距离阈值: {self.klt_approach_distance}m")
        
        # 导航到接近位置
        success = self.robot_controller.move_to_position_robust(approach_pos)
        
        if success:
            # 检查是否足够接近KLT框子
            current_pos, _ = self.robot_controller._get_robot_pose()
            distance_to_klt = np.linalg.norm(current_pos[:2] - klt_position[:2])
            
            print(f"        当前到KLT框子距离: {distance_to_klt:.3f}m")
            
            if distance_to_klt <= self.klt_approach_distance + 0.2:  # 允许一定误差
                # 足够接近，可以投放
                self._perform_release_to_klt_box(grasp_obj, klt_box)
                print(f"        运送到KLT框子完成")
            else:
                print(f"        距离KLT框子太远: {distance_to_klt:.3f}m > {self.klt_approach_distance + 0.2:.3f}m")
        else:
            print(f"        导航到KLT框子失败")
    
    def _perform_release_to_klt_box(self, grasp_obj: SceneObject, klt_box: SceneObject):
        """执行释放到KLT框子"""
        print(f"          执行投放到KLT框子...")
        
        # 机械臂放置动作
        self._move_arm_to_pose("pickup")
        self._control_gripper("open")
        self._move_arm_to_pose("home")
        
        # 在KLT框子内随机分散位置释放物体
        klt_position = klt_box.position.copy()
        klt_dimensions = klt_box.collision_boundary.dimensions
        
        # 计算KLT框子内的安全释放区域（避开边缘）
        safe_margin = 0.2  # 离边缘的安全距离
        x_range = max(0.2, (klt_dimensions[0] - safe_margin * 2))  # 确保最小范围
        y_range = max(0.2, (klt_dimensions[1] - safe_margin * 2))
        
        # 在KLT框子内随机选择释放位置
        random_x_offset = random.uniform(-x_range/2, x_range/2)
        random_y_offset = random.uniform(-y_range/2, y_range/2)
        
        drop_position = klt_position.copy()
        drop_position[0] += random_x_offset
        drop_position[1] += random_y_offset
        drop_position[2] = klt_position[2] + 1.0  # 在KLT框子上方一定高度释放
        
        print(f"          随机分散释放位置: [{drop_position[0]:.3f}, {drop_position[1]:.3f}, {drop_position[2]:.3f}]")
        print(f"          相对KLT框子偏移: [{random_x_offset:.3f}, {random_y_offset:.3f}]")
        
        # 创建新的投放物体（具有物理属性的动态物体）
        drop_name = f"delivered_{grasp_obj.name}"
        drop_prim_path = f"/World/{drop_name}"
        
        # 使用DynamicCuboid创建具有物理掉落效果的物体
        drop_obj = DynamicCuboid(
            prim_path=drop_prim_path,
            name=drop_name,
            position=drop_position,
            scale=grasp_obj.collision_boundary.dimensions * 0.9,  # 稍微小一点放入框内
            color=grasp_obj.color * 0.7  # 稍微暗一点表示已放置
        )
        
        self.world.scene.add(drop_obj)
        
        # 等待物体创建完成
        for _ in range(3):
            self.world.step(render=False)
        
        # 给物体添加随机的初始速度，增加分散效果
        try:
            # 随机的水平初始速度
            random_velocity_x = random.uniform(-0.5, 0.5)
            random_velocity_y = random.uniform(-0.5, 0.5)
            random_velocity_z = random.uniform(-0.2, 0.2)  # 轻微的垂直速度变化
            
            # 设置线性速度
            drop_obj.set_linear_velocity(np.array([random_velocity_x, random_velocity_y, random_velocity_z]))
            
            # 添加轻微的角速度，让物体有旋转效果
            random_angular_velocity = np.array([
                random.uniform(-1.0, 1.0),
                random.uniform(-1.0, 1.0), 
                random.uniform(-1.0, 1.0)
            ])
            drop_obj.set_angular_velocity(random_angular_velocity)
            
            print(f"          添加随机初始速度: 线性[{random_velocity_x:.2f}, {random_velocity_y:.2f}, {random_velocity_z:.2f}]")
            
        except Exception as e:
            print(f"          设置初始速度失败: {e}")
        
        # 观察物体掉落过程
        for _ in range(15):
            self.world.step(render=True)
            time.sleep(0.03)
        
        print(f"          物体在KLT框子内随机位置掉落（具有物理碰撞）")
        
        # 更新统计
        self.carrying_object = None
        self.coverage_stats['delivered_objects'] += 1
    
    def _return_to_coverage(self):
        """返回覆盖位置"""
        if self.return_position is not None:
            print(f"        返回覆盖位置: [{self.return_position[0]:.3f}, {self.return_position[1]:.3f}]")
            self.robot_controller.move_to_position_robust(self.return_position)
            self.return_position = None
            print(f"        返回完成，继续覆盖")
    
    def _control_gripper(self, action):
        """控制夹爪"""
        gripper_names = ["panda_finger_joint1", "panda_finger_joint2"]
        available_gripper_joints = [name for name in gripper_names if name in self.mobile_base.dof_names]
        
        articulation_controller = self.mobile_base.get_articulation_controller()
        gripper_pos = 0.0 if action == "close" else 0.04
        
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        controlled_joints = 0
        for joint_name in available_gripper_joints:
            idx = self.mobile_base.dof_names.index(joint_name)
            joint_positions[idx] = gripper_pos
            controlled_joints += 1
        
        action_obj = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action_obj)
        
        for _ in range(15):
            self.world.step(render=False)
        
        print(f"    夹爪控制完成: {action} ({controlled_joints}个关节)")
    
    def _show_four_object_results(self):
        """显示四类对象结果 - KLT框子版本"""
        coverage_marks = len(self.visualizer.fluent_coverage_visualizer.coverage_marks)
        
        # 统计各类对象
        obstacle_count = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.OBSTACLE])
        sweep_total = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.SWEEP])
        grasp_total = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.GRASP])
        klt_count = len([obj for obj in self.scene_objects if obj.object_type == ObjectType.TASK])
        
        print(f"\n=== 四类对象覆盖结果（KLT框子任务区域版） ===")
        print(f"覆盖路径点数: {len(self.coverage_path)}")
        print(f"超丝滑覆盖标记区域: {coverage_marks}个")
        print(f"")
        print(f"环境对象统计:")
        print(f"  障碍物: {obstacle_count}个 (避障)")
        print(f"  清扫目标: {sweep_total}个")
        print(f"  抓取物体: {grasp_total}个") 
        print(f"  KLT框子任务区域: {klt_count}个")
        print(f"")
        print(f"KLT框子配置:")
        print(f"  USD资产: {self.klt_box_usd_path}")
        print(f"  接近距离阈值: {self.klt_approach_distance}m")
        print(f"")
        print(f"任务执行统计:")
        print(f"  清扫完成: {self.coverage_stats['swept_objects']}/{sweep_total}")
        print(f"  抓取完成: {self.coverage_stats['grasped_objects']}/{grasp_total}")
        print(f"  投放到KLT框子: {self.coverage_stats['delivered_objects']}/{grasp_total}")
        print(f"")
        sweep_rate = (self.coverage_stats['swept_objects'] / sweep_total * 100) if sweep_total > 0 else 0
        grasp_rate = (self.coverage_stats['delivered_objects'] / grasp_total * 100) if grasp_total > 0 else 0
        print(f"任务完成率:")
        print(f"  清扫任务: {sweep_rate:.1f}%")
        print(f"  KLT框子投放任务: {grasp_rate:.1f}%")
        print("KLT框子四类对象覆盖任务完成（集成超丝滑实时覆盖可视化）!")
    
    def run_demo(self):
        """运行演示"""
        print("\n" + "="*80)
        print("四类对象真实移动覆盖算法机器人系统 - KLT框子任务区域版")
        print("障碍物避障 | 清扫目标消失 | 抓取运送 | KLT框子投放")
        print("优化路径规划 | 距离判断到达 | 智能弓字形避障")
        print("集成超丝滑实时覆盖区域可视化 | KLT框子USD资产")
        print("="*80)
        
        pos, yaw = self.robot_controller._get_robot_pose()
        print(f"机器人初始位置: [{pos[0]:.3f}, {pos[1]:.3f}], 朝向: {np.degrees(yaw):.1f}°")
        
        self.plan_coverage_mission()
        
        self.execute_four_object_coverage()
        
        self._move_arm_to_pose("home")
        
        print("\nKLT框子四类对象覆盖系统演示完成!")
    
    def cleanup(self):
        """清理系统"""
        print("清理系统资源...")
        print_memory_usage("清理前")
        
        if self.visualizer:
            self.visualizer.cleanup()
        
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
    print("KLT框子任务区域版机器人覆盖系统")
    print(f"机器人半径: {ROBOT_RADIUS}m")
    print(f"安全边距: {SAFETY_MARGIN}m") 
    print(f"交互距离: {INTERACTION_DISTANCE}m")
    print("优化特性: 高效弓字形避障路径，KLT框子物理碰撞")
    print("")
    print("KLT框子特性:")
    print("  USD资产加载")
    print("  具有物理碰撞")
    print("  距离判断到达")
    print("  框内物体掉落投放")
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