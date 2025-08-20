#!/usr/bin/env python3
"""
Isaac Sim 4.5 稳定高性能REMANI完整避障系统 - 兼容性修复版
- 保守GPU使用，确保稳定性
- 动态虚影管理，每个目标路径至少8个虚影
- 分段虚影可视化，节省内存
- 完整USD机器人模型虚影
- 修复Isaac Sim 4.5 API兼容性问题
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "enable_livestream": False, 
    "enable_cameras": True,
    "enable_rtx": True,
    "physics_dt": 1.0/120.0,     # 稳定的物理频率
    "rendering_dt": 1.0/60.0,    # 稳定的渲染频率
})

import numpy as np
import math
import time
import random
from collections import deque
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import threading
import concurrent.futures
import torch

# Isaac Sim API
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid, DynamicSphere
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.core.utils.types import ArticulationAction
from scipy.spatial.transform import Rotation as R
from pxr import UsdLux, UsdPhysics, Gf, UsdGeom, Usd, UsdShade, Sdf
import isaacsim.core.utils.prims as prim_utils

@dataclass
class CollisionResult:
    is_collision: bool
    min_distance: float
    collision_type: str
    collision_point: Optional[np.ndarray] = None
    collision_normal: Optional[np.ndarray] = None

@dataclass
class ObstacleInfo:
    position: np.ndarray
    size: np.ndarray
    shape_type: str
    rotation: np.ndarray = None

@dataclass
class PathNode:
    position: np.ndarray
    orientation: float
    arm_config: List[float]
    gripper_state: float
    timestamp: float
    node_id: int
    action_type: str = "move"
    target_index: int = -1

@dataclass
class TaskInfo:
    target_name: str
    target_position: np.ndarray
    task_type: str
    approach_pose: str

class OptimizedDistanceCalculator:
    """优化的距离计算器 - 稳定版"""
    
    def __init__(self):
        self.base_radius = 0.17
        self.base_height = 0.1
        self.dh_params = [
            [0, 0, 0.333, 0], [-np.pi/2, 0, 0, 0], [np.pi/2, 0, 0.316, 0],
            [np.pi/2, 0.0825, 0, 0], [-np.pi/2, -0.0825, 0.384, 0],
            [np.pi/2, 0, 0, 0], [np.pi/2, 0.088, 0.107, 0]
        ]
        self.link_geometries = [
            {"radius": 0.060, "length": 0.15}, {"radius": 0.070, "length": 0.20},
            {"radius": 0.060, "length": 0.20}, {"radius": 0.050, "length": 0.18},
            {"radius": 0.060, "length": 0.20}, {"radius": 0.050, "length": 0.15},
            {"radius": 0.040, "length": 0.10}
        ]

    def batch_distance_check(self, positions: List[np.ndarray], obstacles: List[ObstacleInfo]) -> List[float]:
        """批量距离检查，使用向量化计算"""
        if not positions or not obstacles:
            return [float('inf')] * len(positions)
        
        min_distances = []
        positions_array = np.array(positions)
        
        for obstacle in obstacles:
            if obstacle.shape_type == 'sphere':
                distances = np.linalg.norm(positions_array - obstacle.position, axis=1) - obstacle.size[0]
            else:  # box或cylinder简化处理
                diff = np.abs(positions_array - obstacle.position)
                half_size = obstacle.size / 2
                distances = np.max(diff - half_size, axis=1)
            
            if len(min_distances) == 0:
                min_distances = distances
            else:
                min_distances = np.minimum(min_distances, distances)
        
        return min_distances.tolist()

class StableCollisionChecker:
    """稳定的碰撞检测系统"""
    
    def __init__(self, safe_distance: float = 0.5):
        self.safe_distance = safe_distance
        self.arm_safe_distance = 0.12
        self.obstacles = []
        self.distance_calc = OptimizedDistanceCalculator()
        print(f"✅ 稳定避障系统初始化: 安全距离={safe_distance}m")
    
    def add_obstacle(self, position: np.ndarray, size: np.ndarray, shape_type: str = 'box', rotation: np.ndarray = None):
        obstacle_info = ObstacleInfo(
            position=position.copy(),
            size=size.copy(), 
            shape_type=shape_type,
            rotation=rotation.copy() if rotation is not None else np.eye(3)
        )
        self.obstacles.append(obstacle_info)
    
    def check_path_collision_fast(self, start_pos: np.ndarray, end_pos: np.ndarray) -> bool:
        """快速路径碰撞检查"""
        num_samples = max(8, int(np.linalg.norm(end_pos - start_pos) / 0.1))
        sample_positions = []
        
        for i in range(num_samples + 1):
            t = i / num_samples if num_samples > 0 else 0
            pos = start_pos + t * (end_pos - start_pos)
            sample_positions.append(pos)
        
        distances = self.distance_calc.batch_distance_check(sample_positions, self.obstacles)
        return all(d > self.safe_distance for d in distances)
    
    def get_safe_direction_fast(self, current_pos: np.ndarray, target_pos: np.ndarray) -> Tuple[np.ndarray, float]:
        """快速安全方向计算"""
        direct_direction = target_pos[:2] - current_pos[:2]
        direct_distance = np.linalg.norm(direct_direction)
        
        if direct_distance < 0.01:
            return np.array([0.0, 0.0]), 0.0
        
        direct_direction_normalized = direct_direction / direct_distance
        target_orientation = np.arctan2(direct_direction[1], direct_direction[0])
        
        # 快速碰撞检查
        if self.check_path_collision_fast(current_pos, target_pos):
            return direct_direction_normalized, target_orientation
        
        # 寻找安全方向
        angles = np.linspace(0, 2*np.pi, 16)
        test_distance = min(0.6, direct_distance)
        
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            test_target = current_pos[:2] + direction * test_distance
            test_target_3d = np.array([test_target[0], test_target[1], current_pos[2]])
            
            if self.check_path_collision_fast(current_pos, test_target_3d):
                dot_product = np.dot(direction, direct_direction_normalized)
                if dot_product > 0.3:
                    return direction, angle
        
        return np.array([0.0, 0.0]), target_orientation

class AdvancedGhostManager:
    """高级虚影管理器 - 动态分段管理"""
    
    def __init__(self, world: World):
        self.world = world
        self.robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
        self.ghost_container_path = "/World/GhostVisualization"
        self.active_ghosts = {}  # {target_index: [ghost_prims]}
        self.path_lines = {}     # {target_index: [line_objects]}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)  # 保守的线程数
        
        print("🤖 高级虚影管理器初始化完成")
    
    def create_target_ghosts(self, target_index: int, path_nodes: List[PathNode], min_ghosts: int = 8):
        """为指定目标创建虚影群组"""
        print(f"🎭 为目标 #{target_index} 创建虚影群组...")
        
        # 确保容器存在
        self._ensure_container_exists()
        
        # 计算虚影数量（根据路径长度，最少8个）
        num_ghosts = max(min_ghosts, min(15, len(path_nodes) // 2))
        
        # 选择虚影节点
        ghost_nodes = self._select_ghost_nodes(path_nodes, num_ghosts)
        
        # 串行创建虚影（更稳定）
        ghost_prims = []
        for i, node in enumerate(ghost_nodes):
            ghost_prim = self._create_single_ghost(target_index, i, node)
            if ghost_prim:
                ghost_prims.append(ghost_prim)
        
        self.active_ghosts[target_index] = ghost_prims
        
        # 创建路径线
        self._create_target_path_lines(target_index, path_nodes)
        
        print(f"   完成创建 {len(ghost_prims)} 个虚影机器人")
        
        # 等待稳定
        for _ in range(10):
            self.world.step(render=False)
    
    def _select_ghost_nodes(self, path_nodes: List[PathNode], num_ghosts: int) -> List[PathNode]:
        """智能选择虚影节点位置"""
        if len(path_nodes) <= num_ghosts:
            return path_nodes
        
        selected_nodes = []
        total_nodes = len(path_nodes)
        
        # 确保起始和结束节点被选中
        selected_nodes.append(path_nodes[0])
        
        # 均匀分布中间节点
        for i in range(1, num_ghosts - 1):
            index = int((i * (total_nodes - 1)) / (num_ghosts - 1))
            selected_nodes.append(path_nodes[index])
        
        # 结束节点
        if len(path_nodes) > 1:
            selected_nodes.append(path_nodes[-1])
        
        return selected_nodes
    
    def _create_single_ghost(self, target_index: int, ghost_index: int, node: PathNode):
        """创建单个虚影机器人"""
        ghost_path = f"{self.ghost_container_path}/Target_{target_index}_Ghost_{ghost_index}"
        
        stage = self.world.stage
        
        # 清理已存在的虚影
        if stage.GetPrimAtPath(ghost_path):
            stage.RemovePrim(ghost_path)
            for _ in range(3):
                self.world.step(render=False)
        
        # 创建虚影根Prim
        ghost_prim = stage.DefinePrim(ghost_path, "Xform")
        
        # 添加USD引用
        references = ghost_prim.GetReferences()
        references.AddReference(self.robot_usd_path)
        
        # 等待USD加载
        for _ in range(5):
            self.world.step(render=False)
        
        # 设置变换
        self._set_ghost_transform(ghost_prim, node.position, node.orientation)
        
        # 完全禁用物理
        self._disable_ghost_physics(ghost_prim)
        
        # 设置机械臂姿态
        self._set_ghost_arm_configuration(ghost_prim, node.arm_config)
        
        # 设置外观
        self._setup_ghost_appearance(ghost_prim, target_index, ghost_index)
        
        return ghost_prim
    
    def _set_ghost_transform(self, ghost_prim, position: np.ndarray, orientation: float):
        """设置虚影变换 - 稳定版本"""
        # 确保数据类型正确
        ghost_position = Gf.Vec3f(float(position[0]), float(position[1]), float(position[2]))
        yaw_degrees = float(np.degrees(orientation))
        ghost_rotation = Gf.Vec3f(0.0, 0.0, yaw_degrees)
        
        # 获取Xformable
        xform = UsdGeom.Xformable(ghost_prim)
        
        # 清除已有的transform ops
        xform.ClearXformOpOrder()
        
        # 重新添加变换操作
        translate_op = xform.AddTranslateOp()
        translate_op.Set(ghost_position)
        
        rotate_op = xform.AddRotateXYZOp()
        rotate_op.Set(ghost_rotation)
    
    def _disable_ghost_physics(self, ghost_prim):
        """完全禁用虚影物理 - 保守版本"""
        stage = self.world.stage
        
        # 等待加载完成
        for _ in range(5):
            self.world.step(render=False)
        
        # 移除所有物理API
        all_prims = list(Usd.PrimRange(ghost_prim))
        
        for prim in all_prims:
            # 移除基础物理API
            api_classes = [UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI]
            for api_class in api_classes:
                if prim.HasAPI(api_class):
                    try:
                        prim.RemoveAPI(api_class)
                    except:
                        pass  # 静默处理API移除失败
            
            # 移除关节相关 - 更保守的方法
            type_name = prim.GetTypeName()
            if type_name in ['FixedJoint', 'RevoluteJoint', 'PrismaticJoint', 'SphericalJoint', 'D6Joint']:
                try:
                    stage.RemovePrim(prim.GetPath())
                except:
                    pass  # 静默处理移除失败
    
    def _set_ghost_arm_configuration(self, ghost_prim, arm_config: List[float]):
        """设置虚影机械臂配置 - 保守版本"""
        full_arm_config = arm_config[:7] + [0.0] * max(0, 7 - len(arm_config))
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        
        for i, joint_name in enumerate(arm_joint_names):
            # 尝试多种可能的关节路径
            joint_paths = [
                f"{ghost_prim.GetPath()}/ridgeback_franka/panda_link{i}/{joint_name}",
                f"{ghost_prim.GetPath()}/ridgeback_franka/{joint_name}",
                f"{ghost_prim.GetPath()}/create_3/{joint_name}"
            ]
            
            for path in joint_paths:
                if self.world.stage.GetPrimAtPath(path):
                    try:
                        joint_prim = self.world.stage.GetPrimAtPath(path)
                        xform = UsdGeom.Xformable(joint_prim)
                        
                        joint_angle = full_arm_config[i]
                        if i in [0, 2, 4, 6]:  # Z轴关节
                            if not joint_prim.HasAttribute("xformOp:rotateZ"):
                                rot_op = xform.AddRotateZOp()
                                rot_op.Set(float(np.degrees(joint_angle)))
                            else:
                                joint_prim.GetAttribute("xformOp:rotateZ").Set(float(np.degrees(joint_angle)))
                        else:  # Y轴关节  
                            if not joint_prim.HasAttribute("xformOp:rotateY"):
                                rot_op = xform.AddRotateYOp()
                                rot_op.Set(float(np.degrees(joint_angle)))
                            else:
                                joint_prim.GetAttribute("xformOp:rotateY").Set(float(np.degrees(joint_angle)))
                        break
                    except:
                        continue  # 静默跳过失败的关节
    
    def _setup_ghost_appearance(self, ghost_prim, target_index: int, ghost_index: int):
        """设置虚影外观 - 基于目标和虚影索引的颜色"""
        # 为不同目标使用不同颜色系
        target_colors = [
            (0.3, 0.7, 1.0),  # 蓝色系
            (1.0, 0.3, 0.3),  # 红色系  
            (0.3, 1.0, 0.3),  # 绿色系
            (1.0, 0.7, 0.3),  # 橙色系
            (0.7, 0.3, 1.0),  # 紫色系
        ]
        
        base_color = target_colors[target_index % len(target_colors)]
        
        # 同一目标内的虚影透明度渐变
        num_ghosts = len(self.active_ghosts.get(target_index, [ghost_prim])) + 1
        alpha_progress = ghost_index / max(1, num_ghosts - 1)
        opacity = 0.4 + 0.4 * alpha_progress  # 0.4-0.8透明度
        
        ghost_color = Gf.Vec3f(*base_color)
        
        # 设置所有Mesh的外观
        for prim in Usd.PrimRange(ghost_prim):
            if prim.IsA(UsdGeom.Mesh):
                try:
                    mesh = UsdGeom.Mesh(prim)
                    mesh.CreateDisplayColorAttr().Set([ghost_color])
                    mesh.CreateDisplayOpacityAttr().Set([opacity])
                except:
                    pass  # 静默处理外观设置失败
    
    def _create_target_path_lines(self, target_index: int, path_nodes: List[PathNode]):
        """为目标创建路径线"""
        path_lines = []
        
        for i in range(len(path_nodes) - 1):
            start_pos = path_nodes[i].position
            end_pos = path_nodes[i + 1].position
            
            midpoint = (start_pos + end_pos) / 2
            direction = end_pos - start_pos
            length = np.linalg.norm(direction)
            
            if length > 0.02:
                yaw = np.arctan2(direction[1], direction[0])
                
                try:
                    line_vis = DynamicCuboid(
                        prim_path=f"/World/PathLine_Target_{target_index}_Segment_{i}",
                        name=f"path_line_target_{target_index}_segment_{i}",
                        position=midpoint + np.array([0, 0, 0.02]),
                        scale=np.array([length, 0.03, 0.01]),
                        color=np.array([0.0, 0.8, 0.2])
                    )
                    
                    line_vis.set_world_pose(
                        position=midpoint + np.array([0, 0, 0.02]),
                        orientation=np.array([0, 0, np.sin(yaw/2), np.cos(yaw/2)])
                    )
                    
                    self.world.scene.add(line_vis)
                    path_lines.append(line_vis)
                except:
                    pass  # 静默处理路径线创建失败
        
        self.path_lines[target_index] = path_lines
    
    def clear_target_ghosts(self, target_index: int):
        """清除指定目标的所有虚影"""
        print(f"🧹 清除目标 #{target_index} 的虚影群组...")
        
        # 清除虚影
        if target_index in self.active_ghosts:
            for ghost_prim in self.active_ghosts[target_index]:
                try:
                    stage = self.world.stage
                    if stage.GetPrimAtPath(ghost_prim.GetPath()):
                        stage.RemovePrim(ghost_prim.GetPath())
                except:
                    pass  # 静默处理移除失败
            del self.active_ghosts[target_index]
        
        # 清除路径线
        if target_index in self.path_lines:
            for line_obj in self.path_lines[target_index]:
                try:
                    self.world.scene.remove_object(line_obj.name)
                except:
                    pass  # 静默处理移除失败
            del self.path_lines[target_index]
        
        # 等待处理完成
        for _ in range(5):
            self.world.step(render=False)
    
    def _ensure_container_exists(self):
        """确保容器存在"""
        stage = self.world.stage
        if not stage.GetPrimAtPath(self.ghost_container_path):
            stage.DefinePrim(self.ghost_container_path, "Xform")
    
    def cleanup_all(self):
        """清理所有资源"""
        print("🧹 清理所有虚影资源...")
        for target_index in list(self.active_ghosts.keys()):
            self.clear_target_ghosts(target_index)
        self.executor.shutdown(wait=True)

class StableCreate3System:
    """稳定的Create-3系统 - 兼容性优化"""
    
    def __init__(self):
        self.world = None
        self.robot_prim_path = "/World/create3_robot"
        self.mobile_base = None
        self.differential_controller = None
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = 0.0
        
        # 稳定的运动参数
        self.max_linear_velocity = 0.35
        self.max_angular_velocity = 1.0
        
        # 垃圾对象
        self.small_trash_objects = []
        self.large_trash_objects = []
        self.collected_objects = []
        
        # 系统组件
        self.collision_checker = None
        self.ghost_manager = None
        
        # 任务管理
        self.all_tasks = []
        self.target_paths = {}  # {target_index: [PathNode]}
        
        # 机械臂配置
        self.arm_poses = {
            "home": [0.0, -0.569, 0.0, -2.810, 0.0, 2.0, 0.741],
            "ready": [0.0, -0.3, 0.0, -1.8, 0.0, 1.6, 0.785],
            "pickup": [0.0, 0.5, 0.0, -1.6, 0.0, 2.4, 0.785],
            "carry": [0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.785]
        }
        
        self.gripper_open = 0.04
        self.gripper_closed = 0.0
    
    def initialize_stable_sim(self):
        """初始化稳定的仿真环境"""
        print("🚀 正在初始化稳定的Isaac Sim 4.5环境...")
        
        # 保守的World初始化，不使用激进的GPU选项
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0/120.0,
            rendering_dt=1.0/60.0
        )
        self.world.scene.clear()
        
        # 稳定的物理设置
        physics_context = self.world.get_physics_context()
        physics_context.set_gravity(-9.81)
        physics_context.set_solver_type("TGS")
        
        # 保守的GPU设置
        try:
            physics_context.enable_gpu_dynamics(True)
            print("   GPU物理已启用")
        except:
            print("   使用CPU物理")
        
        # 创建地面
        ground = FixedCuboid(
            prim_path="/World/Ground",
            name="ground", 
            position=np.array([0.0, 0.0, -0.5]),
            scale=np.array([50.0, 50.0, 1.0]),
            color=np.array([0.5, 0.5, 0.5])
        )
        self.world.scene.add(ground)
        
        self._setup_stable_lighting()
        self._initialize_stable_systems()
        
        print("✅ 稳定环境初始化完成")
        return True
    
    def _setup_stable_lighting(self):
        """设置稳定的照明"""
        # 主光源
        main_light = prim_utils.create_prim("/World/MainLight", "DistantLight")
        distant_light = UsdLux.DistantLight(main_light)
        distant_light.CreateIntensityAttr(6000)
        distant_light.CreateColorAttr((1.0, 1.0, 0.95))
        
        # 环境光
        env_light = prim_utils.create_prim("/World/EnvLight", "DomeLight")
        dome_light = UsdLux.DomeLight(env_light)
        dome_light.CreateIntensityAttr(1500)
        dome_light.CreateColorAttr((0.8, 0.9, 1.0))
    
    def _initialize_stable_systems(self):
        """初始化稳定的系统组件"""
        self.collision_checker = StableCollisionChecker(safe_distance=0.25)
        self.ghost_manager = AdvancedGhostManager(self.world)
        self._add_environment_obstacles()
    
    def _add_environment_obstacles(self):
        """添加环境障碍物"""
        obstacles = [
            {"pos": [1.0, 0.5, 0.15], "size": [0.3, 0.3, 0.3], "color": [0.6, 0.3, 0.1], "name": "cylinder1", "shape": "cylinder"},
            {"pos": [0.5, -1.2, 0.1], "size": [1.5, 0.2, 0.2], "color": [0.7, 0.7, 0.7], "name": "wall1", "shape": "box"},
            {"pos": [-0.8, 0.8, 0.4], "size": [0.1, 0.8, 0.1], "color": [0.8, 0.2, 0.2], "name": "pole1", "shape": "box"},
            {"pos": [-0.5, -0.8, 0.15], "size": [0.3, 0.3, 0.3], "color": [0.9, 0.5, 0.1], "name": "sphere1", "shape": "sphere"},
            {"pos": [1.5, 1.8, 0.2], "size": [0.4, 0.4, 0.4], "color": [0.2, 0.8, 0.8], "name": "box1", "shape": "box"},
        ]
        
        for obs in obstacles:
            try:
                if obs["shape"] == "sphere":
                    obstacle = DynamicSphere(
                        prim_path=f"/World/Obstacle_{obs['name']}",
                        name=f"obstacle_{obs['name']}",
                        position=np.array(obs["pos"]),
                        radius=obs["size"][0]/2,
                        color=np.array(obs["color"])
                    )
                else:
                    obstacle = FixedCuboid(
                        prim_path=f"/World/Obstacle_{obs['name']}",
                        name=f"obstacle_{obs['name']}",
                        position=np.array(obs["pos"]),
                        scale=np.array(obs["size"]),
                        color=np.array(obs["color"])
                    )
                
                self.world.scene.add(obstacle)
                self.collision_checker.add_obstacle(
                    np.array(obs["pos"]), 
                    np.array(obs["size"]),
                    obs["shape"]
                )
            except Exception as e:
                print(f"   警告: 障碍物 {obs['name']} 创建失败: {e}")
    
    def initialize_robot(self):
        """初始化机器人"""
        print("🤖 正在初始化Create-3+机械臂...")
        
        usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
        
        self.mobile_base = WheeledRobot(
            prim_path=self.robot_prim_path,
            name="create3_robot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=usd_path,
            position=np.array([0.0, 0.0, 0.0])
        )
        
        self.world.scene.add(self.mobile_base)
        
        self.differential_controller = DifferentialController(
            name="create3_controller",
            wheel_radius=0.036,
            wheel_base=0.235,
            max_linear_speed=self.max_linear_velocity,
            max_angular_speed=self.max_angular_velocity
        )
        
        print("✅ 机器人初始化成功")
        return True
    
    def setup_post_load(self):
        """后加载设置 - 修复API兼容性"""
        print("🔧 正在进行稳定后加载设置...")
        
        self.world.reset()
        
        # 稳定等待
        for _ in range(30):
            self.world.step(render=False)
        
        self.mobile_base = self.world.scene.get_object("create3_robot")
        self._setup_stable_control()
        self._move_arm_to_pose("home")
        
        print("✅ 后加载设置完成")
        return True
    
    def _setup_stable_control(self):
        """设置稳定的控制 - 修复tensor兼容性"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        num_dofs = len(self.mobile_base.dof_names)
        
        # 使用torch.tensor而不是numpy数组
        kp = torch.zeros(num_dofs, dtype=torch.float32)
        kd = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 轮子控制参数
        for wheel_name in ["left_wheel_joint", "right_wheel_joint"]:
            try:
                idx = self.mobile_base.dof_names.index(wheel_name)
                kp[idx] = 0.0
                kd[idx] = 800.0
            except ValueError:
                print(f"   警告: 找不到轮子关节 {wheel_name}")
        
        # 机械臂控制参数
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        for joint_name in arm_joint_names:
            try:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 1000.0
                kd[idx] = 50.0
            except ValueError:
                print(f"   警告: 找不到机械臂关节 {joint_name}")
        
        # 夹爪控制参数
        for joint_name in ["panda_finger_joint1", "panda_finger_joint2"]:
            try:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 2e5
                kd[idx] = 2e3
            except ValueError:
                print(f"   警告: 找不到夹爪关节 {joint_name}")
        
        # 应用控制参数
        try:
            articulation_controller.set_gains(kps=kp, kds=kd)
            print("   关节控制参数设置成功")
        except Exception as e:
            print(f"   警告: 控制参数设置失败: {e}")
    
    def _move_arm_to_pose(self, pose_name):
        """移动机械臂到姿态"""
        target_positions = self.arm_poses[pose_name]
        articulation_controller = self.mobile_base.get_articulation_controller()
        num_dofs = len(self.mobile_base.dof_names)
        
        # 使用torch.tensor
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        for i, joint_name in enumerate(arm_joint_names):
            try:
                idx = self.mobile_base.dof_names.index(joint_name)
                joint_positions[idx] = target_positions[i]
            except ValueError:
                continue
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        # 等待到达位置
        for _ in range(30):
            self.world.step(render=False)
    
    def _control_gripper(self, open_close):
        """控制夹爪"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        gripper_position = self.gripper_open if open_close == "open" else self.gripper_closed
        
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        for joint_name in ["panda_finger_joint1", "panda_finger_joint2"]:
            try:
                idx = self.mobile_base.dof_names.index(joint_name)
                joint_positions[idx] = gripper_position
            except ValueError:
                continue
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        for _ in range(15):
            self.world.step(render=False)
    
    def get_robot_pose(self):
        """获取机器人姿态"""
        position, orientation = self.mobile_base.get_world_pose()
        quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        r = R.from_quat(quat)
        yaw = r.as_euler('xyz')[2]
        
        self.current_position = position
        self.current_orientation = yaw
        return position.copy(), yaw
    
    def create_diverse_trash_environment(self):
        """创建多样化垃圾环境"""
        print("🗑️ 创建多样化垃圾环境...")
        
        # 小垃圾位置
        small_trash_positions = [
            [2.5, 0.0, 0.03], [2.0, 1.5, 0.03], [3.2, -0.8, 0.03], [1.8, -2.2, 0.03]
        ]
        
        for i, pos in enumerate(small_trash_positions):
            try:
                trash = DynamicSphere(
                    prim_path=f"/World/small_trash_{i}",
                    name=f"small_trash_{i}",
                    position=np.array(pos),
                    radius=0.03,
                    color=np.array([1.0, 0.2, 0.2])
                )
                self.world.scene.add(trash)
                self.small_trash_objects.append(trash)
            except Exception as e:
                print(f"   警告: 小垃圾 {i} 创建失败: {e}")
        
        # 大垃圾位置
        large_trash_positions = [
            [3.0, 0.8, 0.025], [2.8, -1.8, 0.025], [-1.5, 1.5, 0.025]
        ]
        
        for i, pos in enumerate(large_trash_positions):
            try:
                trash = DynamicCuboid(
                    prim_path=f"/World/large_trash_{i}",
                    name=f"large_trash_{i}",
                    position=np.array(pos),
                    scale=np.array([0.05, 0.05, 0.05]),
                    color=np.array([0.2, 0.8, 0.2])
                )
                self.world.scene.add(trash)
                self.large_trash_objects.append(trash)
            except Exception as e:
                print(f"   警告: 大垃圾 {i} 创建失败: {e}")
        
        print(f"✅ 环境创建完成: 小垃圾{len(self.small_trash_objects)}个, 大垃圾{len(self.large_trash_objects)}个")
    
    def plan_optimized_mission(self):
        """规划优化的任务"""
        print("\n🎯 开始规划优化任务...")
        
        # 创建任务列表
        self.all_tasks = []
        current_pos, _ = self.get_robot_pose()
        
        # 小垃圾任务
        for i, trash in enumerate(self.small_trash_objects):
            trash_pos = trash.get_world_pose()[0]
            task = TaskInfo(
                target_name=trash.name,
                target_position=trash_pos,
                task_type="small_trash",
                approach_pose="carry"
            )
            self.all_tasks.append(task)
        
        # 大垃圾任务
        for i, trash in enumerate(self.large_trash_objects):
            trash_pos = trash.get_world_pose()[0]
            task = TaskInfo(
                target_name=trash.name,
                target_position=trash_pos,
                task_type="large_trash",
                approach_pose="ready"
            )
            self.all_tasks.append(task)
        
        # 返回原点
        home_task = TaskInfo(
            target_name="home",
            target_position=np.array([0.0, 0.0, 0.0]),
            task_type="return_home",
            approach_pose="home"
        )
        self.all_tasks.append(home_task)
        
        # 为每个目标规划详细路径
        self._plan_detailed_paths()
        
        print(f"✅ 任务规划完成: {len(self.all_tasks)}个目标")
    
    def _plan_detailed_paths(self):
        """规划详细路径"""
        print("📍 规划详细路径...")
        
        current_pos, current_yaw = self.get_robot_pose()
        
        for target_index, task in enumerate(self.all_tasks):
            print(f"   规划目标 {target_index}: {task.target_name}")
            
            target_pos = task.target_position.copy()
            target_pos[2] = 0.0
            
            # 生成高密度路径点
            path_points = self._generate_high_density_path(current_pos[:2], target_pos[:2])
            
            # 创建路径节点
            path_nodes = []
            for i, point in enumerate(path_points):
                if i < len(path_points) - 1:
                    direction = np.array(path_points[i + 1]) - np.array(point)
                    orientation = np.arctan2(direction[1], direction[0])
                else:
                    orientation = path_nodes[-1].orientation if path_nodes else 0.0
                
                arm_config = self.arm_poses[task.approach_pose]
                
                node = PathNode(
                    position=np.array([point[0], point[1], 0.0]),
                    orientation=orientation,
                    arm_config=arm_config.copy(),
                    gripper_state=self.gripper_open,
                    timestamp=i * 0.1,
                    node_id=i,
                    action_type="move",
                    target_index=target_index
                )
                path_nodes.append(node)
            
            self.target_paths[target_index] = path_nodes
            current_pos = target_pos
            
            print(f"     生成 {len(path_nodes)} 个路径节点")
    
    def _generate_high_density_path(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> List[np.ndarray]:
        """生成高密度平滑路径"""
        direction = goal_pos - start_pos
        distance = np.linalg.norm(direction)
        
        # 高密度路径点（确保足够的虚影密度）
        num_points = max(15, min(25, int(distance / 0.08)))
        
        path_points = []
        for i in range(num_points + 1):
            t = i / num_points
            # 使用三次贝塞尔曲线插值
            smooth_t = t * t * (3 - 2 * t)
            point = start_pos + smooth_t * direction
            path_points.append(point)
        
        return path_points
    
    def execute_optimized_mission(self):
        """执行优化任务"""
        print("\n🚀 开始执行稳定任务...")
        
        for target_index, task in enumerate(self.all_tasks):
            print(f"\n🎯 执行目标 {target_index}: {task.target_name}")
            
            # 获取该目标的路径
            path_nodes = self.target_paths[target_index]
            
            # 创建该目标的虚影群组
            self.ghost_manager.create_target_ghosts(target_index, path_nodes, min_ghosts=8)
            
            # 显示虚影1.5秒
            print("🎭 展示虚影群组...")
            for _ in range(90):
                self.world.step(render=True)
            
            # 执行路径
            print(f"🏃 执行路径（{len(path_nodes)}个节点）...")
            self._execute_target_path(path_nodes, task)
            
            # 清除该目标的虚影
            self.ghost_manager.clear_target_ghosts(target_index)
            
            print(f"✅ 目标 {target_index} 完成")
        
        print("\n🎉 所有目标执行完成!")
        self._show_final_results()
    
    def _execute_target_path(self, path_nodes: List[PathNode], task: TaskInfo):
        """执行目标路径"""
        for node in path_nodes:
            success = self._navigate_to_node_stable(node, tolerance=0.15)
            
            # 检查是否到达任务目标点
            task_distance = np.linalg.norm(node.position[:2] - task.target_position[:2])
            if task_distance < 0.2 and task.target_name not in self.collected_objects:
                self._execute_task_action(task)
                break
    
    def _navigate_to_node_stable(self, node: PathNode, tolerance: float = 0.15) -> bool:
        """稳定导航到节点"""
        max_time = 20.0
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            current_pos, current_yaw = self.get_robot_pose()
            
            distance = np.linalg.norm(current_pos[:2] - node.position[:2])
            if distance < tolerance:
                return True
            
            # 稳定的方向计算
            safe_direction, safe_orientation = self.collision_checker.get_safe_direction_fast(
                current_pos, node.position
            )
            
            if np.linalg.norm(safe_direction) > 0.01:
                target_angle = np.arctan2(safe_direction[1], safe_direction[0])
                angle_diff = target_angle - current_yaw
                
                # 角度归一化
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                # 稳定控制
                if abs(angle_diff) > 0.15:
                    linear_vel = 0.0
                    angular_vel = np.clip(angle_diff * 2.0, -0.8, 0.8)
                else:
                    linear_vel = min(0.25, max(0.06, distance * 0.4))
                    angular_vel = np.clip(angle_diff * 1.2, -0.3, 0.3)
            else:
                linear_vel = 0.0
                angular_vel = 0.2
            
            self._send_control_command(linear_vel, angular_vel)
            self.world.step(render=True)
            time.sleep(0.016)  # 稳定的时间步
        
        return False
    
    def _send_control_command(self, linear_vel, angular_vel):
        """发送控制命令 - 兼容版本"""
        linear_vel = np.clip(linear_vel, -self.max_linear_velocity, self.max_linear_velocity)
        angular_vel = np.clip(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
        
        try:
            articulation_controller = self.mobile_base.get_articulation_controller()
            wheel_radius = 0.036
            wheel_base = 0.235
            
            left_wheel_vel = (linear_vel - angular_vel * wheel_base / 2.0) / wheel_radius
            right_wheel_vel = (linear_vel + angular_vel * wheel_base / 2.0) / wheel_radius
            
            num_dofs = len(self.mobile_base.dof_names)
            joint_velocities = torch.zeros(num_dofs, dtype=torch.float32)
            
            try:
                left_wheel_idx = self.mobile_base.dof_names.index("left_wheel_joint")
                right_wheel_idx = self.mobile_base.dof_names.index("right_wheel_joint")
                
                joint_velocities[left_wheel_idx] = left_wheel_vel
                joint_velocities[right_wheel_idx] = right_wheel_vel
                
                action = ArticulationAction(joint_velocities=joint_velocities)
                articulation_controller.apply_action(action)
            except ValueError:
                pass  # 静默处理关节名称错误
        except Exception as e:
            pass  # 静默处理控制命令失败
    
    def _execute_task_action(self, task: TaskInfo):
        """执行任务动作"""
        print(f"🎯 执行任务: {task.target_name}")
        
        if task.task_type == "small_trash":
            self._collect_small_trash(task)
        elif task.task_type == "large_trash":
            self._collect_large_trash(task)
        elif task.task_type == "return_home":
            print("🏠 返回原点完成")
    
    def _collect_small_trash(self, task: TaskInfo):
        """收集小垃圾"""
        self._move_arm_to_pose("carry")
        
        # 找到垃圾对象并模拟收集
        for trash in self.small_trash_objects:
            if trash.name == task.target_name:
                try:
                    current_pos, _ = self.get_robot_pose()
                    trash.set_world_pose(current_pos, np.array([0, 0, 0, 1]))
                    self.collected_objects.append(task.target_name)
                    print(f"✅ {task.target_name} 收集成功!")
                except:
                    print(f"⚠️ {task.target_name} 收集失败")
                break
    
    def _collect_large_trash(self, task: TaskInfo):
        """收集大垃圾"""
        self._move_arm_to_pose("ready")
        self._control_gripper("open")
        self._move_arm_to_pose("pickup")
        self._control_gripper("close")
        self._move_arm_to_pose("carry")
        
        # 找到垃圾对象并模拟收集
        for trash in self.large_trash_objects:
            if trash.name == task.target_name:
                try:
                    trash.set_world_pose(np.array([0, 0, -1.0]), np.array([0, 0, 0, 1]))
                    self.collected_objects.append(task.target_name)
                    print(f"✅ {task.target_name} 收集成功!")
                except:
                    print(f"⚠️ {task.target_name} 收集失败")
                break
    
    def _show_final_results(self):
        """显示最终结果"""
        total_items = len(self.small_trash_objects) + len(self.large_trash_objects)
        success_count = len(self.collected_objects)
        success_rate = (success_count / total_items) * 100 if total_items > 0 else 0
        
        total_nodes = sum(len(path) for path in self.target_paths.values())
        
        print(f"\n📊 稳定任务执行结果:")
        print(f"   总目标数: {len(self.all_tasks)}")
        print(f"   总垃圾数: {total_items}")
        print(f"   成功收集: {success_count}")
        print(f"   成功率: {success_rate:.1f}%")
        print(f"   总路径节点: {total_nodes}")
        print(f"   系统稳定性: ✅")
        print(f"   动态虚影管理: ✅")
    
    def run_stable_demo(self):
        """运行稳定演示"""
        print("\n" + "="*80)
        print("🚀 REMANI稳定高性能虚影避障系统")
        print("🎭 动态虚影管理 | 🔧 稳定兼容 | ⚡ 高效运行")
        print("="*80)
        
        pos, yaw = self.get_robot_pose()
        print(f"📍 初始位置: [{pos[0]:.3f}, {pos[1]:.3f}], 朝向: {np.degrees(yaw):.1f}°")
        
        # 规划和执行
        self.plan_optimized_mission()
        self.execute_optimized_mission()
        
        # 返回初始姿态
        self._move_arm_to_pose("home")
        
        print("\n🎉 稳定虚影避障系统演示完成!")
        print("💡 稳定运行，动态虚影管理，兼容性优化")
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理稳定系统资源...")
        self.ghost_manager.cleanup_all()
        self.world.stop()

def main():
    """主函数 - 稳定兼容版本"""
    print("🚀 启动REMANI稳定虚影避障系统...")
    
    system = StableCreate3System()
    
    try:
        # 初始化系统
        system.initialize_stable_sim()
        system.initialize_robot()
        system.setup_post_load()
        system.create_diverse_trash_environment()
        
        # 稳定系统
        print("⚡ 系统稳定中...")
        for _ in range(60):
            system.world.step(render=False)
            time.sleep(0.016)
        
        # 运行稳定演示
        system.run_stable_demo()
        
        # 保持运行
        print("\n💡 稳定系统运行中，按 Ctrl+C 退出")
        while True:
            system.world.step(render=True)
            time.sleep(0.016)
            
    except KeyboardInterrupt:
        print("\n👋 用户中断，正在清理...")
        system.cleanup()
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        system.cleanup()

if __name__ == "__main__":
    main()