#!/usr/bin/env python3
"""
Isaac Sim 4.5 轻量级虚影避障系统 - 稳定优化版
- 改进机器人运动控制，解决卡顿问题
- 优化轮子控制，实现稳定直线运动
- 修复虚影位置同步问题
- 移除有问题的路径线功能
- 添加运动平滑和稳定性辅助措施
- 修复垃圾对象移除问题，避免物理仿真破坏
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "enable_livestream": False, 
    "enable_cameras": True,
    "enable_rtx": True,
    "physics_dt": 1.0/120.0,  # 保持高频物理
    "rendering_dt": 1.0/60.0,
})

import numpy as np
import math
import time
import random
from collections import deque
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import gc
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

class LightweightPathPlanner:
    """轻量级路径规划器"""
    
    def __init__(self, world_size: float = 8.0, resolution: float = 0.15):
        self.world_size = world_size
        self.resolution = resolution
        self.grid_size = int(world_size / resolution)
        self.obstacles = []
        self.obstacle_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        print(f"🗺️ 轻量级路径规划器: {self.grid_size}x{self.grid_size}网格")
    
    def add_obstacle(self, position: np.ndarray, size: np.ndarray, shape_type: str = 'box'):
        """添加障碍物"""
        self.obstacles.append({'pos': position, 'size': size, 'type': shape_type})
        
        center_x = int((position[0] + self.world_size/2) / self.resolution)
        center_y = int((position[1] + self.world_size/2) / self.resolution)
        
        safety_margin = 0.6  # 增大安全边距
        if shape_type == 'sphere':
            radius = int((size[0] + safety_margin) / self.resolution)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        x, y = center_x + dx, center_y + dy
                        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                            self.obstacle_grid[x, y] = True
        else:
            half_x = int((size[0]/2 + safety_margin) / self.resolution)
            half_y = int((size[1]/2 + safety_margin) / self.resolution)
            for dx in range(-half_x, half_x + 1):
                for dy in range(-half_y, half_y + 1):
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        self.obstacle_grid[x, y] = True
    
    def find_safe_path(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> List[np.ndarray]:
        """寻找安全路径"""
        start_grid = self.world_to_grid(start_pos)
        goal_grid = self.world_to_grid(goal_pos)
        
        # 简化的A*搜索
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        
        max_iterations = 1000  # 限制搜索次数
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)[1]
            
            if current == goal_grid:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(self.grid_to_world(*current))
                    current = came_from[current]
                path.append(self.grid_to_world(*start_grid))
                path.reverse()
                return self._smooth_path_simple(path)
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (0 <= neighbor[0] < self.grid_size and 
                    0 <= neighbor[1] < self.grid_size and
                    not self.obstacle_grid[neighbor[0], neighbor[1]]):
                    
                    move_cost = 1.4 if abs(dx) + abs(dy) == 2 else 1.0
                    tentative_g = g_score[current] + move_cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self._heuristic(neighbor, goal_grid)
                        heapq.heappush(open_set, (f_score, neighbor))
        
        # 无路径时返回简单路径
        print(f"⚠️ 使用简化路径")
        return self._create_simple_path(start_pos, goal_pos)
    
    def world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """世界坐标转网格坐标"""
        x = int((world_pos[0] + self.world_size/2) / self.resolution)
        y = int((world_pos[1] + self.world_size/2) / self.resolution)
        return np.clip(x, 0, self.grid_size-1), np.clip(y, 0, self.grid_size-1)
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """网格坐标转世界坐标"""
        x = (grid_x * self.resolution) - self.world_size/2
        y = (grid_y * self.resolution) - self.world_size/2
        return np.array([x, y, 0.0])
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """启发式函数"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _smooth_path_simple(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """简化路径平滑"""
        if len(path) <= 3:
            return path
        
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = min(i + 3, len(path) - 1)  # 限制搜索范围
            smoothed.append(path[j])
            i = j
        
        return smoothed
    
    def _create_simple_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """创建简单路径"""
        direction = goal - start
        distance = np.linalg.norm(direction[:2])
        
        if distance < 0.1:
            return [start, goal]
        
        num_points = max(3, min(8, int(distance / 0.3)))
        path = []
        
        for i in range(num_points + 1):
            t = i / num_points
            point = start + t * direction
            path.append(point)
        
        return path

class EfficientGhostManager:
    """高效虚影管理器 - 移除路径线功能"""
    
    def __init__(self, world: World):
        self.world = world
        self.robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"  # 实际机器人
        self.ghost_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm3.usd"   # 虚影专用
        self.ghost_container_path = "/World/GhostVisualization"
        self.active_ghosts = {}
        self.max_ghosts_per_target = 6
        print("👻 虚影管理器初始化 - 专用无物理资产")
    
    def create_target_ghosts(self, target_index: int, path_nodes: List[PathNode]):
        """创建目标虚影"""
        print(f"🎭 为目标 #{target_index} 创建无物理虚影...")
        
        self._cleanup_previous_ghosts()
        self._ensure_container_exists()
        
        selected_nodes = self._select_efficient_nodes(path_nodes)
        
        ghost_prims = []
        for i, node in enumerate(selected_nodes):
            ghost_prim = self._create_efficient_ghost(target_index, i, node)
            if ghost_prim:
                ghost_prims.append(ghost_prim)
            
            if i >= self.max_ghosts_per_target - 1:
                break
        
        self.active_ghosts[target_index] = ghost_prims
        # 移除路径线创建，避免物理错误
        # self._create_simple_path_lines(target_index, path_nodes)
        
        print(f"   完成: {len(ghost_prims)} 个无物理虚影")
        
        for _ in range(5):
            self.world.step(render=False)
    
    def _select_efficient_nodes(self, path_nodes: List[PathNode]) -> List[PathNode]:
        """高效选择虚影节点"""
        if len(path_nodes) <= self.max_ghosts_per_target:
            return path_nodes
        
        selected = [path_nodes[0]]  # 起始点
        
        # 均匀分布选择
        step = len(path_nodes) // (self.max_ghosts_per_target - 1)
        for i in range(1, self.max_ghosts_per_target - 1):
            index = min(i * step, len(path_nodes) - 1)
            selected.append(path_nodes[index])
        
        selected.append(path_nodes[-1])  # 终点
        return selected
    
    def _create_efficient_ghost(self, target_index: int, ghost_index: int, node: PathNode):
        """创建虚影"""
        ghost_path = f"{self.ghost_container_path}/Target_{target_index}_Ghost_{ghost_index}"
        stage = self.world.stage
        
        if stage.GetPrimAtPath(ghost_path):
            stage.RemovePrim(ghost_path)
            self.world.step(render=False)
        
        try:
            # 使用专用无物理虚影资产
            ghost_prim = stage.DefinePrim(ghost_path, "Xform")
            references = ghost_prim.GetReferences()
            references.AddReference(self.ghost_usd_path)
            
            # 等待加载
            for _ in range(3):
                self.world.step(render=False)
            
            # 设置位置和姿态
            self._set_ghost_transform_simple(ghost_prim, node.position, node.orientation)
            self._set_arm_pose_simple(ghost_prim, node.arm_config)
            
            # 设置灰色透明外观
            self._apply_gray_transparency(ghost_prim, ghost_index)
            
            return ghost_prim
            
        except Exception as e:
            print(f"   虚影 #{ghost_index} 创建失败")
            return None
    
    def _set_ghost_transform_simple(self, ghost_prim, position: np.ndarray, orientation: float):
        """简化变换设置"""
        ghost_position = Gf.Vec3f(float(position[0]), float(position[1]), float(position[2]))
        yaw_degrees = float(np.degrees(orientation))
        
        xform = UsdGeom.Xformable(ghost_prim)
        xform.ClearXformOpOrder()
        
        translate_op = xform.AddTranslateOp()
        translate_op.Set(ghost_position)
        
        if abs(yaw_degrees) > 1.0:  # 只有显著旋转才设置
            rotate_op = xform.AddRotateZOp()
            rotate_op.Set(yaw_degrees)
    
    def _set_arm_pose_simple(self, ghost_prim, arm_config: List[float]):
        """简化机械臂姿态设置"""
        if len(arm_config) < 7:
            return
        
        # 只设置主要关节
        main_joints = [
            ("panda_joint1", arm_config[0], "Z"),
            ("panda_joint2", arm_config[1], "Y"),
            ("panda_joint3", arm_config[2], "Z"),
            ("panda_joint4", arm_config[3], "Y"),
            ("panda_joint7", arm_config[6], "Z")
        ]
        
        for joint_name, angle, axis in main_joints:
            joint_path = f"{ghost_prim.GetPath()}/ridgeback_franka/{joint_name}"
            if self.world.stage.GetPrimAtPath(joint_path):
                joint_prim = self.world.stage.GetPrimAtPath(joint_path)
                xform = UsdGeom.Xformable(joint_prim)
                
                if axis == "Z":
                    rot_op = xform.AddRotateZOp()
                else:
                    rot_op = xform.AddRotateYOp()
                rot_op.Set(float(np.degrees(angle)))
    
    def _apply_gray_transparency(self, ghost_prim, ghost_index: int):
        """应用灰色透明外观"""
        # 灰色系颜色 - 从浅灰到深灰
        gray_values = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
        gray_value = gray_values[ghost_index % len(gray_values)]
        ghost_color = Gf.Vec3f(gray_value, gray_value, gray_value)
        
        # 透明度 - 从透明到半透明
        opacity = 0.2 + 0.3 * (ghost_index / max(1, self.max_ghosts_per_target - 1))
        
        print(f"   设置虚影外观: 灰度={gray_value:.1f}, 透明度={opacity:.2f}")
        
        try:
            # 处理所有mesh组件
            mesh_count = 0
            for prim in Usd.PrimRange(ghost_prim):
                if prim.IsA(UsdGeom.Mesh) and mesh_count < 50:
                    try:
                        mesh = UsdGeom.Mesh(prim)
                        
                        # 设置显示颜色（灰色）
                        display_color_attr = mesh.CreateDisplayColorAttr()
                        display_color_attr.Set([ghost_color])
                        
                        # 设置透明度
                        display_opacity_attr = mesh.CreateDisplayOpacityAttr()
                        display_opacity_attr.Set([opacity])
                        
                        # 设置为透明渲染模式
                        prim.CreateAttribute("primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray).Set([opacity])
                        prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set([ghost_color])
                        
                        # 强制材质更新 - 清除原有材质绑定
                        material_attrs = [
                            "material:binding",
                            "material:binding:collection",
                            "material:binding:preview"
                        ]
                        
                        for attr_name in material_attrs:
                            if prim.HasAttribute(attr_name):
                                prim.RemoveProperty(attr_name)
                        
                        # 确保mesh可见但透明
                        visibility_attr = prim.CreateAttribute("visibility", Sdf.ValueTypeNames.Token)
                        visibility_attr.Set("inherited")
                        
                        mesh_count += 1
                        
                    except Exception as e:
                        continue
            
            print(f"   成功设置 {mesh_count} 个mesh为灰色透明")
            
            # 等待渲染更新
            for _ in range(3):
                self.world.step(render=False)
                
        except Exception as e:
            print(f"   虚影外观设置失败: {e}")
            # 备用简化设置
            try:
                ghost_prim.CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray).Set([ghost_color])
                ghost_prim.CreateAttribute("primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray).Set([opacity])
            except:
                pass
    
    def _cleanup_previous_ghosts(self):
        """清理之前的虚影"""
        stage = self.world.stage
        
        if stage.GetPrimAtPath(self.ghost_container_path):
            stage.RemovePrim(self.ghost_container_path)
            self.world.step(render=False)
        
        self.active_ghosts.clear()
    
    def clear_target_ghosts(self, target_index: int):
        """清除目标虚影"""
        if target_index in self.active_ghosts:
            for ghost_prim in self.active_ghosts[target_index]:
                stage = self.world.stage
                if stage.GetPrimAtPath(ghost_prim.GetPath()):
                    stage.RemovePrim(ghost_prim.GetPath())
            del self.active_ghosts[target_index]
        
        self.world.step(render=False)
    
    def _ensure_container_exists(self):
        """确保容器存在"""
        stage = self.world.stage
        if not stage.GetPrimAtPath(self.ghost_container_path):
            stage.DefinePrim(self.ghost_container_path, "Xform")
    
    def cleanup_all(self):
        """清理所有资源"""
        self._cleanup_previous_ghosts()

class StabilizedRobotController:
    """稳定化机器人控制器 - 解决运动问题"""
    
    def __init__(self, mobile_base, differential_controller):
        self.mobile_base = mobile_base
        self.differential_controller = differential_controller
        
        # 改进的控制参数 - 大幅提高转弯速度
        self.max_linear_velocity = 0.25  
        self.max_angular_velocity = 5.0  # 提高约8倍：0.6 → 5.0
        
        # 运动平滑参数
        self.velocity_filter = deque(maxlen=5)
        self.angular_filter = deque(maxlen=5)
        
        # 稳定性检查 - 适应高速转弯
        self.last_position = None
        self.stuck_counter = 0
        self.stuck_threshold = 40  # 降低阈值，因为高速转弯时间更短
        
        # 控制历史
        self.control_history = deque(maxlen=10)
        
        print("🎮 稳定化控制器初始化")
    
    def send_stable_command(self, target_linear_vel: float, target_angular_vel: float):
        """发送稳定的控制命令"""
        # 速度限制
        target_linear_vel = np.clip(target_linear_vel, -self.max_linear_velocity, self.max_linear_velocity)
        target_angular_vel = np.clip(target_angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
        
        # 速度平滑
        self.velocity_filter.append(target_linear_vel)
        self.angular_filter.append(target_angular_vel)
        
        smooth_linear = np.mean(list(self.velocity_filter))
        smooth_angular = np.mean(list(self.angular_filter))
        
        # 记录控制历史
        self.control_history.append((smooth_linear, smooth_angular))
        
        # 发送控制命令
        self._apply_wheel_control(smooth_linear, smooth_angular)
    
    def _apply_wheel_control(self, linear_vel: float, angular_vel: float):
        """应用轮子控制 - 改进版"""
        try:
            articulation_controller = self.mobile_base.get_articulation_controller()
            
            # 物理参数 - 来自Create-3规格
            wheel_radius = 0.036  # 3.6cm
            wheel_base = 0.235    # 23.5cm
            
            # 计算轮子速度
            left_wheel_vel = (linear_vel - angular_vel * wheel_base / 2.0) / wheel_radius
            right_wheel_vel = (linear_vel + angular_vel * wheel_base / 2.0) / wheel_radius
            
            # 对称性检查 - 确保直线运动时两轮速度相等
            if abs(angular_vel) < 0.05:  # 基本直线运动
                avg_vel = (left_wheel_vel + right_wheel_vel) / 2.0
                left_wheel_vel = avg_vel
                right_wheel_vel = avg_vel
            
            # 创建关节动作
            num_dofs = len(self.mobile_base.dof_names)
            joint_velocities = torch.zeros(num_dofs, dtype=torch.float32)
            
            # 设置轮子速度
            try:
                left_wheel_idx = self.mobile_base.dof_names.index("left_wheel_joint")
                right_wheel_idx = self.mobile_base.dof_names.index("right_wheel_joint")
                
                joint_velocities[left_wheel_idx] = float(left_wheel_vel)
                joint_velocities[right_wheel_idx] = float(right_wheel_vel)
                
                # 应用动作
                action = ArticulationAction(joint_velocities=joint_velocities)
                articulation_controller.apply_action(action)
                
            except (ValueError, IndexError) as e:
                print(f"   轮子控制错误: {e}")
                
        except Exception as e:
            if "invalidated" in str(e) or "backend" in str(e):
                # 物理仿真视图无效化，跳过控制
                pass
            else:
                print(f"   控制应用失败: {e}")
    
    def check_movement_stability(self, current_position: np.ndarray) -> bool:
        """检查运动稳定性 - 改进版"""
        if self.last_position is not None:
            movement = np.linalg.norm(current_position[:2] - self.last_position[:2])
            
            # 更宽松的卡住检测
            if movement < 0.005:  # 进一步降低敏感度
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            
            # 增加卡住判定时间
            if self.stuck_counter >= self.stuck_threshold:
                print("   检测到机器人长时间卡住，尝试恢复...")
                self._unstuck_recovery()
                self.stuck_counter = 0
                return False
        
        self.last_position = current_position.copy()
        return True
    
    def _unstuck_recovery(self):
        """解卡恢复 - 增强版"""
        print("   执行解卡恢复...")
        
        # 1. 完全停止
        for _ in range(10):
            self.send_stable_command(0.0, 0.0)
        
        # 2. 尝试多方向小幅移动 - 高速转弯版
        recovery_actions = [
            (-0.1, 0.0),   # 后退
            (0.0, 3.0),    # 高速左转
            (0.0, -3.0),   # 高速右转
            (-0.08, 2.0),  # 后退+高速左转
            (-0.08, -2.0), # 后退+高速右转
        ]
        
        for linear, angular in recovery_actions:
            for _ in range(8):
                self.send_stable_command(linear, angular)
            # 短暂停止
            for _ in range(3):
                self.send_stable_command(0.0, 0.0)
        
        # 3. 最终停止
        for _ in range(8):
            self.send_stable_command(0.0, 0.0)
        
        print("   解卡恢复完成")

class LightweightRobotSystem:
    """轻量级机器人系统 - 优化版"""
    
    def __init__(self):
        self.world = None
        self.robot_prim_path = "/World/create3_robot"
        self.mobile_base = None
        self.differential_controller = None
        self.stabilized_controller = None  # 新增稳定控制器
        
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = 0.0
        
        # 垃圾对象
        self.small_trash_objects = []
        self.large_trash_objects = []
        self.collected_objects = []
        
        # 系统组件
        self.path_planner = None
        self.ghost_manager = None
        
        # 任务管理
        self.all_tasks = []
        self.target_paths = {}
        
        # 机械臂配置
        self.arm_poses = {
            "home": [0.0, -0.569, 0.0, -2.810, 0.0, 2.0, 0.741],
            "ready": [0.0, -0.3, 0.0, -1.8, 0.0, 1.6, 0.785],
            "pickup": [0.0, 0.5, 0.0, -1.6, 0.0, 2.4, 0.785],
            "carry": [0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.785]
        }
        
        self.gripper_open = 0.04
        self.gripper_closed = 0.0
    
    def initialize_system(self):
        """初始化系统"""
        print("🚀 初始化轻量级Isaac Sim 4.5环境...")
        
        try:
            self.world = World(
                stage_units_in_meters=1.0,
                physics_dt=1.0/120.0,
                rendering_dt=1.0/60.0
            )
            self.world.scene.clear()
            
            # 优化物理设置
            physics_context = self.world.get_physics_context()
            physics_context.set_gravity(-9.81)
            physics_context.set_solver_type("TGS")
            physics_context.enable_gpu_dynamics(True)
            
            # Isaac Sim 4.5兼容的稳定性设置
            try:
                # 尝试设置求解器参数（如果方法存在）
                if hasattr(physics_context, 'set_solver_position_iteration_count'):
                    physics_context.set_solver_position_iteration_count(8)
                if hasattr(physics_context, 'set_solver_velocity_iteration_count'):
                    physics_context.set_solver_velocity_iteration_count(4)
            except Exception as e:
                print(f"   物理参数设置跳过: {e}")
            
            # 创建地面
            ground = FixedCuboid(
                prim_path="/World/Ground",
                name="ground", 
                position=np.array([0.0, 0.0, -0.5]),
                scale=np.array([50.0, 50.0, 1.0]),
                color=np.array([0.5, 0.5, 0.5])
            )
            self.world.scene.add(ground)
            
            self._setup_lighting()
            self._initialize_systems()
            
            print("✅ 轻量级环境初始化完成")
            return True
            
        except Exception as e:
            print(f"❌ 系统初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_lighting(self):
        """设置照明"""
        main_light = prim_utils.create_prim("/World/MainLight", "DistantLight")
        distant_light = UsdLux.DistantLight(main_light)
        distant_light.CreateIntensityAttr(5000)
        distant_light.CreateColorAttr((1.0, 1.0, 0.95))
        
        env_light = prim_utils.create_prim("/World/EnvLight", "DomeLight")
        dome_light = UsdLux.DomeLight(env_light)
        dome_light.CreateIntensityAttr(1200)
        dome_light.CreateColorAttr((0.8, 0.9, 1.0))
    
    def _initialize_systems(self):
        """初始化系统组件"""
        self.path_planner = LightweightPathPlanner(world_size=8.0, resolution=0.15)
        self.ghost_manager = EfficientGhostManager(self.world)
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
            self.path_planner.add_obstacle(
                np.array(obs["pos"]), 
                np.array(obs["size"]),
                obs["shape"]
            )
    
    def initialize_robot(self):
        """初始化机器人"""
        print("🤖 初始化Create-3+机械臂...")
        
        try:
            # 使用有物理属性的实际机器人资产
            robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
            
            self.mobile_base = WheeledRobot(
                prim_path=self.robot_prim_path,
                name="create3_robot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=robot_usd_path,
                position=np.array([0.0, 0.0, 0.0])
            )
            
            self.world.scene.add(self.mobile_base)
            
            self.differential_controller = DifferentialController(
                name="create3_controller",
                wheel_radius=0.036,
                wheel_base=0.235,
                max_linear_speed=0.25,  
                max_angular_speed=5.0  # 大幅提高角速度
            )
            
            print("✅ 机器人初始化成功")
            return True
            
        except Exception as e:
            print(f"❌ 机器人初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def setup_post_load(self):
        """后加载设置"""
        print("🔧 后加载设置...")
        
        try:
            self.world.reset()
            
            # 额外稳定步骤
            for _ in range(50):  # 增加稳定步数
                self.world.step(render=False)
            
            self.mobile_base = self.world.scene.get_object("create3_robot")
            if self.mobile_base is None:
                raise Exception("无法获取机器人对象")
                
            self._setup_improved_control()
            self._move_arm_to_pose("home")
            
            # 初始化稳定控制器
            self.stabilized_controller = StabilizedRobotController(
                self.mobile_base, self.differential_controller
            )
            
            print("✅ 后加载设置完成")
            return True
            
        except Exception as e:
            print(f"❌ 后加载设置失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _setup_improved_control(self):
        """设置改进的控制"""
        try:
            articulation_controller = self.mobile_base.get_articulation_controller()
            num_dofs = len(self.mobile_base.dof_names)
            
            kp = torch.zeros(num_dofs, dtype=torch.float32)
            kd = torch.zeros(num_dofs, dtype=torch.float32)
            
            # 轮子控制 - 降低阻尼，提高响应性
            for wheel_name in ["left_wheel_joint", "right_wheel_joint"]:
                try:
                    idx = self.mobile_base.dof_names.index(wheel_name)
                    kp[idx] = 0.0
                    kd[idx] = 600.0  # 降低阻尼
                except ValueError:
                    print(f"   轮子关节 {wheel_name} 未找到")
            
            # 机械臂控制
            arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
            for joint_name in arm_joint_names:
                try:
                    idx = self.mobile_base.dof_names.index(joint_name)
                    kp[idx] = 1000.0
                    kd[idx] = 50.0
                except ValueError:
                    print(f"   机械臂关节 {joint_name} 未找到")
            
            # 夹爪控制
            for joint_name in ["panda_finger_joint1", "panda_finger_joint2"]:
                try:
                    idx = self.mobile_base.dof_names.index(joint_name)
                    kp[idx] = 2e5
                    kd[idx] = 2e3
                except ValueError:
                    print(f"   夹爪关节 {joint_name} 未找到")
            
            articulation_controller.set_gains(kps=kp, kds=kd)
            print("   改进的关节控制参数设置完成")
            
        except Exception as e:
            print(f"   控制设置错误: {e}")
            raise
    
    def _move_arm_to_pose(self, pose_name):
        """移动机械臂到姿态"""
        target_positions = self.arm_poses[pose_name]
        articulation_controller = self.mobile_base.get_articulation_controller()
        num_dofs = len(self.mobile_base.dof_names)
        
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        for i, joint_name in enumerate(arm_joint_names):
            idx = self.mobile_base.dof_names.index(joint_name)
            joint_positions[idx] = target_positions[i]
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        for _ in range(25):
            self.world.step(render=False)
    
    def _control_gripper(self, open_close):
        """控制夹爪"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        gripper_position = self.gripper_open if open_close == "open" else self.gripper_closed
        
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        for joint_name in ["panda_finger_joint1", "panda_finger_joint2"]:
            idx = self.mobile_base.dof_names.index(joint_name)
            joint_positions[idx] = gripper_position
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        for _ in range(12):
            self.world.step(render=False)
    
    def get_robot_pose(self):
        """获取机器人姿态 - 改进版"""
        try:
            position, orientation = self.mobile_base.get_world_pose()
            
            # 确保position是numpy数组
            if not isinstance(position, np.ndarray):
                position = np.array(position)
            
            # 四元数转欧拉角
            quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
            r = R.from_quat(quat)
            yaw = r.as_euler('xyz')[2]
            
            self.current_position = position
            self.current_orientation = yaw
            
            return position.copy(), yaw
            
        except Exception as e:
            print(f"   位置获取错误: {e}")
            return self.current_position.copy(), self.current_orientation
    
    def create_trash_environment(self):
        """创建垃圾环境"""
        print("🗑️ 创建垃圾环境...")
        
        # 减少垃圾数量以降低复杂度
        small_trash_positions = [
            [2.5, 0.0, 0.03], [2.0, 1.5, 0.03], [3.0, -0.5, 0.03]
        ]
        
        for i, pos in enumerate(small_trash_positions):
            trash = DynamicSphere(
                prim_path=f"/World/small_trash_{i}",
                name=f"small_trash_{i}",
                position=np.array(pos),
                radius=0.03,
                color=np.array([1.0, 0.2, 0.2])
            )
            self.world.scene.add(trash)
            self.small_trash_objects.append(trash)
        
        # 大垃圾位置
        large_trash_positions = [
            [2.8, 1.0, 0.025], [2.5, -2.0, 0.025]
        ]
        
        for i, pos in enumerate(large_trash_positions):
            trash = DynamicCuboid(
                prim_path=f"/World/large_trash_{i}",
                name=f"large_trash_{i}",
                position=np.array(pos),
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0.2, 0.8, 0.2])
            )
            self.world.scene.add(trash)
            self.large_trash_objects.append(trash)
        
        print(f"✅ 环境创建完成: 小垃圾{len(self.small_trash_objects)}个, 大垃圾{len(self.large_trash_objects)}个")
    
    def plan_lightweight_mission(self):
        """轻量级任务规划"""
        print("\n🎯 开始轻量级任务规划...")
        
        self.all_tasks = []
        current_pos, _ = self.get_robot_pose()
        
        # 小垃圾任务
        for trash in self.small_trash_objects:
            trash_pos = trash.get_world_pose()[0]
            task = TaskInfo(
                target_name=trash.name,
                target_position=trash_pos,
                task_type="small_trash",
                approach_pose="carry"
            )
            self.all_tasks.append(task)
        
        # 大垃圾任务
        for trash in self.large_trash_objects:
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
        
        self._plan_lightweight_paths()
        print(f"✅ 轻量级任务规划完成: {len(self.all_tasks)}个目标")
    
    def _plan_lightweight_paths(self):
        """轻量级路径规划 - 改进版"""
        print("🗺️ 轻量级路径规划...")
        
        current_pos, current_yaw = self.get_robot_pose()
        
        for target_index, task in enumerate(self.all_tasks):
            print(f"   规划目标 {target_index}: {task.target_name}")
            
            target_pos = task.target_position.copy()
            target_pos[2] = 0.0
            
            # 打印起始和目标位置
            print(f"     起始: [{current_pos[0]:.2f}, {current_pos[1]:.2f}]")
            print(f"     目标: [{target_pos[0]:.2f}, {target_pos[1]:.2f}]")
            
            # 使用轻量级路径规划
            safe_path = self.path_planner.find_safe_path(current_pos, target_pos)
            
            # 生成路径节点
            path_nodes = []
            for i, point in enumerate(safe_path):
                if i < len(safe_path) - 1:
                    direction = np.array(safe_path[i + 1]) - np.array(point)
                    orientation = np.arctan2(direction[1], direction[0])
                else:
                    orientation = path_nodes[-1].orientation if path_nodes else current_yaw
                
                arm_config = self.arm_poses[task.approach_pose]
                
                # 确保虚影位置与规划路径一致
                node_position = np.array([point[0], point[1], 0.0])
                
                node = PathNode(
                    position=node_position,
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
            
            # 更新当前位置为目标位置，准备下一次规划
            current_pos = target_pos.copy()
            
            print(f"     生成安全路径: {len(path_nodes)} 个节点")
            
            # 打印前几个节点的位置用于调试
            for j, node in enumerate(path_nodes[:3]):
                print(f"       节点{j}: [{node.position[0]:.2f}, {node.position[1]:.2f}]")
    
    def execute_lightweight_mission(self):
        """执行轻量级任务"""
        print("\n🚀 开始执行轻量级任务...")
        
        for target_index, task in enumerate(self.all_tasks):
            print(f"\n🎯 执行目标 {target_index}: {task.target_name}")
            
            # 在执行新目标前，获取当前位置信息
            current_pos, current_yaw = self.get_robot_pose()
            print(f"   当前机器人位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}], 朝向: {np.degrees(current_yaw):.1f}°")
            print(f"   目标位置: [{task.target_position[0]:.3f}, {task.target_position[1]:.3f}]")
            
            # 获取路径
            path_nodes = self.target_paths[target_index]
            
            # 创建轻量级虚影
            self.ghost_manager.create_target_ghosts(target_index, path_nodes)
            
            # 展示虚影1.5秒
            print("👻 展示无物理虚影...")
            for _ in range(90):
                self.world.step(render=True)
            
            # 执行路径
            print(f"🏃 执行路径（{len(path_nodes)}个节点）...")
            self._execute_stable_path(path_nodes, task)
            
            # 清除虚影
            self.ghost_manager.clear_target_ghosts(target_index)
            
            # 稳定系统
            for _ in range(8):
                self.world.step(render=False)
            
            print(f"✅ 目标 {target_index} 完成")
        
        print("\n🎉 所有目标执行完成!")
        self._show_results()
    
    def _execute_stable_path(self, path_nodes: List[PathNode], task: TaskInfo):
        """执行稳定路径 - 改进版"""
        for i, node in enumerate(path_nodes):
            # 稳定导航到节点
            success = self._navigate_to_node_stable(node, tolerance=0.15)
            
            if not success:
                print(f"   节点 {i} 导航失败，尝试继续...")
                continue
            
            # 检查任务完成
            task_distance = np.linalg.norm(node.position[:2] - task.target_position[:2])
            if task_distance < 0.4 and task.target_name not in self.collected_objects:
                print(f"🎯 到达任务目标: {task.target_name}")
                self._execute_task_action(task)
                
                # 任务完成后的位置校准和稳定
                print(f"   任务完成，进行位置校准...")
                self._post_task_calibration()
                
                return True
            
            # 进度显示
            if i % 3 == 0:
                progress = (i / len(path_nodes)) * 100
                print(f"   路径进度: {progress:.1f}%")
        
        return True
    
    def _navigate_to_node_stable(self, node: PathNode, tolerance: float = 0.15) -> bool:
        """稳定导航到节点 - 改进转弯控制"""
        max_time = 12.0  # 增加最大时间
        start_time = time.time()
        
        print(f"   导航到节点: [{node.position[0]:.2f}, {node.position[1]:.2f}]")
        
        while time.time() - start_time < max_time:
            try:
                current_pos, current_yaw = self.get_robot_pose()
                
                # 检查到达
                distance = np.linalg.norm(current_pos[:2] - node.position[:2])
                if distance < tolerance:
                    # 停止机器人
                    self.stabilized_controller.send_stable_command(0.0, 0.0)
                    print(f"   成功到达节点，距离: {distance:.3f}m")
                    return True
                
                # 计算控制量
                direction = node.position[:2] - current_pos[:2]
                target_angle = np.arctan2(direction[1], direction[0])
                angle_diff = target_angle - current_yaw
                
                # 角度归一化
                while angle_diff > np.pi:
                    angle_diff -= 2 * np.pi
                while angle_diff < -np.pi:
                    angle_diff += 2 * np.pi
                
                # 改进的控制策略 - 高速转弯
                if abs(angle_diff) > 0.1:  # 降低转弯阈值
                    # 纯转弯，大幅提高角速度
                    linear_vel = 0.0
                    angular_vel = np.clip(angle_diff * 15.0, -4.5, 4.5)  # 增益提高约7.5倍
                    print(f"   高速转弯: 角度差={np.degrees(angle_diff):.1f}°, 角速度={angular_vel:.2f}")
                else:
                    # 前进，但保持一定的角度修正
                    linear_vel = min(0.2, max(0.06, distance * 0.6))
                    angular_vel = np.clip(angle_diff * 8.0, -2.0, 2.0)  # 前进时的角度修正也提高
                    print(f"   前进: 距离={distance:.2f}m, 线速度={linear_vel:.2f}")
                
                # 发送稳定控制命令
                self.stabilized_controller.send_stable_command(linear_vel, angular_vel)
                
                # 检查稳定性（但不要太频繁干扰）
                if not self.stabilized_controller.check_movement_stability(current_pos):
                    print(f"   稳定性检查失败，重置控制")
                    # 短暂停止后继续
                    for _ in range(3):
                        self.stabilized_controller.send_stable_command(0.0, 0.0)
                        self.world.step(render=True)
                
                # 步进仿真
                self.world.step(render=True)
                
            except Exception as e:
                print(f"   导航异常: {e}")
                continue
        
        # 超时停止
        print(f"   导航超时，当前距离: {distance:.3f}m")
        self.stabilized_controller.send_stable_command(0.0, 0.0)
        return False
    
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
        
        for trash in self.small_trash_objects:
            if trash.name == task.target_name:
                # 安全处理垃圾对象
                self._safely_remove_trash(trash)
                self.collected_objects.append(task.target_name)
                print(f"✅ {task.target_name} 收集成功!")
                break
    
    def _collect_large_trash(self, task: TaskInfo):
        """收集大垃圾"""
        self._move_arm_to_pose("ready")
        self._control_gripper("open")
        self._move_arm_to_pose("pickup")
        self._control_gripper("close")
        self._move_arm_to_pose("carry")
        
        for trash in self.large_trash_objects:
            if trash.name == task.target_name:
                # 安全处理垃圾对象
                self._safely_remove_trash(trash)
                self.collected_objects.append(task.target_name)
                print(f"✅ {task.target_name} 收集成功!")
                break
    
    def _safely_remove_trash(self, trash_object):
        """安全移除垃圾对象 - 完全避免scene.remove_object"""
        try:
            print(f"   正在隐藏垃圾对象: {trash_object.name}")
            
            # 方法1: 禁用物理属性（但保留对象）
            try:
                trash_object.disable_rigid_body_physics()
                print(f"   禁用物理属性: {trash_object.name}")
            except Exception as e:
                print(f"   物理禁用失败: {e}")
            
            # 方法2: 移动到地下远处
            far_away_position = np.array([100.0, 100.0, -5.0])
            trash_object.set_world_pose(far_away_position, np.array([0, 0, 0, 1]))
            print(f"   移动到远处: {trash_object.name}")
            
            # 方法3: 设置为不可见
            try:
                trash_object.set_visibility(False)
                print(f"   设置为不可见: {trash_object.name}")
            except Exception as e:
                print(f"   可见性设置失败: {e}")
            
            # 完全避免从场景中移除，这会破坏物理仿真
            print(f"   垃圾对象处理完成: {trash_object.name}")
            
            # 等待物理更新
            for _ in range(3):
                self.world.step(render=False)
                
        except Exception as e:
            print(f"   垃圾处理过程中出现错误: {e}")
            # 备用简单处理
            try:
                far_away_position = np.array([100.0, 100.0, -5.0])
                trash_object.set_world_pose(far_away_position, np.array([0, 0, 0, 1]))
                print(f"   使用备用方法隐藏: {trash_object.name}")
            except:
                print(f"   无法处理垃圾对象: {trash_object.name}")
    
    def _post_task_calibration(self):
        """任务完成后的位置校准"""
        try:
            print(f"   开始位置校准...")
            
            # 1. 完全停止机器人
            for _ in range(15):
                self.stabilized_controller.send_stable_command(0.0, 0.0)
                self.world.step(render=False)
            
            # 2. 获取当前位置
            current_pos, current_yaw = self.get_robot_pose()
            print(f"   当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}], 朝向: {np.degrees(current_yaw):.1f}°")
            
            # 3. 重置控制器状态
            if hasattr(self.stabilized_controller, 'stuck_counter'):
                self.stabilized_controller.stuck_counter = 0
            if hasattr(self.stabilized_controller, 'last_position'):
                self.stabilized_controller.last_position = current_pos.copy()
            
            # 4. 清空控制历史
            if hasattr(self.stabilized_controller, 'velocity_filter'):
                self.stabilized_controller.velocity_filter.clear()
            if hasattr(self.stabilized_controller, 'angular_filter'):
                self.stabilized_controller.angular_filter.clear()
            
            # 5. 小幅调整确保系统响应 - 高速版
            for _ in range(3):  # 减少次数但增加速度
                self.stabilized_controller.send_stable_command(0.0, 1.5)  # 快速转动
                self.world.step(render=False)
            
            for _ in range(3):
                self.stabilized_controller.send_stable_command(0.0, -1.5)  # 反向快速转动
                self.world.step(render=False)
                
            # 6. 最终停止
            for _ in range(10):
                self.stabilized_controller.send_stable_command(0.0, 0.0)
                self.world.step(render=False)
            
            print(f"   位置校准完成")
            
        except Exception as e:
            print(f"   位置校准过程中出现错误: {e}")
            # 备用停止
            for _ in range(10):
                try:
                    self.stabilized_controller.send_stable_command(0.0, 0.0)
                    self.world.step(render=False)
                except:
                    break
    
    def _show_results(self):
        """显示结果"""
        total_items = len(self.small_trash_objects) + len(self.large_trash_objects)
        success_count = len(self.collected_objects)
        success_rate = (success_count / total_items) * 100 if total_items > 0 else 0.0
        
        total_nodes = sum(len(path) for path in self.target_paths.values())
        
        print(f"\n📊 轻量级任务执行结果:")
        print(f"   总目标数: {len(self.all_tasks)}")
        print(f"   总垃圾数: {total_items}")
        print(f"   成功收集: {success_count}")
        print(f"   成功率: {success_rate:.1f}%")
        print(f"   总路径节点: {total_nodes}")
        print(f"   轻量级A*路径规划: ✅")
        print(f"   专用无物理虚影资产: ✅")
        print(f"   稳定运动控制: ✅")
        print(f"   灰色透明虚影显示: ✅")
    
    def run_lightweight_demo(self):
        """运行轻量级演示"""
        print("\n" + "="*80)
        print("🚀 轻量级虚影避障系统 - Isaac Sim 4.5 稳定版")
        print("🗺️ 轻量级A*路径规划 | 👻 专用无物理资产 | 🎮 稳定运动控制")
        print("="*80)
        
        pos, yaw = self.get_robot_pose()
        print(f"📍 初始位置: [{pos[0]:.3f}, {pos[1]:.3f}], 朝向: {np.degrees(yaw):.1f}°")
        
        self.plan_lightweight_mission()
        self.execute_lightweight_mission()
        
        self._move_arm_to_pose("home")
        
        print("\n🎉 轻量级虚影避障系统演示完成!")
        print("💡 稳定运动控制，无路径线干扰，专用虚影资产")
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理系统资源...")
        try:
            if self.ghost_manager is not None:
                self.ghost_manager.cleanup_all()
        except Exception as e:
            print(f"   虚影清理错误: {e}")
        
        try:
            if self.world is not None:
                self.world.stop()
        except Exception as e:
            print(f"   世界停止错误: {e}")

def main():
    """主函数"""
    print("🚀 启动轻量级虚影避障系统...")
    
    system = LightweightRobotSystem()
    
    try:
        # 初始化系统
        if not system.initialize_system():
            print("❌ 系统初始化失败")
            return
            
        if not system.initialize_robot():
            print("❌ 机器人初始化失败") 
            return
            
        if not system.setup_post_load():
            print("❌ 后加载设置失败")
            return
            
        system.create_trash_environment()
        
        # 稳定系统
        print("⚡ 系统稳定中...")
        for _ in range(50):  # 增加稳定时间
            system.world.step(render=False)
            time.sleep(0.01)  # 减少延迟
        
        # 运行轻量级演示
        system.run_lightweight_demo()
        
        # 保持运行
        print("\n💡 轻量级系统运行中，按 Ctrl+C 退出")
        while True:
            system.world.step(render=True)
            time.sleep(0.016)
            
    except KeyboardInterrupt:
        print("\n👋 用户中断，正在清理...")
    except Exception as e:
        print(f"\n❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 安全清理
        try:
            system.cleanup()
        except Exception as cleanup_error:
            print(f"   清理过程错误: {cleanup_error}")

if __name__ == "__main__":
    main()