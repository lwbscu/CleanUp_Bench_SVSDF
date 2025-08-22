#!/usr/bin/env python3
"""
Isaac Sim 4.5 动态路径显示系统
- 内存充足时使用虚影显示路径
- 达到阈值后切换到轻量级线条显示
- 确保资源不会溢出
"""

import psutil
import torch

def print_memory_usage(stage_name: str = ""):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"💾 {stage_name} 内存: {memory_mb:.1f}MB")
    return memory_mb

def print_stage_statistics(stage, stage_name: str = ""):
    """打印USD stage统计信息"""
    total_prims = 0
    ghost_prims = 0
    robot_prims = 0
    
    for prim in stage.Traverse():
        total_prims += 1
        prim_path = str(prim.GetPath())
        if "Ghost" in prim_path:
            ghost_prims += 1
        elif "create3" in prim_path or "robot" in prim_path:
            robot_prims += 1
    
    print(f"📊 {stage_name} Stage统计: 总Prim={total_prims}, 虚影={ghost_prims}, 机器人={robot_prims}")

# =============================================================================
# 🎮 用户参数设置
# =============================================================================
# 机器人运动控制参数
MAX_LINEAR_VELOCITY = 0.18     # 机器人最大直线运动速度(m/s) - 控制前进后退的最大速度
MAX_ANGULAR_VELOCITY = 2.8     # 机器人最大角速度(rad/s) - 控制转弯时的最大旋转速度

# PID控制增益参数  
TURN_GAIN = 6.0                # 转弯控制增益 - 纯转弯时角度误差的放大系数，值越大转弯越敏感
FORWARD_ANGLE_GAIN = 3.0       # 前进时角度修正增益 - 直线行驶时的航向角修正系数

# 路径可视化显示参数
GHOST_DISPLAY_STEPS = 35       # 虚影路径展示步数 - 创建虚影后在屏幕上静态展示的仿真步数
GHOSTS_PER_TARGET = 4          # 每个目标的默认虚影数量 - 实际会根据路径长度动态调整为3-5个

# 导航控制参数
NAVIGATION_TOLERANCE = 0.15    # 导航到达容差(m) - 机器人到目标点的距离小于此值时认为到达
MAX_NAVIGATION_TIME = 8.0      # 单个导航点最大超时时间(s) - 避免机器人在某点卡死过久

# 系统稳定性参数
STABILIZE_STEPS = 20           # 系统稳定化步数 - 初始化后让物理引擎稳定运行的步数
MEMORY_THRESHOLD_MB = 5500     # 内存阈值(MB) - 超过此内存使用量时自动从虚影显示切换到轻量级线条显示
# =============================================================================

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "enable_livestream": False, 
    "enable_cameras": True,
    "enable_rtx": True,
    "physics_dt": 1.0/120.0,
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
        print(f"🗺️ 路径规划器: {self.grid_size}x{self.grid_size}网格")
    
    def add_obstacle(self, position: np.ndarray, size: np.ndarray, shape_type: str = 'box'):
        """添加障碍物"""
        self.obstacles.append({'pos': position, 'size': size, 'type': shape_type})
        
        center_x = int((position[0] + self.world_size/2) / self.resolution)
        center_y = int((position[1] + self.world_size/2) / self.resolution)
        
        safety_margin = 0.6
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
        
        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        max_iterations = 800
        iterations = 0
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)[1]
            
            if current == goal_grid:
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
        
        return self._create_simple_path(start_pos, goal_pos)
    
    def world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        x = int((world_pos[0] + self.world_size/2) / self.resolution)
        y = int((world_pos[1] + self.world_size/2) / self.resolution)
        return np.clip(x, 0, self.grid_size-1), np.clip(y, 0, self.grid_size-1)
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        x = (grid_x * self.resolution) - self.world_size/2
        y = (grid_y * self.resolution) - self.world_size/2
        return np.array([x, y, 0.0])
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _smooth_path_simple(self, path: List[np.ndarray]) -> List[np.ndarray]:
        if len(path) <= 3:
            return path
        
        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = min(i + 3, len(path) - 1)
            smoothed.append(path[j])
            i = j
        
        return smoothed
    
    def _create_simple_path(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
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

class DynamicPathVisualizer:
    """动态路径可视化器 - 智能切换虚影和线条"""
    
    def __init__(self, world: World):
        self.world = world
        self.current_strategy = "ghost"  # ghost 或 line
        self.memory_threshold = MEMORY_THRESHOLD_MB
        
        # 虚影相关
        self.robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
        self.ghost_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm3.usd"
        self.ghost_container_path = "/World/GhostVisualization"
        
        # 线条相关
        self.line_container_path = "/World/PathLines"
        self.line_prims = []
        
        print(f"🎨 动态路径可视化器初始化")
        print(f"   内存阈值: {self.memory_threshold}MB")
        print(f"   初始策略: {self.current_strategy}")
    
    def check_memory_and_decide_strategy(self) -> str:
        """检查内存并决定策略"""
        current_memory = print_memory_usage("策略检查")
        
        if current_memory > self.memory_threshold:
            if self.current_strategy == "ghost":
                print(f"🔄 内存超阈值({current_memory:.1f}MB > {self.memory_threshold}MB)，切换到线条显示")
                self.current_strategy = "line"
                self._clear_ghosts()
        else:
            if self.current_strategy == "line":
                print(f"🔄 内存充足({current_memory:.1f}MB < {self.memory_threshold}MB)，切换到虚影显示")
                self.current_strategy = "ghost"
                self._clear_lines()
        
        return self.current_strategy
    
    def visualize_path(self, target_index: int, path_nodes: List[PathNode]):
        """可视化路径 - 动态选择策略"""
        strategy = self.check_memory_and_decide_strategy()
        
        print(f"🎨 目标{target_index} 使用策略: {strategy}")
        
        if strategy == "ghost":
            self._create_ghost_visualization(target_index, path_nodes)
        else:
            self._create_line_visualization(target_index, path_nodes)
    
    def clear_current_visualization(self, target_index: int):
        """清除当前可视化"""
        if self.current_strategy == "ghost":
            self._clear_ghosts()
        else:
            self._clear_lines()
    
    def _create_ghost_visualization(self, target_index: int, path_nodes: List[PathNode]):
        """创建虚影可视化"""
        print(f"👻 创建虚影可视化...")
        
        # 清理旧容器
        self._delete_entire_container(self.ghost_container_path)
        
        # 创建新容器
        stage = self.world.stage
        container_prim = stage.DefinePrim(self.ghost_container_path, "Xform")
        
        # 动态选择虚影数量（3-5个，根据路径长度）
        ghost_count = self._calculate_dynamic_ghost_count(path_nodes)
        selected_nodes = self._select_nodes(path_nodes, ghost_count)
        
        print(f"   路径长度: {len(path_nodes)}节点 → 虚影数量: {ghost_count}个")
        
        # 创建虚影
        for i, node in enumerate(selected_nodes):
            ghost_path = f"{self.ghost_container_path}/Target_{target_index}_Ghost_{i}"
            self._create_single_ghost(ghost_path, node)
            self.world.step(render=False)
    
    def _calculate_dynamic_ghost_count(self, path_nodes: List[PathNode]) -> int:
        """根据路径长度动态计算虚影数量"""
        path_length = len(path_nodes)
        
        if path_length <= 10:
            return 3  # 短路径，3个虚影
        elif path_length <= 15:
            return 4  # 中等路径，4个虚影
        else:
            return 5  # 长路径，5个虚影
    
    def _create_line_visualization(self, target_index: int, path_nodes: List[PathNode]):
        """创建线条可视化 - 2D贴地路径"""
        print(f"📏 创建2D贴地路径线条...")
        
        # 清理旧线条
        self._clear_lines()
        
        # 创建线条容器
        stage = self.world.stage
        if not stage.GetPrimAtPath(self.line_container_path):
            container_prim = stage.DefinePrim(self.line_container_path, "Xform")
        
        # 创建路径线条
        self._create_path_lines(target_index, path_nodes)
        
        # 创建关键点标记
        self._create_waypoint_markers(target_index, path_nodes)
    
    def _create_path_lines(self, target_index: int, path_nodes: List[PathNode]):
        """创建路径线条 - 2D贴地显示"""
        if len(path_nodes) < 2:
            return
        
        stage = self.world.stage
        line_path = f"{self.line_container_path}/PathLine_{target_index}"
        
        # 创建线条几何
        line_prim = stage.DefinePrim(line_path, "BasisCurves")
        line_geom = UsdGeom.BasisCurves(line_prim)
        
        # 设置曲线属性
        line_geom.CreateTypeAttr().Set("linear")
        line_geom.CreateBasisAttr().Set("bspline")
        
        # 构建点列表 - 贴地面显示
        points = []
        curve_vertex_counts = []
        
        # 创建连续线段，贴近地面
        for i in range(len(path_nodes) - 1):
            start_pos = path_nodes[i].position
            end_pos = path_nodes[i + 1].position
            
            # 非常贴近地面，只抬高0.02m避免z-fighting
            start_pos_ground = Gf.Vec3f(float(start_pos[0]), float(start_pos[1]), 0.02)
            end_pos_ground = Gf.Vec3f(float(end_pos[0]), float(end_pos[1]), 0.02)
            
            points.extend([start_pos_ground, end_pos_ground])
            curve_vertex_counts.append(2)
        
        # 设置几何数据
        line_geom.CreatePointsAttr().Set(points)
        line_geom.CreateCurveVertexCountsAttr().Set(curve_vertex_counts)
        
        # 设置线条宽度 - 很细的线条
        line_geom.CreateWidthsAttr().Set([0.05] * len(points))  # 0.5cm宽的细线

        # 设置线条材质
        self._setup_line_material(line_prim, target_index)
        
        self.line_prims.append(line_path)
        print(f"   创建贴地路径线条: {len(points)//2}段")
    
    def _create_waypoint_markers(self, target_index: int, path_nodes: List[PathNode]):
        """创建路径点标记 - 小巧贴地显示"""
        stage = self.world.stage
        selected_nodes = self._select_nodes(path_nodes, min(6, len(path_nodes)))
        
        for i, node in enumerate(selected_nodes):
            marker_path = f"{self.line_container_path}/Waypoint_{target_index}_{i}"
            
            # 创建小球标记
            marker_prim = stage.DefinePrim(marker_path, "Sphere")
            sphere_geom = UsdGeom.Sphere(marker_prim)
            
            # 设置很小的半径 - 不遮挡视野
            sphere_geom.CreateRadiusAttr().Set(0.1)  # 只有1cm半径

            # 设置位置 - 贴近地面
            marker_pos = Gf.Vec3d(float(node.position[0]), float(node.position[1]), 0.03)  # 只抬高3cm
            xform = UsdGeom.Xformable(marker_prim)
            translate_op = xform.AddTranslateOp()
            translate_op.Set(marker_pos)
            
            # 设置颜色 - 更亮一些方便看到
            if i == 0:
                color = [0.2, 1.0, 0.2]  # 绿色起点
            elif i == len(selected_nodes)-1:
                color = [1.0, 0.2, 0.2]  # 红色终点
            else:
                color = [0.2, 0.6, 1.0]  # 蓝色中间点
            
            self._setup_marker_material(marker_prim, color)
            
            self.line_prims.append(marker_path)
        
        print(f"   创建贴地路径点标记: {len(selected_nodes)}个")
    
    def _setup_line_material(self, line_prim, target_index: int):
        """设置线条材质 - 2D地面路径风格"""
        # 创建材质
        material_path = f"/World/Materials/LineMaterial_{target_index}"
        stage = self.world.stage
        
        material_prim = stage.DefinePrim(material_path, "Material")
        material = UsdShade.Material(material_prim)
        
        # 创建shader
        shader_prim = stage.DefinePrim(f"{material_path}/Shader", "Shader")
        shader = UsdShade.Shader(shader_prim)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        # 设置明亮的路径颜色 - 像地面导航线
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.8, 1.0))  # 明亮青色
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0.05, 0.4, 0.5))  # 轻微发光
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)  # 较高粗糙度，避免反光
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)  # 非金属
        
        # 连接输出
        material_output = material.CreateSurfaceOutput()
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material_output.ConnectToSource(shader_output)
        
        # 绑定材质
        UsdShade.MaterialBindingAPI(line_prim).Bind(material)
    
    def _setup_marker_material(self, marker_prim, color: List[float]):
        """设置标记材质"""
        # 简化材质设置，直接使用displayColor
        marker_geom = UsdGeom.Gprim(marker_prim)
        marker_geom.CreateDisplayColorAttr().Set([color])
    
    def _create_single_ghost(self, ghost_path: str, node: PathNode):
        """创建单个虚影"""
        stage = self.world.stage
        
        # 创建虚影prim
        ghost_prim = stage.DefinePrim(ghost_path, "Xform")
        
        # 添加引用
        references = ghost_prim.GetReferences()
        references.AddReference(self.ghost_usd_path)
        
        # 设置变换 - 使用正确的USD类型
        ghost_position = Gf.Vec3d(float(node.position[0]), float(node.position[1]), float(node.position[2]))
        yaw_degrees = float(np.degrees(node.orientation))
        
        xform = UsdGeom.Xformable(ghost_prim)
        xform.ClearXformOpOrder()
        
        translate_op = xform.AddTranslateOp()
        translate_op.Set(ghost_position)
        
        if abs(yaw_degrees) > 1.0:
            rotate_op = xform.AddRotateZOp()
            rotate_op.Set(yaw_degrees)
    
    def _select_nodes(self, path_nodes: List[PathNode], count: int) -> List[PathNode]:
        """选择关键节点"""
        if len(path_nodes) <= count:
            return path_nodes
        
        selected = [path_nodes[0]]  # 起始点
        
        # 均匀分布
        step = len(path_nodes) // (count - 1)
        for i in range(1, count - 1):
            index = min(i * step, len(path_nodes) - 1)
            selected.append(path_nodes[index])
        
        selected.append(path_nodes[-1])  # 终点
        return selected
    
    def _clear_ghosts(self):
        """清除虚影"""
        self._delete_entire_container(self.ghost_container_path)
    
    def _clear_lines(self):
        """清除线条"""
        stage = self.world.stage
        
        # 删除所有线条prims
        for line_path in self.line_prims:
            if stage.GetPrimAtPath(line_path):
                stage.RemovePrim(line_path)
        
        self.line_prims.clear()
        
        # 删除线条容器
        if stage.GetPrimAtPath(self.line_container_path):
            stage.RemovePrim(self.line_container_path)
        
        # 删除材质
        materials_path = "/World/Materials"
        if stage.GetPrimAtPath(materials_path):
            materials_prim = stage.GetPrimAtPath(materials_path)
            for child in materials_prim.GetChildren():
                if "LineMaterial" in str(child.GetPath()):
                    stage.RemovePrim(child.GetPath())
    
    def _delete_entire_container(self, container_path: str):
        """删除整个容器"""
        stage = self.world.stage
        
        if stage.GetPrimAtPath(container_path):
            stage.RemovePrim(container_path)
            for _ in range(5):
                self.world.step(render=False)
    
    def cleanup_all(self):
        """清理所有资源"""
        print("🧹 清理可视化资源...")
        self._clear_ghosts()
        self._clear_lines()

class StabilizedRobotController:
    """稳定机器人控制器"""
    
    def __init__(self, mobile_base, differential_controller):
        self.mobile_base = mobile_base
        self.differential_controller = differential_controller
        self.max_linear_velocity = MAX_LINEAR_VELOCITY  
        self.max_angular_velocity = MAX_ANGULAR_VELOCITY
        
        self.velocity_filter = deque(maxlen=5)
        self.angular_filter = deque(maxlen=5)
        
        self.last_position = None
        self.stuck_counter = 0
        self.stuck_threshold = 100
        
        print("🎮 稳定控制器初始化")
    
    def send_stable_command(self, target_linear_vel: float, target_angular_vel: float):
        """发送稳定控制命令"""
        target_linear_vel = np.clip(target_linear_vel, -self.max_linear_velocity, self.max_linear_velocity)
        target_angular_vel = np.clip(target_angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
        
        self.velocity_filter.append(target_linear_vel)
        self.angular_filter.append(target_angular_vel)
        
        smooth_linear = np.mean(list(self.velocity_filter))
        smooth_angular = np.mean(list(self.angular_filter))
        
        self._apply_wheel_control(smooth_linear, smooth_angular)
    
    def _apply_wheel_control(self, linear_vel: float, angular_vel: float):
        """应用轮子控制"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        wheel_radius = 0.036
        wheel_base = 0.235
        
        left_wheel_vel = (linear_vel - angular_vel * wheel_base / 2.0) / wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * wheel_base / 2.0) / wheel_radius
        
        # 直线运动对称性
        if abs(angular_vel) < 0.05:
            avg_vel = (left_wheel_vel + right_wheel_vel) / 2.0
            left_wheel_vel = avg_vel
            right_wheel_vel = avg_vel
        
        num_dofs = len(self.mobile_base.dof_names)
        joint_velocities = torch.zeros(num_dofs, dtype=torch.float32)
        
        left_wheel_idx = self.mobile_base.dof_names.index("left_wheel_joint")
        right_wheel_idx = self.mobile_base.dof_names.index("right_wheel_joint")
        
        joint_velocities[left_wheel_idx] = float(left_wheel_vel)
        joint_velocities[right_wheel_idx] = float(right_wheel_vel)
        
        action = ArticulationAction(joint_velocities=joint_velocities)
        articulation_controller.apply_action(action)
    
    def check_movement_stability(self, current_position: np.ndarray) -> bool:
        """检查运动稳定性"""
        if self.last_position is not None:
            movement = np.linalg.norm(current_position[:2] - self.last_position[:2])
            
            if movement < 0.005:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            
            if self.stuck_counter >= self.stuck_threshold:
                print("   检测到卡住，执行恢复...")
                self._unstuck_recovery()
                self.stuck_counter = 0
                return False
        
        self.last_position = current_position.copy()
        return True
    
    def _unstuck_recovery(self):
        """解卡恢复"""
        print("   执行解卡...")
        
        # 完全停止
        for _ in range(8):
            self.send_stable_command(0.0, 0.0)
        
        # 恢复动作
        recovery_actions = [
            (-0.1, 0.0),   # 后退
            (0.0, 2.5),    # 左转
            (0.0, -2.5),   # 右转
            (-0.08, 2.0),  # 后退左转
            (-0.08, -2.0), # 后退右转
        ]
        
        for linear, angular in recovery_actions:
            for _ in range(6):
                self.send_stable_command(linear, angular)
            for _ in range(2):
                self.send_stable_command(0.0, 0.0)
        
        # 最终停止
        for _ in range(6):
            self.send_stable_command(0.0, 0.0)

class OptimizedRobotSystem:
    """优化版机器人系统 - 动态路径显示"""
    
    def __init__(self):
        self.world = None
        self.robot_prim_path = "/World/create3_robot"
        self.mobile_base = None
        self.differential_controller = None
        self.stabilized_controller = None
        
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = 0.0
        
        self.small_trash_objects = []
        self.large_trash_objects = []
        self.collected_objects = []
        
        self.path_planner = None
        self.path_visualizer = None  # 新的动态可视化器
        
        self.all_tasks = []
        self.target_paths = {}
        
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
        print("🚀 初始化动态路径显示系统...")
        
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0/120.0,
            rendering_dt=1.0/60.0
        )
        self.world.scene.clear()
        
        physics_context = self.world.get_physics_context()
        physics_context.set_gravity(-9.81)
        physics_context.set_solver_type("TGS")
        physics_context.enable_gpu_dynamics(True)
        
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
        
        print("✅ 系统初始化完成")
        print_memory_usage("系统初始化完成")
        return True
    
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
        self.path_visualizer = DynamicPathVisualizer(self.world)  # 使用动态可视化器
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
            max_linear_speed=MAX_LINEAR_VELOCITY,  
            max_angular_speed=MAX_ANGULAR_VELOCITY
        )
        
        print("✅ 机器人初始化成功")
        print_memory_usage("机器人初始化完成")
        return True
    
    def setup_post_load(self):
        """后加载设置"""
        print("🔧 后加载设置...")
        
        self.world.reset()
        
        for _ in range(STABILIZE_STEPS):
            self.world.step(render=False)
        
        self.mobile_base = self.world.scene.get_object("create3_robot")
        self._setup_improved_control()
        self._move_arm_to_pose("home")
        
        self.stabilized_controller = StabilizedRobotController(
            self.mobile_base, self.differential_controller
        )
        
        print("✅ 后加载设置完成")
        print_memory_usage("后加载设置完成")
        return True
    
    def _setup_improved_control(self):
        """设置控制参数"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        num_dofs = len(self.mobile_base.dof_names)
        
        kp = torch.zeros(num_dofs, dtype=torch.float32)
        kd = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 轮子控制
        for wheel_name in ["left_wheel_joint", "right_wheel_joint"]:
            idx = self.mobile_base.dof_names.index(wheel_name)
            kp[idx] = 0.0
            kd[idx] = 500.0
        
        # 机械臂控制
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        for joint_name in arm_joint_names:
            idx = self.mobile_base.dof_names.index(joint_name)
            kp[idx] = 1000.0
            kd[idx] = 50.0
        
        # 夹爪控制
        for joint_name in ["panda_finger_joint1", "panda_finger_joint2"]:
            idx = self.mobile_base.dof_names.index(joint_name)
            kp[idx] = 2e5
            kd[idx] = 2e3
        
        articulation_controller.set_gains(kps=kp, kds=kd)
        print("   控制参数设置完成")
    
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
        
        for _ in range(20):
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
        
        for _ in range(10):
            self.world.step(render=False)
    
    def get_robot_pose(self):
        """获取机器人姿态"""
        position, orientation = self.mobile_base.get_world_pose()
        
        if not isinstance(position, np.ndarray):
            position = np.array(position)
        
        quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        r = R.from_quat(quat)
        yaw = r.as_euler('xyz')[2]
        
        self.current_position = position
        self.current_orientation = yaw
        
        return position.copy(), yaw
    
    def create_trash_environment(self):
        """创建垃圾环境"""
        print("🗑️ 创建垃圾环境...")
        
        # 创建少量垃圾用于测试
        small_trash_positions = [
            [2.5, 0.0, 0.03], [2.0, 1.5, 0.03]
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
        
        large_trash_positions = [
            [2.8, 1.0, 0.025]
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
        print_memory_usage("垃圾环境创建完成")
    
    def plan_mission(self):
        """任务规划"""
        print("\n🎯 开始任务规划...")
        
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
        
        self._plan_paths()
        print(f"✅ 任务规划完成: {len(self.all_tasks)}个目标")
        print_memory_usage("任务规划完成")
    
    def _plan_paths(self):
        """路径规划"""
        print("🗺️ 路径规划...")
        
        current_pos, current_yaw = self.get_robot_pose()
        
        for target_index, task in enumerate(self.all_tasks):
            print(f"   规划目标 {target_index}: {task.target_name}")
            
            target_pos = task.target_position.copy()
            target_pos[2] = 0.0
            
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
            current_pos = target_pos.copy()
            print(f"     路径节点: {len(path_nodes)} 个")
    
    def execute_mission(self):
        """执行任务"""
        print("\n🚀 开始执行动态路径显示任务...")
        print_memory_usage("任务开始前")
        print_stage_statistics(self.world.stage, "任务开始前")
        
        for target_index, task in enumerate(self.all_tasks):
            print(f"\n🎯 执行目标 {target_index}: {task.target_name}")
            
            current_pos, current_yaw = self.get_robot_pose()
            print(f"   当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
            print(f"   目标位置: [{task.target_position[0]:.3f}, {task.target_position[1]:.3f}]")
            
            path_nodes = self.target_paths[target_index]
            
            # 使用动态可视化器
            print(f"🎨 ====== 目标{target_index}路径可视化开始 ======")
            print_memory_usage(f"目标{target_index}可视化前")
            self.path_visualizer.visualize_path(target_index, path_nodes)
            print_memory_usage(f"目标{target_index}可视化后")
            print(f"🎨 ====== 目标{target_index}路径可视化完成 ======")
            
            # 展示路径
            display_steps = GHOST_DISPLAY_STEPS if self.path_visualizer.current_strategy == "ghost" else 15
            strategy_name = "虚影" if self.path_visualizer.current_strategy == "ghost" else "2D贴地线条"
            print(f"👁️ 展示路径 ({display_steps}步, 策略:{strategy_name})...")
            for step in range(display_steps):
                self.world.step(render=True)
                if step % 5 == 0:
                    print(f"   展示进度: {step}/{display_steps}")
            
            # 执行路径
            print(f"🏃 执行路径（{len(path_nodes)}个节点）...")
            self._execute_path(path_nodes, task)
            
            # 清除可视化
            print(f"🧹 ====== 目标{target_index}路径清理开始 ======")
            print_memory_usage(f"目标{target_index}清理前")
            self.path_visualizer.clear_current_visualization(target_index)
            print_memory_usage(f"目标{target_index}清理后")
            print(f"🧹 ====== 目标{target_index}路径清理完成 ======")
            
            # 强制垃圾回收
            print(f"🔄 执行强制垃圾回收...")
            for i in range(5):
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.world.step(render=False)
            
            print(f"✅ 目标 {target_index} 完成")
            print_stage_statistics(self.world.stage, f"目标{target_index}完成后")
            print_memory_usage(f"目标{target_index}最终内存")
        
        print("\n🎉 所有目标执行完成!")
        print_memory_usage("所有任务完成后")
        print_stage_statistics(self.world.stage, "所有任务完成后")
        self._show_results()
    
    def _execute_path(self, path_nodes: List[PathNode], task: TaskInfo):
        """执行路径"""
        for i, node in enumerate(path_nodes):
            success = self._navigate_to_node(node, tolerance=NAVIGATION_TOLERANCE)
            
            if not success:
                print(f"   节点 {i} 导航失败，继续...")
                continue
            
            # 检查任务完成
            task_distance = np.linalg.norm(node.position[:2] - task.target_position[:2])
            if task_distance < 0.4 and task.target_name not in self.collected_objects:
                print(f"🎯 到达任务目标: {task.target_name}")
                self._execute_task_action(task)
                self._post_task_calibration()
                return True
            
            # 进度显示
            if i % 3 == 0:
                progress = (i / len(path_nodes)) * 100
                print(f"   路径进度: {progress:.1f}%")
        
        return True
    
    def _navigate_to_node(self, node: PathNode, tolerance: float = None) -> bool:
        """导航到节点"""
        if tolerance is None:
            tolerance = NAVIGATION_TOLERANCE
            
        max_time = MAX_NAVIGATION_TIME
        start_time = time.time()
        step_counter = 0
        
        print(f"   导航到: [{node.position[0]:.2f}, {node.position[1]:.2f}]")
        
        while time.time() - start_time < max_time:
            current_pos, current_yaw = self.get_robot_pose()
            step_counter += 1
            
            # 检查到达
            distance = np.linalg.norm(current_pos[:2] - node.position[:2])
            if distance < tolerance:
                self.stabilized_controller.send_stable_command(0.0, 0.0)
                print(f"   到达节点，距离: {distance:.3f}m")
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
            
            # 控制策略
            if abs(angle_diff) > 0.1:
                linear_vel = 0.0
                angular_vel = np.clip(angle_diff * TURN_GAIN, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
                if step_counter % 500 == 0:
                    print(f"   转弯: 角度差={np.degrees(angle_diff):.1f}°")
            else:
                linear_vel = min(MAX_LINEAR_VELOCITY, max(0.06, distance * 0.6))
                angular_vel = np.clip(angle_diff * FORWARD_ANGLE_GAIN, -2.0, 2.0)
                if step_counter % 500 == 0:
                    print(f"   前进: 距离={distance:.2f}m")
            
            self.stabilized_controller.send_stable_command(linear_vel, angular_vel)
            
            # 稳定性检查
            if not self.stabilized_controller.check_movement_stability(current_pos):
                print(f"   稳定性检查失败")
                for _ in range(3):
                    self.stabilized_controller.send_stable_command(0.0, 0.0)
                    self.world.step(render=True)
            
            self.world.step(render=True)
        
        # 超时
        print(f"   导航超时")
        self.stabilized_controller.send_stable_command(0.0, 0.0)
        return False
    
    def _execute_task_action(self, task: TaskInfo):
        """执行任务动作"""
        print(f"🎯 执行任务: {task.target_name}")
        
        if task.task_type == "small_trash":
            self._collect_small_trash(task)
        elif task.task_type == "large_trash":
            self._collect_large_trash(task)
    
    def _collect_small_trash(self, task: TaskInfo):
        """收集小垃圾"""
        self._move_arm_to_pose("carry")
        
        for trash in self.small_trash_objects:
            if trash.name == task.target_name:
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
                self._safely_remove_trash(trash)
                self.collected_objects.append(task.target_name)
                print(f"✅ {task.target_name} 收集成功!")
                break
    
    def _safely_remove_trash(self, trash_object):
        """安全移除垃圾对象"""
        print(f"   隐藏垃圾: {trash_object.name}")
        
        # 禁用物理
        trash_object.disable_rigid_body_physics()
        
        # 移动到远处
        far_away_position = np.array([100.0, 100.0, -5.0])
        trash_object.set_world_pose(far_away_position, np.array([0, 0, 0, 1]))
        
        # 设置不可见
        trash_object.set_visibility(False)
        
        # 等待更新
        for _ in range(3):
            self.world.step(render=False)
    
    def _post_task_calibration(self):
        """任务后校准"""
        print(f"   位置校准...")
        
        # 完全停止
        for _ in range(12):
            self.stabilized_controller.send_stable_command(0.0, 0.0)
            self.world.step(render=False)
        
        # 重置控制器状态
        self.stabilized_controller.stuck_counter = 0
        current_pos, _ = self.get_robot_pose()
        self.stabilized_controller.last_position = current_pos.copy()
        
        # 清空滤波器
        self.stabilized_controller.velocity_filter.clear()
        self.stabilized_controller.angular_filter.clear()
        
        # 小幅调整
        for _ in range(2):
            self.stabilized_controller.send_stable_command(0.0, 1.2)
            self.world.step(render=False)
        
        for _ in range(2):
            self.stabilized_controller.send_stable_command(0.0, -1.2)
            self.world.step(render=False)
        
        # 最终停止
        for _ in range(8):
            self.stabilized_controller.send_stable_command(0.0, 0.0)
            self.world.step(render=False)
        
        print(f"   校准完成")
    
    def _show_results(self):
        """显示结果"""
        total_items = len(self.small_trash_objects) + len(self.large_trash_objects)
        success_count = len(self.collected_objects)
        success_rate = (success_count / total_items) * 100 if total_items > 0 else 0.0
        
        total_nodes = sum(len(path) for path in self.target_paths.values())
        
        print(f"\n📊 动态路径显示任务执行结果:")
        print(f"   总目标数: {len(self.all_tasks)}")
        print(f"   总垃圾数: {total_items}")
        print(f"   成功收集: {success_count}")
        print(f"   成功率: {success_rate:.1f}%")
        print(f"   总路径节点: {total_nodes}")
        print(f"   内存阈值: {MEMORY_THRESHOLD_MB}MB")
        print(f"🎨 动态策略: 智能切换虚影/2D贴地线条")
        print(f"👻 虚影数量: 3-5个（根据路径长度动态调整）")
        print(f"📏 线条显示: 超薄贴地，不遮挡机器人视野")
        print(f"✅ 资源占用得到有效控制")
        print(f"✅ 避免内存溢出问题")
    
    def run_demo(self):
        """运行演示"""
        print("\n" + "="*80)
        print("🚀 动态路径显示系统 - Isaac Sim 4.5")
        print("🎨 智能切换策略 | 👻 虚影 ⟷ 📏 2D贴地线条 | 🛡️ 防止资源溢出")
        print("="*80)
        
        pos, yaw = self.get_robot_pose()
        print(f"📍 初始位置: [{pos[0]:.3f}, {pos[1]:.3f}], 朝向: {np.degrees(yaw):.1f}°")
        
        self.plan_mission()
        self.execute_mission()
        
        self._move_arm_to_pose("home")
        
        print("\n🎉 动态路径显示系统演示完成!")
        print("💡 智能策略确保了资源的高效利用")
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理系统资源...")
        print_memory_usage("最终清理前")
        
        if self.path_visualizer is not None:
            self.path_visualizer.cleanup_all()
            
        print("   强制垃圾回收...")
        for i in range(10):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.world is not None:
            print("   清理物理世界...")
            for _ in range(8):
                self.world.step(render=False)
            
            self.world.stop()
            print("   世界停止完成")
        
        for i in range(5):
            gc.collect()
        
        print_memory_usage("最终清理后")
        print("✅ 资源清理完成")

def main():
    """主函数"""
    print("🚀 启动动态路径显示系统...")
    print(f"⚙️ 运动参数: 线速度={MAX_LINEAR_VELOCITY}m/s, 角速度={MAX_ANGULAR_VELOCITY}rad/s")
    print(f"⚙️ 显示设置: 虚影3-5个(根据路径长度), 2D贴地线条")
    print(f"⚙️ 内存管理: 阈值={MEMORY_THRESHOLD_MB}MB")
    print(f"🎨 智能策略: 动态切换虚影/2D贴地线条显示")
    
    system = OptimizedRobotSystem()
    
    system.initialize_system()
    system.initialize_robot()
    system.setup_post_load()
    system.create_trash_environment()
    
    # 稳定系统
    print("⚡ 系统稳定中...")
    for _ in range(STABILIZE_STEPS):
        system.world.step(render=False)
        time.sleep(0.01)
    
    # 运行演示
    system.run_demo()
    
    # 保持运行一段时间用于观察
    print("\n💡 系统运行中，按 Ctrl+C 退出")
    for i in range(100):
        system.world.step(render=True)
        time.sleep(0.1)
    
    system.cleanup()

if __name__ == "__main__":
    main()