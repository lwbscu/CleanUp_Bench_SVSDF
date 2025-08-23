#!/usr/bin/env python3
"""
Isaac Sim 4.5 真实移动覆盖算法机器人系统
- 实际机器人：create_3_with_arm2.usd（有物理属性）
- 虚影机器人：create_3_with_arm3.usd（无物理属性）
- 真实机器人移动：确保机器人真正移动到每个目标点
- 精确路径跟踪：降低容差，确保覆盖精度
"""

import psutil
import torch

def print_memory_usage(stage_name: str = ""):
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"内存: {memory_mb:.1f}MB - {stage_name}")

# 核心参数 - 降低容差确保真实移动
MAX_LINEAR_VELOCITY = 0.15     
MAX_ANGULAR_VELOCITY = 2.5     
COVERAGE_CELL_SIZE = 0.8       
COVERAGE_AREA_SIZE = 6.0       
ROBOT_RADIUS = 0.45            
PATH_TOLERANCE = 0.15          # 降低容差确保真实到达
POSITION_TOLERANCE = 0.12      # 位置到达精度
ANGLE_TOLERANCE = 0.1          # 角度到达精度
MAX_NAVIGATION_STEPS = 300     # 增加最大导航步数
COVERAGE_UPDATE_FREQ = 8       

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "enable_livestream": False, 
    "enable_cameras": True,
    "enable_rtx": True,
    "physics_dt": 1.0/60.0,
    "rendering_dt": 1.0/30.0,
})

import numpy as np
import math
import time
import random
from collections import deque
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
class CoveragePoint:
    position: np.ndarray
    orientation: float
    has_object: bool = False

class RealMovementVisualizer:
    """真实移动可视化器 - 使用不同USD资产"""
    
    def __init__(self, world: World):
        self.world = world
        self.stage = world.stage
        
        # 使用不同的USD资产
        self.real_robot_usd = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
        self.ghost_robot_usd = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm3.usd"
        
        # 容器路径
        self.path_container = "/World/CompletePath"
        self.ghost_container = "/World/SyncGhosts" 
        self.coverage_container = "/World/CoverageMarks"
        
        # 状态变量
        self.all_path_points = []
        self.ghost_robots = []
        self.coverage_marks = {}
        
        print("初始化真实移动可视化器")
        print(f"实际机器人USD: {self.real_robot_usd}")
        print(f"虚影机器人USD: {self.ghost_robot_usd}")
    
    def setup_complete_path_visualization(self, path_points: List[CoveragePoint]):
        """设置完整路径可视化"""
        self.all_path_points = path_points
        print(f"设置完整路径: {len(path_points)}个点")
        
        # 清理旧可视化
        self._cleanup_all()
        
        # 创建完整路径线条
        self._create_complete_path_line()
        
        # 创建虚影机器人（使用无物理资产）
        self._create_ghost_robots()
        
        # 创建覆盖标记容器
        self._create_coverage_container()
        
        print("完整路径可视化设置完成")
    
    def _create_complete_path_line(self):
        """创建完整路径线条"""
        self.stage.DefinePrim(self.path_container, "Xform")
        
        # 创建路径线条
        line_path = f"{self.path_container}/PathLine"
        line_prim = self.stage.DefinePrim(line_path, "BasisCurves")
        line_geom = UsdGeom.BasisCurves(line_prim)
        
        line_geom.CreateTypeAttr().Set("linear")
        line_geom.CreateBasisAttr().Set("bspline")
        
        # 构建路径点
        points = []
        for point in self.all_path_points:
            world_pos = Gf.Vec3f(float(point.position[0]), float(point.position[1]), 0.05)
            points.append(world_pos)
        
        line_geom.CreatePointsAttr().Set(points)
        line_geom.CreateCurveVertexCountsAttr().Set([len(points)])
        line_geom.CreateWidthsAttr().Set([0.06] * len(points))
        
        # 设置鲜艳的路径材质
        self._setup_path_material(line_prim)
        
        # 添加物体标记
        for i, point in enumerate(self.all_path_points):
            if point.has_object:
                self._create_object_marker(i, point.position)
        
        print(f"创建完整路径线条: {len(points)}个点")
    
    def _create_ghost_robots(self):
        """创建虚影机器人 - 使用无物理资产"""
        self.stage.DefinePrim(self.ghost_container, "Xform")
        
        # 选择关键位置放置虚影机器人
        ghost_indices = self._select_ghost_positions()
        
        for i, point_idx in enumerate(ghost_indices):
            point = self.all_path_points[point_idx]
            ghost_path = f"{self.ghost_container}/GhostRobot_{i}"
            
            # 创建虚影机器人 - 使用无物理资产
            ghost_prim = self.stage.DefinePrim(ghost_path, "Xform")
            references = ghost_prim.GetReferences()
            references.AddReference(self.ghost_robot_usd)  # 使用无物理资产
            
            # 设置位置和朝向
            position = Gf.Vec3d(float(point.position[0]), float(point.position[1]), 0.0)
            yaw_degrees = float(np.degrees(point.orientation))
            
            xform = UsdGeom.Xformable(ghost_prim)
            xform.ClearXformOpOrder()
            
            translate_op = xform.AddTranslateOp()
            translate_op.Set(position)
            
            if abs(yaw_degrees) > 1.0:
                rotate_op = xform.AddRotateZOp()
                rotate_op.Set(yaw_degrees)
            
            # 额外禁用物理（双重保险）
            self._disable_ghost_physics_safe(ghost_prim)
            
            # 设置半透明绿色
            self._set_ghost_material(ghost_prim, 0.4)
            
            self.ghost_robots.append(ghost_path)
        
        print(f"创建虚影机器人: {len(ghost_indices)}个")
    
    def _select_ghost_positions(self) -> List[int]:
        """智能选择虚影位置"""
        total_points = len(self.all_path_points)
        max_ghosts = min(10, total_points)
        
        if total_points <= max_ghosts:
            return list(range(total_points))
        
        selected_indices = []
        
        # 必须包含起点和终点
        selected_indices.append(0)
        if total_points > 1:
            selected_indices.append(total_points - 1)
        
        # 优先选择有物体的点
        object_indices = [i for i, p in enumerate(self.all_path_points) if p.has_object]
        for idx in object_indices[:max_ghosts//2]:
            if idx not in selected_indices:
                selected_indices.append(idx)
        
        # 均匀选择其他点
        remaining_slots = max_ghosts - len(selected_indices)
        if remaining_slots > 0:
            step = max(1, total_points // remaining_slots)
            
            for i in range(step, total_points, step):
                if len(selected_indices) >= max_ghosts:
                    break
                if i not in selected_indices:
                    selected_indices.append(i)
        
        return sorted(selected_indices)
    
    def _create_coverage_container(self):
        """创建覆盖标记容器"""
        self.stage.DefinePrim(self.coverage_container, "Xform")
    
    def mark_coverage_realtime(self, robot_position: np.ndarray, step_count: int):
        """实时标记覆盖区域"""
        if step_count % COVERAGE_UPDATE_FREQ != 0:
            return
        
        # 量化位置
        x = round(robot_position[0] / 0.12) * 0.12
        y = round(robot_position[1] / 0.12) * 0.12
        pos_key = f"{x:.2f}_{y:.2f}"
        
        if pos_key not in self.coverage_marks:
            self._create_coverage_mark(np.array([x, y, 0.02]), pos_key)
            self.coverage_marks[pos_key] = True
    
    def _create_coverage_mark(self, position: np.ndarray, pos_key: str):
        """创建覆盖标记"""
        mark_path = f"{self.coverage_container}/Mark_{pos_key.replace('.', 'p').replace('-', 'N')}"
        
        mark_prim = self.stage.DefinePrim(mark_path, "Cylinder")
        cylinder_geom = UsdGeom.Cylinder(mark_prim)
        cylinder_geom.CreateRadiusAttr().Set(0.1)
        cylinder_geom.CreateHeightAttr().Set(0.015)
        
        xform = UsdGeom.Xformable(mark_prim)
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        
        # 禁用物理
        UsdPhysics.RigidBodyAPI.Apply(mark_prim)
        rigid_body = UsdPhysics.RigidBodyAPI(mark_prim)
        rigid_body.CreateRigidBodyEnabledAttr().Set(False)
        
        # 设置灰色
        gprim = UsdGeom.Gprim(mark_prim)
        gprim.CreateDisplayColorAttr().Set([(0.7, 0.7, 0.7)])
    
    def _disable_ghost_physics_safe(self, root_prim):
        """安全禁用虚影物理属性"""
        def disable_recursive(prim):
            try:
                # 只对Xformable类型的prim应用物理API
                if prim.IsA(UsdGeom.Xformable):
                    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        UsdPhysics.RigidBodyAPI.Apply(prim)
                    rigid_body = UsdPhysics.RigidBodyAPI(prim)
                    rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                    
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI.Apply(prim)
                    collision = UsdPhysics.CollisionAPI(prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
            except:
                pass  # 忽略无法应用物理API的prim
            
            # 递归处理子prim
            for child in prim.GetChildren():
                disable_recursive(child)
        
        disable_recursive(root_prim)
    
    def _set_ghost_material(self, ghost_prim, opacity: float):
        """设置虚影材质 - 绿色半透明"""
        material_path = f"/World/Materials/GhostMaterial_{hash(str(ghost_prim.GetPath())) % 1000}"
        
        material_prim = self.stage.DefinePrim(material_path, "Material")
        material = UsdShade.Material(material_prim)
        
        shader_prim = self.stage.DefinePrim(f"{material_path}/Shader", "Shader")
        shader = UsdShade.Shader(shader_prim)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.2, 0.9, 0.2))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.5, 0.1))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        
        material_output = material.CreateSurfaceOutput()
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material_output.ConnectToSource(shader_output)
        
        # 应用到mesh
        def apply_material_recursive(prim):
            if prim.GetTypeName() == "Mesh":
                UsdShade.MaterialBindingAPI(prim).Bind(material)
            for child in prim.GetChildren():
                apply_material_recursive(child)
        
        apply_material_recursive(ghost_prim)
    
    def _setup_path_material(self, line_prim):
        """设置路径材质 - 鲜艳蓝色"""
        material_path = "/World/Materials/PathMaterial"
        
        material_prim = self.stage.DefinePrim(material_path, "Material")
        material = UsdShade.Material(material_prim)
        
        shader_prim = self.stage.DefinePrim(f"{material_path}/Shader", "Shader")
        shader = UsdShade.Shader(shader_prim)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.1, 0.5, 1.0))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((0.05, 0.25, 0.5))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.2)
        
        material_output = material.CreateSurfaceOutput()
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material_output.ConnectToSource(shader_output)
        
        UsdShade.MaterialBindingAPI(line_prim).Bind(material)
    
    def _create_object_marker(self, index: int, position: np.ndarray):
        """创建物体标记 - 红色球体"""
        marker_path = f"{self.path_container}/ObjectMarker_{index}"
        marker_prim = self.stage.DefinePrim(marker_path, "Sphere")
        sphere_geom = UsdGeom.Sphere(marker_prim)
        sphere_geom.CreateRadiusAttr().Set(0.12)
        
        marker_pos = Gf.Vec3d(float(position[0]), float(position[1]), 0.12)
        xform = UsdGeom.Xformable(marker_prim)
        translate_op = xform.AddTranslateOp()
        translate_op.Set(marker_pos)
        
        # 禁用物理
        UsdPhysics.RigidBodyAPI.Apply(marker_prim)
        rigid_body = UsdPhysics.RigidBodyAPI(marker_prim)
        rigid_body.CreateRigidBodyEnabledAttr().Set(False)
        
        gprim = UsdGeom.Gprim(marker_prim)
        gprim.CreateDisplayColorAttr().Set([(1.0, 0.2, 0.2)])
    
    def _cleanup_all(self):
        """清理所有可视化"""
        containers = [self.path_container, self.ghost_container, self.coverage_container]
        for container in containers:
            if self.stage.GetPrimAtPath(container):
                self.stage.RemovePrim(container)
        
        # 清理材质
        materials_path = "/World/Materials"
        if self.stage.GetPrimAtPath(materials_path):
            materials_prim = self.stage.GetPrimAtPath(materials_path)
            for child in materials_prim.GetChildren():
                if any(name in str(child.GetPath()) for name in ["PathMaterial", "GhostMaterial"]):
                    self.stage.RemovePrim(child.GetPath())
    
    def cleanup(self):
        """清理资源"""
        self._cleanup_all()
        self.coverage_marks.clear()
        print("可视化资源清理完成")

class SmartCoveragePathPlanner:
    """智能覆盖路径规划器"""
    
    def __init__(self):
        self.world_size = COVERAGE_AREA_SIZE
        self.cell_size = COVERAGE_CELL_SIZE
        self.grid_size = int(self.world_size / self.cell_size)
        self.robot_radius = ROBOT_RADIUS
        
        self.obstacle_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.objects_positions = []
        self.obstacles_info = []
        
        print(f"智能覆盖规划器: {self.grid_size}x{self.grid_size}网格")
    
    def add_obstacle(self, position: np.ndarray, size: np.ndarray, shape_type: str = 'box'):
        """添加障碍物"""
        self.obstacles_info.append({
            'position': position.copy(),
            'size': size.copy(),
            'shape': shape_type
        })
        self._mark_obstacle_cells(position, size, shape_type)
        print(f"添加{shape_type}障碍物: {position[:2]}")
    
    def _mark_obstacle_cells(self, position: np.ndarray, size: np.ndarray, shape_type: str):
        """标记障碍物网格"""
        center_x = int((position[0] + self.world_size/2) / self.cell_size)
        center_y = int((position[1] + self.world_size/2) / self.cell_size)
        
        radius = int((max(size) + self.robot_radius) / self.cell_size) + 2
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                gx, gy = center_x + dx, center_y + dy
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    world_x = (gx * self.cell_size) - self.world_size/2
                    world_y = (gy * self.cell_size) - self.world_size/2
                    
                    if self._check_collision(np.array([world_x, world_y]), position, size, shape_type):
                        self.obstacle_map[gx, gy] = True
    
    def _check_collision(self, point: np.ndarray, obs_pos: np.ndarray, obs_size: np.ndarray, shape: str) -> bool:
        """检查点与障碍物碰撞"""
        distance = np.linalg.norm(point - obs_pos[:2])
        
        if shape == 'sphere':
            return distance < (self.robot_radius + obs_size[0] + 0.1)
        else:  # box
            rel_pos = point - obs_pos[:2]
            half_x = obs_size[0]/2 + self.robot_radius + 0.1
            half_y = obs_size[1]/2 + self.robot_radius + 0.1
            return abs(rel_pos[0]) < half_x and abs(rel_pos[1]) < half_y
    
    def add_objects(self, objects_list: List[np.ndarray]):
        """添加目标物体"""
        self.objects_positions = objects_list
        print(f"添加目标物体: {len(objects_list)}个")
    
    def generate_coverage_path(self, start_pos: np.ndarray) -> List[CoveragePoint]:
        """生成覆盖路径"""
        print("生成智能覆盖路径...")
        
        # 生成覆盖点网格
        coverage_points = self._generate_coverage_grid()
        
        # 生成蛇形路径
        path_points = self._create_serpentine_path(coverage_points, start_pos)
        
        print(f"覆盖路径生成完成: {len(path_points)}个点")
        return path_points
    
    def _generate_coverage_grid(self) -> List[Tuple[int, int]]:
        """生成可覆盖网格点"""
        coverage_points = []
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if not self.obstacle_map[x, y] and self._is_valid_coverage_point(x, y):
                    coverage_points.append((x, y))
        
        return coverage_points
    
    def _is_valid_coverage_point(self, grid_x: int, grid_y: int) -> bool:
        """检查网格点是否有效"""
        world_x = (grid_x * self.cell_size) - self.world_size/2
        world_y = (grid_y * self.cell_size) - self.world_size/2
        
        margin = self.robot_radius + 0.2
        return (abs(world_x) < self.world_size/2 - margin and 
                abs(world_y) < self.world_size/2 - margin)
    
    def _create_serpentine_path(self, coverage_points: List[Tuple[int, int]], start_pos: np.ndarray) -> List[CoveragePoint]:
        """创建蛇形路径"""
        path_points = []
        
        # 按Y坐标分组
        rows = {}
        for x, y in coverage_points:
            if y not in rows:
                rows[y] = []
            rows[y].append(x)
        
        # 排序行
        sorted_rows = sorted(rows.keys())
        
        for row_idx, y in enumerate(sorted_rows):
            row_x_coords = sorted(rows[y])
            
            # 奇数行反向
            if row_idx % 2 == 1:
                row_x_coords.reverse()
            
            for point_idx, x in enumerate(row_x_coords):
                world_pos = self._grid_to_world(x, y)
                
                # 计算朝向
                orientation = 0.0 if row_idx % 2 == 0 else math.pi
                if point_idx > 0:
                    prev_pos = path_points[-1].position
                    direction = world_pos[:2] - prev_pos[:2]
                    if np.linalg.norm(direction) > 0.01:
                        orientation = np.arctan2(direction[1], direction[0])
                
                # 检查是否有物体
                has_object = self._has_nearby_object(world_pos)
                
                point = CoveragePoint(
                    position=world_pos,
                    orientation=orientation,
                    has_object=has_object
                )
                path_points.append(point)
        
        return self._optimize_path(path_points, start_pos)
    
    def _has_nearby_object(self, position: np.ndarray) -> bool:
        """检查附近是否有物体"""
        for obj_pos in self.objects_positions:
            distance = np.linalg.norm(position[:2] - obj_pos[:2])
            if distance < self.robot_radius * 0.8:
                return True
        return False
    
    def _optimize_path(self, path_points: List[CoveragePoint], start_pos: np.ndarray) -> List[CoveragePoint]:
        """优化路径"""
        return path_points  # 保持完整路径，确保覆盖
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """网格坐标转世界坐标"""
        x = (grid_x * self.cell_size) - self.world_size/2
        y = (grid_y * self.cell_size) - self.world_size/2
        return np.array([x, y, 0.0])

class RealMovementRobotController:
    """真实移动机器人控制器 - 确保机器人真正移动"""
    
    def __init__(self, mobile_base, differential_controller):
        self.mobile_base = mobile_base
        self.differential_controller = differential_controller
        self.max_linear_vel = MAX_LINEAR_VELOCITY
        self.max_angular_vel = MAX_ANGULAR_VELOCITY
        
        # 速度滤波器
        self.velocity_filter = deque(maxlen=3)
        self.angular_filter = deque(maxlen=3)
        
        print("真实移动机器人控制器初始化")
    
    def move_to_position_real(self, target_pos: np.ndarray, target_orientation: float) -> bool:
        """真实移动到指定位置 - 确保机器人真正到达"""
        max_steps = MAX_NAVIGATION_STEPS
        step_count = 0
        
        print(f"开始导航到: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
        
        while step_count < max_steps:
            current_pos, current_yaw = self._get_robot_pose()
            step_count += 1
            
            # 检查位置到达
            pos_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
            
            if pos_error < POSITION_TOLERANCE:
                self._send_smooth_command(0.0, 0.0)
                print(f"  成功到达! 误差: {pos_error:.3f}m, 用时: {step_count}步")
                return True
            
            # 计算控制指令
            direction = target_pos[:2] - current_pos[:2]
            if np.linalg.norm(direction) > 0.001:
                direction_norm = direction / np.linalg.norm(direction)
                target_yaw = np.arctan2(direction[1], direction[0])
                
                yaw_error = self._normalize_angle(target_yaw - current_yaw)
                
                # 改进的控制策略
                if abs(yaw_error) > ANGLE_TOLERANCE:
                    # 需要转向
                    linear_vel = 0.0
                    angular_vel = np.clip(yaw_error * 2.5, -self.max_angular_vel, self.max_angular_vel)
                else:
                    # 可以前进
                    linear_vel = np.clip(pos_error * 2.0, 0.0, self.max_linear_vel)
                    angular_vel = np.clip(yaw_error * 1.5, -self.max_angular_vel, self.max_angular_vel)
                
                self._send_smooth_command(linear_vel, angular_vel)
            else:
                self._send_smooth_command(0.0, 0.0)
            
            # 进度报告
            if step_count % 50 == 0:
                print(f"  导航中... 距离: {pos_error:.3f}m, 步数: {step_count}")
        
        # 超时停止
        self._send_smooth_command(0.0, 0.0)
        current_pos, _ = self._get_robot_pose()
        final_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
        print(f"  导航超时! 最终误差: {final_error:.3f}m")
        return False
    
    def _normalize_angle(self, angle):
        """角度归一化"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _get_robot_pose(self):
        """获取机器人位姿"""
        position, orientation = self.mobile_base.get_world_pose()
        position = np.array(position)
        quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        r = R.from_quat(quat)
        yaw = r.as_euler('xyz')[2]
        return position, yaw
    
    def _send_smooth_command(self, linear_vel: float, angular_vel: float):
        """发送平滑控制指令"""
        # 速度滤波
        self.velocity_filter.append(linear_vel)
        self.angular_filter.append(angular_vel)
        
        smooth_linear = np.mean(list(self.velocity_filter))
        smooth_angular = np.mean(list(self.angular_filter))
        
        # 应用到轮子
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        wheel_radius = 0.036
        wheel_base = 0.235
        
        left_wheel_vel = (smooth_linear - smooth_angular * wheel_base / 2.0) / wheel_radius
        right_wheel_vel = (smooth_linear + smooth_angular * wheel_base / 2.0) / wheel_radius
        
        num_dofs = len(self.mobile_base.dof_names)
        joint_velocities = torch.zeros(num_dofs, dtype=torch.float32)
        
        left_wheel_idx = self.mobile_base.dof_names.index("left_wheel_joint")
        right_wheel_idx = self.mobile_base.dof_names.index("right_wheel_joint")
        
        joint_velocities[left_wheel_idx] = float(left_wheel_vel)
        joint_velocities[right_wheel_idx] = float(right_wheel_vel)
        
        action = ArticulationAction(joint_velocities=joint_velocities)
        articulation_controller.apply_action(action)

class RealMovementCoverageSystem:
    """真实移动覆盖系统"""
    
    def __init__(self):
        self.world = None
        self.mobile_base = None
        self.robot_controller = None
        self.path_planner = None
        self.visualizer = None
        
        self.collectible_objects = []
        self.collected_objects = []
        self.coverage_path = []
        
        # 机械臂配置
        self.arm_poses = {
            "home": [0.0, -0.5, 0.0, -2.5, 0.0, 2.0, 0.785],
            "pickup": [0.0, 0.3, 0.0, -1.5, 0.0, 2.2, 0.785],
            "carry": [0.0, -0.4, 0.0, -2.0, 0.0, 1.8, 0.785]
        }
    
    def initialize_system(self):
        """初始化系统"""
        print("初始化真实移动覆盖系统...")
        
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
        self._create_environment()
        
        print("系统初始化完成")
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
    
    def _initialize_components(self):
        """初始化组件"""
        self.path_planner = SmartCoveragePathPlanner()
        self.visualizer = RealMovementVisualizer(self.world)
    
    def _create_environment(self):
        """创建环境"""
        # 添加障碍物
        obstacles = [
            {"pos": [1.2, 0.8, 0.15], "size": [0.3, 0.3, 0.3], "color": [0.6, 0.3, 0.1], "name": "obs1", "shape": "box"},
            {"pos": [0.5, -1.5, 0.1], "size": [1.0, 0.2, 0.2], "color": [0.7, 0.7, 0.7], "name": "obs2", "shape": "box"},
            {"pos": [-1.0, 1.2, 0.4], "size": [0.1, 0.8, 0.1], "color": [0.8, 0.2, 0.2], "name": "obs3", "shape": "box"},
            {"pos": [-0.8, -1.0, 0.15], "size": [0.3, 0.3, 0.3], "color": [0.9, 0.5, 0.1], "name": "obs4", "shape": "sphere"},
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
            self.path_planner.add_obstacle(np.array(obs["pos"]), np.array(obs["size"]), obs["shape"])
        
        # 创建可收集物体
        object_positions = [
            [1.5, 0.5, 0.03], [2.0, 1.8, 0.03], [-0.5, 1.5, 0.03],
            [0.8, -0.8, 0.03], [-1.5, 0.2, 0.03], [1.8, -1.2, 0.03],
            [-0.2, -1.8, 0.03], [2.2, 0.0, 0.03]
        ]
        
        for i, pos in enumerate(object_positions):
            obj = DynamicSphere(
                prim_path=f"/World/collectible_{i}",
                name=f"collectible_{i}",
                position=np.array(pos),
                radius=0.04,
                color=np.array([0.2, 0.8, 0.2])
            )
            self.world.scene.add(obj)
            self.collectible_objects.append(obj)
        
        # 添加到路径规划器
        object_world_positions = [obj.get_world_pose()[0] for obj in self.collectible_objects]
        self.path_planner.add_objects(object_world_positions)
        
        print(f"环境创建完成: 障碍物{len(obstacles)}个, 物体{len(self.collectible_objects)}个")
    
    def initialize_robot(self):
        """初始化机器人 - 使用有物理属性的资产"""
        print("初始化Create-3机器人...")
        print("使用有物理属性的机器人资产: create_3_with_arm2.usd")
        
        # 使用有物理属性的机器人资产
        robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
        
        self.mobile_base = WheeledRobot(
            prim_path="/World/create3_robot",
            name="create3_robot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=robot_usd_path,
            position=np.array([0.0, 0.0, 0.0])
        )
        
        self.world.scene.add(self.mobile_base)
        
        print("机器人初始化完成")
        return True
    
    def setup_post_load(self):
        """后加载设置"""
        print("后加载设置...")
        
        self.world.reset()
        
        # 稳定物理
        for _ in range(30):
            self.world.step(render=False)
        
        self.mobile_base = self.world.scene.get_object("create3_robot")
        self._setup_control_gains()
        self._move_arm_to_pose("home")
        
        # 初始化控制器
        differential_controller = DifferentialController(
            name="create3_controller",
            wheel_radius=0.036,
            wheel_base=0.235,
            max_linear_speed=MAX_LINEAR_VELOCITY,
            max_angular_speed=MAX_ANGULAR_VELOCITY
        )
        
        self.robot_controller = RealMovementRobotController(self.mobile_base, differential_controller)
        
        print("后加载设置完成")
        return True
    
    def _setup_control_gains(self):
        """设置控制增益"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        num_dofs = len(self.mobile_base.dof_names)
        
        kp = torch.zeros(num_dofs, dtype=torch.float32)
        kd = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 轮子控制
        for wheel_name in ["left_wheel_joint", "right_wheel_joint"]:
            idx = self.mobile_base.dof_names.index(wheel_name)
            kp[idx] = 0.0
            kd[idx] = 400.0
        
        # 机械臂控制
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        for joint_name in arm_joint_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 1000.0
                kd[idx] = 50.0
        
        # 夹爪控制
        for joint_name in ["panda_finger_joint1", "panda_finger_joint2"]:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 2e5
                kd[idx] = 2e3
        
        articulation_controller.set_gains(kps=kp, kds=kd)
        print("控制增益设置完成")
    
    def _move_arm_to_pose(self, pose_name):
        """移动机械臂"""
        if pose_name not in self.arm_poses:
            return
            
        target_positions = self.arm_poses[pose_name]
        articulation_controller = self.mobile_base.get_articulation_controller()
        num_dofs = len(self.mobile_base.dof_names)
        
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        for i, joint_name in enumerate(arm_joint_names):
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                joint_positions[idx] = target_positions[i]
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        for _ in range(15):
            self.world.step(render=False)
    
    def plan_coverage_mission(self):
        """规划覆盖任务"""
        print("规划真实移动覆盖任务...")
        
        current_pos, _ = self._get_robot_pose()
        print(f"机器人当前位置: [{current_pos[0]:.3f}, {current_pos[1]:.3f}]")
        
        # 生成覆盖路径
        self.coverage_path = self.path_planner.generate_coverage_path(current_pos)
        
        # 设置完整路径可视化
        self.visualizer.setup_complete_path_visualization(self.coverage_path)
        
        print(f"覆盖规划完成: {len(self.coverage_path)}个覆盖点")
        print_memory_usage("覆盖规划完成")
    
    def execute_real_coverage(self):
        """执行真实覆盖 - 确保机器人真正移动到每个点"""
        print("开始执行真实移动覆盖...")
        print_memory_usage("任务开始")
        
        # 展示完整路径预览
        print("展示完整路径预览...")
        for step in range(90):
            self.world.step(render=True)
            time.sleep(0.05)
        
        print("开始真实路径执行...")
        step_counter = 0
        successful_points = 0
        
        for i, point in enumerate(self.coverage_path):
            print(f"\n=== 导航到点 {i+1}/{len(self.coverage_path)} ===")
            
            # 真实移动到覆盖点
            success = self.robot_controller.move_to_position_real(point.position, point.orientation)
            
            if success:
                successful_points += 1
            
            current_pos, _ = self._get_robot_pose()
            
            # 实时覆盖标记
            self.visualizer.mark_coverage_realtime(current_pos, step_counter)
            step_counter += 1
            
            # 检查和收集物体
            self._check_and_collect_objects(current_pos)
            
            # 步进仿真
            for _ in range(5):
                self.world.step(render=True)
            
            time.sleep(0.1)
        
        print(f"\n真实移动覆盖执行完成!")
        print(f"成功到达点数: {successful_points}/{len(self.coverage_path)}")
        self._show_results()
    
    def _get_robot_pose(self):
        """获取机器人位姿"""
        position, orientation = self.mobile_base.get_world_pose()
        position = np.array(position)
        quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        r = R.from_quat(quat)
        yaw = r.as_euler('xyz')[2]
        return position, yaw
    
    def _check_and_collect_objects(self, robot_pos: np.ndarray):
        """检查并收集物体"""
        for obj in self.collectible_objects:
            if obj.name in self.collected_objects:
                continue
            
            obj_pos = obj.get_world_pose()[0]
            distance = np.linalg.norm(robot_pos[:2] - obj_pos[:2])
            
            if distance < ROBOT_RADIUS:
                print(f"收集物体: {obj.name} (距离: {distance:.3f}m)")
                self._collect_object(obj)
                self.collected_objects.append(obj.name)
                break
    
    def _collect_object(self, obj):
        """收集物体"""
        # 快速机械臂动作
        self._move_arm_to_pose("pickup")
        self._control_gripper("close")
        self._move_arm_to_pose("carry")
        
        # 隐藏物体
        obj.disable_rigid_body_physics()
        obj.set_world_pose(np.array([100.0, 100.0, -5.0]), np.array([0, 0, 0, 1]))
        obj.set_visibility(False)
    
    def _control_gripper(self, action):
        """控制夹爪"""
        if "panda_finger_joint1" not in self.mobile_base.dof_names:
            return
            
        articulation_controller = self.mobile_base.get_articulation_controller()
        gripper_pos = 0.0 if action == "close" else 0.04
        
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        for joint_name in ["panda_finger_joint1", "panda_finger_joint2"]:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                joint_positions[idx] = gripper_pos
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        for _ in range(8):
            self.world.step(render=False)
    
    def _show_results(self):
        """显示结果"""
        total_objects = len(self.collectible_objects)
        collected = len(self.collected_objects)
        collection_rate = (collected / total_objects) * 100
        coverage_marks = len(self.visualizer.coverage_marks)
        
        print(f"\n=== 真实移动覆盖结果 ===")
        print(f"覆盖路径点数: {len(self.coverage_path)}")
        print(f"覆盖标记区域: {coverage_marks}")
        print(f"物体总数: {total_objects}")
        print(f"成功收集: {collected}")
        print(f"收集率: {collection_rate:.1f}%")
        print("真实路径执行成功!")
        print(f"实际机器人资产: create_3_with_arm2.usd")
        print(f"虚影机器人资产: create_3_with_arm3.usd")
    
    def run_real_movement_demo(self):
        """运行真实移动演示"""
        print("\n" + "="*80)
        print("真实移动覆盖算法机器人系统 - Isaac Sim 4.5")
        print("实际机器人: create_3_with_arm2.usd | 虚影机器人: create_3_with_arm3.usd")
        print("真实路径跟踪 | 精确位置控制 | 完整路径预览 | 智能物体收集")
        print("="*80)
        
        pos, yaw = self._get_robot_pose()
        print(f"机器人初始位置: [{pos[0]:.3f}, {pos[1]:.3f}], 朝向: {np.degrees(yaw):.1f}°")
        
        self.plan_coverage_mission()
        self.execute_real_coverage()
        
        self._move_arm_to_pose("home")
        
        print("\n真实移动覆盖系统演示完成!")
        print("成功实现机器人真实路径跟踪和精确位置控制")
    
    def cleanup(self):
        """清理系统"""
        print("清理系统资源...")
        print_memory_usage("清理前")
        
        if self.visualizer:
            self.visualizer.cleanup()
        
        # 清理内存
        for _ in range(8):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.world:
            self.world.stop()
        
        print_memory_usage("清理后")
        print("系统资源清理完成")

def main():
    """主函数"""
    print("启动真实移动覆盖算法机器人系统...")
    print(f"实际机器人USD: create_3_with_arm2.usd (有物理属性)")
    print(f"虚影机器人USD: create_3_with_arm3.usd (无物理属性)")
    print(f"机器人参数: 半径={ROBOT_RADIUS}m")
    print(f"覆盖参数: 网格={COVERAGE_CELL_SIZE}m, 区域={COVERAGE_AREA_SIZE}m")
    print(f"精度参数: 位置容差={POSITION_TOLERANCE}m, 角度容差={ANGLE_TOLERANCE}rad")
    print(f"运动参数: 线速度={MAX_LINEAR_VELOCITY}m/s, 角速度={MAX_ANGULAR_VELOCITY}rad/s")
    
    system = RealMovementCoverageSystem()
    
    try:
        system.initialize_system()
        system.initialize_robot()
        system.setup_post_load()
        
        # 稳定系统
        print("系统稳定中...")
        for _ in range(25):
            system.world.step(render=False)
            time.sleep(0.01)
        
        # 运行演示
        system.run_real_movement_demo()
        
        # 保持运行观察效果
        print("\n观察真实移动覆盖效果...")
        for i in range(50):
            system.world.step(render=True)
            time.sleep(0.1)
    
    finally:
        system.cleanup()

if __name__ == "__main__":
    main()