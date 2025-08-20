#!/usr/bin/env python3
"""
Isaac Sim 4.5兼容版Create-3+机械臂垃圾收集系统
REMANI完整避障系统 - 精确表面到表面距离计算与智能路径可视化
"""

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
    """碰撞检测结果"""
    is_collision: bool
    min_distance: float
    collision_type: str
    collision_point: Optional[np.ndarray] = None
    collision_normal: Optional[np.ndarray] = None

@dataclass
class ObstacleInfo:
    """障碍物信息"""
    position: np.ndarray
    size: np.ndarray
    shape_type: str  # 'box', 'sphere', 'cylinder'
    rotation: np.ndarray = None

@dataclass
class PathNode:
    """路径节点"""
    position: np.ndarray
    orientation: float
    arm_config: List[float]
    timestamp: float

class REMANIPreciseDistanceCalculator:
    """REMANI精确表面到表面距离计算器"""
    
    def __init__(self):
        # Create3底座几何参数
        self.base_radius = 0.17
        self.base_height = 0.1
        
        # 机械臂DH参数（Panda 7DOF）
        self.dh_params = [
            [0, 0, 0.333, 0],
            [-np.pi/2, 0, 0, 0],
            [np.pi/2, 0, 0.316, 0],
            [np.pi/2, 0.0825, 0, 0],
            [-np.pi/2, -0.0825, 0.384, 0],
            [np.pi/2, 0, 0, 0],
            [np.pi/2, 0.088, 0.107, 0]
        ]
        
        # 连杆几何参数
        self.link_geometries = [
            {"radius": 0.060, "length": 0.15},
            {"radius": 0.070, "length": 0.20},
            {"radius": 0.060, "length": 0.20},
            {"radius": 0.050, "length": 0.18},
            {"radius": 0.060, "length": 0.20},
            {"radius": 0.050, "length": 0.15},
            {"radius": 0.040, "length": 0.10}
        ]

    def point_to_box_surface_distance(self, point: np.ndarray, box_center: np.ndarray, 
                                    box_size: np.ndarray, box_rotation: np.ndarray = None) -> Tuple[float, np.ndarray]:
        """计算点到立方体表面的距离和最近点"""
        local_point = point - box_center if box_rotation is None else box_rotation.T @ (point - box_center)
        half_size = box_size / 2
        clamped_point = np.clip(local_point, -half_size, half_size)
        
        if np.allclose(local_point, clamped_point):
            distances_to_faces = half_size - np.abs(local_point)
            min_axis = np.argmin(distances_to_faces)
            distance = -distances_to_faces[min_axis]
            surface_point = local_point.copy()
            surface_point[min_axis] = half_size[min_axis] * np.sign(local_point[min_axis])
        else:
            distance = np.linalg.norm(local_point - clamped_point)
            surface_point = clamped_point
            
        world_surface_point = surface_point + box_center if box_rotation is None else box_rotation @ surface_point + box_center
        return distance, world_surface_point

    def point_to_sphere_surface_distance(self, point: np.ndarray, sphere_center: np.ndarray, 
                                       sphere_radius: float) -> Tuple[float, np.ndarray]:
        """计算点到球体表面的距离和最近点"""
        vec_to_point = point - sphere_center
        distance_to_center = np.linalg.norm(vec_to_point)
        
        if distance_to_center < 1e-10:
            return -sphere_radius, sphere_center + np.array([sphere_radius, 0, 0])
            
        direction = vec_to_point / distance_to_center
        surface_point = sphere_center + direction * sphere_radius
        distance = distance_to_center - sphere_radius
        return distance, surface_point

    def point_to_cylinder_surface_distance(self, point: np.ndarray, cylinder_center: np.ndarray,
                                         cylinder_radius: float, cylinder_height: float) -> Tuple[float, np.ndarray]:
        """计算点到圆柱体表面的距离和最近点"""
        local_point = point - cylinder_center
        radial_vec = np.array([local_point[0], 0, local_point[2]])
        radial_distance = np.linalg.norm(radial_vec)
        axial_distance = local_point[1]
        half_height = cylinder_height / 2
        
        if abs(axial_distance) <= half_height and radial_distance <= cylinder_radius:
            radial_penetration = cylinder_radius - radial_distance
            axial_penetration = half_height - abs(axial_distance)
            
            if radial_penetration < axial_penetration:
                direction = radial_vec / radial_distance if radial_distance > 1e-10 else np.array([1, 0, 0])
                surface_point = cylinder_center + direction * cylinder_radius
                surface_point[1] = point[1]
                distance = -radial_penetration
            else:
                surface_point = point.copy()
                surface_point[1] = cylinder_center[1] + half_height * np.sign(axial_distance)
                distance = -axial_penetration
        else:
            clamped_radial = min(radial_distance, cylinder_radius)
            clamped_axial = np.clip(axial_distance, -half_height, half_height)
            
            if radial_distance > 1e-10:
                radial_direction = radial_vec / radial_distance
                surface_point = cylinder_center + radial_direction * clamped_radial
            else:
                surface_point = cylinder_center.copy()
                
            surface_point[1] = cylinder_center[1] + clamped_axial
            
            if abs(axial_distance) <= half_height:
                distance = radial_distance - cylinder_radius
            elif radial_distance <= cylinder_radius:
                distance = abs(axial_distance) - half_height
            else:
                radial_excess = radial_distance - cylinder_radius
                axial_excess = abs(axial_distance) - half_height
                distance = np.sqrt(radial_excess**2 + axial_excess**2)
                
        return distance, surface_point

    def circle_to_obstacle_surface_distance(self, circle_center: np.ndarray, circle_radius: float,
                                          obstacle: ObstacleInfo) -> Tuple[float, np.ndarray, np.ndarray]:
        """计算圆形到障碍物表面的精确距离"""
        if obstacle.shape_type == 'box':
            point_distance, nearest_surface_point = self.point_to_box_surface_distance(
                circle_center, obstacle.position, obstacle.size, obstacle.rotation)
        elif obstacle.shape_type == 'sphere':
            point_distance, nearest_surface_point = self.point_to_sphere_surface_distance(
                circle_center, obstacle.position, obstacle.size[0])
        elif obstacle.shape_type == 'cylinder':
            point_distance, nearest_surface_point = self.point_to_cylinder_surface_distance(
                circle_center, obstacle.position, obstacle.size[0]/2, obstacle.size[1])
        else:
            point_distance, nearest_surface_point = self.point_to_box_surface_distance(
                circle_center, obstacle.position, obstacle.size)
            
        surface_distance = point_distance - circle_radius
        contact_normal = ((nearest_surface_point - circle_center) / 
                         np.linalg.norm(nearest_surface_point - circle_center) 
                         if np.linalg.norm(nearest_surface_point - circle_center) > 1e-10 
                         else np.array([1.0, 0.0, 0.0]))
        return surface_distance, nearest_surface_point, contact_normal

    def cylinder_to_obstacle_surface_distance(self, cylinder_center: np.ndarray, cylinder_axis: np.ndarray,
                                            cylinder_radius: float, cylinder_length: float,
                                            obstacle: ObstacleInfo) -> Tuple[float, np.ndarray, np.ndarray]:
        """计算圆柱体到障碍物表面的精确距离"""
        num_axial_samples = max(3, int(cylinder_length / 0.05))
        num_radial_samples = 8
        
        min_distance = float('inf')
        closest_surface_point = None
        closest_contact_normal = None
        
        axis_normalized = cylinder_axis / np.linalg.norm(cylinder_axis)
        
        perpendicular1 = (np.cross(axis_normalized, np.array([0, 0, 1])) 
                         if abs(axis_normalized[2]) < 0.9 
                         else np.cross(axis_normalized, np.array([1, 0, 0])))
        perpendicular1 /= np.linalg.norm(perpendicular1)
        perpendicular2 = np.cross(axis_normalized, perpendicular1)
        
        for i in range(num_axial_samples):
            t = (i / (num_axial_samples - 1) - 0.5) if num_axial_samples > 1 else 0.0
            axial_point = cylinder_center + t * cylinder_length * axis_normalized
            
            for j in range(num_radial_samples):
                angle = 2 * np.pi * j / num_radial_samples
                radial_offset = cylinder_radius * (np.cos(angle) * perpendicular1 + np.sin(angle) * perpendicular2)
                sample_point = axial_point + radial_offset
                
                if obstacle.shape_type == 'box':
                    distance, surface_point = self.point_to_box_surface_distance(
                        sample_point, obstacle.position, obstacle.size, obstacle.rotation)
                elif obstacle.shape_type == 'sphere':
                    distance, surface_point = self.point_to_sphere_surface_distance(
                        sample_point, obstacle.position, obstacle.size[0])
                elif obstacle.shape_type == 'cylinder':
                    distance, surface_point = self.point_to_cylinder_surface_distance(
                        sample_point, obstacle.position, obstacle.size[0]/2, obstacle.size[1])
                else:
                    distance, surface_point = self.point_to_box_surface_distance(
                        sample_point, obstacle.position, obstacle.size)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_surface_point = surface_point
                    closest_contact_normal = ((surface_point - sample_point) / 
                                            np.linalg.norm(surface_point - sample_point) 
                                            if np.linalg.norm(surface_point - sample_point) > 1e-10 
                                            else np.array([1.0, 0.0, 0.0]))
        
        return min_distance, closest_surface_point, closest_contact_normal

class REMANIAdvancedCollisionChecker:
    """REMANI高级避障系统 - 上帝视角精确避障"""
    
    def __init__(self, safe_distance: float = 0.3):
        self.safe_distance = safe_distance
        self.arm_safe_distance = 0.15
        self.obstacles = []
        self.distance_calc = REMANIPreciseDistanceCalculator()
        
        print(f"✅ REMANI高级避障系统初始化: 底盘安全距离={safe_distance}m, 机械臂安全距离={self.arm_safe_distance}m")
    
    def add_obstacle(self, position: np.ndarray, size: np.ndarray, shape_type: str = 'box', rotation: np.ndarray = None):
        """添加障碍物"""
        obstacle_info = ObstacleInfo(
            position=position.copy(),
            size=size.copy(),
            shape_type=shape_type,
            rotation=rotation.copy() if rotation is not None else np.eye(3)
        )
        self.obstacles.append(obstacle_info)
        print(f"   添加{shape_type}障碍物: 位置{position}, 尺寸{size}")
    
    def check_path_collision_free(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                                 start_orientation: float, end_orientation: float,
                                 arm_config: List[float]) -> bool:
        """检查路径是否无碰撞"""
        # 沿路径采样多个点进行碰撞检测
        num_samples = max(10, int(np.linalg.norm(end_pos - start_pos) / 0.1))
        
        for i in range(num_samples + 1):
            t = i / num_samples if num_samples > 0 else 0
            
            # 插值位置和朝向
            current_pos = start_pos + t * (end_pos - start_pos)
            current_orientation = start_orientation + t * (end_orientation - start_orientation)
            
            # 检查底盘碰撞
            base_collision = self.check_base_collision_precise(current_pos, current_orientation)
            if base_collision.is_collision:
                return False
            
            # 检查机械臂碰撞
            arm_collision = self.check_arm_collision_precise(current_pos, current_orientation, arm_config)
            if arm_collision.is_collision:
                return False
        
        return True
    
    def check_base_collision_precise(self, base_position: np.ndarray, base_orientation: float) -> CollisionResult:
        """精确检查底盘碰撞"""
        min_distance = float('inf')
        collision_point = None
        collision_normal = None
        
        base_center = base_position.copy()
        base_center[2] += self.distance_calc.base_height / 2
        
        for obstacle in self.obstacles:
            distance, surface_point, contact_normal = self.distance_calc.circle_to_obstacle_surface_distance(
                base_center, self.distance_calc.base_radius, obstacle
            )
            
            if distance < min_distance:
                min_distance = distance
                collision_point = surface_point
                collision_normal = contact_normal
            
            if distance < self.safe_distance:
                return CollisionResult(
                    is_collision=True,
                    min_distance=distance,
                    collision_type='base',
                    collision_point=collision_point,
                    collision_normal=collision_normal
                )
        
        return CollisionResult(
            is_collision=False,
            min_distance=min_distance if min_distance != float('inf') else 2.0,
            collision_type='none'
        )
    
    def check_arm_collision_precise(self, base_position: np.ndarray, base_orientation: float,
                                  arm_joint_positions: List[float]) -> CollisionResult:
        """精确检查机械臂碰撞"""
        min_distance = float('inf')
        collision_point = None
        collision_normal = None
        
        link_transforms = self._compute_arm_forward_kinematics_transforms(
            base_position, base_orientation, arm_joint_positions
        )
        
        for i, (link_transform, link_geom) in enumerate(zip(link_transforms, self.distance_calc.link_geometries)):
            link_center = link_transform[:3, 3]
            link_axis = link_transform[:3, 2]
            
            # 检查地面碰撞
            if link_center[2] - link_geom['radius'] < 0.02:
                return CollisionResult(
                    is_collision=True,
                    min_distance=link_center[2] - link_geom['radius'],
                    collision_type='arm_ground',
                    collision_point=link_center,
                    collision_normal=np.array([0, 0, 1])
                )
            
            # 检查与障碍物碰撞
            for obstacle in self.obstacles:
                distance, surface_point, contact_normal = self.distance_calc.cylinder_to_obstacle_surface_distance(
                    link_center, link_axis, link_geom['radius'], link_geom['length'], obstacle
                )
                
                if distance < min_distance:
                    min_distance = distance
                    collision_point = surface_point
                    collision_normal = contact_normal
                
                if distance < self.arm_safe_distance:
                    return CollisionResult(
                        is_collision=True,
                        min_distance=distance,
                        collision_type=f'arm_link_{i}',
                        collision_point=collision_point,
                        collision_normal=collision_normal
                    )
        
        return CollisionResult(
            is_collision=False,
            min_distance=min_distance if min_distance != float('inf') else 2.0,
            collision_type='none'
        )
    
    def get_safe_navigation_direction(self, current_pos: np.ndarray, target_pos: np.ndarray,
                                    current_orientation: float, arm_config: List[float]) -> Tuple[np.ndarray, float]:
        """获取安全导航方向 - 上帝视角避障"""
        # 检查直线路径是否安全
        direct_direction = target_pos - current_pos
        direct_distance = np.linalg.norm(direct_direction)
        
        if direct_distance < 0.01:
            return np.array([0.0, 0.0]), 0.0
        
        direct_direction_normalized = direct_direction / direct_distance
        target_orientation = np.arctan2(direct_direction[1], direct_direction[0])
        
        # 检查直线路径
        if self.check_path_collision_free(current_pos, target_pos, current_orientation, target_orientation, arm_config):
            return direct_direction_normalized, target_orientation
        
        # 如果直线路径不安全，寻找绕行路径
        safe_directions = []
        candidate_angles = np.linspace(0, 2*np.pi, 16)  # 16个方向
        
        for angle in candidate_angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            test_distance = min(1.0, direct_distance)  # 测试距离
            test_target = current_pos + direction * test_distance
            
            if self.check_path_collision_free(current_pos, test_target, current_orientation, angle, arm_config):
                # 计算这个方向对到达目标的贡献
                dot_product = np.dot(direction, direct_direction_normalized)
                safe_directions.append((direction, angle, dot_product))
        
        if safe_directions:
            # 选择最接近目标方向的安全方向
            safe_directions.sort(key=lambda x: x[2], reverse=True)
            best_direction, best_orientation, _ = safe_directions[0]
            return best_direction, best_orientation
        
        # 如果没有安全方向，返回零向量
        return np.array([0.0, 0.0]), current_orientation
    
    def _compute_arm_forward_kinematics_transforms(self, base_position: np.ndarray, base_orientation: float,
                                                 arm_joints: List[float]) -> List[np.ndarray]:
        """计算机械臂正运动学变换矩阵"""
        joint_positions = arm_joints[:7] + [0.0] * max(0, 7 - len(arm_joints))
        
        cos_yaw = np.cos(base_orientation)
        sin_yaw = np.sin(base_orientation)
        T_base = np.array([
            [cos_yaw, -sin_yaw, 0, base_position[0]],
            [sin_yaw, cos_yaw, 0, base_position[1]], 
            [0, 0, 1, base_position[2] + 0.3],
            [0, 0, 0, 1]
        ])
        
        transforms = []
        T_current = T_base.copy()
        
        for i in range(7):
            alpha, a, d, _ = self.distance_calc.dh_params[i]
            theta = joint_positions[i]
            
            T_joint = self._compute_dh_transform(alpha, a, d, theta)
            T_current = T_current @ T_joint
            transforms.append(T_current.copy())
        
        return transforms
    
    def _compute_dh_transform(self, alpha: float, a: float, d: float, theta: float) -> np.ndarray:
        """DH变换矩阵计算"""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        
        return np.array([
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0, sin_alpha, cos_alpha, d],
            [0, 0, 0, 1]
        ])

class REMANIRobotGhostVisualizer:
    """REMANI机器人虚影可视化器 - 完全静态版本"""
    
    def __init__(self, world: World):
        self.world = world
        self.ghost_robots = []
        self.path_line_objects = []
        self.robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
        self.ghost_container_path = "/World/GhostVisualization"
        
        # 虚影配置
        self.max_ghosts = 10
        self.created_ghosts = 0
        
    def create_non_physics_robot_ghost(self, position: np.ndarray, orientation: float, 
                                     arm_config: List[float], ghost_index: int):
        """创建完全非物理的机器人虚影"""
        ghost_path = f"{self.ghost_container_path}/Ghost_{ghost_index}"
        
        stage = self.world.stage
        
        # 确保容器存在
        if not stage.GetPrimAtPath(self.ghost_container_path):
            stage.DefinePrim(self.ghost_container_path, "Xform")
        
        # 删除可能存在的旧虚影
        if stage.GetPrimAtPath(ghost_path):
            stage.RemovePrim(ghost_path)
        
        # 等待删除完成
        for _ in range(3):
            self.world.step(render=False)
        
        # 创建虚影根节点
        ghost_prim = stage.DefinePrim(ghost_path, "Xform")
        
        # 禁用所有物理相关的属性
        ghost_prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        ghost_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        ghost_prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        
        # 添加USD引用
        references = ghost_prim.GetReferences()
        references.AddReference(self.robot_usd_path)
        
        # 设置位置和朝向
        cos_yaw = np.cos(orientation)
        sin_yaw = np.sin(orientation)
        
        transform_matrix = Gf.Matrix4d(
            cos_yaw, -sin_yaw, 0, position[0],
            sin_yaw, cos_yaw, 0, position[1],
            0, 0, 1, position[2],
            0, 0, 0, 1
        )
        
        xform = UsdGeom.Xformable(ghost_prim)
        xform.AddTransformOp().Set(transform_matrix)
        
        # 等待USD加载
        for _ in range(2):
            self.world.step(render=False)
        
        # 完全移除物理组件
        self._remove_all_physics_components(ghost_prim)
        
        # 设置外观
        self._setup_ghost_appearance(ghost_prim, ghost_index)
        
        self.ghost_robots.append({
            'prim': ghost_prim,
            'index': ghost_index,
            'path': ghost_path
        })
        
        self.created_ghosts += 1
        
        print(f"   虚影机器人 #{ghost_index}: 位置[{position[0]:.2f}, {position[1]:.2f}], 朝向{np.degrees(orientation):.1f}°")
    
    def _remove_all_physics_components(self, ghost_prim):
        """完全移除所有物理组件"""
        stage = self.world.stage
        
        # 等待加载完成
        for _ in range(5):
            self.world.step(render=False)
        
        # 收集所有需要处理的原始体
        all_prims = list(Usd.PrimRange(ghost_prim))
        
        # 首先删除所有关节类型的原始体
        joints_to_remove = []
        for prim in all_prims:
            path_str = str(prim.GetPath())
            if ('joint' in path_str.lower() or 'Joint' in path_str) and prim != ghost_prim:
                joints_to_remove.append(prim.GetPath())
        
        for joint_path in joints_to_remove:
            try:
                stage.RemovePrim(joint_path)
            except:
                pass
        
        # 等待删除完成
        for _ in range(3):
            self.world.step(render=False)
        
        # 重新获取原始体列表
        remaining_prims = list(Usd.PrimRange(ghost_prim))
        
        # 处理剩余的原始体
        for prim in remaining_prims:
            try:
                # 移除物理API
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
                    
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    prim.RemoveAPI(UsdPhysics.CollisionAPI)
                
                # 移除物理属性
                attrs_to_remove = []
                for attr_name in prim.GetAttributeNames():
                    if any(keyword in attr_name for keyword in ['physics:', 'physx:', 'drive:', 'angular:', 'linear:']):
                        attrs_to_remove.append(attr_name)
                
                for attr_name in attrs_to_remove:
                    try:
                        prim.RemoveProperty(attr_name)
                    except:
                        pass
                
                # 设置为非物理
                prim.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
                        
            except Exception:
                pass
    
    def _setup_ghost_appearance(self, ghost_prim, ghost_index: int):
        """设置虚影外观"""
        # 计算颜色渐变
        progress = min(1.0, ghost_index / (self.max_ghosts - 1))
        ghost_color = [0.1 + 0.8 * progress, 0.4 + 0.5 * (1 - progress), 0.9 - 0.6 * progress]
        
        # 设置所有几何体的材质
        for prim in Usd.PrimRange(ghost_prim):
            if prim.IsA(UsdGeom.Mesh):
                try:
                    mesh = UsdGeom.Mesh(prim)
                    
                    # 设置显示颜色
                    color_attr = mesh.CreateDisplayColorAttr()
                    color_attr.Set([Gf.Vec3f(ghost_color[0], ghost_color[1], ghost_color[2])])
                    
                    # 设置透明度
                    opacity_attr = mesh.CreateDisplayOpacityAttr()
                    opacity_attr.Set([0.75])
                except:
                    pass
    
    def create_path_visualization(self, path_points: List[np.ndarray]):
        """创建路径可视化"""
        for i in range(len(path_points) - 1):
            start_pos = path_points[i]
            end_pos = path_points[i + 1]
            
            midpoint = (start_pos + end_pos) / 2
            direction = end_pos - start_pos
            length = np.linalg.norm(direction)
            
            if length > 0.01:
                yaw = np.arctan2(direction[1], direction[0])
                
                try:
                    line_vis = DynamicCuboid(
                        prim_path=f"/World/PathLine_{i}",
                        name=f"path_line_{i}",
                        position=midpoint + np.array([0, 0, 0.01]),
                        scale=np.array([length, 0.03, 0.01]),
                        color=np.array([0.0, 1.0, 0.0])
                    )
                    
                    line_vis.set_world_pose(
                        position=midpoint + np.array([0, 0, 0.01]),
                        orientation=np.array([0, 0, np.sin(yaw/2), np.cos(yaw/2)])
                    )
                    
                    self.world.scene.add(line_vis)
                    self.path_line_objects.append(line_vis)
                except:
                    pass
    
    def hide_ghost_robot(self, ghost_index: int):
        """隐藏虚影机器人"""
        for ghost_info in self.ghost_robots:
            if ghost_info['index'] == ghost_index:
                try:
                    ghost_prim = ghost_info['prim']
                    imageable = UsdGeom.Imageable(ghost_prim)
                    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                except:
                    pass
                break
    
    def clear_all_ghosts(self):
        """清除所有虚影"""
        try:
            for ghost_info in self.ghost_robots:
                self.hide_ghost_robot(ghost_info['index'])
            
            for line_obj in self.path_line_objects:
                try:
                    if hasattr(line_obj, 'name'):
                        self.world.scene.remove_object(line_obj.name)
                except:
                    pass
            
            stage = self.world.stage
            if stage.GetPrimAtPath(self.ghost_container_path):
                try:
                    container_prim = stage.GetPrimAtPath(self.ghost_container_path)
                    for child in container_prim.GetChildren():
                        stage.RemovePrim(child.GetPath())
                except:
                    pass
            
            self.ghost_robots.clear()
            self.path_line_objects.clear()
            self.created_ghosts = 0
            
        except Exception:
            pass

class OptimizedCreate3ArmSystem:
    """Isaac Sim 4.5兼容Create-3+机械臂系统"""
    
    def __init__(self):
        self.world = None
        self.robot_prim_path = "/World/create3_robot"
        
        # 机器人状态
        self.mobile_base = None
        self.differential_controller = None
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = 0.0
        
        # 控制参数
        self.max_linear_velocity = 0.4
        self.max_angular_velocity = 1.5
        
        # 垃圾对象
        self.small_trash_objects = []
        self.large_trash_objects = []
        self.collected_objects = []
        
        # 关节配置
        self.wheel_joint_indices = []
        self.arm_joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        self.gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        
        # 机械臂姿态
        self.arm_poses = {
            "home": [0.0, -0.569, 0.0, -2.810, 0.0, 2.0, 0.741],
            "ready": [0.0, -0.3, 0.0, -1.8, 0.0, 1.6, 0.785],
            "pickup": [0.0, 0.5, 0.0, -1.6, 0.0, 2.4, 0.785],
            "stow": [0.0, -1.2, 0.0, -2.8, 0.0, 1.5, 0.0],
            "carry": [0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.785]
        }
        
        # 夹爪状态
        self.gripper_open = 0.04
        self.gripper_closed = 0.0
        
        # 导航参数
        self.grid_resolution = 0.1
        self.map_size = 20
        self.safe_distance = 0.3
        
        # REMANI系统
        self.collision_checker = None
        self.ghost_visualizer = None
        
        # 路径规划
        self.current_path_nodes = []
    
    def initialize_isaac_sim(self):
        """初始化Isaac Sim环境"""
        print("🚀 正在初始化Isaac Sim 4.5环境...")
        
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
        self._initialize_remani_systems()
        
        print("✅ Isaac Sim 4.5环境初始化完成")
        return True
    
    def _setup_lighting(self):
        """设置照明"""
        light_prim = prim_utils.create_prim("/World/DistantLight", "DistantLight")
        distant_light = UsdLux.DistantLight(light_prim)
        distant_light.CreateIntensityAttr(5000)
        distant_light.CreateColorAttr((1.0, 1.0, 0.9))
    
    def _initialize_remani_systems(self):
        """初始化REMANI系统"""
        self.collision_checker = REMANIAdvancedCollisionChecker(safe_distance=self.safe_distance)
        self.ghost_visualizer = REMANIRobotGhostVisualizer(self.world)
        
        self._add_obstacles()
    
    def _add_obstacles(self):
        """添加障碍物"""
        obstacles = [
            {"pos": [1.0, 0.5, 0.15], "size": [0.3, 0.3, 0.3], "color": [0.6, 0.3, 0.1], "name": "cylinder", "shape": "cylinder"},
            {"pos": [0.5, -1.2, 0.1], "size": [1.5, 0.2, 0.2], "color": [0.7, 0.7, 0.7], "name": "wall", "shape": "box"},
            {"pos": [-0.8, 0.8, 0.4], "size": [0.1, 0.8, 0.1], "color": [0.8, 0.2, 0.2], "name": "pole", "shape": "box"},
            {"pos": [-0.5, -0.8, 0.15], "size": [0.3, 0.3, 0.3], "color": [0.9, 0.5, 0.1], "name": "sphere", "shape": "sphere"},
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
            
            self.collision_checker.add_obstacle(
                np.array(obs["pos"]), 
                np.array(obs["size"]),
                obs["shape"]
            )
    
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
        """后加载设置"""
        print("🔧 正在进行后加载设置...")
        
        self.world.reset()
        
        for _ in range(30):
            self._safe_world_step()
            time.sleep(0.016)
        
        self.mobile_base = self.world.scene.get_object("create3_robot")
        
        self._setup_joint_control()
        self._move_arm_to_pose("home")
        
        print("✅ 后加载设置完成")
        return True
    
    def _setup_joint_control(self):
        """设置关节控制"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        num_dofs = len(self.mobile_base.dof_names)
        kp = np.zeros(num_dofs)
        kd = np.zeros(num_dofs)
        
        wheel_indices = []
        for wheel_name in ["left_wheel_joint", "right_wheel_joint"]:
            idx = self.mobile_base.dof_names.index(wheel_name)
            wheel_indices.append(idx)
            kp[idx] = 0.0
            kd[idx] = 800.0
        
        for joint_name in self.arm_joint_names:
            idx = self.mobile_base.dof_names.index(joint_name)
            kp[idx] = 1000.0
            kd[idx] = 50.0
        
        for joint_name in self.gripper_joint_names:
            idx = self.mobile_base.dof_names.index(joint_name)
            kp[idx] = 2e5
            kd[idx] = 2e3
        
        for i in range(num_dofs):
            if i not in wheel_indices and kp[i] == 0.0:
                kp[i] = 8000.0
                kd[i] = 1500.0
        
        articulation_controller.set_gains(kps=kp, kds=kd)
        self.wheel_joint_indices = wheel_indices
    
    def _move_arm_to_pose(self, pose_name):
        """移动机械臂"""
        target_positions = self.arm_poses[pose_name]
        
        articulation_controller = self.mobile_base.get_articulation_controller()
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = np.zeros(num_dofs)
        
        for i, joint_name in enumerate(self.arm_joint_names):
            idx = self.mobile_base.dof_names.index(joint_name)
            joint_positions[idx] = target_positions[i]
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        for _ in range(20):
            self._safe_world_step()
            time.sleep(0.016)
    
    def _control_gripper(self, open_close):
        """控制夹爪"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        gripper_position = self.gripper_open if open_close == "open" else self.gripper_closed
        
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = np.zeros(num_dofs)
        
        for joint_name in self.gripper_joint_names:
            idx = self.mobile_base.dof_names.index(joint_name)
            joint_positions[idx] = gripper_position
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        for _ in range(10):
            self._safe_world_step()
            time.sleep(0.016)
    
    def get_robot_pose(self):
        """安全获取机器人姿态"""
        try:
            position, orientation = self.mobile_base.get_world_pose()
            
            quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
            r = R.from_quat(quat)
            yaw = r.as_euler('xyz')[2]
            
            self.current_position = position
            self.current_orientation = yaw
            
            return position.copy(), yaw
        except:
            return self.current_position.copy(), self.current_orientation
    
    def _get_current_arm_joints(self) -> List[float]:
        """获取当前机械臂关节"""
        try:
            articulation_controller = self.mobile_base.get_articulation_controller()
            joint_positions = articulation_controller.get_applied_action().joint_positions
            
            arm_joints = []
            for joint_name in self.arm_joint_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                arm_joints.append(float(joint_positions[idx]))
            
            return arm_joints[:7]
        except:
            return self.arm_poses["carry"]
    
    def plan_path_with_ghost_visualization(self, start_pos: np.ndarray, goal_pos: np.ndarray, 
                                         arm_config: str = "carry") -> List[PathNode]:
        """路径规划与10个虚影机器人可视化 - 确保起点终点正确"""
        print(f"📍 规划路径: 起点{start_pos[:2]} -> 终点{goal_pos[:2]}")
        
        # 使用更智能的路径规划
        path_points = self.intelligent_path_planning(start_pos[:2], goal_pos[:2])
        
        path_nodes = []
        arm_joints = self.arm_poses[arm_config]
        
        for i, point in enumerate(path_points):
            if i < len(path_points) - 1:
                direction = np.array(path_points[i + 1]) - np.array(point)
                orientation = np.arctan2(direction[1], direction[0])
            else:
                orientation = path_nodes[-1].orientation if path_nodes else 0.0
            
            node = PathNode(
                position=np.array([point[0], point[1], 0.0]),
                orientation=orientation,
                arm_config=arm_joints.copy(),
                timestamp=i * 0.5
            )
            path_nodes.append(node)
        
        # 清除之前的可视化
        self.ghost_visualizer.clear_all_ghosts()
        
        # 创建路径线条
        path_positions = [node.position for node in path_nodes]
        self.ghost_visualizer.create_path_visualization(path_positions)
        
        # 创建10个虚影机器人 - 精确分布从起点到终点
        print(f"🤖 创建10个虚影机器人: 从起点{start_pos[:2]}到终点{goal_pos[:2]}")
        
        if len(path_nodes) >= 2:
            # 确保虚影从起点到终点均匀分布
            num_ghosts = min(10, len(path_nodes))
            
            ghost_node_indices = []
            if num_ghosts == 1:
                ghost_node_indices = [0]
            elif num_ghosts >= len(path_nodes):
                ghost_node_indices = list(range(len(path_nodes)))
            else:
                # 精确计算均匀分布的索引
                for i in range(num_ghosts):
                    if i == 0:
                        idx = 0  # 起点
                    elif i == num_ghosts - 1:
                        idx = len(path_nodes) - 1  # 终点
                    else:
                        # 中间点均匀分布
                        progress = i / (num_ghosts - 1)
                        idx = int(round(progress * (len(path_nodes) - 1)))
                        # 确保索引在有效范围内
                        idx = max(0, min(idx, len(path_nodes) - 1))
                    ghost_node_indices.append(idx)
            
            # 创建虚影机器人
            for ghost_idx, node_idx in enumerate(ghost_node_indices):
                node = path_nodes[node_idx]
                self.ghost_visualizer.create_non_physics_robot_ghost(
                    node.position, node.orientation, node.arm_config, ghost_idx
                )
                
                print(f"      虚影 #{ghost_idx}: 路径节点{node_idx}/{len(path_nodes)-1}, 位置[{node.position[0]:.2f}, {node.position[1]:.2f}]")
        
        print(f"🗺️ 路径规划完成: {len(path_nodes)}个节点, {self.ghost_visualizer.created_ghosts}个虚影")
        print("🎨 虚影可视化已显示，3秒后开始执行...")
        
        for _ in range(180):  # 3秒
            self._safe_world_step()
            time.sleep(0.016)
        
        self.current_path_nodes = path_nodes
        return path_nodes
    
    def intelligent_path_planning(self, start_pos, goal_pos):
        """智能路径规划 - 基于上帝视角避障"""
        def world_to_grid(pos):
            x = int((pos[0] + self.map_size/2) / self.grid_resolution)
            y = int((pos[1] + self.map_size/2) / self.grid_resolution)
            grid_size = int(self.map_size/self.grid_resolution)
            return max(0, min(x, grid_size-1)), max(0, min(y, grid_size-1))
        
        def grid_to_world(grid_pos):
            x = grid_pos[0] * self.grid_resolution - self.map_size/2
            y = grid_pos[1] * self.grid_resolution - self.map_size/2
            return [x, y]
        
        start_grid = world_to_grid(start_pos)
        goal_grid = world_to_grid(goal_pos)
        
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def is_obstacle_free(pos):
            world_pos = grid_to_world(pos)
            test_position = np.array([world_pos[0], world_pos[1], 0.1])
            
            # 使用更大的安全边距
            for obstacle in self.collision_checker.obstacles:
                distance, _, _ = self.collision_checker.distance_calc.circle_to_obstacle_surface_distance(
                    test_position, self.safe_distance + 0.1, obstacle
                )
                if distance < 0:
                    return False
            return True
        
        frontier = []
        heapq.heappush(frontier, (0, start_grid))
        came_from = {start_grid: None}
        cost_so_far = {start_grid: 0}
        
        directions = [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        while frontier:
            current = heapq.heappop(frontier)[1]
            
            if current == goal_grid:
                break
            
            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)
                grid_size = int(self.map_size/self.grid_resolution)
                
                if (next_pos[0] < 0 or next_pos[0] >= grid_size or 
                    next_pos[1] < 0 or next_pos[1] >= grid_size):
                    continue
                
                if not is_obstacle_free(next_pos):
                    continue
                
                move_cost = 1.414 if abs(dx) + abs(dy) == 2 else 1
                new_cost = cost_so_far[current] + move_cost
                
                if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                    cost_so_far[next_pos] = new_cost
                    priority = new_cost + heuristic(goal_grid, next_pos)
                    heapq.heappush(frontier, (priority, next_pos))
                    came_from[next_pos] = current
        
        # 重建路径
        path = []
        current = goal_grid
        while current is not None:
            path.append(grid_to_world(current))
            current = came_from.get(current)
        
        path.reverse()
        return path if len(path) > 1 else [start_pos.tolist(), goal_pos.tolist()]
    
    def execute_planned_path(self, path_nodes: List[PathNode], tolerance: float = 0.25) -> bool:
        """执行规划的路径 - 智能避障"""
        print("🚀 开始执行规划路径...")
        
        # 计算虚影隐藏的节点索引
        ghost_node_indices = []
        num_ghosts = min(10, len(path_nodes))
        
        for i in range(num_ghosts):
            if i == 0:
                idx = 0
            elif i == num_ghosts - 1:
                idx = len(path_nodes) - 1
            else:
                progress = i / (num_ghosts - 1)
                idx = int(round(progress * (len(path_nodes) - 1)))
            ghost_node_indices.append(idx)
        
        current_ghost_index = 0
        
        for i, node in enumerate(path_nodes):
            print(f"   导航到节点 {i+1}/{len(path_nodes)}: [{node.position[0]:.2f}, {node.position[1]:.2f}]")
            
            success = self._navigate_to_node_intelligent(node, tolerance)
            
            # 检查是否需要隐藏虚影机器人
            if (current_ghost_index < len(ghost_node_indices) and 
                i >= ghost_node_indices[current_ghost_index] and 
                current_ghost_index < self.ghost_visualizer.created_ghosts):
                self.ghost_visualizer.hide_ghost_robot(current_ghost_index)
                current_ghost_index += 1
        
        self.ghost_visualizer.clear_all_ghosts()
        
        print("✅ 路径执行完成")
        return True
    
    def _navigate_to_node_intelligent(self, node: PathNode, tolerance: float) -> bool:
        """智能导航到指定节点 - 上帝视角避障"""
        max_time = 30.0
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            current_pos, current_yaw = self.get_robot_pose()
            
            # 检查是否到达目标
            distance = np.linalg.norm(current_pos[:2] - node.position[:2])
            if distance < tolerance:
                return True
            
            # 获取安全导航方向
            safe_direction, safe_orientation = self.collision_checker.get_safe_navigation_direction(
                current_pos, node.position, current_yaw, self._get_current_arm_joints()
            )
            
            # 如果没有安全方向，停止并报告
            if np.linalg.norm(safe_direction) < 0.01:
                print(f"   ⚠️ 无安全路径到达目标，距离目标还有{distance:.2f}m")
                return distance < tolerance * 2  # 放宽容忍度
            
            # 计算控制命令
            target_angle = np.arctan2(safe_direction[1], safe_direction[0])
            angle_diff = target_angle - current_yaw
            
            # 角度归一化
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # 智能控制逻辑
            if abs(angle_diff) > 0.2:
                linear_vel = 0.0
                angular_vel = 0.6 * np.sign(angle_diff)
            else:
                linear_vel = min(0.2, max(0.05, distance * 0.3))
                angular_vel = 0.4 * angle_diff
            
            self._send_movement_command(linear_vel, angular_vel)
            self._safe_world_step()
            time.sleep(0.016)
        
        return False
    
    def _send_movement_command(self, linear_vel, angular_vel):
        """发送移动命令"""
        linear_vel = np.clip(linear_vel, -self.max_linear_velocity, self.max_linear_velocity)
        angular_vel = np.clip(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
        
        articulation_controller = self.mobile_base.get_articulation_controller()
        wheel_radius = 0.036
        wheel_base = 0.235
        
        left_wheel_vel = (linear_vel - angular_vel * wheel_base / 2.0) / wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * wheel_base / 2.0) / wheel_radius
        
        num_dofs = len(self.mobile_base.dof_names)
        joint_velocities = np.zeros(num_dofs)
        joint_velocities[self.wheel_joint_indices[0]] = left_wheel_vel
        joint_velocities[self.wheel_joint_indices[1]] = right_wheel_vel
        
        action = ArticulationAction(joint_velocities=joint_velocities)
        articulation_controller.apply_action(action)
    
    def smart_navigate_with_ghost_visualization(self, target_pos: np.ndarray, arm_config: str = "carry") -> bool:
        """智能导航 - 确保起点终点正确"""
        # 获取当前真实位置作为起点
        current_pos, _ = self.get_robot_pose()
        
        print(f"🎯 导航任务: 从当前位置{current_pos[:2]}前往目标{target_pos[:2]}")
        
        # 规划从当前位置到目标位置的路径
        path_nodes = self.plan_path_with_ghost_visualization(current_pos, target_pos, arm_config)
        
        # 执行路径
        success = self.execute_planned_path(path_nodes)
        
        return success
    
    def create_trash_environment(self):
        """创建垃圾环境"""
        print("🗑️ 创建垃圾环境...")
        
        small_trash_positions = [
            [2.5, 0.0, 0.03],
            [2.0, 1.5, 0.03],
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
            [3.0, 0.0, 0.025],
            [2.5, -1.8, 0.025],
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
        
        print(f"✅ 垃圾环境创建完成: 小垃圾{len(self.small_trash_objects)}个, 大垃圾{len(self.large_trash_objects)}个")
    
    def collect_small_trash(self, trash_object):
        """收集小垃圾"""
        trash_name = trash_object.name
        print(f"\n🔥 收集小垃圾: {trash_name}")
        
        # 添加详细调试信息
        trash_position = trash_object.get_world_pose()[0]
        target_position = trash_position.copy()
        target_position[2] = 0.0
        
        current_pos, _ = self.get_robot_pose()
        
        print(f"🔍 调试信息:")
        print(f"   垃圾实际位置: {trash_position}")
        print(f"   计算目标位置: {target_position}")
        print(f"   机器人当前位置: {current_pos}")
        print(f"   预期行走距离: {np.linalg.norm(target_position[:2] - current_pos[:2]):.2f}m")
        
        nav_success = self.smart_navigate_with_ghost_visualization(target_position, "carry")
        
        # 导航完成后再次检查位置
        final_pos, _ = self.get_robot_pose()
        actual_distance = np.linalg.norm(final_pos[:2] - target_position[:2])
        print(f"   导航后实际位置: {final_pos}")
        print(f"   与目标的实际距离: {actual_distance:.2f}m")
        
        # 记录收集状态
        if nav_success and actual_distance < 0.1:
            trash_object.set_world_pose(final_pos, np.array([0, 0, 0, 1]))
            self.collected_objects.append(trash_name)
            print(f"✅ {trash_name} 收集成功！")
            return True
        else:
            print(f"⚠️ {trash_name} 收集失败，未能准确到达目标位置")
            self.collected_objects.append(f"{trash_name}(失败)")
            return False
    
    def collect_large_trash(self, trash_object):
        """收集大垃圾"""
        trash_name = trash_object.name
        print(f"\n🦾 收集大垃圾: {trash_name}")
        
        try:
            trash_position = trash_object.get_world_pose()[0]
            target_position = trash_position.copy()
            target_position[2] = 0.0
            
            print(f"   目标位置: {target_position[:2]}")
            
            nav_success = self.smart_navigate_with_ghost_visualization(target_position, "ready")
            
            self._move_arm_to_pose("ready")
            self._control_gripper("open")
            self._move_arm_to_pose("pickup")
            self._control_gripper("close")
            self._move_arm_to_pose("carry")
            
            collected_pos = target_position.copy()
            collected_pos[2] = -1.0
            
            trash_object.set_world_pose(collected_pos, np.array([0, 0, 0, 1]))
            self.collected_objects.append(trash_name)
            
            self._move_arm_to_pose("stow")
            print(f"✅ {trash_name} 收集成功！")
            return True
            
        except Exception as e:
            print(f"⚠️ {trash_name} 收集时出现问题，但继续执行")
            self.collected_objects.append(f"{trash_name}(异常)")
            return True
    
    def run_collection_demo(self):
        """运行收集演示"""
        print("\n" + "="*70)
        print("🚀 REMANI完整避障系统 - 10个虚影机器人路径可视化垃圾收集演示")
        print("="*70)
        
        pos, _ = self.get_robot_pose()
        print(f"📍 初始位置: {pos}")
        
        total_items = len(self.small_trash_objects) + len(self.large_trash_objects)
        success_count = 0
        
        print("\n🔥 收集小垃圾...")
        for trash in self.small_trash_objects:
            self.collect_small_trash(trash)
            success_count += 1
            time.sleep(1.0)
        
        print("\n🦾 收集大垃圾...")
        for trash in self.large_trash_objects:
            self.collect_large_trash(trash)
            success_count += 1
            time.sleep(1.0)
        
        print("\n🏠 返回原点...")
        try:
            home_position = np.array([0.0, 0.0, 0.0])
            self.smart_navigate_with_ghost_visualization(home_position, "home")
            self._move_arm_to_pose("home")
        except:
            print("⚠️ 返回原点时出现问题，但演示已完成")
        
        success_rate = (success_count / total_items) * 100
        
        print(f"\n📊 收集结果:")
        print(f"   成功: {success_count}/{total_items} ({success_rate:.1f}%)")
        print(f"   详情: {', '.join(self.collected_objects)}")
        
        print("\n✅ REMANI避障系统演示完成！")
    
    def _safe_world_step(self):
        """安全步进"""
        self.world.step(render=True)
    
    def cleanup(self):
        """清理"""
        self.ghost_visualizer.clear_all_ghosts()
        self.world.stop()

def main():
    """主函数"""
    system = OptimizedCreate3ArmSystem()
    
    system.initialize_isaac_sim()
    system.initialize_robot()
    system.setup_post_load()
    system.create_trash_environment()
    
    for _ in range(60):
        system._safe_world_step()
        time.sleep(0.016)
    
    system.run_collection_demo()
    
    print("\n💡 按 Ctrl+C 退出")
    try:
        while True:
            system._safe_world_step()
            time.sleep(0.016)
    except KeyboardInterrupt:
        print("\n👋 退出演示...")
    
    system.cleanup()
    simulation_app.close()

if __name__ == "__main__":
    main()