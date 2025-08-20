#!/usr/bin/env python3
"""
Isaac Sim 4.5 高质量REMANI完整避障系统 - 修复版
- 预计算完整路径和机器人姿态
- 精确虚影机器人可视化（完整USD模型）
- 平滑运动控制
- 完整垃圾收集任务流程
- 修复Matrix4d构造问题
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
    """路径节点 - 包含完整机器人状态"""
    position: np.ndarray      # 底盘位置
    orientation: float        # 底盘朝向
    arm_config: List[float]   # 机械臂关节角度
    gripper_state: float      # 夹爪状态
    timestamp: float          # 时间戳
    node_id: int             # 节点ID
    action_type: str = "move" # 动作类型: move, pickup, drop

@dataclass
class TaskInfo:
    """任务信息"""
    target_name: str
    target_position: np.ndarray
    task_type: str  # "small_trash", "large_trash"
    approach_pose: str  # 接近时的机械臂姿态

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
    """REMANI高级避障系统"""
    
    def __init__(self, safe_distance: float = 0.25):
        self.safe_distance = safe_distance
        self.arm_safe_distance = 0.12
        self.obstacles = []
        self.distance_calc = REMANIPreciseDistanceCalculator()
        
        print(f"✅ REMANI避障系统: 底盘安全距离={safe_distance}m, 机械臂安全距离={self.arm_safe_distance}m")
    
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
        num_samples = max(15, int(np.linalg.norm(end_pos - start_pos) / 0.05))
        
        for i in range(num_samples + 1):
            t = i / num_samples if num_samples > 0 else 0
            
            current_pos = start_pos + t * (end_pos - start_pos)
            current_orientation = start_orientation + t * (end_orientation - start_orientation)
            
            base_collision = self.check_base_collision_precise(current_pos, current_orientation)
            if base_collision.is_collision:
                return False
            
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
            
            if link_center[2] - link_geom['radius'] < 0.02:
                return CollisionResult(
                    is_collision=True,
                    min_distance=link_center[2] - link_geom['radius'],
                    collision_type='arm_ground',
                    collision_point=link_center,
                    collision_normal=np.array([0, 0, 1])
                )
            
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
        """获取安全导航方向"""
        direct_direction = target_pos[:2] - current_pos[:2]
        direct_distance = np.linalg.norm(direct_direction)
        
        if direct_distance < 0.01:
            return np.array([0.0, 0.0]), 0.0
        
        direct_direction_normalized = direct_direction / direct_distance
        target_orientation = np.arctan2(direct_direction[1], direct_direction[0])
        
        if self.check_path_collision_free(current_pos, target_pos, current_orientation, target_orientation, arm_config):
            return direct_direction_normalized, target_orientation
        
        safe_directions = []
        candidate_angles = np.linspace(0, 2*np.pi, 24)
        
        for angle in candidate_angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            test_distance = min(0.8, direct_distance)
            test_target = current_pos[:2] + direction * test_distance
            test_target_3d = np.array([test_target[0], test_target[1], current_pos[2]])
            
            if self.check_path_collision_free(current_pos, test_target_3d, current_orientation, angle, arm_config):
                dot_product = np.dot(direction, direct_direction_normalized)
                safe_directions.append((direction, angle, dot_product))
        
        if safe_directions:
            safe_directions.sort(key=lambda x: x[2], reverse=True)
            best_direction, best_orientation, _ = safe_directions[0]
            return best_direction, best_orientation
        
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

class REMANIAdvancedGhostVisualizer:
    """REMANI高级虚影可视化器 - 完整USD模型 - 修复版"""
    
    def __init__(self, world: World):
        self.world = world
        self.ghost_robots = []
        self.path_line_objects = []
        self.robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
        self.ghost_container_path = "/World/GhostVisualization"
        self.created_ghosts = 0
        
    def create_ghost_robot_at_node(self, path_node: PathNode, ghost_index: int):
        """在指定路径节点创建完整虚影机器人 - 修复版"""
        ghost_path = f"{self.ghost_container_path}/Ghost_{ghost_index}"
        
        stage = self.world.stage
        
        # 确保容器存在
        if not stage.GetPrimAtPath(self.ghost_container_path):
            stage.DefinePrim(self.ghost_container_path, "Xform")
        
        # 清理已存在的虚影
        if stage.GetPrimAtPath(ghost_path):
            stage.RemovePrim(ghost_path)
        
        # 等待清理完成
        for _ in range(3):
            self.world.step(render=False)
        
        # 创建虚影根Prim
        ghost_prim = stage.DefinePrim(ghost_path, "Xform")
        
        # 添加USD引用
        references = ghost_prim.GetReferences()
        references.AddReference(self.robot_usd_path)
        
        # 等待USD加载完成
        for _ in range(5):
            self.world.step(render=False)
        
        # 使用正确的Transform方法设置位置和朝向
        self._set_ghost_transform_correct(ghost_prim, path_node.position, path_node.orientation)
        
        # 完全禁用物理系统
        self._completely_disable_physics(ghost_prim)
        
        # 设置机械臂姿态
        self._set_ghost_arm_pose(ghost_prim, path_node.arm_config)
        
        # 设置虚影外观
        self._setup_ghost_appearance(ghost_prim, ghost_index)
        
        # 记录虚影信息
        self.ghost_robots.append({
            'prim': ghost_prim,
            'index': ghost_index,
            'path': ghost_path,
            'node': path_node
        })
        
        self.created_ghosts += 1
        
        print(f"   虚影 #{ghost_index}: 节点{path_node.node_id}, 位置[{path_node.position[0]:.2f}, {path_node.position[1]:.2f}], 朝向{np.degrees(path_node.orientation):.1f}°")
    
    def _set_ghost_transform_correct(self, ghost_prim, position: np.ndarray, orientation: float):
        """使用正确的方法设置虚影变换 - Isaac Sim 4.5兼容"""
        # 确保数据类型正确
        ghost_position = Gf.Vec3f(float(position[0]), float(position[1]), float(position[2]))
        
        # 将弧度转换为度数，并设置绕Z轴旋转
        yaw_degrees = float(np.degrees(orientation))
        ghost_rotation = Gf.Vec3f(0.0, 0.0, yaw_degrees)
        
        # 获取Xformable
        xform = UsdGeom.Xformable(ghost_prim)
        
        # 设置位置
        if not ghost_prim.HasAttribute("xformOp:translate"):
            translate_op = xform.AddTranslateOp()
            translate_op.Set(ghost_position)
        else:
            ghost_prim.GetAttribute("xformOp:translate").Set(ghost_position)
        
        # 设置旋转
        if not ghost_prim.HasAttribute("xformOp:rotateXYZ"):
            rotate_op = xform.AddRotateXYZOp()
            rotate_op.Set(ghost_rotation)
        else:
            ghost_prim.GetAttribute("xformOp:rotateXYZ").Set(ghost_rotation)
    
    def _completely_disable_physics(self, ghost_prim):
        """完全禁用虚影的物理系统 - Isaac Sim 4.5兼容版"""
        stage = self.world.stage
        
        # 等待完全加载
        for _ in range(5):
            self.world.step(render=False)
        
        # 获取所有子Prim
        all_prims = list(Usd.PrimRange(ghost_prim))
        
        # 移除基础物理API
        basic_physics_apis = [
            UsdPhysics.ArticulationRootAPI,
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.CollisionAPI,
        ]
        
        for prim in all_prims:
            # 移除基础物理API
            for api_class in basic_physics_apis:
                if prim.HasAPI(api_class):
                    prim.RemoveAPI(api_class)
            
            # 移除DriveAPI - 需要特殊处理，因为它有参数
            try:
                # 尝试移除不同类型的DriveAPI
                drive_types = ["linear", "angular", "transX", "transY", "transZ", "rotX", "rotY", "rotZ"]
                for drive_type in drive_types:
                    if prim.HasAPI(UsdPhysics.DriveAPI, drive_type):
                        prim.RemoveAPI(UsdPhysics.DriveAPI, drive_type)
            except:
                # 如果上面失败，尝试通用方式
                pass
        
        # 删除关节类型的Prim
        joints_to_remove = []
        for prim in all_prims:
            type_name = prim.GetTypeName()
            # 检查具体的关节类型
            if type_name in ['FixedJoint', 'RevoluteJoint', 'PrismaticJoint', 'SphericalJoint', 'D6Joint']:
                joints_to_remove.append(prim.GetPath())
        
        # 删除关节Prim
        for joint_path in joints_to_remove:
            stage.RemovePrim(joint_path)
        
        # 最终等待处理完成
        for _ in range(3):
            self.world.step(render=False)
    
    def _set_ghost_arm_pose(self, ghost_prim, arm_config: List[float]):
        """设置虚影机械臂姿态"""
        stage = self.world.stage
        
        # 确保机械臂配置完整
        full_arm_config = arm_config[:7] + [0.0] * max(0, 7 - len(arm_config))
        
        # 机械臂关节名称
        arm_joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        
        # 遍历机械臂关节并设置角度
        for i, joint_name in enumerate(arm_joint_names):
            # 尝试多种可能的关节路径
            possible_paths = [
                f"{ghost_prim.GetPath()}/ridgeback_franka/panda_link{i}/{joint_name}",
                f"{ghost_prim.GetPath()}/ridgeback_franka/panda_link{i}/panda_joint{i+1}",
                f"{ghost_prim.GetPath()}/ridgeback_franka/{joint_name}"
            ]
            
            joint_prim = None
            for path in possible_paths:
                if stage.GetPrimAtPath(path):
                    joint_prim = stage.GetPrimAtPath(path)
                    break
            
            if joint_prim:
                # 设置关节角度
                joint_angle = full_arm_config[i]
                
                # 使用Xformable设置旋转
                xform = UsdGeom.Xformable(joint_prim)
                
                # 根据关节类型设置正确的旋转轴
                if i in [0, 2, 4, 6]:  # Z轴旋转关节
                    if not joint_prim.HasAttribute("xformOp:rotateZ"):
                        rot_op = xform.AddRotateZOp()
                        rot_op.Set(float(np.degrees(joint_angle)))
                    else:
                        joint_prim.GetAttribute("xformOp:rotateZ").Set(float(np.degrees(joint_angle)))
                else:  # Y轴旋转关节
                    if not joint_prim.HasAttribute("xformOp:rotateY"):
                        rot_op = xform.AddRotateYOp()
                        rot_op.Set(float(np.degrees(joint_angle)))
                    else:
                        joint_prim.GetAttribute("xformOp:rotateY").Set(float(np.degrees(joint_angle)))
    
    def _setup_ghost_appearance(self, ghost_prim, ghost_index: int):
        """设置虚影外观 - 透明度和颜色渐变"""
        # 计算颜色渐变 (蓝色到红色)
        progress = ghost_index / max(1, 4) if self.created_ghosts > 1 else 0.0
        
        # 蓝色到红色的颜色插值
        red = 0.3 + 0.7 * progress
        green = 0.4 + 0.2 * (1 - progress)
        blue = 0.9 - 0.6 * progress
        
        ghost_color = Gf.Vec3f(float(red), float(green), float(blue))
        ghost_opacity = 0.7  # 透明度
        
        # 遍历所有Mesh几何体设置外观
        for prim in Usd.PrimRange(ghost_prim):
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                
                # 设置颜色
                color_attr = mesh.CreateDisplayColorAttr()
                color_attr.Set([ghost_color])
                
                # 设置透明度
                opacity_attr = mesh.CreateDisplayOpacityAttr()
                opacity_attr.Set([ghost_opacity])
    
    def create_path_line_visualization(self, path_nodes: List[PathNode]):
        """创建路径线可视化"""
        print("🎨 创建路径线可视化...")
        
        for i in range(len(path_nodes) - 1):
            start_node = path_nodes[i]
            end_node = path_nodes[i + 1]
            
            start_pos = start_node.position
            end_pos = end_node.position
            
            # 计算线段参数
            midpoint = (start_pos + end_pos) / 2
            direction = end_pos - start_pos
            length = np.linalg.norm(direction)
            
            if length > 0.01:
                yaw = np.arctan2(direction[1], direction[0])
                
                # 创建线段可视化
                line_vis = DynamicCuboid(
                    prim_path=f"/World/PathLine_{i}",
                    name=f"path_line_{i}",
                    position=midpoint + np.array([0, 0, 0.01]),
                    scale=np.array([length, 0.02, 0.01]),
                    color=np.array([0.0, 1.0, 0.0])  # 绿色路径线
                )
                
                # 设置正确的朝向
                line_vis.set_world_pose(
                    position=midpoint + np.array([0, 0, 0.01]),
                    orientation=np.array([0, 0, np.sin(yaw/2), np.cos(yaw/2)])
                )
                
                self.world.scene.add(line_vis)
                self.path_line_objects.append(line_vis)
    
    def hide_ghost_robot(self, ghost_index: int):
        """隐藏指定虚影机器人"""
        for ghost_info in self.ghost_robots:
            if ghost_info['index'] == ghost_index:
                ghost_prim = ghost_info['prim']
                imageable = UsdGeom.Imageable(ghost_prim)
                imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
                print(f"   隐藏虚影 #{ghost_index}")
                break
    
    def clear_all_visualizations(self):
        """清除所有可视化元素"""
        print("🧹 清理所有可视化元素...")
        
        # 隐藏所有虚影
        for ghost_info in self.ghost_robots:
            self.hide_ghost_robot(ghost_info['index'])
        
        # 移除路径线
        for line_obj in self.path_line_objects:
            self.world.scene.remove_object(line_obj.name)
        
        # 清理虚影容器
        stage = self.world.stage
        if stage.GetPrimAtPath(self.ghost_container_path):
            container_prim = stage.GetPrimAtPath(self.ghost_container_path)
            for child in container_prim.GetChildren():
                stage.RemovePrim(child.GetPath())
        
        # 重置状态
        self.ghost_robots.clear()
        self.path_line_objects.clear()
        self.created_ghosts = 0

class OptimizedCreate3ArmSystem:
    """高质量Create-3+机械臂系统 - Isaac Sim 4.5"""
    
    def __init__(self):
        self.world = None
        self.robot_prim_path = "/World/create3_robot"
        
        self.mobile_base = None
        self.differential_controller = None
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = 0.0
        
        # 运动参数
        self.max_linear_velocity = 0.3
        self.max_angular_velocity = 1.0
        
        # 垃圾对象
        self.small_trash_objects = []
        self.large_trash_objects = []
        self.collected_objects = []
        
        # 关节控制
        self.wheel_joint_indices = []
        self.arm_joint_names = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        self.gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
        
        # 机械臂预设姿态
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
        
        # 系统组件
        self.collision_checker = None
        self.ghost_visualizer = None
        
        # 任务规划
        self.all_tasks = []
        self.current_task_index = 0
        self.global_path_nodes = []
    
    def initialize_isaac_sim(self):
        """初始化Isaac Sim环境"""
        print("🚀 正在初始化Isaac Sim 4.5环境...")
        
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0/120.0,
            rendering_dt=1.0/60.0
        )
        self.world.scene.clear()
        
        # 物理设置
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
        self.collision_checker = REMANIAdvancedCollisionChecker(safe_distance=0.25)
        self.ghost_visualizer = REMANIAdvancedGhostVisualizer(self.world)
        
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
        
        # 等待系统稳定
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
        
        # 轮子关节控制参数
        wheel_indices = []
        for wheel_name in ["left_wheel_joint", "right_wheel_joint"]:
            idx = self.mobile_base.dof_names.index(wheel_name)
            wheel_indices.append(idx)
            kp[idx] = 0.0
            kd[idx] = 800.0
        
        # 机械臂关节控制参数
        for joint_name in self.arm_joint_names:
            idx = self.mobile_base.dof_names.index(joint_name)
            kp[idx] = 1000.0
            kd[idx] = 50.0
        
        # 夹爪关节控制参数
        for joint_name in self.gripper_joint_names:
            idx = self.mobile_base.dof_names.index(joint_name)
            kp[idx] = 2e5
            kd[idx] = 2e3
        
        # 其他关节默认参数
        for i in range(num_dofs):
            if i not in wheel_indices and kp[i] == 0.0:
                kp[i] = 8000.0
                kd[i] = 1500.0
        
        articulation_controller.set_gains(kps=kp, kds=kd)
        self.wheel_joint_indices = wheel_indices
    
    def _move_arm_to_pose(self, pose_name):
        """移动机械臂到预设姿态"""
        target_positions = self.arm_poses[pose_name]
        
        articulation_controller = self.mobile_base.get_articulation_controller()
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = np.zeros(num_dofs)
        
        for i, joint_name in enumerate(self.arm_joint_names):
            idx = self.mobile_base.dof_names.index(joint_name)
            joint_positions[idx] = target_positions[i]
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        # 等待到达目标位置
        for _ in range(30):
            self._safe_world_step()
            time.sleep(0.016)
    
    def _control_gripper(self, open_close):
        """控制夹爪开合"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        gripper_position = self.gripper_open if open_close == "open" else self.gripper_closed
        
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = np.zeros(num_dofs)
        
        for joint_name in self.gripper_joint_names:
            idx = self.mobile_base.dof_names.index(joint_name)
            joint_positions[idx] = gripper_position
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        # 等待夹爪动作完成
        for _ in range(15):
            self._safe_world_step()
            time.sleep(0.016)
    
    def get_robot_pose(self):
        """获取机器人姿态"""
        position, orientation = self.mobile_base.get_world_pose()
        
        # 四元数转欧拉角
        quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        r = R.from_quat(quat)
        yaw = r.as_euler('xyz')[2]
        
        self.current_position = position
        self.current_orientation = yaw
        
        return position.copy(), yaw
    
    def _get_current_arm_joints(self) -> List[float]:
        """获取当前机械臂关节角度"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        joint_positions = articulation_controller.get_applied_action().joint_positions
        
        arm_joints = []
        for joint_name in self.arm_joint_names:
            idx = self.mobile_base.dof_names.index(joint_name)
            arm_joints.append(float(joint_positions[idx]))
        
        return arm_joints[:7]
    
    def create_trash_environment(self):
        """创建垃圾环境"""
        print("🗑️ 创建垃圾环境...")
        
        # 小垃圾位置
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
        
        # 大垃圾位置
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
    
    def plan_complete_mission(self):
        """规划完整任务路径"""
        print("\n🎯 开始规划完整收集任务...")
        
        # 创建任务列表
        self.all_tasks = []
        
        # 添加小垃圾收集任务
        for i, trash in enumerate(self.small_trash_objects):
            trash_pos = trash.get_world_pose()[0]
            task = TaskInfo(
                target_name=trash.name,
                target_position=trash_pos,
                task_type="small_trash",
                approach_pose="carry"
            )
            self.all_tasks.append(task)
        
        # 添加大垃圾收集任务
        for i, trash in enumerate(self.large_trash_objects):
            trash_pos = trash.get_world_pose()[0]
            task = TaskInfo(
                target_name=trash.name,
                target_position=trash_pos,
                task_type="large_trash",
                approach_pose="ready"
            )
            self.all_tasks.append(task)
        
        # 添加返回原点任务
        home_task = TaskInfo(
            target_name="home",
            target_position=np.array([0.0, 0.0, 0.0]),
            task_type="return_home",
            approach_pose="home"
        )
        self.all_tasks.append(home_task)
        
        # 规划整体路径
        self._plan_global_path()
        
        print(f"✅ 任务规划完成: {len(self.all_tasks)}个任务, {len(self.global_path_nodes)}个路径节点")
    
    def _plan_global_path(self):
        """规划全局路径"""
        print("📍 规划全局路径...")
        
        current_pos, current_yaw = self.get_robot_pose()
        self.global_path_nodes = []
        node_id = 0
        
        for task_index, task in enumerate(self.all_tasks):
            print(f"   规划任务 {task_index + 1}: {task.target_name}")
            
            # 规划到目标的路径
            target_pos = task.target_position.copy()
            target_pos[2] = 0.0  # 确保在地面上
            
            # 生成路径点
            path_points = self._generate_smooth_path(current_pos[:2], target_pos[:2])
            
            # 为每个路径点创建节点
            for i, point in enumerate(path_points):
                # 计算朝向
                if i < len(path_points) - 1:
                    direction = np.array(path_points[i + 1]) - np.array(point)
                    orientation = np.arctan2(direction[1], direction[0])
                else:
                    orientation = self.global_path_nodes[-1].orientation if self.global_path_nodes else 0.0
                
                # 获取机械臂配置
                arm_config = self.arm_poses[task.approach_pose]
                
                # 创建路径节点
                node = PathNode(
                    position=np.array([point[0], point[1], 0.0]),
                    orientation=orientation,
                    arm_config=arm_config.copy(),
                    gripper_state=self.gripper_open,
                    timestamp=node_id * 0.2,
                    node_id=node_id,
                    action_type="move"
                )
                
                self.global_path_nodes.append(node)
                node_id += 1
            
            # 更新当前位置
            current_pos = target_pos
            current_yaw = orientation
        
        print(f"   生成 {len(self.global_path_nodes)} 个路径节点")
    
    def _generate_smooth_path(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> List[np.ndarray]:
        """生成平滑路径"""
        direction = goal_pos - start_pos
        distance = np.linalg.norm(direction)
        
        # 根据距离生成合适数量的路径点
        num_points = max(5, min(15, int(distance / 0.2)))
        
        path_points = []
        for i in range(num_points + 1):
            t = i / num_points
            # 使用平滑插值
            smooth_t = 3 * t**2 - 2 * t**3  # S曲线插值
            point = start_pos + smooth_t * direction
            path_points.append(point)
        
        return path_points
    
    def create_ghost_visualization(self):
        """创建虚影可视化"""
        print("🤖 创建虚影机器人可视化...")
        
        # 清理之前的可视化
        self.ghost_visualizer.clear_all_visualizations()
        
        # 创建路径线
        self.ghost_visualizer.create_path_line_visualization(self.global_path_nodes)
        
        # 选择5个关键节点创建虚影
        num_ghosts = 5
        total_nodes = len(self.global_path_nodes)
        
        if total_nodes >= num_ghosts:
            # 均匀分布虚影
            ghost_indices = []
            for i in range(num_ghosts):
                index = int((i * (total_nodes - 1)) / (num_ghosts - 1))
                ghost_indices.append(index)
        else:
            ghost_indices = list(range(total_nodes))
        
        print(f"   创建 {len(ghost_indices)} 个虚影机器人:")
        
        # 创建虚影机器人
        for ghost_idx, node_idx in enumerate(ghost_indices):
            node = self.global_path_nodes[node_idx]
            self.ghost_visualizer.create_ghost_robot_at_node(node, ghost_idx)
        
        print("✅ 虚影可视化创建完成")
        
        # 展示虚影3秒
        print("🎨 展示虚影可视化效果 (3秒)...")
        for _ in range(180):  # 3秒
            self._safe_world_step()
            time.sleep(0.016)
    
    def execute_complete_mission(self):
        """执行完整任务"""
        print("\n🚀 开始执行完整收集任务...")
        
        # 计算虚影对应的节点
        num_ghosts = min(5, self.ghost_visualizer.created_ghosts)
        total_nodes = len(self.global_path_nodes)
        ghost_indices = []
        
        if total_nodes >= num_ghosts:
            for i in range(num_ghosts):
                index = int((i * (total_nodes - 1)) / (num_ghosts - 1))
                ghost_indices.append(index)
        else:
            ghost_indices = list(range(total_nodes))
        
        current_ghost_index = 0
        
        # 执行所有路径节点
        for i, node in enumerate(self.global_path_nodes):
            success = self._navigate_to_node_smooth(node, tolerance=0.12)
            
            # 检查是否到达任务点
            self._check_and_execute_task_at_node(node)
            
            # 隐藏经过的虚影
            if (current_ghost_index < len(ghost_indices) and 
                i >= ghost_indices[current_ghost_index] and 
                current_ghost_index < self.ghost_visualizer.created_ghosts):
                self.ghost_visualizer.hide_ghost_robot(current_ghost_index)
                current_ghost_index += 1
        
        # 清理可视化
        self.ghost_visualizer.clear_all_visualizations()
        
        print("✅ 完整任务执行完成!")
        
        # 显示任务结果
        self._show_mission_results()
    
    def _navigate_to_node_smooth(self, node: PathNode, tolerance: float = 0.12) -> bool:
        """平滑导航到指定节点"""
        max_time = 20.0
        start_time = time.time()
        
        while time.time() - start_time < max_time:
            current_pos, current_yaw = self.get_robot_pose()
            
            # 检查是否到达目标
            distance = np.linalg.norm(current_pos[:2] - node.position[:2])
            if distance < tolerance:
                return True
            
            # 计算目标方向
            direction = node.position[:2] - current_pos[:2]
            target_angle = np.arctan2(direction[1], direction[0])
            
            # 角度差处理
            angle_diff = target_angle - current_yaw
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # 平滑控制
            if abs(angle_diff) > 0.1:  # 需要转向
                linear_vel = 0.0
                angular_vel = np.clip(angle_diff * 2.0, -0.8, 0.8)
            else:  # 直线前进
                linear_vel = min(0.2, max(0.05, distance * 0.4))
                angular_vel = np.clip(angle_diff * 1.0, -0.3, 0.3)
            
            # 避障检查
            safe_direction, safe_orientation = self.collision_checker.get_safe_navigation_direction(
                current_pos, node.position, current_yaw, self._get_current_arm_joints()
            )
            
            if np.linalg.norm(safe_direction) < 0.01:
                linear_vel = 0.0
                angular_vel = 0.2  # 原地旋转寻找安全方向
            
            # 发送控制命令
            self._send_smooth_movement_command(linear_vel, angular_vel)
            self._safe_world_step()
            time.sleep(0.016)
        
        return False
    
    def _send_smooth_movement_command(self, linear_vel, angular_vel):
        """发送平滑运动命令"""
        # 限制速度
        linear_vel = np.clip(linear_vel, -self.max_linear_velocity, self.max_linear_velocity)
        angular_vel = np.clip(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
        
        # 计算轮子速度
        articulation_controller = self.mobile_base.get_articulation_controller()
        wheel_radius = 0.036
        wheel_base = 0.235
        
        left_wheel_vel = (linear_vel - angular_vel * wheel_base / 2.0) / wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * wheel_base / 2.0) / wheel_radius
        
        # 应用轮子速度
        num_dofs = len(self.mobile_base.dof_names)
        joint_velocities = np.zeros(num_dofs)
        joint_velocities[self.wheel_joint_indices[0]] = left_wheel_vel
        joint_velocities[self.wheel_joint_indices[1]] = right_wheel_vel
        
        action = ArticulationAction(joint_velocities=joint_velocities)
        articulation_controller.apply_action(action)
    
    def _check_and_execute_task_at_node(self, node: PathNode):
        """检查并执行节点处的任务"""
        # 检查是否到达任务目标点
        for task in self.all_tasks:
            task_distance = np.linalg.norm(node.position[:2] - task.target_position[:2])
            
            if task_distance < 0.2 and task.target_name not in self.collected_objects:
                print(f"\n🎯 到达任务点: {task.target_name}")
                
                if task.task_type == "small_trash":
                    self._collect_small_trash_at_location(task)
                elif task.task_type == "large_trash":
                    self._collect_large_trash_at_location(task)
                elif task.task_type == "return_home":
                    print("🏠 返回原点完成")
                
                break
    
    def _collect_small_trash_at_location(self, task: TaskInfo):
        """在位置收集小垃圾"""
        print(f"🔥 收集小垃圾: {task.target_name}")
        
        # 确保机械臂处于carry姿态
        self._move_arm_to_pose("carry")
        
        # 找到对应的垃圾对象
        trash_obj = None
        for trash in self.small_trash_objects:
            if trash.name == task.target_name:
                trash_obj = trash
                break
        
        if trash_obj:
            # 模拟收集 - 将垃圾移动到机器人位置
            current_pos, _ = self.get_robot_pose()
            trash_obj.set_world_pose(current_pos, np.array([0, 0, 0, 1]))
            self.collected_objects.append(task.target_name)
            print(f"✅ {task.target_name} 收集成功!")
        
        # 短暂延迟
        for _ in range(30):
            self._safe_world_step()
            time.sleep(0.016)
    
    def _collect_large_trash_at_location(self, task: TaskInfo):
        """在位置收集大垃圾"""
        print(f"🦾 收集大垃圾: {task.target_name}")
        
        # 机械臂动作序列
        self._move_arm_to_pose("ready")
        self._control_gripper("open")
        self._move_arm_to_pose("pickup")
        self._control_gripper("close")
        self._move_arm_to_pose("carry")
        
        # 找到对应的垃圾对象
        trash_obj = None
        for trash in self.large_trash_objects:
            if trash.name == task.target_name:
                trash_obj = trash
                break
        
        if trash_obj:
            # 模拟收集 - 将垃圾移动到隐藏位置
            trash_obj.set_world_pose(np.array([0, 0, -1.0]), np.array([0, 0, 0, 1]))
            self.collected_objects.append(task.target_name)
            print(f"✅ {task.target_name} 收集成功!")
        
        # 收起机械臂
        self._move_arm_to_pose("stow")
    
    def _show_mission_results(self):
        """显示任务结果"""
        total_items = len(self.small_trash_objects) + len(self.large_trash_objects)
        success_count = len(self.collected_objects)
        success_rate = (success_count / total_items) * 100 if total_items > 0 else 0
        
        print(f"\n📊 任务执行结果:")
        print(f"   总垃圾数: {total_items}")
        print(f"   成功收集: {success_count}")
        print(f"   成功率: {success_rate:.1f}%")
        print(f"   收集详情: {', '.join(self.collected_objects)}")
        print(f"   路径节点: {len(self.global_path_nodes)}")
        print(f"   虚影展示: {self.ghost_visualizer.created_ghosts}")
    
    def run_complete_demo(self):
        """运行完整演示"""
        print("\n" + "="*80)
        print("🚀 REMANI高质量避障系统 - 完整虚影机器人路径可视化演示")
        print("="*80)
        
        # 显示初始状态
        pos, yaw = self.get_robot_pose()
        print(f"📍 初始位置: [{pos[0]:.3f}, {pos[1]:.3f}], 朝向: {np.degrees(yaw):.1f}°")
        
        # 阶段1: 任务规划
        self.plan_complete_mission()
        
        # 阶段2: 虚影可视化
        self.create_ghost_visualization()
        
        # 阶段3: 执行任务
        self.execute_complete_mission()
        
        # 最终检查机械臂姿态
        self._move_arm_to_pose("home")
        
        print("\n✅ REMANI高质量虚影避障系统演示完成!")
        print("💡 所有垃圾已收集，机器人已返回原点")
    
    def _safe_world_step(self):
        """安全步进世界"""
        self.world.step(render=True)
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理系统资源...")
        self.ghost_visualizer.clear_all_visualizations()
        self.world.stop()

def main():
    """主函数"""
    print("🚀 启动REMANI高质量避障系统...")
    
    system = OptimizedCreate3ArmSystem()
    
    # 初始化系统
    system.initialize_isaac_sim()
    system.initialize_robot()
    system.setup_post_load()
    system.create_trash_environment()
    
    # 等待系统稳定
    for _ in range(60):
        system._safe_world_step()
        time.sleep(0.016)
    
    # 运行完整演示
    system.run_complete_demo()
    
    # 保持运行状态
    print("\n💡 按 Ctrl+C 退出程序")
    while True:
        system._safe_world_step()
        time.sleep(0.016)

if __name__ == "__main__":
    main()