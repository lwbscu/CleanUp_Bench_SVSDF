#!/usr/bin/env python3
"""
Isaac Sim 4.5兼容版Create-3+机械臂垃圾收集系统
REMANI完整避障系统 - 精确表面到表面距离计算
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
from pxr import UsdLux, UsdPhysics, Gf
import isaacsim.core.utils.prims as prim_utils

@dataclass
class CollisionResult:
    """碰撞检测结果"""
    is_collision: bool
    min_distance: float
    collision_type: str
    collision_point: Optional[np.ndarray] = None

@dataclass
class ObstacleInfo:
    """障碍物信息"""
    position: np.ndarray
    size: np.ndarray
    shape_type: str  # 'box', 'sphere', 'cylinder'

class REMANISurfaceDistanceCalculator:
    """REMANI风格精确表面距离计算器"""
    
    def __init__(self):
        # Create3底座几何参数（精确尺寸）
        self.base_length = 0.34
        self.base_width = 0.26
        self.base_height = 0.1
        self.base_radius = 0.17  # 外接圆半径
        
        # 机械臂几何参数
        self.arm_thickness = 0.08
        self.arm_safe_margin = 0.20  # 增大安全距离
        
        # DH参数（Panda 7DOF）
        self.dh_params = [
            [0, 0, 0.333, 0],
            [-np.pi/2, 0, 0, 0],
            [np.pi/2, 0, 0.316, 0],
            [np.pi/2, 0.0825, 0, 0],
            [-np.pi/2, -0.0825, 0.384, 0],
            [np.pi/2, 0, 0, 0],
            [np.pi/2, 0.088, 0.107, 0]
        ]
        
        # 连杆几何（半径为外轮廓）
        self.link_geometries = [
            {"length": 0.15, "radius": 0.06},
            {"length": 0.20, "radius": 0.07}, 
            {"length": 0.20, "radius": 0.06},
            {"length": 0.18, "radius": 0.05},
            {"length": 0.20, "radius": 0.06},
            {"length": 0.15, "radius": 0.05},
            {"length": 0.10, "radius": 0.04}
        ]
    
    def surface_distance_point_to_box(self, point: np.ndarray, box_center: np.ndarray, box_size: np.ndarray) -> float:
        """计算点到立方体外表面的精确距离"""
        # 转到盒子局部坐标
        local_point = point - box_center
        half_size = box_size / 2
        
        # 计算各轴方向到表面的距离
        dx = max(0, abs(local_point[0]) - half_size[0])
        dy = max(0, abs(local_point[1]) - half_size[1])
        dz = max(0, abs(local_point[2]) - half_size[2])
        
        # 如果点在盒子外部
        if dx > 0 or dy > 0 or dz > 0:
            return np.sqrt(dx*dx + dy*dy + dz*dz)
        
        # 如果点在盒子内部，返回到最近表面的距离（负值）
        internal_distances = [
            half_size[0] - abs(local_point[0]),
            half_size[1] - abs(local_point[1]),
            half_size[2] - abs(local_point[2])
        ]
        return -min(internal_distances)
    
    def surface_distance_point_to_sphere(self, point: np.ndarray, sphere_center: np.ndarray, sphere_radius: float) -> float:
        """计算点到球体外表面的距离"""
        center_distance = np.linalg.norm(point - sphere_center)
        return center_distance - sphere_radius
    
    def surface_distance_point_to_cylinder(self, point: np.ndarray, cylinder_center: np.ndarray, cylinder_size: np.ndarray) -> float:
        """计算点到圆柱体外表面的距离"""
        radius = cylinder_size[0] / 2
        height = cylinder_size[1]  # 使用y轴作为高度
        
        local_point = point - cylinder_center
        
        # 径向距离（x-z平面）
        radial_distance = np.sqrt(local_point[0]**2 + local_point[2]**2)
        # 轴向距离
        axial_distance = abs(local_point[1]) - height/2
        
        # 外部距离计算
        if radial_distance <= radius and abs(local_point[1]) <= height/2:
            # 点在圆柱内部
            radial_penetration = radius - radial_distance
            axial_penetration = height/2 - abs(local_point[1])
            return -min(radial_penetration, axial_penetration)
        elif radial_distance > radius and abs(local_point[1]) <= height/2:
            # 侧面外部
            return radial_distance - radius
        elif radial_distance <= radius and abs(local_point[1]) > height/2:
            # 端面外部
            return max(0, axial_distance)
        else:
            # 边角外部
            radial_excess = max(0, radial_distance - radius)
            axial_excess = max(0, axial_distance)
            return np.sqrt(radial_excess**2 + axial_excess**2)
    
    def robot_surface_to_obstacle_surface(self, robot_point: np.ndarray, robot_radius: float, 
                                        obstacle: ObstacleInfo) -> float:
        """计算机器人表面到障碍物表面的精确距离"""
        if obstacle.shape_type == 'box':
            obstacle_distance = self.surface_distance_point_to_box(
                robot_point, obstacle.position, obstacle.size)
        elif obstacle.shape_type == 'sphere':
            obstacle_distance = self.surface_distance_point_to_sphere(
                robot_point, obstacle.position, obstacle.size[0])
        elif obstacle.shape_type == 'cylinder':
            obstacle_distance = self.surface_distance_point_to_cylinder(
                robot_point, obstacle.position, obstacle.size)
        else:
            obstacle_distance = self.surface_distance_point_to_box(
                robot_point, obstacle.position, obstacle.size)
        
        # 表面到表面的距离 = 点到障碍物表面距离 - 机器人半径
        return obstacle_distance - robot_radius

class REMANICollisionChecker:
    """REMANI完整避障系统 - 精确表面距离"""
    
    def __init__(self, grid_resolution: float, map_size: float, safe_distance: float):
        self.grid_resolution = grid_resolution
        self.map_size = map_size
        self.safe_distance = safe_distance
        self.map_cells = int(map_size / grid_resolution)
        
        # 障碍物存储
        self.obstacles = []
        
        # 距离计算器
        self.distance_calc = REMANISurfaceDistanceCalculator()
        
        print(f"✅ REMANI避障系统初始化: 安全距离={safe_distance}m")
    
    def add_obstacle(self, position: np.ndarray, size: np.ndarray, shape_type: str = 'box'):
        """添加障碍物"""
        obstacle_info = ObstacleInfo(
            position=position.copy(),
            size=size.copy(),
            shape_type=shape_type
        )
        self.obstacles.append(obstacle_info)
        print(f"   添加{shape_type}障碍物: 位置{position}, 尺寸{size}")
    
    def check_base_collision(self, position: np.ndarray, orientation: float) -> CollisionResult:
        """检查移动底盘碰撞"""
        # 生成底盘轮廓检查点
        base_points = self._generate_base_collision_points(position, orientation)
        
        min_distance = float('inf')
        collision_point = None
        
        for point in base_points:
            for obstacle in self.obstacles:
                # 计算表面到表面距离
                distance = self.distance_calc.robot_surface_to_obstacle_surface(
                    point, self.distance_calc.base_radius * 0.3, obstacle  # 使用底盘厚度的一半
                )
                
                if distance < min_distance:
                    min_distance = distance
                    collision_point = point
                
                # 碰撞检查
                if distance < self.safe_distance:
                    return CollisionResult(
                        is_collision=True,
                        min_distance=distance,
                        collision_type='base',
                        collision_point=collision_point
                    )
        
        return CollisionResult(
            is_collision=False,
            min_distance=min_distance if min_distance != float('inf') else 2.0,
            collision_type='none'
        )
    
    def check_arm_collision(self, base_position: np.ndarray, base_orientation: float,
                          arm_joint_positions: List[float]) -> CollisionResult:
        """检查机械臂碰撞"""
        arm_collision_data = self._compute_arm_forward_kinematics(
            base_position, base_orientation, arm_joint_positions
        )
        
        min_distance = float('inf')
        collision_point = None
        ground_clearance = 0.05
        
        for link_data in arm_collision_data:
            points = link_data['points']
            link_radius = link_data['radius']
            
            for point in points:
                # 地面碰撞检查
                if point[2] < ground_clearance:
                    return CollisionResult(
                        is_collision=True,
                        min_distance=point[2] - ground_clearance,
                        collision_type='arm_ground',
                        collision_point=point
                    )
                
                # 障碍物碰撞检查
                for obstacle in self.obstacles:
                    distance = self.distance_calc.robot_surface_to_obstacle_surface(
                        point, link_radius, obstacle
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        collision_point = point
                    
                    if distance < self.distance_calc.arm_safe_margin:
                        return CollisionResult(
                            is_collision=True,
                            min_distance=distance,
                            collision_type='arm',
                            collision_point=collision_point
                        )
        
        return CollisionResult(
            is_collision=False,
            min_distance=min_distance if min_distance != float('inf') else 2.0,
            collision_type='none'
        )
    
    def _generate_base_collision_points(self, position: np.ndarray, orientation: float) -> List[np.ndarray]:
        """生成底盘碰撞检测点"""
        points = []
        
        # 底盘轮廓采样
        num_points = 12
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            local_x = self.distance_calc.base_radius * np.cos(angle)
            local_y = self.distance_calc.base_radius * np.sin(angle)
            
            # 转换到世界坐标
            cos_yaw = np.cos(orientation)
            sin_yaw = np.sin(orientation)
            
            world_x = position[0] + cos_yaw * local_x - sin_yaw * local_y
            world_y = position[1] + sin_yaw * local_x + cos_yaw * local_y
            
            # 多个高度层
            for height_offset in [0.03, 0.06, 0.09]:
                points.append(np.array([world_x, world_y, position[2] + height_offset]))
        
        # 添加中心点
        points.append(position.copy())
        
        return points
    
    def _compute_arm_forward_kinematics(self, base_position: np.ndarray, base_orientation: float, 
                                      arm_joints: List[float]) -> List[Dict]:
        """计算机械臂正运动学"""
        collision_data = []
        
        # 确保关节数量
        joint_positions = arm_joints[:7] + [0.0] * max(0, 7 - len(arm_joints))
        
        # 基座变换
        cos_yaw = np.cos(base_orientation)
        sin_yaw = np.sin(base_orientation)
        T_base = np.array([
            [cos_yaw, -sin_yaw, 0, base_position[0]],
            [sin_yaw, cos_yaw, 0, base_position[1]], 
            [0, 0, 1, base_position[2] + 0.3],  # 机械臂基座高度
            [0, 0, 0, 1]
        ])
        
        T_current = T_base.copy()
        
        # 正运动学计算
        for i in range(7):
            alpha, a, d, _ = self.distance_calc.dh_params[i]
            theta = joint_positions[i]
            
            # DH变换
            T_joint = self._compute_dh_transform(alpha, a, d, theta)
            T_current = T_current @ T_joint
            
            # 生成连杆检查点
            link_points = self._generate_link_collision_points(i, T_current)
            
            collision_data.append({
                'points': link_points,
                'radius': self.distance_calc.link_geometries[i]['radius'],
                'link_index': i
            })
        
        return collision_data
    
    def _compute_dh_transform(self, alpha: float, a: float, d: float, theta: float) -> np.ndarray:
        """DH变换矩阵"""
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
    
    def _generate_link_collision_points(self, link_index: int, T_link: np.ndarray) -> List[np.ndarray]:
        """生成连杆碰撞检测点"""
        points = []
        
        link_length = self.distance_calc.link_geometries[link_index]['length']
        link_radius = self.distance_calc.link_geometries[link_index]['radius']
        
        # 轴向采样点
        num_axial = max(3, int(link_length / 0.04) + 1)
        for i in range(num_axial):
            t = i * link_length / (num_axial - 1) if num_axial > 1 else 0
            
            # 中心轴点
            local_center = np.array([0, 0, t, 1])
            world_center = T_link @ local_center
            points.append(world_center[:3])
            
            # 径向采样点
            num_radial = 6
            for j in range(num_radial):
                angle = 2 * np.pi * j / num_radial
                offset_x = link_radius * 0.8 * np.cos(angle)  # 稍微收缩避免过于保守
                offset_y = link_radius * 0.8 * np.sin(angle)
                
                local_point = np.array([offset_x, offset_y, t, 1])
                world_point = T_link @ local_point
                points.append(world_point[:3])
        
        return points

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
        
        # 平滑控制
        self.current_linear_vel = 0.0
        self.current_angular_vel = 0.0
        self.velocity_smoothing = 0.15
        
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
        self.grid_resolution = 0.15
        self.map_size = 20
        self.safe_distance = 0.5  # 安全距离
        
        # REMANI避障系统
        self.collision_checker = None
        
        # 避障控制
        self.last_avoidance_time = 0
        self.avoidance_cooldown = 3.0
    
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
        
        # 添加地面
        ground = FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.5]),
            scale=np.array([50.0, 50.0, 1.0]),
            color=np.array([0.5, 0.5, 0.5])
        )
        self.world.scene.add(ground)
        
        self._setup_lighting()
        self._initialize_obstacle_map()
        
        print("✅ Isaac Sim 4.5环境初始化完成")
        return True
    
    def _setup_lighting(self):
        """设置照明"""
        light_prim = prim_utils.create_prim("/World/DistantLight", "DistantLight")
        distant_light = UsdLux.DistantLight(light_prim)
        distant_light.CreateIntensityAttr(5000)
        distant_light.CreateColorAttr((1.0, 1.0, 0.9))
    
    def _initialize_obstacle_map(self):
        """初始化REMANI避障系统"""
        self.collision_checker = REMANICollisionChecker(
            grid_resolution=self.grid_resolution,
            map_size=self.map_size,
            safe_distance=self.safe_distance
        )
        
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
            
            # 添加到避障系统
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
        
        # 稳定化
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
        
        # 轮子关节
        wheel_indices = []
        for wheel_name in ["left_wheel_joint", "right_wheel_joint"]:
            if wheel_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(wheel_name)
                wheel_indices.append(idx)
                kp[idx] = 0.0
                kd[idx] = 800.0
        
        # 机械臂关节
        for joint_name in self.arm_joint_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 1000.0
                kd[idx] = 50.0
        
        # 夹爪关节
        for joint_name in self.gripper_joint_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 2e5
                kd[idx] = 2e3
        
        # 其他关节
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
            if joint_name in self.mobile_base.dof_names and i < len(target_positions):
                idx = self.mobile_base.dof_names.index(joint_name)
                joint_positions[idx] = target_positions[i]
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        # 等待稳定
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
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                joint_positions[idx] = gripper_position
        
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        for _ in range(10):
            self._safe_world_step()
            time.sleep(0.016)
    
    def get_robot_pose(self):
        """获取机器人姿态"""
        position, orientation = self.mobile_base.get_world_pose()
        
        quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        if np.linalg.norm(quat) > 0:
            r = R.from_quat(quat)
            yaw = r.as_euler('xyz')[2]
        else:
            yaw = 0.0
        
        self.current_position = position
        self.current_orientation = yaw
        
        return position.copy(), yaw
    
    def _get_current_arm_joints(self) -> List[float]:
        """获取当前机械臂关节"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        joint_positions = articulation_controller.get_applied_action().joint_positions
        
        if joint_positions is None:
            return [0.0] * 7
        
        arm_joints = []
        for joint_name in self.arm_joint_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                if idx < len(joint_positions):
                    arm_joints.append(float(joint_positions[idx]))
                else:
                    arm_joints.append(0.0)
            else:
                arm_joints.append(0.0)
        
        return arm_joints[:7]
    
    def _send_movement_command(self, linear_vel, angular_vel):
        """发送移动命令"""
        linear_vel = np.clip(linear_vel, -self.max_linear_velocity, self.max_linear_velocity)
        angular_vel = np.clip(angular_vel, -self.max_angular_velocity, self.max_angular_velocity)
        
        if len(self.wheel_joint_indices) == 2:
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
            return True
        
        return False
    
    def _stop_robot(self):
        """停止机器人"""
        self._send_movement_command(0.0, 0.0)
    
    def check_collision_and_avoid(self) -> bool:
        """REMANI避障检查"""
        current_time = time.time()
        
        # 避障冷却
        if current_time - self.last_avoidance_time < self.avoidance_cooldown:
            return False
        
        current_pos, current_yaw = self.get_robot_pose()
        
        # 底盘碰撞检查
        base_collision = self.collision_checker.check_base_collision(current_pos, current_yaw)
        
        # 机械臂碰撞检查
        arm_collision = self.collision_checker.check_arm_collision(
            current_pos, current_yaw, self._get_current_arm_joints()
        )
        
        collision_detected = False
        
        if base_collision.is_collision:
            print(f"⚠️ 底盘避障: 距离={base_collision.min_distance:.3f}m")
            self._execute_base_avoidance()
            collision_detected = True
        
        if arm_collision.is_collision:
            print(f"⚠️ 机械臂避障: 距离={arm_collision.min_distance:.3f}m")
            self._execute_arm_avoidance()
            collision_detected = True
        
        if collision_detected:
            self.last_avoidance_time = current_time
        
        return collision_detected
    
    def _execute_base_avoidance(self):
        """执行底盘避障"""
        print("🚗 执行底盘避障...")
        
        # 停止
        self._stop_robot()
        time.sleep(0.2)
        
        # 后退
        for _ in range(40):
            self._send_movement_command(-0.3, 0.0)
            self._safe_world_step()
            time.sleep(0.016)
        
        # 转向
        turn_direction = 1.0 if random.random() > 0.5 else -1.0
        for _ in range(50):
            self._send_movement_command(0.0, 1.2 * turn_direction)
            self._safe_world_step()
            time.sleep(0.016)
        
        self._stop_robot()
        print("✅ 底盘避障完成")
    
    def _execute_arm_avoidance(self):
        """执行机械臂避障"""
        print("🦾 执行机械臂避障...")
        self._move_arm_to_pose("stow")
        print("✅ 机械臂避障完成")
    
    def a_star_path_planning(self, start_pos, goal_pos):
        """A*路径规划"""
        def world_to_grid(pos):
            x = int((pos[0] + self.map_size/2) / self.grid_resolution)
            y = int((pos[1] + self.map_size/2) / self.grid_resolution)
            return max(0, min(x, int(self.map_size/self.grid_resolution)-1)), max(0, min(y, int(self.map_size/self.grid_resolution)-1))
        
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
            for obstacle in self.collision_checker.obstacles:
                if obstacle.shape_type == 'box':
                    dist = self.collision_checker.distance_calc.surface_distance_point_to_box(
                        np.array([world_pos[0], world_pos[1], 0.1]), obstacle.position, obstacle.size
                    )
                elif obstacle.shape_type == 'sphere':
                    dist = self.collision_checker.distance_calc.surface_distance_point_to_sphere(
                        np.array([world_pos[0], world_pos[1], 0.1]), obstacle.position, obstacle.size[0]
                    )
                elif obstacle.shape_type == 'cylinder':
                    dist = self.collision_checker.distance_calc.surface_distance_point_to_cylinder(
                        np.array([world_pos[0], world_pos[1], 0.1]), obstacle.position, obstacle.size
                    )
                else:
                    dist = self.collision_checker.distance_calc.surface_distance_point_to_box(
                        np.array([world_pos[0], world_pos[1], 0.1]), obstacle.position, obstacle.size
                    )
                
                if dist < 0.6:  # 路径规划安全距离
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
                
                if (next_pos[0] < 0 or next_pos[0] >= int(self.map_size/self.grid_resolution) or 
                    next_pos[1] < 0 or next_pos[1] >= int(self.map_size/self.grid_resolution)):
                    continue
                
                if not is_obstacle_free(next_pos):
                    continue
                
                new_cost = cost_so_far[current] + (1.414 if abs(dx) + abs(dy) == 2 else 1)
                
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
        
        return path if len(path) > 1 else [start_pos, goal_pos]
    
    def smart_navigate_to_target(self, target_pos: np.ndarray, max_time: float = 30.0, tolerance: float = 0.4) -> bool:
        """智能导航"""
        print(f"🎯 导航到目标: [{target_pos[0]:.2f}, {target_pos[1]:.2f}]")
        
        current_pos, current_yaw = self.get_robot_pose()
        path = self.a_star_path_planning(current_pos[:2], target_pos[:2])
        
        start_time = time.time()
        path_index = 1
        
        while time.time() - start_time < max_time and path_index < len(path):
            current_pos, current_yaw = self.get_robot_pose()
            
            # 避障检查
            if self.check_collision_and_avoid():
                time.sleep(1.0)
                path = self.a_star_path_planning(current_pos[:2], target_pos[:2])
                path_index = 1
                continue
            
            # 获取当前目标
            current_target = path[path_index]
            direction = np.array(current_target) - current_pos[:2]
            distance = np.linalg.norm(direction)
            
            # 路径点切换
            if distance < 0.3:
                path_index += 1
                if path_index >= len(path):
                    break
                continue
            
            # 最终目标检查
            final_distance = np.linalg.norm(current_pos[:2] - target_pos[:2])
            if final_distance < tolerance:
                self._stop_robot()
                print(f"✅ 导航成功！距离: {final_distance:.3f}m")
                return True
            
            # 控制计算
            target_angle = np.arctan2(direction[1], direction[0])
            angle_diff = target_angle - current_yaw
            
            # 角度归一化
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # 控制策略
            if abs(angle_diff) > 0.6:
                linear_vel = 0.0
                angular_vel = 1.0 * np.sign(angle_diff)
            elif abs(angle_diff) > 0.2:
                linear_vel = 0.2
                angular_vel = 0.8 * np.sign(angle_diff)
            else:
                linear_vel = min(0.4, max(0.15, distance * 0.6))
                angular_vel = 0.5 * angle_diff
            
            # 平滑控制
            self.current_linear_vel = (self.velocity_smoothing * self.current_linear_vel + 
                                      (1 - self.velocity_smoothing) * linear_vel)
            self.current_angular_vel = (self.velocity_smoothing * self.current_angular_vel + 
                                       (1 - self.velocity_smoothing) * angular_vel)
            
            self._send_movement_command(self.current_linear_vel, self.current_angular_vel)
            
            self._safe_world_step()
            time.sleep(0.016)
        
        # 最终检查
        final_pos, _ = self.get_robot_pose()
        final_distance = np.linalg.norm(final_pos[:2] - target_pos[:2])
        
        if final_distance < tolerance * 1.5:
            print(f"✅ 导航接近成功！距离: {final_distance:.3f}m")
            return True
        else:
            print(f"⚠️ 导航失败，距离: {final_distance:.3f}m")
            return False
    
    def create_trash_environment(self):
        """创建垃圾环境"""
        print("🗑️ 创建垃圾环境...")
        
        # 小垃圾
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
        
        # 大垃圾
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
        print(f"🔥 收集小垃圾: {trash_name}")
        
        trash_position = trash_object.get_world_pose()[0]
        target_position = trash_position.copy()
        target_position[2] = 0.0
        
        nav_success = self.smart_navigate_to_target(target_position, max_time=25, tolerance=0.5)
        
        if nav_success:
            # 模拟吸附
            collected_pos = target_position.copy()
            collected_pos[2] = -1.0
            trash_object.set_world_pose(collected_pos, trash_object.get_world_pose()[1])
            self.collected_objects.append(trash_name)
            print(f"✅ {trash_name} 收集成功！")
            return True
        else:
            self.collected_objects.append(f"{trash_name}(失败)")
            return False
    
    def collect_large_trash(self, trash_object):
        """收集大垃圾"""
        trash_name = trash_object.name
        print(f"🦾 收集大垃圾: {trash_name}")
        
        trash_position = trash_object.get_world_pose()[0]
        target_position = trash_position.copy()
        target_position[2] = 0.0
        
        nav_success = self.smart_navigate_to_target(target_position, max_time=30, tolerance=0.6)
        
        if nav_success:
            # 抓取动作
            self._stop_robot()
            self._move_arm_to_pose("ready")
            self._control_gripper("open")
            self._move_arm_to_pose("pickup")
            self._control_gripper("close")
            self._move_arm_to_pose("carry")
            
            # 模拟收集
            collected_pos = target_position.copy()
            collected_pos[2] = -1.0
            trash_object.set_world_pose(collected_pos, trash_object.get_world_pose()[1])
            self.collected_objects.append(trash_name)
            
            self._move_arm_to_pose("stow")
            print(f"✅ {trash_name} 收集成功！")
            return True
        else:
            self.collected_objects.append(f"{trash_name}(失败)")
            return False
    
    def run_collection_demo(self):
        """运行收集演示"""
        print("\n" + "="*70)
        print("🚀 REMANI完整避障系统 - 垃圾收集演示")
        print("="*70)
        
        # 获取初始位置
        pos, _ = self.get_robot_pose()
        print(f"📍 初始位置: {pos}")
        
        # 收集统计
        total_items = len(self.small_trash_objects) + len(self.large_trash_objects)
        success_count = 0
        
        # 收集小垃圾
        print("\n🔥 收集小垃圾...")
        for trash in self.small_trash_objects:
            if self.collect_small_trash(trash):
                success_count += 1
            time.sleep(0.5)
        
        # 收集大垃圾
        print("\n🦾 收集大垃圾...")
        for trash in self.large_trash_objects:
            if self.collect_large_trash(trash):
                success_count += 1
            time.sleep(0.5)
        
        # 返回原点
        print("\n🏠 返回原点...")
        home_position = np.array([0.0, 0.0, 0.0])
        self.smart_navigate_to_target(home_position)
        self._move_arm_to_pose("home")
        
        # 结果报告
        success_rate = (success_count / total_items) * 100 if total_items > 0 else 0
        
        print(f"\n📊 收集结果:")
        print(f"   成功: {success_count}/{total_items} ({success_rate:.1f}%)")
        print(f"   详情: {', '.join(self.collected_objects)}")
        
        print("\n✅ REMANI避障系统演示完成！")
    
    def _safe_world_step(self):
        """安全步进"""
        if self.world:
            self.world.step(render=True)
    
    def cleanup(self):
        """清理"""
        self._stop_robot()
        if self.world:
            self.world.stop()

def main():
    """主函数"""
    system = OptimizedCreate3ArmSystem()
    
    system.initialize_isaac_sim()
    system.initialize_robot()
    system.setup_post_load()
    system.create_trash_environment()
    
    # 稳定化
    for _ in range(60):
        system._safe_world_step()
        time.sleep(0.016)
    
    system.run_collection_demo()
    
    # 保持运行
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