#!/usr/bin/env python3
"""
数据结构和类型定义模块
定义系统中使用的所有数据结构、枚举类型和配置参数
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass

# 系统配置参数
MAX_LINEAR_VELOCITY = 0.3      # 最大线速度 (m/s)
MAX_ANGULAR_VELOCITY = 1.5     # 最大角速度 (rad/s)
COVERAGE_CELL_SIZE = 0.5       # 覆盖网格单元大小 (m)
COVERAGE_AREA_SIZE = 10.0      # 覆盖区域大小 (m)
ROBOT_RADIUS = 0.45            # 机器人半径 (m)
PATH_TOLERANCE = 0.2           # 路径容差 (m)
POSITION_TOLERANCE = 0.2       # 位置容差 (m)
ANGLE_TOLERANCE = 0.2          # 角度容差 (rad)
MAX_NAVIGATION_STEPS = 10000   # 最大导航步数
MAX_GHOST_ROBOTS = 10          # 最大幽灵机器人数量
SAFETY_MARGIN = 0.2            # 安全边距 (m)
INTERACTION_DISTANCE = 1     # 交互距离 (m)

# 超丝滑可视化参数 - 大幅提升流畅度
FINE_GRID_SIZE = 0.03          # 从0.1减小到0.03，标记更密集
COVERAGE_UPDATE_FREQUENCY = 1  # 每步都更新，无延迟
COVERAGE_MARK_RADIUS = 0.45    # 从0.45减小到0.35，减少重叠
MARK_DISTANCE_THRESHOLD = 0.02 # 新增：更小的距离阈值，更连贯

class ObjectType(Enum):
    """对象类型枚举"""
    OBSTACLE = "obstacle"
    SWEEP = "sweep"
    GRASP = "grasp"
    TASK = "task"

@dataclass
class CollisionBoundary:
    """碰撞边界"""
    center: np.ndarray
    shape_type: str  # 'box', 'sphere', 'cylinder'
    dimensions: np.ndarray  # [width, length, height] for box, [radius] for sphere
    rotation: float = 0.0  # rotation around z-axis

@dataclass
class SceneObject:
    """场景对象"""
    name: str
    object_type: ObjectType
    position: np.ndarray
    collision_boundary: CollisionBoundary
    isaac_object: object = None
    color: np.ndarray = None
    is_active: bool = True
    original_position: np.ndarray = None

@dataclass
class CoveragePoint:
    position: np.ndarray
    orientation: float
    has_object: bool = False