#!/usr/bin/env python3
"""
Isaac Sim 4.5 智能覆盖算法机器人系统 - 流畅可视化优化版
- 覆盖算法进行区域地毯式移动
- 智能结合物体抓取与清扫
- 实时流畅的覆盖区域可视化标记系统
- 动态路径显示
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
MAX_LINEAR_VELOCITY = 0.15     # 覆盖时的最大直线速度
MAX_ANGULAR_VELOCITY = 2.5     # 覆盖时的最大角速度

# 覆盖算法参数
COVERAGE_CELL_SIZE = 0.8       # 覆盖网格大小(m) - 根据底盘直径(0.9m)设计，留小量重叠
COVERAGE_AREA_SIZE = 6.0       # 覆盖区域大小(m)
OVERLAP_DISTANCE = 5         # 覆盖重叠距离(m)

# 路径可视化显示参数
GHOST_DISPLAY_STEPS = 25       # 虚影路径展示步数
GHOSTS_PER_SEGMENT = 4         # 每个覆盖段的虚影数量

# 物体收集参数
COLLECTION_DISTANCE = 0.45     # 物体收集距离(m) - 等于底盘半径
COVERAGE_MARK_RADIUS = 0.45    # 覆盖标记半径(m) - 实际底盘半径

# 流畅可视化参数
FINE_GRID_SIZE = 0.1          # 精细网格大小(m) - 流畅可视化
COVERAGE_UPDATE_FREQUENCY = 5  # 覆盖标记更新频率（每N步更新一次）

# 系统稳定性参数
STABILIZE_STEPS = 15           # 系统稳定化步数
MEMORY_THRESHOLD_MB = 5500     # 内存阈值(MB)
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
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
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
    coverage_priority: float
    has_object: bool = False
    object_type: str = ""
    node_id: int = 0

@dataclass
class CoverageSegment:
    points: List[CoveragePoint]
    segment_type: str  # "main_line", "turn", "approach_object"
    priority: float

class FluentCoverageVisualizer:
    """流畅覆盖区域可视化器 - 实时跟随机器人移动"""
    
    def __init__(self, world: World):
        self.world = world
        self.coverage_marks = {}  # 精细位置 -> 覆盖次数
        self.mark_prims = {}      # 精细位置 -> prim路径
        self.coverage_container = "/World/CoverageMarks"
        self.last_marked_position = None
        self.mark_counter = 0
        
        print("🎨 流畅覆盖可视化器初始化")
    
    def mark_coverage_realtime(self, robot_position: np.ndarray):
        """实时标记覆盖区域 - 流畅跟随机器人移动"""
        self.mark_counter += 1
        
        # 提高更新频率，每步都可能更新
        if self.mark_counter % COVERAGE_UPDATE_FREQUENCY != 0:
            return
            
        # 使用精细网格进行流畅标记
        fine_grid_pos = self._fine_quantize_position(robot_position)
        
        # 检查是否需要新标记
        if self._should_create_new_mark(fine_grid_pos):
            self._create_fluent_coverage_mark(fine_grid_pos)
            self.last_marked_position = fine_grid_pos.copy()
    
    def _should_create_new_mark(self, current_pos: np.ndarray) -> bool:
        """判断是否应该创建新标记"""
        if self.last_marked_position is None:
            return True
            
        # 距离上次标记位置足够远时创建新标记
        distance = np.linalg.norm(current_pos[:2] - self.last_marked_position[:2])
        return distance >= FINE_GRID_SIZE
    
    def _fine_quantize_position(self, position: np.ndarray) -> np.ndarray:
        """精细量化位置 - 更小的网格实现流畅效果"""
        # 使用精细网格，确保标记连贯流畅
        x = round(position[0] / FINE_GRID_SIZE) * FINE_GRID_SIZE
        y = round(position[1] / FINE_GRID_SIZE) * FINE_GRID_SIZE
        return np.array([x, y, 0.02])  # 略高于地面
    
    def _create_fluent_coverage_mark(self, position: np.ndarray):
        """创建流畅的覆盖标记"""
        stage = self.world.stage
        
        # 确保容器存在
        if not stage.GetPrimAtPath(self.coverage_container):
            stage.DefinePrim(self.coverage_container, "Xform")
        
        # 创建唯一的标记路径
        mark_id = len(self.coverage_marks)
        x_str = f"{position[0]:.2f}".replace(".", "p").replace("-", "N")
        y_str = f"{position[1]:.2f}".replace(".", "p").replace("-", "N")
        pos_key = f"{x_str}_{y_str}"
        mark_path = f"{self.coverage_container}/FluentMark_{mark_id}_{pos_key}"
        
        # 记录覆盖
        if pos_key in self.coverage_marks:
            self.coverage_marks[pos_key] += 1
            return  # 已存在标记，直接返回
        else:
            self.coverage_marks[pos_key] = 1
        
        coverage_count = self.coverage_marks[pos_key]
        
        # 创建圆形标记
        mark_prim = stage.DefinePrim(mark_path, "Cylinder")
        cylinder_geom = UsdGeom.Cylinder(mark_prim)
        
        # 设置为扁平圆盘，半径稍小于底盘半径实现更精细的标记
        mark_radius = COVERAGE_MARK_RADIUS * 0.8  # 稍小一些，更精细
        cylinder_geom.CreateRadiusAttr().Set(mark_radius)
        cylinder_geom.CreateHeightAttr().Set(0.01)  # 更薄的圆盘
        
        # 设置位置 - 修复变换操作冲突
        xform = UsdGeom.Xformable(mark_prim)
        xform.ClearXformOpOrder()  # 清除现有变换操作
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        
        # 禁用物理
        UsdPhysics.RigidBodyAPI.Apply(mark_prim)
        rigid_body = UsdPhysics.RigidBodyAPI(mark_prim)
        rigid_body.CreateRigidBodyEnabledAttr().Set(False)
        
        # 设置流畅渐变颜色
        self._set_fluent_color(mark_prim, coverage_count)
        
        # 记录标记
        self.mark_prims[pos_key] = mark_path
        
        # 流畅进度显示
        if len(self.coverage_marks) % 20 == 0:
            print(f"🎨 流畅覆盖进度: {len(self.coverage_marks)}个精细标记")
    
    def _set_fluent_color(self, mark_prim, coverage_count: int):
        """设置流畅渐变颜色"""
        # 从亮绿色到深绿色的渐变，表示覆盖深度
        intensity = min(coverage_count / 3.0, 1.0)  # 最多3次覆盖达到最深色
        
        # 绿色渐变：浅绿 -> 深绿
        green_value = 0.8 - (intensity * 0.5)  # 0.8 -> 0.3
        red_value = 0.2 + (intensity * 0.3)    # 0.2 -> 0.5  
        blue_value = 0.2
        
        gprim = UsdGeom.Gprim(mark_prim)
        gprim.CreateDisplayColorAttr().Set([(red_value, green_value, blue_value)])
    
    def cleanup(self):
        """清理覆盖标记"""
        stage = self.world.stage
        
        # 删除所有标记prims
        for pos_key, mark_path in self.mark_prims.items():
            if stage.GetPrimAtPath(mark_path):
                stage.RemovePrim(mark_path)
        
        # 删除容器
        if stage.GetPrimAtPath(self.coverage_container):
            stage.RemovePrim(self.coverage_container)
            
        self.coverage_marks.clear()
        self.mark_prims.clear()
        self.last_marked_position = None
        
        print("🧹 流畅覆盖标记清理完成")

class CoveragePathPlanner:
    """智能覆盖路径规划器 - 适配0.45m底盘半径"""
    
    def __init__(self, world_size: float = COVERAGE_AREA_SIZE, cell_size: float = COVERAGE_CELL_SIZE):
        self.world_size = world_size
        self.cell_size = cell_size
        self.grid_size = int(world_size / cell_size)
        self.obstacles = []
        self.obstacle_grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.objects_positions = []  # 待收集物体位置
        
        print(f"🗺️ 覆盖规划器: {self.grid_size}x{self.grid_size}网格, 单元格{cell_size}m")
        print(f"   底盘半径: {COVERAGE_MARK_RADIUS}m, 覆盖直径: {COVERAGE_MARK_RADIUS*2}m")
    
    def add_obstacle(self, position: np.ndarray, size: np.ndarray, shape_type: str = 'box'):
        """添加障碍物 - 考虑0.45m底盘半径"""
        self.obstacles.append({'pos': position, 'size': size, 'type': shape_type})
        
        center_x = int((position[0] + self.world_size/2) / self.cell_size)
        center_y = int((position[1] + self.world_size/2) / self.cell_size)
        
        # 安全距离 = 底盘半径 + 额外安全边距
        safety_margin = COVERAGE_MARK_RADIUS + 0.2
        
        if shape_type == 'sphere':
            radius = int((size[0] + safety_margin) / self.cell_size)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx*dx + dy*dy <= radius*radius:
                        x, y = center_x + dx, center_y + dy
                        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                            self.obstacle_grid[x, y] = True
        else:
            half_x = int((size[0]/2 + safety_margin) / self.cell_size)
            half_y = int((size[1]/2 + safety_margin) / self.cell_size)
            for dx in range(-half_x, half_x + 1):
                for dy in range(-half_y, half_y + 1):
                    x, y = center_x + dx, center_y + dy
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        self.obstacle_grid[x, y] = True
    
    def add_objects(self, objects_list: List[np.ndarray]):
        """添加待收集物体位置"""
        self.objects_positions = objects_list
        print(f"📦 添加 {len(objects_list)} 个待收集物体")
    
    def generate_coverage_path(self, start_pos: np.ndarray) -> List[CoverageSegment]:
        """生成智能覆盖路径"""
        print("🌀 生成覆盖路径...")
        
        # 获取有效覆盖区域
        valid_cells = self._get_valid_coverage_cells()
        print(f"   有效覆盖单元格: {len(valid_cells)}")
        
        # 生成蛇形覆盖路径
        coverage_segments = self._generate_serpentine_path(start_pos, valid_cells)
        
        # 优化路径，集成物体收集
        optimized_segments = self._optimize_with_object_collection(coverage_segments)
        
        total_points = sum(len(seg.points) for seg in optimized_segments)
        print(f"   生成覆盖段: {len(optimized_segments)}, 总点数: {total_points}")
        
        return optimized_segments
    
    def _get_valid_coverage_cells(self) -> List[Tuple[int, int]]:
        """获取有效的覆盖单元格"""
        valid_cells = []
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if not self.obstacle_grid[x, y]:
                    # 检查是否在工作区域内
                    world_x = (x * self.cell_size) - self.world_size/2
                    world_y = (y * self.cell_size) - self.world_size/2
                    
                    if abs(world_x) < self.world_size/2 - 0.5 and abs(world_y) < self.world_size/2 - 0.5:
                        valid_cells.append((x, y))
        
        return valid_cells
    
    def _generate_serpentine_path(self, start_pos: np.ndarray, valid_cells: List[Tuple[int, int]]) -> List[CoverageSegment]:
        """生成蛇形覆盖路径"""
        segments = []
        
        # 按Y坐标分组
        rows = {}
        for x, y in valid_cells:
            if y not in rows:
                rows[y] = []
            rows[y].append(x)
        
        # 排序行
        sorted_rows = sorted(rows.keys())
        
        # 生成蛇形路径
        for i, row_y in enumerate(sorted_rows):
            row_x_coords = sorted(rows[row_y])
            
            # 奇数行反向
            if i % 2 == 1:
                row_x_coords.reverse()
            
            # 创建该行的覆盖点
            row_points = []
            for x in row_x_coords:
                world_pos = self._grid_to_world(x, row_y)
                
                # 计算朝向
                if len(row_points) > 0:
                    direction = world_pos[:2] - row_points[-1].position[:2]
                    orientation = np.arctan2(direction[1], direction[0])
                else:
                    orientation = 0.0 if i % 2 == 0 else np.pi
                
                coverage_point = CoveragePoint(
                    position=world_pos,
                    orientation=orientation,
                    coverage_priority=1.0,
                    node_id=len(row_points)
                )
                row_points.append(coverage_point)
            
            # 创建覆盖段
            segment = CoverageSegment(
                points=row_points,
                segment_type="main_line",
                priority=1.0
            )
            segments.append(segment)
            
            # 添加转弯段（除了最后一行）
            if i < len(sorted_rows) - 1:
                turn_segment = self._create_turn_segment(row_points[-1], sorted_rows[i+1])
                segments.append(turn_segment)
        
        return segments
    
    def _create_turn_segment(self, last_point: CoveragePoint, next_row_y: int) -> CoverageSegment:
        """创建转弯段"""
        # 简单的转弯：停留在当前位置调整朝向
        turn_point = CoveragePoint(
            position=last_point.position.copy(),
            orientation=last_point.orientation + np.pi/2,  # 转90度
            coverage_priority=0.5,
            node_id=0
        )
        
        return CoverageSegment(
            points=[turn_point],
            segment_type="turn",
            priority=0.5
        )
    
    def _optimize_with_object_collection(self, segments: List[CoverageSegment]) -> List[CoverageSegment]:
        """优化路径以集成物体收集"""
        optimized_segments = []
        
        for segment in segments:
            # 检查段内是否有物体需要收集
            objects_in_segment = self._find_objects_near_segment(segment)
            
            if objects_in_segment:
                # 在段内添加物体收集点
                enhanced_segment = self._enhance_segment_with_objects(segment, objects_in_segment)
                optimized_segments.append(enhanced_segment)
            else:
                optimized_segments.append(segment)
        
        return optimized_segments
    
    def _find_objects_near_segment(self, segment: CoverageSegment) -> List[np.ndarray]:
        """查找段附近的物体 - 基于底盘半径"""
        nearby_objects = []
        
        for obj_pos in self.objects_positions:
            for point in segment.points:
                distance = np.linalg.norm(point.position[:2] - obj_pos[:2])
                if distance < COVERAGE_MARK_RADIUS * 1.5:  # 扩大检测范围到底盘半径的1.5倍
                    nearby_objects.append(obj_pos)
                    break
        
        return nearby_objects
    
    def _enhance_segment_with_objects(self, segment: CoverageSegment, objects: List[np.ndarray]) -> CoverageSegment:
        """在段中增强物体收集 - 基于底盘半径"""
        enhanced_points = []
        
        for point in segment.points:
            enhanced_points.append(point)
            
            # 检查附近是否有物体 - 使用底盘半径作为收集距离
            for obj_pos in objects:
                distance = np.linalg.norm(point.position[:2] - obj_pos[:2])
                if distance < COVERAGE_MARK_RADIUS:  # 使用底盘半径作为收集距离
                    # 标记该点有物体
                    point.has_object = True
                    point.object_type = "collectible"
                    point.coverage_priority = 2.0  # 提高优先级
        
        segment.points = enhanced_points
        return segment
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """网格坐标转世界坐标"""
        x = (grid_x * self.cell_size) - self.world_size/2
        y = (grid_y * self.cell_size) - self.world_size/2
        return np.array([x, y, 0.0])

class DynamicPathVisualizer:
    """动态路径可视化器 - 适配覆盖算法"""
    
    def __init__(self, world: World):
        self.world = world
        self.current_strategy = "ghost"
        self.memory_threshold = MEMORY_THRESHOLD_MB
        
        # 虚影相关
        self.robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm2.usd"
        self.ghost_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm3.usd"
        self.ghost_container_path = "/World/GhostVisualization"
        
        # 线条相关
        self.line_container_path = "/World/PathLines"
        self.line_prims = []
        
        print(f"🎨 动态路径可视化器初始化（覆盖模式）")
    
    def check_memory_and_decide_strategy(self) -> str:
        """检查内存并决定策略"""
        current_memory = print_memory_usage("策略检查")
        
        if current_memory > self.memory_threshold:
            if self.current_strategy == "ghost":
                print(f"🔄 内存超阈值，切换到线条显示")
                self.current_strategy = "line"
                self._clear_ghosts()
        else:
            if self.current_strategy == "line":
                print(f"🔄 内存充足，切换到虚影显示")
                self.current_strategy = "ghost"
                self._clear_lines()
        
        return self.current_strategy
    
    def visualize_coverage_segment(self, segment_index: int, segment: CoverageSegment):
        """可视化覆盖段"""
        strategy = self.check_memory_and_decide_strategy()
        
        print(f"🎨 段{segment_index} 使用策略: {strategy}")
        
        if strategy == "ghost":
            self._create_ghost_visualization_for_segment(segment_index, segment)
        else:
            self._create_line_visualization_for_segment(segment_index, segment)
    
    def _create_ghost_visualization_for_segment(self, segment_index: int, segment: CoverageSegment):
        """为覆盖段创建虚影可视化"""
        print(f"👻 创建段{segment_index}虚影...")
        
        # 清理旧容器
        self._delete_entire_container(self.ghost_container_path)
        
        # 创建新容器
        stage = self.world.stage
        stage.DefinePrim(self.ghost_container_path, "Xform")
        
        # 选择关键点创建虚影
        selected_points = self._select_key_points(segment.points, GHOSTS_PER_SEGMENT)
        
        print(f"   段长度: {len(segment.points)}点 → 虚影数量: {len(selected_points)}个")
        
        # 创建虚影
        for i, point in enumerate(selected_points):
            ghost_path = f"{self.ghost_container_path}/Segment_{segment_index}_Ghost_{i}"
            self._create_single_ghost(ghost_path, point)
            self.world.step(render=False)
    
    def _create_line_visualization_for_segment(self, segment_index: int, segment: CoverageSegment):
        """为覆盖段创建线条可视化"""
        print(f"📏 创建段{segment_index}线条...")
        
        # 清理旧线条
        self._clear_lines()
        
        # 创建线条容器
        stage = self.world.stage
        if not stage.GetPrimAtPath(self.line_container_path):
            stage.DefinePrim(self.line_container_path, "Xform")
        
        # 创建覆盖路径线条
        self._create_coverage_path_lines(segment_index, segment)
        
        # 创建物体标记
        self._create_object_markers(segment_index, segment)
    
    def _create_coverage_path_lines(self, segment_index: int, segment: CoverageSegment):
        """创建覆盖路径线条"""
        if len(segment.points) < 2:
            return
        
        stage = self.world.stage
        line_path = f"{self.line_container_path}/CoverageLine_{segment_index}"
        
        # 创建线条几何
        line_prim = stage.DefinePrim(line_path, "BasisCurves")
        line_geom = UsdGeom.BasisCurves(line_prim)
        
        line_geom.CreateTypeAttr().Set("linear")
        line_geom.CreateBasisAttr().Set("bspline")
        
        # 构建覆盖路径点
        points = []
        for point in segment.points:
            pos_ground = Gf.Vec3f(float(point.position[0]), float(point.position[1]), 0.02)
            points.append(pos_ground)
        
        # 设置几何数据
        line_geom.CreatePointsAttr().Set(points)
        line_geom.CreateCurveVertexCountsAttr().Set([len(points)])
        line_geom.CreateWidthsAttr().Set([0.03] * len(points))  # 细线条
        
        # 设置覆盖路径颜色
        color = [0.2, 1.0, 0.2] if segment.segment_type == "main_line" else [1.0, 0.8, 0.2]
        self._setup_line_material(line_prim, segment_index, color)
        
        self.line_prims.append(line_path)
        print(f"   创建覆盖路径线条: {len(points)}点")
    
    def _create_object_markers(self, segment_index: int, segment: CoverageSegment):
        """创建物体标记"""
        stage = self.world.stage
        
        for i, point in enumerate(segment.points):
            if point.has_object:
                marker_path = f"{self.line_container_path}/ObjectMarker_{segment_index}_{i}"
                
                # 创建物体标记
                marker_prim = stage.DefinePrim(marker_path, "Sphere")
                sphere_geom = UsdGeom.Sphere(marker_prim)
                sphere_geom.CreateRadiusAttr().Set(0.08)
                
                # 设置位置
                marker_pos = Gf.Vec3d(float(point.position[0]), float(point.position[1]), 0.05)
                xform = UsdGeom.Xformable(marker_prim)
                translate_op = xform.AddTranslateOp()
                translate_op.Set(marker_pos)
                
                # 设置亮红色表示有物体
                gprim = UsdGeom.Gprim(marker_prim)
                gprim.CreateDisplayColorAttr().Set([[1.0, 0.3, 0.3]])
                
                self.line_prims.append(marker_path)
    
    def _select_key_points(self, points: List[CoveragePoint], count: int) -> List[CoveragePoint]:
        """选择关键覆盖点"""
        if len(points) <= count:
            return points
        
        # 优先选择有物体的点
        object_points = [p for p in points if p.has_object]
        regular_points = [p for p in points if not p.has_object]
        
        selected = []
        
        # 先添加物体点
        selected.extend(object_points[:count//2])
        
        # 再均匀选择常规点
        remaining_count = count - len(selected)
        if remaining_count > 0 and regular_points:
            step = max(1, len(regular_points) // remaining_count)
            for i in range(0, len(regular_points), step):
                if len(selected) < count:
                    selected.append(regular_points[i])
        
        return selected[:count]
    
    def _create_single_ghost(self, ghost_path: str, point: CoveragePoint):
        """创建单个虚影"""
        stage = self.world.stage
        
        # 创建虚影prim
        ghost_prim = stage.DefinePrim(ghost_path, "Xform")
        
        # 添加引用
        references = ghost_prim.GetReferences()
        references.AddReference(self.ghost_usd_path)
        
        # 设置变换
        ghost_position = Gf.Vec3d(float(point.position[0]), float(point.position[1]), float(point.position[2]))
        yaw_degrees = float(np.degrees(point.orientation))
        
        xform = UsdGeom.Xformable(ghost_prim)
        xform.ClearXformOpOrder()
        
        translate_op = xform.AddTranslateOp()
        translate_op.Set(ghost_position)
        
        if abs(yaw_degrees) > 1.0:
            rotate_op = xform.AddRotateZOp()
            rotate_op.Set(yaw_degrees)
    
    def _setup_line_material(self, line_prim, segment_index: int, color: List[float]):
        """设置线条材质"""
        material_path = f"/World/Materials/CoverageMaterial_{segment_index}"
        stage = self.world.stage
        
        material_prim = stage.DefinePrim(material_path, "Material")
        material = UsdShade.Material(material_prim)
        
        shader_prim = stage.DefinePrim(f"{material_path}/Shader", "Shader")
        shader = UsdShade.Shader(shader_prim)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(tuple(color))
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(tuple([c*0.3 for c in color]))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        
        material_output = material.CreateSurfaceOutput()
        shader_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material_output.ConnectToSource(shader_output)
        
        UsdShade.MaterialBindingAPI(line_prim).Bind(material)
    
    def _clear_ghosts(self):
        """清除虚影"""
        self._delete_entire_container(self.ghost_container_path)
    
    def _clear_lines(self):
        """清除线条"""
        stage = self.world.stage
        
        for line_path in self.line_prims:
            if stage.GetPrimAtPath(line_path):
                stage.RemovePrim(line_path)
        
        self.line_prims.clear()
        
        if stage.GetPrimAtPath(self.line_container_path):
            stage.RemovePrim(self.line_container_path)
        
        # 清理材质
        materials_path = "/World/Materials"
        if stage.GetPrimAtPath(materials_path):
            materials_prim = stage.GetPrimAtPath(materials_path)
            for child in materials_prim.GetChildren():
                if "CoverageMaterial" in str(child.GetPath()):
                    stage.RemovePrim(child.GetPath())
    
    def _delete_entire_container(self, container_path: str):
        """删除整个容器"""
        stage = self.world.stage
        
        if stage.GetPrimAtPath(container_path):
            stage.RemovePrim(container_path)
            for _ in range(3):
                self.world.step(render=False)
    
    def cleanup_all(self):
        """清理所有资源"""
        print("🧹 清理可视化资源...")
        self._clear_ghosts()
        self._clear_lines()

class StabilizedRobotController:
    """稳定机器人控制器 - 覆盖模式"""
    
    def __init__(self, mobile_base, differential_controller):
        self.mobile_base = mobile_base
        self.differential_controller = differential_controller
        self.max_linear_velocity = MAX_LINEAR_VELOCITY  
        self.max_angular_velocity = MAX_ANGULAR_VELOCITY
        
        self.velocity_filter = deque(maxlen=3)
        self.angular_filter = deque(maxlen=3)
        
        print("🎮 稳定控制器初始化（覆盖模式）")
    
    def send_coverage_command(self, target_linear_vel: float, target_angular_vel: float):
        """发送覆盖移动命令"""
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
        
        # 覆盖模式的对称性控制
        if abs(angular_vel) < 0.1:
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

class FluentCoverageRobotSystem:
    """流畅覆盖算法机器人系统"""
    
    def __init__(self):
        self.world = None
        self.robot_prim_path = "/World/create3_robot"
        self.mobile_base = None
        self.differential_controller = None
        self.stabilized_controller = None
        
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = 0.0
        
        self.collectible_objects = []
        self.collected_objects = []
        
        self.coverage_planner = None
        self.path_visualizer = None
        self.coverage_visualizer = None
        
        self.coverage_segments = []
        
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
        print("🚀 初始化流畅覆盖机器人系统...")
        
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
            color=np.array([0.4, 0.4, 0.4])
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
        self.coverage_planner = CoveragePathPlanner(world_size=COVERAGE_AREA_SIZE, cell_size=COVERAGE_CELL_SIZE)
        self.path_visualizer = DynamicPathVisualizer(self.world)
        self.coverage_visualizer = FluentCoverageVisualizer(self.world)  # 使用流畅可视化器
        self._add_environment_obstacles()
    
    def _add_environment_obstacles(self):
        """添加环境障碍物"""
        obstacles = [
            {"pos": [1.2, 0.8, 0.15], "size": [0.3, 0.3, 0.3], "color": [0.6, 0.3, 0.1], "name": "cylinder1", "shape": "cylinder"},
            {"pos": [0.5, -1.5, 0.1], "size": [1.2, 0.2, 0.2], "color": [0.7, 0.7, 0.7], "name": "wall1", "shape": "box"},
            {"pos": [-1.0, 1.2, 0.4], "size": [0.1, 0.8, 0.1], "color": [0.8, 0.2, 0.2], "name": "pole1", "shape": "box"},
            {"pos": [-0.8, -1.0, 0.15], "size": [0.3, 0.3, 0.3], "color": [0.9, 0.5, 0.1], "name": "sphere1", "shape": "sphere"},
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
            self.coverage_planner.add_obstacle(
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
            kd[idx] = 400.0
        
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
        
        for _ in range(15):
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
        
        for _ in range(8):
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
    
    def create_collectible_environment(self):
        """创建可收集物体环境"""
        print("📦 创建可收集物体环境...")
        
        # 分布在覆盖区域内的物体
        object_positions = [
            [1.5, 0.5, 0.03], [2.0, 1.8, 0.03], [-0.5, 1.5, 0.03],
            [0.8, -0.8, 0.03], [-1.5, 0.2, 0.03], [1.8, -1.2, 0.03],
            [-0.2, -1.8, 0.03], [2.2, 0.0, 0.03]
        ]
        
        for i, pos in enumerate(object_positions):
            # 随机选择物体类型
            if i % 2 == 0:
                # 小球类物体
                obj = DynamicSphere(
                    prim_path=f"/World/collectible_{i}",
                    name=f"collectible_{i}",
                    position=np.array(pos),
                    radius=0.04,
                    color=np.array([0.2, 0.8, 0.2])
                )
            else:
                # 立方体类物体
                obj = DynamicCuboid(
                    prim_path=f"/World/collectible_{i}",
                    name=f"collectible_{i}",
                    position=np.array(pos),
                    scale=np.array([0.06, 0.06, 0.06]),
                    color=np.array([0.8, 0.2, 0.8])
                )
            
            self.world.scene.add(obj)
            self.collectible_objects.append(obj)
        
        # 将物体位置传递给路径规划器
        object_world_positions = [obj.get_world_pose()[0] for obj in self.collectible_objects]
        self.coverage_planner.add_objects(object_world_positions)
        
        print(f"✅ 环境创建完成: 可收集物体{len(self.collectible_objects)}个")
        print_memory_usage("物体环境创建完成")
    
    def plan_coverage_mission(self):
        """覆盖任务规划"""
        print("\n🌀 开始覆盖任务规划...")
        
        current_pos, _ = self.get_robot_pose()
        
        # 生成覆盖路径
        self.coverage_segments = self.coverage_planner.generate_coverage_path(current_pos)
        
        total_points = sum(len(seg.points) for seg in self.coverage_segments)
        print(f"✅ 覆盖规划完成: {len(self.coverage_segments)}个段, {total_points}个点")
        print(f"🎨 流畅可视化: 精细网格{FINE_GRID_SIZE}m, 实时跟随机器人移动")
        print_memory_usage("覆盖规划完成")
    
    def execute_coverage_mission(self):
        """执行覆盖任务"""
        print("\n🚀 开始执行流畅覆盖任务...")
        print_memory_usage("任务开始前")
        
        for segment_index, segment in enumerate(self.coverage_segments):
            print(f"\n🌀 执行覆盖段 {segment_index}: {segment.segment_type}")
            print(f"   段点数: {len(segment.points)}")
            
            # 可视化当前段
            print(f"🎨 ====== 段{segment_index}路径可视化开始 ======")
            self.path_visualizer.visualize_coverage_segment(segment_index, segment)
            print(f"🎨 ====== 段{segment_index}路径可视化完成 ======")
            
            # 展示路径
            display_steps = GHOST_DISPLAY_STEPS if self.path_visualizer.current_strategy == "ghost" else 10
            print(f"👁️ 展示覆盖路径 ({display_steps}步)...")
            for step in range(display_steps):
                self.world.step(render=True)
            
            # 执行覆盖段
            print(f"🏃 执行流畅覆盖移动...")
            self._execute_fluent_coverage_segment(segment)
            
            # 清除可视化
            print(f"🧹 清理段{segment_index}可视化...")
            self.path_visualizer._clear_ghosts()
            self.path_visualizer._clear_lines()
            
            # 垃圾回收
            for i in range(2):
                gc.collect()
                self.world.step(render=False)
            
            print(f"✅ 流畅覆盖段 {segment_index} 完成")
        
        print("\n🎉 流畅覆盖任务执行完成!")
        self._show_fluent_coverage_results()
    
    def _execute_fluent_coverage_segment(self, segment: CoverageSegment):
        """执行流畅覆盖段"""
        for i, point in enumerate(segment.points):
            # 导航到覆盖点
            success = self._navigate_to_coverage_point(point)
            
            current_pos, _ = self.get_robot_pose()
            
            # 实时流畅标记覆盖区域
            self.coverage_visualizer.mark_coverage_realtime(current_pos)
            
            # 检查是否有物体需要收集
            self._check_and_collect_nearby_objects(current_pos)
            
            # 进度显示
            if i % 3 == 0:
                progress = (i / len(segment.points)) * 100
                print(f"   流畅覆盖进度: {progress:.1f}%")
    
    def _navigate_to_coverage_point(self, point: CoveragePoint) -> bool:
        """导航到覆盖点 - 流畅移动优化"""
        # 更精细的容差，配合流畅可视化
        tolerance = FINE_GRID_SIZE  # 使用精细网格大小作为容差
        max_steps = 150
        step_counter = 0
        
        while step_counter < max_steps:
            current_pos, current_yaw = self.get_robot_pose()
            step_counter += 1
            
            # 实时更新流畅覆盖标记
            self.coverage_visualizer.mark_coverage_realtime(current_pos)
            
            # 检查到达
            distance = np.linalg.norm(current_pos[:2] - point.position[:2])
            if distance < tolerance:
                self.stabilized_controller.send_coverage_command(0.0, 0.0)
                return True
            
            # 计算控制量
            direction = point.position[:2] - current_pos[:2]
            target_angle = np.arctan2(direction[1], direction[0])
            angle_diff = target_angle - current_yaw
            
            # 角度归一化
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # 流畅覆盖移动控制策略
            if abs(angle_diff) > 0.15:
                linear_vel = 0.0
                angular_vel = np.clip(angle_diff * 4.0, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
            else:
                linear_vel = min(MAX_LINEAR_VELOCITY, max(0.05, distance * 0.8))
                angular_vel = np.clip(angle_diff * 2.0, -1.5, 1.5)
            
            self.stabilized_controller.send_coverage_command(linear_vel, angular_vel)
            self.world.step(render=True)
        
        # 超时停止
        self.stabilized_controller.send_coverage_command(0.0, 0.0)
        return False
    
    def _check_and_collect_nearby_objects(self, robot_pos: np.ndarray):
        """检查并收集附近物体 - 基于实际底盘半径"""
        for obj in self.collectible_objects:
            if obj.name in self.collected_objects:
                continue
                
            obj_pos = obj.get_world_pose()[0]
            distance = np.linalg.norm(robot_pos[:2] - obj_pos[:2])
            
            # 使用底盘半径作为收集距离
            if distance < COVERAGE_MARK_RADIUS:
                print(f"🎯 收集物体: {obj.name} (距离: {distance:.2f}m)")
                self._collect_object(obj)
                self.collected_objects.append(obj.name)
                break
    
    def _collect_object(self, obj):
        """收集物体"""
        # 快速机械臂动作
        self._move_arm_to_pose("ready")
        self._control_gripper("open")
        self._control_gripper("close")
        self._move_arm_to_pose("carry")
        
        # 隐藏物体
        obj.disable_rigid_body_physics()
        far_away_position = np.array([100.0, 100.0, -5.0])
        obj.set_world_pose(far_away_position, np.array([0, 0, 0, 1]))
        obj.set_visibility(False)
        
        for _ in range(2):
            self.world.step(render=False)
    
    def _show_fluent_coverage_results(self):
        """显示流畅覆盖结果"""
        total_objects = len(self.collectible_objects)
        collected_count = len(self.collected_objects)
        collection_rate = (collected_count / total_objects) * 100
        
        coverage_count = len(self.coverage_visualizer.coverage_marks)
        total_points = sum(len(seg.points) for seg in self.coverage_segments)
        
        print(f"\n📊 流畅覆盖任务执行结果:")
        print(f"   覆盖段数: {len(self.coverage_segments)}")
        print(f"   总覆盖点: {total_points}")
        print(f"   流畅标记区域: {coverage_count}")
        print(f"   总物体数: {total_objects}")
        print(f"   成功收集: {collected_count}")
        print(f"   收集率: {collection_rate:.1f}%")
        print(f"🤖 底盘参数: 半径{COVERAGE_MARK_RADIUS}m, 直径{COVERAGE_MARK_RADIUS*2}m")
        print(f"🌀 覆盖算法: 智能蛇形覆盖，网格{COVERAGE_CELL_SIZE}m")
        print(f"🎨 流畅可视化: 精细网格{FINE_GRID_SIZE}m，实时跟随机器人移动")
        print(f"✨ 流畅标记: 绿色渐变圆盘，颗粒度精细，连贯流畅")
        print(f"🤖 智能结合: 流畅覆盖移动 + 实时标记 + 物体收集")
        print(f"📏 路径优化: 避免过度重叠，提高覆盖效率")
        print(f"✅ 高效完成流畅区域覆盖任务")
    
    def run_fluent_coverage_demo(self):
        """运行流畅覆盖演示"""
        print("\n" + "="*80)
        print("🌀 流畅覆盖算法机器人系统 - Isaac Sim 4.5")
        print("🤖 流畅覆盖 | 📦 智能收集 | 🎨 实时标记 | ⚡ 精细高效")
        print(f"🔧 底盘半径: {COVERAGE_MARK_RADIUS}m | 精细网格: {FINE_GRID_SIZE}m")
        print("="*80)
        
        pos, yaw = self.get_robot_pose()
        print(f"📍 初始位置: [{pos[0]:.3f}, {pos[1]:.3f}], 朝向: {np.degrees(yaw):.1f}°")
        
        self.plan_coverage_mission()
        self.execute_coverage_mission()
        
        self._move_arm_to_pose("home")
        
        print("\n🎉 流畅覆盖系统演示完成!")
        print("💡 成功实现流畅区域覆盖与物体收集的智能结合")
        print(f"🎨 流畅标记实时跟随机器人，精细网格{FINE_GRID_SIZE}m")
    
    def cleanup(self):
        """清理资源"""
        print("🧹 清理系统资源...")
        print_memory_usage("最终清理前")
        
        if self.path_visualizer is not None:
            self.path_visualizer.cleanup_all()
        
        if self.coverage_visualizer is not None:
            self.coverage_visualizer.cleanup()
            
        for i in range(8):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if self.world is not None:
            for _ in range(5):
                self.world.step(render=False)
            self.world.stop()
        
        for i in range(3):
            gc.collect()
        
        print_memory_usage("最终清理后")
        print("✅ 资源清理完成")

def main():
    """主函数"""
    print("🌀 启动流畅覆盖算法机器人系统...")
    print(f"⚙️ 底盘参数: 半径={COVERAGE_MARK_RADIUS}m, 直径={COVERAGE_MARK_RADIUS*2}m")
    print(f"⚙️ 覆盖参数: 网格={COVERAGE_CELL_SIZE}m, 区域={COVERAGE_AREA_SIZE}m")
    print(f"⚙️ 流畅参数: 精细网格={FINE_GRID_SIZE}m, 更新频率={COVERAGE_UPDATE_FREQUENCY}")
    print(f"⚙️ 运动参数: 线速度={MAX_LINEAR_VELOCITY}m/s, 角速度={MAX_ANGULAR_VELOCITY}rad/s")
    print(f"🎨 流畅可视化: 绿色渐变圆盘标记，实时跟随机器人移动，颗粒度精细")
    print(f"🤖 智能算法: 蛇形覆盖 + 流畅标记 + 机会式物体收集")
    print(f"📏 路径优化: 适配{COVERAGE_MARK_RADIUS}m底盘，避免过度重叠，连贯流畅")
    
    system = FluentCoverageRobotSystem()
    
    system.initialize_system()
    system.initialize_robot()
    system.setup_post_load()
    system.create_collectible_environment()
    
    # 稳定系统
    print("⚡ 系统稳定中...")
    for _ in range(STABILIZE_STEPS):
        system.world.step(render=False)
        time.sleep(0.01)
    
    # 运行演示
    system.run_fluent_coverage_demo()
    
    # 保持运行用于观察
    print("\n💡 系统运行中，观察流畅覆盖效果...")
    for i in range(50):
        system.world.step(render=True)
        time.sleep(0.1)
    
    system.cleanup()

if __name__ == "__main__":
    main()