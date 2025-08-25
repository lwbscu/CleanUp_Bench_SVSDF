#!/usr/bin/env python3
"""
路径规划模块
包含四类对象路径规划器，实现高效弓字形避障算法
"""

import numpy as np
import math
from typing import List, Tuple, Optional
from data_structures import *

class FourObjectPathPlanner:
    """四类对象路径规划器 - 高效弓字形避障优化版"""
    
    def __init__(self):
        self.world_size = COVERAGE_AREA_SIZE
        self.cell_size = COVERAGE_CELL_SIZE
        self.grid_size = int(self.world_size / self.cell_size)
        self.robot_radius = ROBOT_RADIUS
        self.safety_margin = SAFETY_MARGIN
        
        # 高分辨率障碍地图用于精确路径检查
        self.fine_grid_size = self.grid_size * 4  # 4倍精度
        self.fine_cell_size = self.world_size / self.fine_grid_size
        
        self.obstacle_map = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        self.fine_obstacle_map = np.zeros((self.fine_grid_size, self.fine_grid_size), dtype=bool)
        self.scene_objects = []
        
        print(f"高效弓字形路径规划器初始化:")
        print(f"  基础网格: {self.grid_size}x{self.grid_size}")
        print(f"  精细网格: {self.fine_grid_size}x{self.fine_grid_size}")
        print(f"  机器人半径: {self.robot_radius}m")
        print(f"  安全边距: {self.safety_margin}m")
    
    def add_scene_object(self, scene_obj: SceneObject):
        """添加场景对象"""
        self.scene_objects.append(scene_obj)
        
        # 只有障碍物需要加入碰撞地图用于路径规划避障
        if scene_obj.object_type == ObjectType.OBSTACLE:
            self._mark_collision_cells(scene_obj)
        
        print(f"添加{scene_obj.object_type.value}对象: {scene_obj.name}")
    
    def _mark_collision_cells(self, scene_obj: SceneObject):
        """标记碰撞单元格 - 双分辨率标记"""
        boundary = scene_obj.collision_boundary
        center = boundary.center
        
        expansion = self.robot_radius + self.safety_margin
        
        if boundary.shape_type == 'sphere':
            radius = boundary.dimensions[0] + expansion
            self._mark_circular_area(center, radius, self.obstacle_map, self.grid_size, self.cell_size)
            self._mark_circular_area(center, radius, self.fine_obstacle_map, self.fine_grid_size, self.fine_cell_size)
        
        elif boundary.shape_type == 'box':
            half_w = boundary.dimensions[0] / 2 + expansion
            half_l = boundary.dimensions[1] / 2 + expansion
            self._mark_rectangular_area(center, half_w, half_l, boundary.rotation, 
                                      self.obstacle_map, self.grid_size, self.cell_size)
            self._mark_rectangular_area(center, half_w, half_l, boundary.rotation,
                                      self.fine_obstacle_map, self.fine_grid_size, self.fine_cell_size)
        
        elif boundary.shape_type == 'cylinder':
            radius = boundary.dimensions[0] + expansion
            self._mark_circular_area(center, radius, self.obstacle_map, self.grid_size, self.cell_size)
            self._mark_circular_area(center, radius, self.fine_obstacle_map, self.fine_grid_size, self.fine_cell_size)
    
    def _mark_circular_area(self, center: np.ndarray, radius: float, grid_map: np.ndarray, 
                           grid_size: int, cell_size: float):
        """标记圆形区域"""
        center_x = int((center[0] + self.world_size/2) / cell_size)
        center_y = int((center[1] + self.world_size/2) / cell_size)
        
        grid_radius = int(radius / cell_size) + 1
        
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                gx, gy = center_x + dx, center_y + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    world_x = (gx * cell_size) - self.world_size/2
                    world_y = (gy * cell_size) - self.world_size/2
                    
                    distance = np.linalg.norm([world_x - center[0], world_y - center[1]])
                    if distance <= radius:
                        grid_map[gx, gy] = True
    
    def _mark_rectangular_area(self, center: np.ndarray, half_w: float, half_l: float, 
                             rotation: float, grid_map: np.ndarray, grid_size: int, cell_size: float):
        """标记矩形区域（考虑旋转）"""
        center_x = int((center[0] + self.world_size/2) / cell_size)
        center_y = int((center[1] + self.world_size/2) / cell_size)
        
        max_extent = max(half_w, half_l) * 1.42  # sqrt(2) approximation
        grid_extent = int(max_extent / cell_size) + 1
        
        cos_rot = np.cos(-rotation)
        sin_rot = np.sin(-rotation)
        
        for dx in range(-grid_extent, grid_extent + 1):
            for dy in range(-grid_extent, grid_extent + 1):
                gx, gy = center_x + dx, center_y + dy
                if 0 <= gx < grid_size and 0 <= gy < grid_size:
                    world_x = (gx * cell_size) - self.world_size/2
                    world_y = (gy * cell_size) - self.world_size/2
                    
                    rel_x = world_x - center[0]
                    rel_y = world_y - center[1]
                    
                    local_x = cos_rot * rel_x - sin_rot * rel_y
                    local_y = sin_rot * rel_x + cos_rot * rel_y
                    
                    if abs(local_x) <= half_w and abs(local_y) <= half_l:
                        grid_map[gx, gy] = True
    
    def check_collision_with_object(self, robot_pos: np.ndarray, scene_obj: SceneObject) -> bool:
        """检查机器人与对象的碰撞"""
        if not scene_obj.is_active:
            return False
            
        boundary = scene_obj.collision_boundary
        center = boundary.center
        
        distance_2d = np.linalg.norm(robot_pos[:2] - center[:2])
        
        if boundary.shape_type == 'sphere':
            return distance_2d <= (boundary.dimensions[0] + INTERACTION_DISTANCE)
        
        elif boundary.shape_type == 'box':
            rel_pos = robot_pos[:2] - center[:2]
            cos_rot = np.cos(-boundary.rotation)
            sin_rot = np.sin(-boundary.rotation)
            
            local_x = cos_rot * rel_pos[0] - sin_rot * rel_pos[1]
            local_y = sin_rot * rel_pos[0] + cos_rot * rel_pos[1]
            
            half_w = boundary.dimensions[0] / 2 + INTERACTION_DISTANCE
            half_l = boundary.dimensions[1] / 2 + INTERACTION_DISTANCE
            
            return abs(local_x) <= half_w and abs(local_y) <= half_l
        
        elif boundary.shape_type == 'cylinder':
            return distance_2d <= (boundary.dimensions[0] + INTERACTION_DISTANCE)
        
        return False
    
    def generate_coverage_path(self, start_pos: np.ndarray) -> List[CoveragePoint]:
        """生成高效弓字形覆盖路径"""
        print("生成高效弓字形覆盖路径...")
        
        coverage_points = self._generate_safe_coverage_grid()
        path_points = self._create_efficient_bow_pattern_path(coverage_points, start_pos)
        validated_path = self._validate_and_fix_path_connectivity(path_points)
        
        print(f"弓字形覆盖路径生成完成: {len(validated_path)}个点")
        return validated_path
    
    def _generate_safe_coverage_grid(self) -> List[Tuple[int, int]]:
        """生成安全可覆盖网格点"""
        coverage_points = []
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.obstacle_map[x, y]:
                    continue
                
                if not self._is_safe_boundary_point(x, y):
                    continue
                
                if not self._is_robot_space_clear(x, y):
                    continue
                
                coverage_points.append((x, y))
        
        print(f"生成安全覆盖点: {len(coverage_points)}个")
        return coverage_points
    
    def _is_safe_boundary_point(self, grid_x: int, grid_y: int) -> bool:
        """检查网格点是否距离边界足够远"""
        world_x = (grid_x * self.cell_size) - self.world_size/2
        world_y = (grid_y * self.cell_size) - self.world_size/2
        
        margin = self.robot_radius + self.safety_margin + 0.1
        return (abs(world_x) < self.world_size/2 - margin and 
                abs(world_y) < self.world_size/2 - margin)
    
    def _is_robot_space_clear(self, center_x: int, center_y: int) -> bool:
        """检查机器人占地空间是否完全清空"""
        robot_grid_radius = max(1, int((self.robot_radius + self.safety_margin * 0.5) / self.cell_size))
        
        for dx in range(-robot_grid_radius, robot_grid_radius + 1):
            for dy in range(-robot_grid_radius, robot_grid_radius + 1):
                check_x, check_y = center_x + dx, center_y + dy
                
                if not (0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size):
                    continue
                
                world_center_x = (center_x * self.cell_size) - self.world_size/2
                world_center_y = (center_y * self.cell_size) - self.world_size/2
                world_check_x = (check_x * self.cell_size) - self.world_size/2
                world_check_y = (check_y * self.cell_size) - self.world_size/2
                
                distance = np.sqrt((world_check_x - world_center_x)**2 + 
                                 (world_check_y - world_center_y)**2)
                
                if distance <= self.robot_radius + self.safety_margin * 0.5 and self.obstacle_map[check_x, check_y]:
                    return False
        
        return True
    
    def _create_efficient_bow_pattern_path(self, coverage_points: List[Tuple[int, int]], 
                                         start_pos: np.ndarray) -> List[CoveragePoint]:
        """创建高效弓字形避障路径 - 核心算法"""
        print("创建高效弓字形避障路径...")
        
        path_points = []
        
        # 按Y坐标分组形成行
        rows = {}
        for x, y in coverage_points:
            if y not in rows:
                rows[y] = []
            rows[y].append(x)
        
        # 按Y坐标排序行
        sorted_row_keys = sorted(rows.keys())
        
        # 处理每一行，实现弓字形模式
        for row_idx, y in enumerate(sorted_row_keys):
            row_x_coords = sorted(rows[y])
            
            # 奇数行反向形成弓字形
            if row_idx % 2 == 1:
                row_x_coords.reverse()
            
            # 找到行中的连续段，跳过障碍物区间
            continuous_segments = self._find_continuous_segments(row_x_coords, y)
            
            # 对每个连续段生成路径点
            for segment in continuous_segments:
                segment_points = self._create_segment_path(segment, y, row_idx % 2 == 1)
                
                # 检查与上一个点的连通性
                if path_points and segment_points:
                    connection_points = self._create_safe_connection(path_points[-1], segment_points[0])
                    path_points.extend(connection_points)
                
                path_points.extend(segment_points)
        
        print(f"弓字形路径创建完成: {len(path_points)}个点")
        return path_points
    
    def _find_continuous_segments(self, x_coords: List[int], y: int) -> List[List[int]]:
        """找到行中的连续可达段"""
        segments = []
        current_segment = []
        
        for i, x in enumerate(x_coords):
            current_segment.append(x)
            
            should_break = False
            
            if i < len(x_coords) - 1:
                next_x = x_coords[i + 1]
                if abs(next_x - x) > 2 or not self._is_segment_connection_safe(x, next_x, y):
                    should_break = True
            else:
                should_break = True
            
            if should_break:
                if len(current_segment) >= 1:
                    segments.append(current_segment.copy())
                current_segment = []
        
        return segments
    
    def _is_segment_connection_safe(self, x1: int, x2: int, y: int) -> bool:
        """检查段内连接是否安全"""
        start_x, end_x = min(x1, x2), max(x1, x2)
        
        for check_x in range(start_x, end_x + 1):
            if self.obstacle_map[check_x, y]:
                return False
        
        return True
    
    def _create_segment_path(self, segment: List[int], y: int, reverse: bool) -> List[CoveragePoint]:
        """为连续段创建路径点"""
        points = []
        
        for x in segment:
            world_pos = self._grid_to_world(x, y)
            
            if reverse:
                orientation = math.pi  # 反向
            else:
                orientation = 0.0  # 正向
            
            has_object = self._has_nearby_target_object(world_pos)
            
            point = CoveragePoint(
                position=world_pos,
                orientation=orientation,
                has_object=has_object
            )
            
            points.append(point)
        
        return points
    
    def _create_safe_connection(self, from_point: CoveragePoint, to_point: CoveragePoint) -> List[CoveragePoint]:
        """创建两点间的安全连接路径"""
        connection_points = []
        
        if self._is_path_safe_high_res(from_point.position, to_point.position):
            return connection_points  # 直接连接，不需要中间点
        
        # 需要绕行，使用90度转弯策略
        mid_points = self._find_90_degree_detour(from_point.position, to_point.position)
        
        for mid_pos in mid_points:
            mid_point = CoveragePoint(
                position=mid_pos,
                orientation=np.arctan2(to_point.position[1] - mid_pos[1], 
                                     to_point.position[0] - mid_pos[0]),
                has_object=False
            )
            connection_points.append(mid_point)
        
        return connection_points
    
    def _find_90_degree_detour(self, start_pos: np.ndarray, end_pos: np.ndarray) -> List[np.ndarray]:
        """找到90度转弯的绕行路径"""
        detour_points = []
        
        option1 = np.array([end_pos[0], start_pos[1], start_pos[2]])  # 先水平
        option2 = np.array([start_pos[0], end_pos[1], start_pos[2]])  # 先垂直
        
        for option in [option1, option2]:
            if (self._is_path_safe_high_res(start_pos, option) and 
                self._is_path_safe_high_res(option, end_pos) and
                self._is_position_safe_high_res(option)):
                detour_points.append(option)
                break
        
        return detour_points
    
    def _is_path_safe_high_res(self, start_pos: np.ndarray, end_pos: np.ndarray) -> bool:
        """使用高分辨率网格检查路径安全性"""
        distance = np.linalg.norm(end_pos[:2] - start_pos[:2])
        steps = max(1, int(distance / (self.fine_cell_size * 0.5)))
        
        for i in range(steps + 1):
            t = i / steps
            check_pos = start_pos + t * (end_pos - start_pos)
            
            if not self._is_position_safe_high_res(check_pos):
                return False
        
        return True
    
    def _is_position_safe_high_res(self, position: np.ndarray) -> bool:
        """使用高分辨率网格检查位置安全性"""
        fine_x = int((position[0] + self.world_size/2) / self.fine_cell_size)
        fine_y = int((position[1] + self.world_size/2) / self.fine_cell_size)
        
        if not (0 <= fine_x < self.fine_grid_size and 0 <= fine_y < self.fine_grid_size):
            return False
        
        robot_fine_radius = max(1, int((self.robot_radius + self.safety_margin * 0.5) / self.fine_cell_size))
        
        for dx in range(-robot_fine_radius, robot_fine_radius + 1):
            for dy in range(-robot_fine_radius, robot_fine_radius + 1):
                check_x, check_y = fine_x + dx, fine_y + dy
                
                if not (0 <= check_x < self.fine_grid_size and 0 <= check_y < self.fine_grid_size):
                    continue
                
                world_center_x = (fine_x * self.fine_cell_size) - self.world_size/2
                world_center_y = (fine_y * self.fine_cell_size) - self.world_size/2
                world_check_x = (check_x * self.fine_cell_size) - self.world_size/2
                world_check_y = (check_y * self.fine_cell_size) - self.world_size/2
                
                distance = np.sqrt((world_check_x - world_center_x)**2 + 
                                 (world_check_y - world_center_y)**2)
                
                if distance <= self.robot_radius + self.safety_margin * 0.5:
                    if self.fine_obstacle_map[check_x, check_y]:
                        return False
        
        return True
    
    def _validate_and_fix_path_connectivity(self, path_points: List[CoveragePoint]) -> List[CoveragePoint]:
        """验证和修复路径连通性"""
        if len(path_points) <= 1:
            return path_points
        
        validated_path = [path_points[0]]
        
        for i in range(1, len(path_points)):
            current = validated_path[-1]
            next_point = path_points[i]
            
            if self._is_path_safe_high_res(current.position, next_point.position):
                validated_path.append(next_point)
            else:
                repair_points = self._repair_connection(current, next_point)
                validated_path.extend(repair_points)
                validated_path.append(next_point)
        
        print(f"路径验证完成: 原{len(path_points)}点 -> 验证{len(validated_path)}点")
        return validated_path
    
    def _repair_connection(self, from_point: CoveragePoint, to_point: CoveragePoint) -> List[CoveragePoint]:
        """修复两点间的连接"""
        repair_path = self._find_safe_path_astar(from_point.position, to_point.position)
        
        repair_points = []
        for pos in repair_path:
            repair_point = CoveragePoint(
                position=pos,
                orientation=np.arctan2(to_point.position[1] - pos[1], 
                                     to_point.position[0] - pos[0]),
                has_object=False
            )
            repair_points.append(repair_point)
        
        return repair_points
    
    def _find_safe_path_astar(self, start_pos: np.ndarray, goal_pos: np.ndarray) -> List[np.ndarray]:
        """使用简化A*算法寻找安全路径"""
        start_grid = (int((start_pos[0] + self.world_size/2) / self.cell_size),
                     int((start_pos[1] + self.world_size/2) / self.cell_size))
        goal_grid = (int((goal_pos[0] + self.world_size/2) / self.cell_size),
                    int((goal_pos[1] + self.world_size/2) / self.cell_size))
        
        open_set = [(0, start_grid, [])]
        closed_set = set()
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 上右下左，90度移动优先
        
        while open_set:
            open_set.sort(key=lambda x: x[0])
            cost, current, path = open_set.pop(0)
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            if current == goal_grid:
                world_path = []
                for grid_pos in path:
                    world_pos = self._grid_to_world(grid_pos[0], grid_pos[1])
                    world_path.append(world_pos)
                return world_path
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (neighbor not in closed_set and 
                    0 <= neighbor[0] < self.grid_size and 
                    0 <= neighbor[1] < self.grid_size and
                    not self.obstacle_map[neighbor[0], neighbor[1]]):
                    
                    new_cost = cost + 1
                    heuristic = abs(neighbor[0] - goal_grid[0]) + abs(neighbor[1] - goal_grid[1])
                    total_cost = new_cost + heuristic
                    
                    new_path = path + [current]
                    open_set.append((total_cost, neighbor, new_path))
        
        return []
    
    def _has_nearby_target_object(self, position: np.ndarray) -> bool:
        """检查附近是否有目标对象"""
        for obj in self.scene_objects:
            if obj.object_type in [ObjectType.SWEEP, ObjectType.GRASP] and obj.is_active:
                distance = np.linalg.norm(position[:2] - obj.position[:2])
                if distance < self.robot_radius * 2:
                    return True
        return False
    
    def _grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        """网格坐标转世界坐标"""
        x = (grid_x * self.cell_size) - self.world_size/2
        y = (grid_y * self.cell_size) - self.world_size/2
        return np.array([x, y, 0.0])
    
    def get_active_objects_by_type(self, obj_type: ObjectType) -> List[SceneObject]:
        """获取指定类型的活跃对象"""
        return [obj for obj in self.scene_objects if obj.object_type == obj_type and obj.is_active]