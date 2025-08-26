#!/usr/bin/env python3
"""
可视化模块
包含超丝滑流畅覆盖可视化器和真实移动可视化器
"""

import numpy as np
from typing import List
from data_structures import *

class FluentCoverageVisualizer:
    """超丝滑流畅覆盖区域可视化器 - 无延迟实时跟踪机器人移动"""
    
    def __init__(self, world):
        from pxr import UsdLux, UsdPhysics, Gf, UsdGeom, UsdShade, Sdf
        
        self.UsdPhysics = UsdPhysics
        self.UsdGeom = UsdGeom
        self.Gf = Gf
        
        self.world = world
        self.coverage_marks = {}  # 精细位置 -> 覆盖次数
        self.mark_prims = {}      # 精细位置 -> prim路径
        self.coverage_container = "/World/CoverageMarks"
        self.last_marked_position = None
        self.mark_counter = 0
        
        print(f"超丝滑流畅覆盖可视化器初始化 - 网格精度: {FINE_GRID_SIZE}m")
    
    def mark_coverage_realtime(self, robot_position: np.ndarray):
        """超丝滑实时标记 - 每次机器人移动都可能创建标记"""
        self.mark_counter += 1
        
        # 使用超精细网格进行丝滑标记
        fine_grid_pos = self._ultra_fine_quantize_position(robot_position)
        
        # 更激进的标记策略 - 距离很小就创建新标记
        if self._should_create_ultra_smooth_mark(fine_grid_pos):
            self._create_ultra_smooth_coverage_mark(fine_grid_pos)
            self.last_marked_position = fine_grid_pos.copy()
    
    def _should_create_ultra_smooth_mark(self, current_pos: np.ndarray) -> bool:
        """判断是否应该创建新标记 - 超敏感距离检测"""
        if self.last_marked_position is None:
            return True
            
        # 使用更小的距离阈值，让标记更连贯
        distance = np.linalg.norm(current_pos[:2] - self.last_marked_position[:2])
        return distance >= MARK_DISTANCE_THRESHOLD
    
    def _ultra_fine_quantize_position(self, position: np.ndarray) -> np.ndarray:
        """超精细量化位置 - 极小网格实现超丝滑效果"""
        # 使用极精细网格，确保标记无缝连接
        x = round(position[0] / FINE_GRID_SIZE) * FINE_GRID_SIZE
        y = round(position[1] / FINE_GRID_SIZE) * FINE_GRID_SIZE
        return np.array([x, y, 0.015])  # 更贴近地面
    
    def _create_ultra_smooth_coverage_mark(self, position: np.ndarray):
        """创建超丝滑的覆盖标记"""
        stage = self.world.stage
        
        # 确保容器存在
        if not stage.GetPrimAtPath(self.coverage_container):
            stage.DefinePrim(self.coverage_container, "Xform")
        
        # 创建唯一标记路径
        mark_id = len(self.coverage_marks)
        x_str = f"{position[0]:.3f}".replace(".", "p").replace("-", "N")
        y_str = f"{position[1]:.3f}".replace(".", "p").replace("-", "N")
        pos_key = f"{x_str}_{y_str}"
        mark_path = f"{self.coverage_container}/SmoothMark_{mark_id}_{pos_key}"
        
        # 记录覆盖
        if pos_key in self.coverage_marks:
            self.coverage_marks[pos_key] += 1
            return  # 已存在，避免重复
        else:
            self.coverage_marks[pos_key] = 1
        
        coverage_count = self.coverage_marks[pos_key]
        
        # 创建超小圆形标记
        mark_prim = stage.DefinePrim(mark_path, "Cylinder")
        cylinder_geom = self.UsdGeom.Cylinder(mark_prim)
        
        # 更小的标记半径，更精细的跟踪
        mark_radius = COVERAGE_MARK_RADIUS * 0.7  # 进一步减小
        cylinder_geom.CreateRadiusAttr().Set(mark_radius)
        cylinder_geom.CreateHeightAttr().Set(0.008)  # 更薄
        
        # 设置位置
        xform = self.UsdGeom.Xformable(mark_prim)
        xform.ClearXformOpOrder()
        translate_op = xform.AddTranslateOp()
        translate_op.Set(self.Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        
        # 简单禁用物理
        try:
            self.UsdPhysics.RigidBodyAPI.Apply(mark_prim)
            rigid_body = self.UsdPhysics.RigidBodyAPI(mark_prim)
            rigid_body.CreateRigidBodyEnabledAttr().Set(False)
        except:
            pass
        
        # 设置5档超精细渐变颜色
        self._set_ultra_smooth_color(mark_prim, coverage_count)
        
        # 记录标记
        self.mark_prims[pos_key] = mark_path
        
        # 更频繁的进度显示
        if len(self.coverage_marks) % 50 == 0:
            print(f"超丝滑覆盖进度: {len(self.coverage_marks)}个超精细标记")
    
    def _set_ultra_smooth_color(self, mark_prim, coverage_count: int):
        """设置5档超精细灰度渐变 - 从极浅灰到深黑"""
        # 5档精细渐变
        coverage_level = min(coverage_count, 5)
        
        # 从0.95(极浅灰)到0.05(深黑)，分5档
        gray_intensity = 0.95 - (coverage_level - 1) * 0.06  # (0.95-0.05)/5 = 0.06
        
        color_value = float(gray_intensity)
        
        gprim = self.UsdGeom.Gprim(mark_prim)
        gprim.CreateDisplayColorAttr().Set([(color_value, color_value, color_value)])
    
    def cleanup(self):
        """清理覆盖标记"""
        stage = self.world.stage
        
        for pos_key, mark_path in self.mark_prims.items():
            if stage.GetPrimAtPath(mark_path):
                stage.RemovePrim(mark_path)
        
        if stage.GetPrimAtPath(self.coverage_container):
            stage.RemovePrim(self.coverage_container)
            
        self.coverage_marks.clear()
        self.mark_prims.clear()
        self.last_marked_position = None
        
        print("超丝滑覆盖标记清理完成")

class RealMovementVisualizer:
    """真实移动可视化器 - 集成超丝滑流畅覆盖可视化"""
    
    def __init__(self, world):
        from pxr import UsdLux, UsdPhysics, Gf, UsdGeom, UsdShade, Sdf
        
        self.UsdPhysics = UsdPhysics
        self.UsdGeom = UsdGeom
        self.UsdShade = UsdShade
        self.Gf = Gf
        self.Sdf = Sdf
        
        self.world = world
        self.stage = world.stage
        
        # USD资产路径
        self.real_robot_usd = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm4.usd"
        self.ghost_robot_usd = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm3.usd"
        
        # 容器路径
        self.path_container = "/World/CompletePath"
        self.ghost_container = "/World/SyncGhosts" 
        
        # 状态变量
        self.all_path_points = []
        self.ghost_robots = []
        
        # 集成超丝滑流畅覆盖可视化器
        self.fluent_coverage_visualizer = FluentCoverageVisualizer(world)
        
        print("初始化真实移动可视化器（集成超丝滑覆盖）")
    
    def setup_complete_path_visualization(self, path_points: List[CoveragePoint]):
        """设置完整路径可视化 - 优化版，避免线条贯穿障碍物"""
        self.all_path_points = path_points
        print(f"设置优化路径可视化: {len(path_points)}个点")
        
        # 清理旧可视化
        self._cleanup_all()
        
        # 创建分段路径线条，避免贯穿
        self._create_segmented_path_lines()
        
        # 创建虚影机器人
        self._create_ghost_robots()
        
        print("优化路径可视化设置完成")
    
    def _create_segmented_path_lines(self):
        """创建分段路径线条，避免贯穿障碍物"""
        self.stage.DefinePrim(self.path_container, "Xform")
        
        if len(self.all_path_points) < 2:
            return
        
        # 将路径分段，每段确保不贯穿障碍物
        path_segments = self._identify_safe_path_segments()
        
        # 为每个安全段创建线条
        for seg_idx, segment in enumerate(path_segments):
            if len(segment) < 2:
                continue
                
            line_path = f"{self.path_container}/PathSegment_{seg_idx}"
            line_prim = self.stage.DefinePrim(line_path, "BasisCurves")
            line_geom = self.UsdGeom.BasisCurves(line_prim)
            
            line_geom.CreateTypeAttr().Set("linear")
            line_geom.CreateBasisAttr().Set("bspline")
            
            # 构建段内路径点
            segment_points = []
            for point_idx in segment:
                point = self.all_path_points[point_idx]
                world_pos = self.Gf.Vec3f(float(point.position[0]), float(point.position[1]), 0.05)
                segment_points.append(world_pos)
            
            line_geom.CreatePointsAttr().Set(segment_points)
            line_geom.CreateCurveVertexCountsAttr().Set([len(segment_points)])
            line_geom.CreateWidthsAttr().Set([0.06] * len(segment_points))
            
            # 设置路径材质
            self._setup_path_material(line_prim)
        
        print(f"创建分段路径线条: {len(path_segments)}段")
    
    def _identify_safe_path_segments(self) -> List[List[int]]:
        """识别安全的路径段，避免贯穿障碍物"""
        segments = []
        current_segment = [0]  # 从第一个点开始
        
        for i in range(1, len(self.all_path_points)):
            prev_point = self.all_path_points[i-1]
            curr_point = self.all_path_points[i]
            
            # 检查两点间连接是否安全（简化检查）
            if self._is_connection_visually_safe(prev_point.position, curr_point.position):
                current_segment.append(i)
            else:
                # 结束当前段，开始新段
                if len(current_segment) >= 2:
                    segments.append(current_segment.copy())
                current_segment = [i-1, i]  # 新段从上一个点开始
        
        # 添加最后一段
        if len(current_segment) >= 2:
            segments.append(current_segment)
        
        return segments
    
    def _is_connection_visually_safe(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """检查两点连线是否视觉安全（简化版，主要检查距离）"""
        distance = np.linalg.norm(pos2[:2] - pos1[:2])
        
        # 距离太大可能跨越障碍物
        if distance > COVERAGE_CELL_SIZE * 3:
            return False
        
        return True
    
    def _create_ghost_robots(self):
        """创建虚影机器人"""
        self.stage.DefinePrim(self.ghost_container, "Xform")
        
        # 选择关键位置放置虚影机器人
        ghost_indices = self._select_ghost_positions()
        
        for i, point_idx in enumerate(ghost_indices):
            point = self.all_path_points[point_idx]
            ghost_path = f"{self.ghost_container}/GhostRobot_{i}"
            
            # 创建虚影机器人
            ghost_prim = self.stage.DefinePrim(ghost_path, "Xform")
            references = ghost_prim.GetReferences()
            references.AddReference(self.ghost_robot_usd)
            
            # 设置位置和朝向
            position = self.Gf.Vec3d(float(point.position[0]), float(point.position[1]), 0.0)
            yaw_degrees = float(np.degrees(point.orientation))
            
            xform = self.UsdGeom.Xformable(ghost_prim)
            xform.ClearXformOpOrder()
            
            translate_op = xform.AddTranslateOp()
            translate_op.Set(position)
            
            if abs(yaw_degrees) > 1.0:
                rotate_op = xform.AddRotateZOp()
                rotate_op.Set(yaw_degrees)
            
            # 禁用物理
            self._disable_ghost_physics_safe(ghost_prim)
            
            # 设置半透明绿色
            self._set_ghost_material(ghost_prim, 0.4)
            
            self.ghost_robots.append(ghost_path)
        
        print(f"创建虚影机器人: {len(ghost_indices)}个")
    
    def _select_ghost_positions(self) -> List[int]:
        """智能选择虚影位置"""
        total_points = len(self.all_path_points)
        max_ghosts = min(MAX_GHOST_ROBOTS, total_points)
        
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
    
    def mark_coverage_realtime(self, robot_position: np.ndarray, step_count: int):
        """超丝滑实时标记覆盖区域 - 无延迟标记"""
        # 调用集成的超丝滑流畅覆盖可视化器
        self.fluent_coverage_visualizer.mark_coverage_realtime(robot_position)
    
    def _disable_ghost_physics_safe(self, root_prim):
        """安全禁用虚影物理属性"""
        def disable_recursive(prim):
            try:
                if prim.IsA(self.UsdGeom.Xformable):
                    if not prim.HasAPI(self.UsdPhysics.RigidBodyAPI):
                        self.UsdPhysics.RigidBodyAPI.Apply(prim)
                    rigid_body = self.UsdPhysics.RigidBodyAPI(prim)
                    rigid_body.CreateRigidBodyEnabledAttr().Set(False)
                    
                    if not prim.HasAPI(self.UsdPhysics.CollisionAPI):
                        self.UsdPhysics.CollisionAPI.Apply(prim)
                    collision = self.UsdPhysics.CollisionAPI(prim)
                    collision.CreateCollisionEnabledAttr().Set(False)
            except:
                pass
            
            for child in prim.GetChildren():
                disable_recursive(child)
        
        disable_recursive(root_prim)
    
    def _set_ghost_material(self, ghost_prim, opacity: float):
        """设置虚影材质"""
        material_path = f"/World/Materials/GhostMaterial_{hash(str(ghost_prim.GetPath())) % 1000}"
        
        material_prim = self.stage.DefinePrim(material_path, "Material")
        material = self.UsdShade.Material(material_prim)
        
        shader_prim = self.stage.DefinePrim(f"{material_path}/Shader", "Shader")
        shader = self.UsdShade.Shader(shader_prim)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set((0.2, 0.9, 0.2))
        shader.CreateInput("emissiveColor", self.Sdf.ValueTypeNames.Color3f).Set((0.1, 0.5, 0.1))
        shader.CreateInput("opacity", self.Sdf.ValueTypeNames.Float).Set(opacity)
        shader.CreateInput("roughness", self.Sdf.ValueTypeNames.Float).Set(0.8)
        
        material_output = material.CreateSurfaceOutput()
        shader_output = shader.CreateOutput("surface", self.Sdf.ValueTypeNames.Token)
        material_output.ConnectToSource(shader_output)
        
        # 应用到mesh
        def apply_material_recursive(prim):
            if prim.GetTypeName() == "Mesh":
                self.UsdShade.MaterialBindingAPI(prim).Bind(material)
            for child in prim.GetChildren():
                apply_material_recursive(child)
        
        apply_material_recursive(ghost_prim)
    
    def _setup_path_material(self, line_prim):
        """设置路径材质"""
        material_path = "/World/Materials/PathMaterial"
        
        material_prim = self.stage.DefinePrim(material_path, "Material")
        material = self.UsdShade.Material(material_prim)
        
        shader_prim = self.stage.DefinePrim(f"{material_path}/Shader", "Shader")
        shader = self.UsdShade.Shader(shader_prim)
        shader.CreateIdAttr("UsdPreviewSurface")
        
        shader.CreateInput("diffuseColor", self.Sdf.ValueTypeNames.Color3f).Set((0.1, 0.5, 1.0))
        shader.CreateInput("emissiveColor", self.Sdf.ValueTypeNames.Color3f).Set((0.05, 0.25, 0.5))
        shader.CreateInput("roughness", self.Sdf.ValueTypeNames.Float).Set(0.2)
        
        material_output = material.CreateSurfaceOutput()
        shader_output = shader.CreateOutput("surface", self.Sdf.ValueTypeNames.Token)
        material_output.ConnectToSource(shader_output)
        
        self.UsdShade.MaterialBindingAPI(line_prim).Bind(material)
    
    def _cleanup_all(self):
        """清理所有可视化"""
        containers = [self.path_container, self.ghost_container]
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
        # 清理超丝滑流畅覆盖可视化器
        if self.fluent_coverage_visualizer:
            self.fluent_coverage_visualizer.cleanup()
        print("可视化资源清理完成")