#!/usr/bin/env python3
"""
激光雷达避障模块
"""

import numpy as np
import math
import rospy
from sensor_msgs.msg import LaserScan, PointCloud2
from typing import List, Tuple, Optional
from data_structures import *

try:
    import sensor_msgs.point_cloud2 as pc2
    PC2_AVAILABLE = True
except ImportError:
    PC2_AVAILABLE = False

class LidarAvoidanceController:
    """激光雷达避障控制器"""
    
    def __init__(self):
        self.min_scan_range = 0.5
        self.danger_distance = 0.8  # 适度减小，避免过度保守
        self.warning_distance = 1.0  # 适度减小
        self.safe_distance = 1.2    # 适度减小
        self.danger_speed_factor = 0.0
        self.warning_speed_factor = 0.3
        self.safe_speed_factor = 0.8
        self.front_angle_range = 60
        self.side_angle_range = 120
        self.current_scan = None
        self.last_scan_time = rospy.Time.now()
        self.avoidance_mode = "NORMAL"
        self.preferred_direction = None
        self.last_distance_output_time = rospy.Time.now()
        self.distance_output_interval = 30.0  # 增加到30秒输出一次，减少调试信息
        
        self.laser_sub = rospy.Subscriber('/robot_lidar_pointcloud', PointCloud2, 
                                        self.pointcloud_callback, queue_size=1)
        self.simulated_scan = None
        
        print("激光雷达避障控制器初始化完成")
    
    def pointcloud_callback(self, pointcloud_msg: PointCloud2):
        """点云数据回调"""
        simulated_scan = self._convert_pointcloud_to_laserscan(pointcloud_msg)
        if simulated_scan:
            self.current_scan = simulated_scan
            self.last_scan_time = rospy.Time.now()
            self._analyze_obstacles()
    
    def _convert_pointcloud_to_laserscan(self, pointcloud_msg: PointCloud2) -> Optional[LaserScan]:
        """将PointCloud2转换为LaserScan格式"""
        scan = LaserScan()
        scan.header = pointcloud_msg.header
        scan.angle_min = -math.pi
        scan.angle_max = math.pi
        scan.angle_increment = math.radians(1.0)
        scan.range_min = 0.1
        scan.range_max = 100
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        
        num_points = int((scan.angle_max - scan.angle_min) / scan.angle_increment)
        scan.ranges = [float('inf')] * num_points
        
        points = []
        if PC2_AVAILABLE:
            for point in pc2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
                x, y, z = point
                points.append((x, y, z))
        else:
            points = self._parse_pointcloud_manually(pointcloud_msg)
        
        for x, y, z in points:
            if abs(z) > 0.5:
                continue
            
            distance = math.sqrt(x*x + y*y)
            if distance < scan.range_min or distance > scan.range_max:
                continue
            
            angle = math.atan2(y, x)
            angle_index = int((angle - scan.angle_min) / scan.angle_increment)
            if 0 <= angle_index < len(scan.ranges):
                if distance < scan.ranges[angle_index]:
                    scan.ranges[angle_index] = distance
        
        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('inf'):
                scan.ranges[i] = scan.range_max
        
        return scan
    
    def _parse_pointcloud_manually(self, pointcloud_msg: PointCloud2) -> List[Tuple[float, float, float]]:
        """手动解析点云数据"""
        points = []
        import struct
        
        point_step = pointcloud_msg.point_step
        for i in range(pointcloud_msg.width):
            offset = i * point_step
            if offset + 12 <= len(pointcloud_msg.data):
                x = struct.unpack('f', pointcloud_msg.data[offset:offset+4])[0]
                y = struct.unpack('f', pointcloud_msg.data[offset+4:offset+8])[0]
                z = struct.unpack('f', pointcloud_msg.data[offset+8:offset+12])[0]
                
                if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
                    points.append((x, y, z))
        
        return points

    def laser_callback(self, scan_msg: LaserScan):
        """激光雷达数据回调"""
        self.current_scan = scan_msg
        self.last_scan_time = rospy.Time.now()
        self._analyze_obstacles()
    
    def _analyze_obstacles(self):
        """分析激光雷达数据中的障碍物分布"""
        if self.current_scan is None:
            return
        
        scan = self.current_scan
        ranges = np.array(scan.ranges)
        
        # 过滤无效数据
        valid_ranges = ranges[(ranges >= scan.range_min) & (ranges <= scan.range_max)]
        
        if len(valid_ranges) == 0:
            return
        
        # 计算角度
        angles = []
        valid_ranges_filtered = []
        
        for i, r in enumerate(ranges):
            if scan.range_min <= r <= scan.range_max:
                angle = scan.angle_min + i * scan.angle_increment
                angles.append(angle)
                valid_ranges_filtered.append(r)
        
        angles = np.array(angles)
        valid_ranges_filtered = np.array(valid_ranges_filtered)
        
        # 分析各个方向的最远和最近距离
        front_max_dist = self._get_sector_max_distance(angles, valid_ranges_filtered, 
                                                     -math.radians(self.front_angle_range/2), 
                                                     math.radians(self.front_angle_range/2))
        front_min_dist = self._get_sector_min_distance(angles, valid_ranges_filtered, 
                                                     -math.radians(self.front_angle_range/2), 
                                                     math.radians(self.front_angle_range/2))
        
        # 左侧 (30-150度)
        left_max_dist = self._get_sector_max_distance(angles, valid_ranges_filtered,
                                                    math.radians(30), 
                                                    math.radians(150))
        left_min_dist = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                    math.radians(30), 
                                                    math.radians(150))
        
        # 右侧 (-150到-30度)
        right_max_dist = self._get_sector_max_distance(angles, valid_ranges_filtered,
                                                     math.radians(-150), 
                                                     math.radians(-30))
        right_min_dist = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                     math.radians(-150), 
                                                     math.radians(-30))
        
        # 后方 (150-210度和-150到-210度)
        back_max_dist = self._get_sector_max_distance(angles, valid_ranges_filtered,
                                                    math.radians(150), 
                                                    math.radians(210))
        back_min_dist = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                    math.radians(150), 
                                                    math.radians(210))
        
        # 左前方 (15-75度)
        left_front_max_dist = self._get_sector_max_distance(angles, valid_ranges_filtered,
                                                          math.radians(15), 
                                                          math.radians(75))
        left_front_min_dist = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                          math.radians(15), 
                                                          math.radians(75))
        
        # 右前方 (-75到-15度)
        right_front_max_dist = self._get_sector_max_distance(angles, valid_ranges_filtered,
                                                           math.radians(-75), 
                                                           math.radians(-15))
        right_front_min_dist = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                           math.radians(-75), 
                                                           math.radians(-15))
        
        # 每隔1秒输出距离信息
        current_time = rospy.Time.now()
        if (current_time - self.last_distance_output_time).to_sec() >= self.distance_output_interval:
            self._output_distance_info(scan)
            self.last_distance_output_time = current_time
        
        # 更新避障模式（使用最小距离）
        self._update_avoidance_mode(front_min_dist, left_min_dist, right_min_dist)
    
    def _output_distance_info(self, scan_data):
        """输出前方、左前方、右前方的最短距离信息"""
        try:
            # 准备角度和距离数据
            ranges = np.array(scan_data.ranges)
            angles = []
            valid_ranges_filtered = []
            
            for i, r in enumerate(ranges):
                if scan_data.range_min <= r <= scan_data.range_max:
                    angle = scan_data.angle_min + i * scan_data.angle_increment
                    angles.append(angle)
                    valid_ranges_filtered.append(r)
            
            angles = np.array(angles)
            valid_ranges_filtered = np.array(valid_ranges_filtered)
            
            # 前方 (-15° 到 +15°)
            front_min_dist = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                     math.radians(-self.front_angle_range/2),
                                                     math.radians(self.front_angle_range/2))
            
            # 左前方 (30° 到 60°)
            left_front_min_dist = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                          math.radians(30),
                                                          math.radians(60))
            
            # 右前方 (-60° 到 -30°)
            right_front_min_dist = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                           math.radians(-60),
                                                           math.radians(-30))
            
            print("=== 激光雷达扫描距离信息 ===")
            print(f"前方最短距离:    {front_min_dist:.2f}m")
            print(f"左前方最短距离:  {left_front_min_dist:.2f}m")
            print(f"右前方最短距离:  {right_front_min_dist:.2f}m")
            print()
            
        except Exception as e:
            rospy.logwarn(f"输出距离信息时出错: {e}")
    
    def _get_sector_min_distance(self, angles: np.ndarray, ranges: np.ndarray, 
                                min_angle: float, max_angle: float) -> float:
        """获取指定角度扇区内的最小距离"""
        # 规范化角度到[-π, π]
        angles_norm = np.arctan2(np.sin(angles), np.cos(angles))
        min_angle = np.arctan2(np.sin(min_angle), np.cos(min_angle))
        max_angle = np.arctan2(np.sin(max_angle), np.cos(max_angle))
        
        if min_angle <= max_angle:
            mask = (angles_norm >= min_angle) & (angles_norm <= max_angle)
        else:  # 跨越±π边界
            mask = (angles_norm >= min_angle) | (angles_norm <= max_angle)
        
        sector_ranges = ranges[mask]
        
        if len(sector_ranges) > 0:
            return np.min(sector_ranges)
        else:
            return float('inf')
    
    def _get_sector_max_distance(self, angles: np.ndarray, ranges: np.ndarray, 
                                min_angle: float, max_angle: float) -> float:
        """获取指定角度扇区内的最大距离"""
        # 规范化角度到[-π, π]
        angles_norm = np.arctan2(np.sin(angles), np.cos(angles))
        min_angle = np.arctan2(np.sin(min_angle), np.cos(min_angle))
        max_angle = np.arctan2(np.sin(max_angle), np.cos(max_angle))
        
        if min_angle <= max_angle:
            mask = (angles_norm >= min_angle) & (angles_norm <= max_angle)
        else:  # 跨越±π边界
            mask = (angles_norm >= min_angle) | (angles_norm <= max_angle)
        
        sector_ranges = ranges[mask]
        
        if len(sector_ranges) > 0:
            return np.max(sector_ranges)
        else:
            return 0.0
    
    def _update_avoidance_mode(self, front_dist: float, left_dist: float, right_dist: float):
        """更新避障模式和偏好方向"""
        min_dist = min(front_dist, left_dist, right_dist)
        
        # 确定避障模式
        if min_dist < self.danger_distance:
            self.avoidance_mode = "DANGER"
        elif min_dist < self.warning_distance:
            self.avoidance_mode = "WARNING"
        else:
            self.avoidance_mode = "NORMAL"
        
        # 确定偏好转向方向（远离最近障碍物）
        if front_dist < self.warning_distance:
            if left_dist > right_dist:
                self.preferred_direction = "LEFT"
            else:
                self.preferred_direction = "RIGHT"
        else:
            self.preferred_direction = None
    
    def get_avoidance_velocity_adjustment(self, target_linear: float, target_angular: float) -> Tuple[float, float]:
        """根据避障模式调整目标速度"""
        if self.current_scan is None:
            return target_linear, target_angular
        
        # 检查数据新鲜度
        time_since_scan = (rospy.Time.now() - self.last_scan_time).to_sec()
        if time_since_scan > 0.5:  # 数据过时，保守处理
            return target_linear * 0.5, target_angular
        
        adjusted_linear = target_linear
        adjusted_angular = target_angular
        
        if self.avoidance_mode == "DANGER":
            # 危险模式：停止前进，快速转向
            adjusted_linear = target_linear * self.danger_speed_factor
            if self.preferred_direction == "LEFT":
                adjusted_angular = abs(target_angular) + 0.5
            elif self.preferred_direction == "RIGHT":
                adjusted_angular = -abs(target_angular) - 0.5
            
        elif self.avoidance_mode == "WARNING":
            # 警告模式：减速并调整方向
            adjusted_linear = target_linear * self.warning_speed_factor
            if self.preferred_direction == "LEFT":
                adjusted_angular = target_angular + 0.3
            elif self.preferred_direction == "RIGHT":
                adjusted_angular = target_angular - 0.3
                
        else:  # NORMAL模式
            adjusted_linear = target_linear * self.safe_speed_factor
        
        # 限制速度范围
        adjusted_linear = np.clip(adjusted_linear, 0.0, MAX_LINEAR_VELOCITY)
        adjusted_angular = np.clip(adjusted_angular, -MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY)
        
        return adjusted_linear, adjusted_angular
    
    def is_path_blocked(self, target_position: np.ndarray, current_position: np.ndarray) -> bool:
        """检查到目标位置的路径是否被阻挡"""
        if self.current_scan is None:
            return False
        
        # 计算目标方向
        direction_vector = target_position[:2] - current_position[:2]
        if np.linalg.norm(direction_vector) < 0.1:
            return False
        
        target_angle = np.arctan2(direction_vector[1], direction_vector[0])
        
        # 检查目标方向是否有障碍物
        scan = self.current_scan
        ranges = np.array(scan.ranges)
        
        # 计算与目标方向最接近的激光束
        angle_tolerance = math.radians(15)  # ±15度容差
        
        for i, r in enumerate(ranges):
            if scan.range_min <= r <= scan.range_max:
                beam_angle = scan.angle_min + i * scan.angle_increment
                angle_diff = abs(np.arctan2(np.sin(beam_angle - target_angle), 
                                          np.cos(beam_angle - target_angle)))
                
                if angle_diff <= angle_tolerance:
                    if r < self.warning_distance:
                        return True
        
        return False
    
    def get_safe_directions(self) -> List[float]:
        """获取当前环境下的安全方向列表"""
        if self.current_scan is None:
            return []
        
        safe_directions = []
        scan = self.current_scan
        ranges = np.array(scan.ranges)
        
        # 以30度为步长检查各个方向
        for angle_deg in range(-180, 180, 30):
            angle_rad = math.radians(angle_deg)
            
            # 检查该方向附近是否安全
            is_safe = True
            check_range = math.radians(20)  # ±20度检查范围
            
            for i, r in enumerate(ranges):
                if scan.range_min <= r <= scan.range_max:
                    beam_angle = scan.angle_min + i * scan.angle_increment
                    angle_diff = abs(np.arctan2(np.sin(beam_angle - angle_rad), 
                                              np.cos(beam_angle - angle_rad)))
                    
                    if angle_diff <= check_range:
                        if r < self.safe_distance:
                            is_safe = False
                            break
            
            if is_safe:
                safe_directions.append(angle_rad)
        
        return safe_directions
    
    def get_avoidance_status(self) -> dict:
        """获取避障状态信息"""
        return {
            'mode': self.avoidance_mode,
            'preferred_direction': self.preferred_direction,
            'data_age': (rospy.Time.now() - self.last_scan_time).to_sec() if self.current_scan else float('inf'),
            'scan_available': self.current_scan is not None
        }

class DynamicPathReplanner:
    """动态路径重规划器"""
    
    def __init__(self, path_planner, avoidance_controller):
        self.path_planner = path_planner
        self.avoidance_controller = avoidance_controller
        self.last_replan_time = rospy.Time.now()
        self.replan_cooldown = 2.0  # 重规划冷却时间（秒）
        
    def should_replan(self, current_path: List, current_pos: np.ndarray, target_index: int) -> bool:
        """判断是否需要重新规划路径"""
        # 冷却时间检查
        if (rospy.Time.now() - self.last_replan_time).to_sec() < self.replan_cooldown:
            return False
        
        # 检查当前路径是否被阻挡
        if target_index < len(current_path):
            target_point = current_path[target_index]
            if self.avoidance_controller.is_path_blocked(target_point.position, current_pos):
                return True
        
        # 检查前方几个路径点是否安全
        check_ahead = min(3, len(current_path) - target_index)
        for i in range(check_ahead):
            if target_index + i < len(current_path):
                point = current_path[target_index + i]
                if self.avoidance_controller.is_path_blocked(point.position, current_pos):
                    return True
        
        return False
    
    def generate_emergency_path(self, current_pos: np.ndarray, final_target: np.ndarray) -> List:
        """生成紧急避障路径"""
        self.last_replan_time = rospy.Time.now()
        
        safe_directions = self.avoidance_controller.get_safe_directions()
        if not safe_directions:
            return []  # 无安全方向，返回空路径
        
        # 选择最接近目标方向的安全方向
        target_direction = np.arctan2(final_target[1] - current_pos[1], 
                                    final_target[0] - current_pos[0])
        
        best_direction = None
        min_angle_diff = float('inf')
        
        for safe_dir in safe_directions:
            angle_diff = abs(np.arctan2(np.sin(safe_dir - target_direction), 
                                      np.cos(safe_dir - target_direction)))
            if angle_diff < min_angle_diff:
                min_angle_diff = angle_diff
                best_direction = safe_dir
        
        if best_direction is None:
            return []
        
        # 生成临时路径点
        emergency_points = []
        step_distance = 0.5
        
        for i in range(3):  # 生成3个临时点
            distance = step_distance * (i + 1)
            temp_pos = current_pos.copy()
            temp_pos[0] += distance * np.cos(best_direction)
            temp_pos[1] += distance * np.sin(best_direction)
            temp_pos[2] = current_pos[2]
            
            from data_structures import CoveragePoint
            emergency_point = CoveragePoint(
                position=temp_pos,
                orientation=best_direction,
                has_object=False
            )
            emergency_points.append(emergency_point)
        
        return emergency_points