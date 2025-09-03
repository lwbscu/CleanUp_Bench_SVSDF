#!/usr/bin/env python3
"""
机器人控制模块
"""

import numpy as np
import math
import torch
import rospy
from data_structures import *

class FixedRobotController:
    def __init__(self, mobile_base, world):
        from isaacsim.core.utils.types import ArticulationAction
        from scipy.spatial.transform import Rotation as R
        
        self.ArticulationAction = ArticulationAction
        self.R = R
        self.mobile_base = mobile_base
        self.world = world
        self.max_linear_vel = MAX_LINEAR_VELOCITY
        self.max_angular_vel = MAX_ANGULAR_VELOCITY
        self.linear_kp = 3.0
        self.angular_kp = 4.0
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.smooth_factor = 0.8
        self.coverage_visualizer = None
        self.lidar_avoidance = None
        self.dynamic_replanner = None
        self.current_path = []
        self.current_path_index = 0
        self.path_blocked_counter = 0
        self.emergency_path = []
        self.using_emergency_path = False
    
    def set_coverage_visualizer(self, visualizer):
        """设置覆盖可视化器引用"""
        self.coverage_visualizer = visualizer
    
    def set_lidar_avoidance(self, lidar_avoidance_controller, dynamic_replanner=None):
        """设置激光雷达避障控制器"""
        self.lidar_avoidance = lidar_avoidance_controller
        self.dynamic_replanner = dynamic_replanner
    
    def set_coverage_path(self, path_points):
        """设置覆盖路径"""
        self.current_path = path_points
        self.current_path_index = 0
        self.using_emergency_path = False
    
    def move_to_position_robust(self, target_pos: np.ndarray, target_orientation: float = 0.0) -> bool:
        """鲁棒的移动到位置方法 - 集成激光雷达避障"""
        print(f"开始导航到: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
        
        max_steps = MAX_NAVIGATION_STEPS
        step_count = 0
        stuck_counter = 0
        prev_position = None
        
        start_pos = self._get_robot_pose()[0]
        start_distance = np.linalg.norm(start_pos[:2] - target_pos[:2])
        
        while step_count < max_steps:
            current_pos, current_yaw = self._get_robot_pose()
            step_count += 1
            
            # 超丝滑覆盖标记 - 每步都标记当前位置
            self._mark_coverage_ultra_smooth(current_pos)
            
            # 检查是否到达
            pos_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
            
            if pos_error < POSITION_TOLERANCE:
                self._send_zero_velocity()
                print(f"  成功到达! 误差: {pos_error:.3f}m, 步数: {step_count}")
                return True
            
            # 优先检查激光雷达近距离危险(<0.57m) - 唯一避障策略
            if self._check_close_range_danger():
                print(f"  激光雷达检测到近距离危险(<0.57m)，启动智能避障转向...")
                self._intelligent_close_range_avoidance(current_pos, target_pos)
                continue  # 执行避障后继续导航循环
            
            # 检测卡住状态
            if prev_position is not None:
                movement = np.linalg.norm(current_pos[:2] - prev_position[:2])
                if movement < 0.001:
                    stuck_counter += 1
                    if stuck_counter > 600:
                        print(f"  检测到卡住状态，启动智能脱困...")
                        self._intelligent_unstuck_strategy(current_pos, target_pos)
                        stuck_counter = 0
                else:
                    stuck_counter = 0
            
            prev_position = current_pos.copy()
            
            # 计算控制指令（集成避障）
            self._compute_and_send_control_with_avoidance(current_pos, current_yaw, target_pos, target_orientation)
            
            # 步进物理仿真
            self.world.step(render=True)
            
            # 周期性进度报告
            if step_count % 500 == 0:
                progress = max(0, (start_distance - pos_error) / start_distance * 100)
                print(f"  导航中... 距离: {pos_error:.3f}m, 进度: {progress:.1f}%, 步数: {step_count}")
        
        # 导航结束
        self._send_zero_velocity()
        final_pos, _ = self._get_robot_pose()
        final_error = np.linalg.norm(final_pos[:2] - target_pos[:2])
        
        if final_error < POSITION_TOLERANCE * 1.5:
            print(f"  接近成功! 最终误差: {final_error:.3f}m")
            return True
        else:
            print(f"  导航失败! 最终误差: {final_error:.3f}m, 用时: {step_count}步")
            return False
    
    def _check_emergency_avoidance_needed(self, current_pos: np.ndarray, target_pos: np.ndarray) -> bool:
        """检查是否需要紧急避障 - 新增方法"""
        if not self.lidar_avoidance:
            return False
        
        # 检查避障状态
        status = self.lidar_avoidance.get_avoidance_status()
        
        # 危险模式需要紧急避障
        if status['mode'] == 'DANGER':
            return True
        
        # 检查路径是否被阻挡
        if self.lidar_avoidance.is_path_blocked(target_pos, current_pos):
            self.path_blocked_counter += 1
            if self.path_blocked_counter > 10:  # 连续10次检测到阻挡
                return True
        else:
            self.path_blocked_counter = 0
        
        return False
    
    def _execute_emergency_avoidance(self, current_pos: np.ndarray, target_pos: np.ndarray) -> bool:
        """执行紧急避障 - 新增方法"""
        if not self.dynamic_replanner:
            # 简单避障：停止并转向安全方向
            safe_directions = self.lidar_avoidance.get_safe_directions()
            if safe_directions:
                # 选择最接近目标的安全方向
                target_angle = np.arctan2(target_pos[1] - current_pos[1], 
                                        target_pos[0] - current_pos[0])
                
                best_direction = min(safe_directions, 
                                   key=lambda x: abs(np.arctan2(np.sin(x - target_angle), 
                                                               np.cos(x - target_angle))))
                
                # 转向安全方向
                current_yaw = self._get_robot_pose()[1]
                angle_error = self._normalize_angle(best_direction - current_yaw)
                
                if abs(angle_error) > ANGLE_TOLERANCE:
                    angular_vel = np.clip(self.angular_kp * angle_error, 
                                        -self.max_angular_vel, self.max_angular_vel)
                    self._send_velocity_command(0.0, angular_vel)
                    return True
            
            return False
        
        # 使用动态重规划器
        if not self.using_emergency_path:
            self.emergency_path = self.dynamic_replanner.generate_emergency_path(current_pos, target_pos)
            if self.emergency_path:
                self.using_emergency_path = True
                print(f"  生成紧急避障路径: {len(self.emergency_path)}个点")
        
        # 跟随紧急路径
        if self.using_emergency_path and self.emergency_path:
            next_point = self.emergency_path[0]
            error = np.linalg.norm(current_pos[:2] - next_point.position[:2])
            
            if error < POSITION_TOLERANCE * 2:
                self.emergency_path.pop(0)
                if not self.emergency_path:
                    self.using_emergency_path = False
                    print(f"  紧急避障路径完成")
            
            return len(self.emergency_path) > 0
        
        return False
    
    def _check_close_range_danger(self) -> bool:
        """检查激光雷达是否检测到近距离危险(<0.57m) - 加强版检测"""
        if not self.lidar_avoidance or not self.lidar_avoidance.current_scan:
            return False
        
        scan = self.lidar_avoidance.current_scan
        ranges = np.array(scan.ranges)
        
        # 检查有效距离数据
        valid_ranges = ranges[(ranges >= scan.range_min) & (ranges <= scan.range_max)]
        if len(valid_ranges) == 0:
            return False
        
        # 详细检查各个重要扇区的最短距离
        angles = []
        valid_ranges_filtered = []
        
        for i, r in enumerate(ranges):
            if scan.range_min <= r <= scan.range_max:
                angle = scan.angle_min + i * scan.angle_increment
                angles.append(angle)
                valid_ranges_filtered.append(r)
        
        if len(valid_ranges_filtered) == 0:
            return False
            
        angles = np.array(angles)
        valid_ranges_filtered = np.array(valid_ranges_filtered)
        
        # 检查关键方向的最短距离
        danger_threshold = 0.57
        
        # 前方扇区 (-30° 到 +30°)
        front_min = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                  math.radians(-30), math.radians(30))
        
        # 左前方扇区 (30° 到 60°) 
        left_front_min = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                       math.radians(30), math.radians(60))
        
        # 右前方扇区 (-60° 到 -30°)
        right_front_min = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                        math.radians(-60), math.radians(-30))
        
        # 检查任一重要扇区是否有近距离危险
        if (front_min < danger_threshold or 
            left_front_min < danger_threshold or 
            right_front_min < danger_threshold):
            
            print(f"    近距离危险检测详情:")
            print(f"      前方最短: {front_min:.2f}m")
            print(f"      左前方最短: {left_front_min:.2f}m") 
            print(f"      右前方最短: {right_front_min:.2f}m")
            print(f"      危险阈值: {danger_threshold}m")
            return True
        
        return False
    
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
    
    def _intelligent_close_range_avoidance(self, current_pos: np.ndarray, target_pos: np.ndarray):
        """激光雷达近距离危险智能避障策略 - 完整版：后退+转向+前进"""
        print(f"    执行完整近距离智能避障策略...")
        
        # 第一步：后退脱离当前障碍物
        print(f"    步骤1: 后退脱离近距离障碍物")
        for _ in range(40):  # 延长后退时间
            self._send_velocity_command(-self.max_linear_vel * 1, 0.0)
            self.world.step(render=True)
            current_pos, _ = self._get_robot_pose()
            self._mark_coverage_ultra_smooth(current_pos)
        
        # 第二步：智能分析最优避障转向方向
        optimal_direction = self._analyze_optimal_escape_direction(current_pos, target_pos)
        print(f"    步骤2: 智能分析得出最优避障转向方向: {math.degrees(optimal_direction):.1f}°")
        
        # 第三步：强化精确转向最优方向
        print(f"    步骤3: 强化精确转向最优避障方向")
        current_yaw = self._get_robot_pose()[1]
        
        for step in range(150):  # 大幅延长转向时间
            angle_error = self._normalize_angle(optimal_direction - current_yaw)
            if abs(angle_error) < ANGLE_TOLERANCE * 0.1:  # 更严格的角度要求
                print(f"    精确转向完成，角度误差: {math.degrees(angle_error):.2f}°")
                break
            
            # 增强转向力度，特别是对迟缓机器人 - 大幅提升角速度
            angular_vel = np.clip(self.angular_kp * angle_error * 5, 
                                -self.max_angular_vel * 2, self.max_angular_vel * 2)
            self._send_velocity_command(0.0, angular_vel)
            self.world.step(render=True)
            
            current_pos, current_yaw = self._get_robot_pose()
            self._mark_coverage_ultra_smooth(current_pos)
            
            # 每20步输出转向进度
            if step % 20 == 0:
                print(f"      转向进度: {step}/150, 角度误差: {math.degrees(angle_error):.2f}°")
        
        # 额外的转向稳定时间
        print(f"    步骤3.5: 转向稳定确保到位")
        for _ in range(30):
            self.world.step(render=True)
            current_pos, _ = self._get_robot_pose()
            self._mark_coverage_ultra_smooth(current_pos)
        
        # 第四步：持续前进充分脱离危险区域
        print(f"    步骤4: 持续前进充分脱离近距离危险区域")
        for _ in range(60):  # 大幅延长前进时间
            self._send_velocity_command(self.max_linear_vel * 0.3, 0.0)
            self.world.step(render=True)
            current_pos, _ = self._get_robot_pose()
            self._mark_coverage_ultra_smooth(current_pos)
        
        # 第五步：验证是否成功脱离危险区域
        self._send_zero_velocity()
        if self._verify_danger_cleared():
            print(f"    ✓ 近距离智能避障成功完成，已脱离危险区域")
        else:
            print(f"    ⚠ 避障后仍有近距离危险，需要进一步处理")
        
        print(f"    继续原路径导航")
    
    def _verify_danger_cleared(self) -> bool:
        """验证是否成功脱离近距离危险区域"""
        if not self.lidar_avoidance or not self.lidar_avoidance.current_scan:
            return True
        
        # 等待激光雷达数据更新
        for _ in range(10):
            self.world.step(render=True)
        
        # 重新检查近距离危险
        scan = self.lidar_avoidance.current_scan
        ranges = np.array(scan.ranges)
        
        # 检查关键扇区
        angles = []
        valid_ranges_filtered = []
        
        for i, r in enumerate(ranges):
            if scan.range_min <= r <= scan.range_max:
                angle = scan.angle_min + i * scan.angle_increment
                angles.append(angle)
                valid_ranges_filtered.append(r)
        
        if len(valid_ranges_filtered) == 0:
            return True
            
        angles = np.array(angles)
        valid_ranges_filtered = np.array(valid_ranges_filtered)
        
        # 检查主要方向是否安全
        front_min = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                  math.radians(-30), math.radians(30))
        left_front_min = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                       math.radians(30), math.radians(60))
        right_front_min = self._get_sector_min_distance(angles, valid_ranges_filtered,
                                                        math.radians(-60), math.radians(-30))
        
        danger_threshold = 0.65  # 稍微放宽验证阈值
        
        all_clear = (front_min >= danger_threshold and 
                    left_front_min >= danger_threshold and 
                    right_front_min >= danger_threshold)
        
        print(f"    脱离验证: 前方{front_min:.2f}m, 左前{left_front_min:.2f}m, 右前{right_front_min:.2f}m")
        
        return all_clear
    
    def _intelligent_unstuck_strategy(self, current_pos: np.ndarray, target_pos: np.ndarray):
        """智能脱困策略 - 综合分析目标方向和避障方向选择最优路径"""
        print(f"    启动智能脱困: 综合分析目标方向和避障安全方向...")
        
        # 第一步：后退脱离障碍物
        print(f"    步骤1: 后退脱离障碍物")
        for _ in range(30):
            self._send_velocity_command(-self.max_linear_vel * 0.2, 0.0)
            self.world.step(render=True)
            current_pos, _ = self._get_robot_pose()
            self._mark_coverage_ultra_smooth(current_pos)
        
        # 第二步：智能分析最优脱困方向
        optimal_direction = self._analyze_optimal_escape_direction(current_pos, target_pos)
        print(f"    步骤2: 智能分析得出最优脱困方向 {math.degrees(optimal_direction):.1f}°")
        
        # 第三步：转向最优方向
        print(f"    步骤3: 精确转向最优方向")
        current_yaw = self._get_robot_pose()[1]
        
        for _ in range(80):
            angle_error = self._normalize_angle(optimal_direction - current_yaw)
            if abs(angle_error) < ANGLE_TOLERANCE:
                break
            
            angular_vel = np.clip(self.angular_kp * angle_error, 
                                -self.max_angular_vel * 0.6, self.max_angular_vel * 0.6)
            self._send_velocity_command(0.0, angular_vel)
            self.world.step(render=True)
            
            current_pos, current_yaw = self._get_robot_pose()
            self._mark_coverage_ultra_smooth(current_pos)
        
        # 第四步：沿最优方向前进脱困
        print(f"    步骤4: 沿最优方向前进脱困")
        for _ in range(50):
            self._send_velocity_command(self.max_linear_vel * 0.35, 0.0)
            self.world.step(render=True)
            current_pos, _ = self._get_robot_pose()
            self._mark_coverage_ultra_smooth(current_pos)
        
        # 停止并完成脱困
        self._send_zero_velocity()
        print(f"    智能脱困策略完成，继续原路径规划")
    
    def _analyze_optimal_escape_direction(self, current_pos: np.ndarray, target_pos: np.ndarray) -> float:
        """智能分析最优脱困方向 - 综合考虑目标方向和安全方向"""
        # 计算朝向目标的理想方向
        target_direction = np.arctan2(target_pos[1] - current_pos[1], 
                                    target_pos[0] - current_pos[0])
        
        # 获取所有安全方向
        safe_directions = []
        if self.lidar_avoidance:
            safe_directions = self.lidar_avoidance.get_safe_directions()
        
        # 如果没有激光雷达数据或安全方向，使用基础方向分析
        if not safe_directions:
            # 基础策略：尝试几个固定方向
            candidate_directions = [
                target_direction,  # 直接朝目标
                target_direction + math.pi/4,   # 目标方向右45度
                target_direction - math.pi/4,   # 目标方向左45度
                target_direction + math.pi/2,   # 目标方向右90度
                target_direction - math.pi/2,   # 目标方向左90度
            ]
            return candidate_directions[1]  # 选择右45度作为默认
        
        # 智能综合分析：为每个安全方向计算综合评分
        best_direction = safe_directions[0]
        best_score = -float('inf')
        
        for safe_dir in safe_directions:
            score = self._calculate_direction_score(safe_dir, target_direction, current_pos, target_pos)
            if score > best_score:
                best_score = score
                best_direction = safe_dir
        
        return best_direction
    
    def _calculate_direction_score(self, candidate_dir: float, target_dir: float, 
                                 current_pos: np.ndarray, target_pos: np.ndarray) -> float:
        """计算方向的综合评分"""
        # 1. 方向一致性评分 (0-1)：越接近目标方向得分越高
        angle_diff = abs(self._normalize_angle(candidate_dir - target_dir))
        direction_score = 1.0 - (angle_diff / math.pi)
        
        # 2. 距离优势评分 (0-1)：选择这个方向是否能更快接近目标
        distance_to_target = np.linalg.norm(target_pos[:2] - current_pos[:2])
        # 预估沿候选方向移动1米后到目标的距离
        future_pos = current_pos[:2] + np.array([np.cos(candidate_dir), np.sin(candidate_dir)])
        future_distance = np.linalg.norm(target_pos[:2] - future_pos)
        distance_score = max(0, (distance_to_target - future_distance) / distance_to_target)
        
        # 3. 安全性评分：这个方向被认为是安全的，给基础安全分
        safety_score = 0.8
        
        # 综合评分：方向一致性权重0.5，距离优势权重0.3，安全性权重0.2
        total_score = (direction_score * 0.5 + 
                      distance_score * 0.3 + 
                      safety_score * 0.2)
        
        return float(total_score)
    
    def _compute_and_send_control_with_avoidance(self, current_pos: np.ndarray, current_yaw: float, 
                                               target_pos: np.ndarray, target_yaw: float):
        """计算并发送控制指令（集成避障） - 修改方法"""
        # 计算基础控制指令
        direction = target_pos[:2] - current_pos[:2]
        distance = np.linalg.norm(direction)
        
        if distance < 0.0001:
            self._send_zero_velocity()
            return
        
        # 计算目标角度
        direction_norm = direction / distance
        desired_yaw = np.arctan2(direction_norm[1], direction_norm[0])
        yaw_error = self._normalize_angle(desired_yaw - current_yaw)
        
        # 控制策略：先转向，后前进
        if abs(yaw_error) > ANGLE_TOLERANCE:
            linear_vel = 0.0
            angular_vel = np.clip(self.angular_kp * yaw_error, 
                                -self.max_angular_vel, self.max_angular_vel)
        else:
            linear_vel = np.clip(self.linear_kp * distance, 
                               0.0, self.max_linear_vel)
            angular_vel = np.clip(self.angular_kp * yaw_error * 0.5, 
                                -self.max_angular_vel, self.max_angular_vel)
        
        # 激光雷达避障调整
        if self.lidar_avoidance:
            linear_vel, angular_vel = self.lidar_avoidance.get_avoidance_velocity_adjustment(linear_vel, angular_vel)
        
        # 平滑速度变化
        linear_vel = self._smooth_velocity(float(linear_vel), self.prev_linear)
        angular_vel = self._smooth_velocity(float(angular_vel), self.prev_angular)
        
        self.prev_linear = linear_vel
        self.prev_angular = angular_vel
        
        # 发送控制指令
        self._send_velocity_command(linear_vel, angular_vel)
    
    def _mark_coverage_ultra_smooth(self, robot_position: np.ndarray):
        """超丝滑覆盖标记 - 每步都尝试标记"""
        if self.coverage_visualizer and hasattr(self.coverage_visualizer, 'fluent_coverage_visualizer'):
            self.coverage_visualizer.fluent_coverage_visualizer.mark_coverage_realtime(robot_position)
    
    def _smooth_velocity(self, new_vel: float, prev_vel: float) -> float:
        """平滑速度变化"""
        return self.smooth_factor * prev_vel + (1 - self.smooth_factor) * new_vel
    
    def _send_velocity_command(self, linear_vel: float, angular_vel: float):
        """发送速度指令到轮子"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        # 轮子参数
        wheel_radius = 0.036
        wheel_base = 0.235
        
        # 计算轮子速度
        left_wheel_vel = (linear_vel - angular_vel * wheel_base / 2.0) / wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * wheel_base / 2.0) / wheel_radius
        
        # 构建关节速度
        num_dofs = len(self.mobile_base.dof_names)
        joint_velocities = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 设置轮子速度
        if "left_wheel_joint" in self.mobile_base.dof_names:
            left_idx = self.mobile_base.dof_names.index("left_wheel_joint")
            joint_velocities[left_idx] = float(left_wheel_vel)
        
        if "right_wheel_joint" in self.mobile_base.dof_names:
            right_idx = self.mobile_base.dof_names.index("right_wheel_joint")
            joint_velocities[right_idx] = float(right_wheel_vel)
        
        # 应用控制指令
        action = self.ArticulationAction(joint_velocities=joint_velocities)
        articulation_controller.apply_action(action)
    
    def _send_zero_velocity(self):
        """发送零速度指令"""
        self._send_velocity_command(0.0, 0.0)
    
    def _normalize_angle(self, angle):
        """角度归一化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _get_robot_pose(self):
        """获取机器人位姿"""
        position, orientation = self.mobile_base.get_world_pose()
        position = np.array(position)
        
        # 四元数转欧拉角
        quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        r = self.R.from_quat(quat)
        yaw = r.as_euler('xyz')[2]
        
        return position, yaw