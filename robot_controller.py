#!/usr/bin/env python3
"""
机器人控制模块 - 集成激光雷达避障与超丝滑覆盖标记
包含机器人移动控制、实时覆盖标记和激光雷达避障功能
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
        
        # 控制参数
        self.linear_kp = 3.0
        self.angular_kp = 4.0
        
        # 速度平滑
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.smooth_factor = 0.8
        
        # 覆盖可视化器引用 - 用于超丝滑标记
        self.coverage_visualizer = None
        
        # 激光雷达避障控制器 - 新增
        self.lidar_avoidance = None
        self.dynamic_replanner = None
        
        # 路径跟踪状态
        self.current_path = []
        self.current_path_index = 0
        self.path_blocked_counter = 0
        self.emergency_path = []
        self.using_emergency_path = False
        
        print("集成激光雷达避障的机器人控制器初始化")
        print(f"线速度上限: {self.max_linear_vel}m/s")
        print(f"角速度上限: {self.max_angular_vel}rad/s")
    
    def set_coverage_visualizer(self, visualizer):
        """设置覆盖可视化器引用"""
        self.coverage_visualizer = visualizer
    
    def set_lidar_avoidance(self, lidar_avoidance_controller, dynamic_replanner=None):
        """设置激光雷达避障控制器 - 新增方法"""
        self.lidar_avoidance = lidar_avoidance_controller
        self.dynamic_replanner = dynamic_replanner
        print("激光雷达避障控制器已集成")
    
    def set_coverage_path(self, path_points):
        """设置覆盖路径 - 新增方法"""
        self.current_path = path_points
        self.current_path_index = 0
        self.using_emergency_path = False
        print(f"设置覆盖路径: {len(path_points)}个点")
    
    def move_to_position_robust(self, target_pos: np.ndarray, target_orientation: float = 0.0) -> bool:
        """鲁棒的移动到位置方法 - 集成激光雷达避障"""
        print(f"开始导航到: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
        
        max_steps = MAX_NAVIGATION_STEPS
        step_count = 0
        stuck_counter = 0
        prev_position = None
        
        start_pos = self._get_robot_pose()[0]
        start_distance = np.linalg.norm(start_pos[:2] - target_pos[:2])
        
        # 检查激光雷达避障状态
        if self.lidar_avoidance:
            avoidance_status = self.lidar_avoidance.get_avoidance_status()
            if avoidance_status['mode'] == 'DANGER':
                print(f"  警告: 检测到危险障碍物，启用紧急避障")
        
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
            
            # 激光雷达避障检查 - 新增
            if self._check_emergency_avoidance_needed(current_pos, target_pos):
                if self._execute_emergency_avoidance(current_pos, target_pos):
                    continue  # 继续紧急避障
                else:
                    print(f"  紧急避障失败，终止导航")
                    break
            
            # 检测卡住状态
            if prev_position is not None:
                movement = np.linalg.norm(current_pos[:2] - prev_position[:2])
                if movement < 0.001:
                    stuck_counter += 1
                    if stuck_counter > 600:
                        print(f"  检测到卡住状态，尝试激光雷达辅助脱困...")
                        if self._lidar_assisted_unstuck(current_pos):
                            stuck_counter = 0
                        else:
                            print(f"  激光雷达辅助脱困失败")
                            break
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
                if self.lidar_avoidance:
                    status = self.lidar_avoidance.get_avoidance_status()
                    print(f"  导航中... 距离: {pos_error:.3f}m, 进度: {progress:.1f}%, 避障模式: {status['mode']}")
                else:
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
    
    def _lidar_assisted_unstuck(self, current_pos: np.ndarray) -> bool:
        """激光雷达辅助脱困 - 新增方法"""
        if not self.lidar_avoidance:
            return False
        
        safe_directions = self.lidar_avoidance.get_safe_directions()
        if not safe_directions:
            return False
        
        # 选择一个安全方向进行脱困
        escape_direction = safe_directions[0]  # 选择第一个安全方向
        
        print(f"    激光雷达辅助脱困: 转向{math.degrees(escape_direction):.1f}度")
        
        # 后退
        for _ in range(30):
            self._send_velocity_command(-self.max_linear_vel * 0.2, 0.0)
            self.world.step(render=True)
            current_pos, _ = self._get_robot_pose()
            self._mark_coverage_ultra_smooth(current_pos)
        
        # 转向安全方向
        current_yaw = self._get_robot_pose()[1]
        target_yaw = escape_direction
        
        for _ in range(60):
            angle_error = self._normalize_angle(target_yaw - current_yaw)
            if abs(angle_error) < ANGLE_TOLERANCE:
                break
            
            angular_vel = np.clip(self.angular_kp * angle_error, 
                                -self.max_angular_vel * 0.5, self.max_angular_vel * 0.5)
            self._send_velocity_command(0.0, angular_vel)
            self.world.step(render=True)
            
            current_pos, current_yaw = self._get_robot_pose()
            self._mark_coverage_ultra_smooth(current_pos)
        
        # 前进一小段距离
        for _ in range(40):
            self._send_velocity_command(self.max_linear_vel * 0.3, 0.0)
            self.world.step(render=True)
            current_pos, _ = self._get_robot_pose()
            self._mark_coverage_ultra_smooth(current_pos)
        
        self._send_zero_velocity()
        print(f"    激光雷达辅助脱困完成")
        return True
    
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
        
        # 激光雷达避障调整 - 新增
        if self.lidar_avoidance:
            linear_vel, angular_vel = self.lidar_avoidance.get_avoidance_velocity_adjustment(linear_vel, angular_vel)
        
        # 平滑速度变化
        linear_vel = self._smooth_velocity(linear_vel, self.prev_linear)
        angular_vel = self._smooth_velocity(angular_vel, self.prev_angular)
        
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