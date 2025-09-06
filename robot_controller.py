#!/usr/bin/env python3
"""
SLAM版本机器人控制模块 - 集成MapEx导航和Cartographer SLAM
"""

import numpy as np
import math
import torch
import rospy
import time
from data_structures import *
from geometry_msgs.msg import Twist

class SLAMRobotController:
    def __init__(self, mobile_base, world, ros_bridge):
        from isaacsim.core.utils.types import ArticulationAction
        from scipy.spatial.transform import Rotation as R
        
        self.ArticulationAction = ArticulationAction
        self.R = R
        self.mobile_base = mobile_base
        self.world = world
        self.ros_bridge = ros_bridge
        
        # 基础控制参数
        self.max_linear_vel = MAX_LINEAR_VELOCITY
        self.max_angular_vel = MAX_ANGULAR_VELOCITY
        self.linear_kp = 3.0
        self.angular_kp = 4.0
        self.prev_linear = 0.0
        self.prev_angular = 0.0
        self.smooth_factor = 0.8
        
        # 可视化器引用
        self.coverage_visualizer = None
        
        # MapEx和SLAM相关
        self.mapex_interface = ros_bridge.get_mapex_interface()
        self.cartographer_interface = ros_bridge.get_cartographer_interface()
        
        # MapEx命令发布器
        self.mapex_cmd_pub = rospy.Publisher('/mapex/robot_cmd', Twist, queue_size=10)
        
        # 导航状态
        self.navigation_timeout = 30.0  # MapEx导航超时
        self.position_tolerance = 0.3   # MapEx导航位置容差
        self.mapex_active = False
        
        print("SLAM机器人控制器初始化完成")
        print("MapEx导航接口已连接")
        print("Cartographer SLAM接口已连接")
    
    def set_coverage_visualizer(self, visualizer):
        """设置覆盖可视化器引用"""
        self.coverage_visualizer = visualizer
    
    def move_to_position_with_mapex(self, target_pos: np.ndarray, target_orientation: float = 0.0) -> bool:
        """使用MapEx导航到目标位置"""
        print(f"MapEx导航到: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
        
        # 发送目标到MapEx
        self.mapex_interface.send_goal(target_pos[0], target_pos[1], target_orientation)
        
        # 等待MapEx导航完成
        start_time = time.time()
        self.mapex_active = True
        
        while (time.time() - start_time) < self.navigation_timeout and self.mapex_active:
            current_pos, _ = self._get_robot_pose()
            
            # 超级平滑覆盖标记 - 每步都标记当前位置
            self._mark_coverage_ultra_smooth(current_pos)
            
            # 检查是否到达目标
            distance_to_target = np.linalg.norm(current_pos[:2] - target_pos[:2])
            if distance_to_target < self.position_tolerance:
                self.mapex_active = False
                print(f"  MapEx导航成功! 误差: {distance_to_target:.3f}m")
                return True
            
            # 获取并执行MapEx的速度命令
            self._execute_mapex_velocity_commands()
            
            # 步进仿真
            self.world.step(render=True)
            
            # 短暂延时
            time.sleep(0.01)
        
        # 导航结束处理
        self.mapex_active = False
        if time.time() - start_time >= self.navigation_timeout:
            print(f"  MapEx导航超时")
            return False
        
        final_pos, _ = self._get_robot_pose()
        final_error = np.linalg.norm(final_pos[:2] - target_pos[:2])
        print(f"  MapEx导航完成，最终误差: {final_error:.3f}m")
        return final_error < self.position_tolerance * 1.5
    
    def _execute_mapex_velocity_commands(self):
        """执行MapEx的速度命令 - 增强版本"""
        # 从MapEx接口获取速度命令
        mapex_velocity = self.mapex_interface.mapex_velocity
        
        linear_vel = float(mapex_velocity.linear.x)
        angular_vel = float(mapex_velocity.angular.z)
        
        # 限制速度范围
        linear_vel = np.clip(linear_vel, -self.max_linear_vel, self.max_linear_vel)
        angular_vel = np.clip(angular_vel, -self.max_angular_vel, self.max_angular_vel)
        
        # 平滑速度变化
        linear_vel = self._smooth_velocity(linear_vel, self.prev_linear)
        angular_vel = self._smooth_velocity(angular_vel, self.prev_angular)
        
        self.prev_linear = linear_vel
        self.prev_angular = angular_vel
        
        # 发送速度指令到机器人
        self._send_velocity_command(linear_vel, angular_vel)
        
        # 调试：显示接收到的原始指令
        if abs(mapex_velocity.linear.x) > 0.01 or abs(mapex_velocity.angular.z) > 0.01:
            print(f"Raw MapEx cmd: lin={mapex_velocity.linear.x:.3f}, ang={mapex_velocity.angular.z:.3f}")

    
    def move_to_position_robust(self, target_pos: np.ndarray, target_orientation: float = 0.0) -> bool:
        """鲁棒的移动到位置方法 - 备用方案，不使用MapEx"""
        print(f"直接导航到: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]")
        
        max_steps = MAX_NAVIGATION_STEPS
        step_count = 0
        
        start_pos = self._get_robot_pose()[0]
        start_distance = np.linalg.norm(start_pos[:2] - target_pos[:2])
        
        while step_count < max_steps:
            current_pos, current_yaw = self._get_robot_pose()
            step_count += 1
            
            # 超级平滑覆盖标记 - 每步都标记当前位置
            self._mark_coverage_ultra_smooth(current_pos)
            
            # 检查是否到达
            pos_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
            
            if pos_error < POSITION_TOLERANCE:
                self._send_zero_velocity()
                print(f"  成功到达! 误差: {pos_error:.3f}m, 步数: {step_count}")
                return True
            
            # 计算控制指令
            self._compute_and_send_control(current_pos, current_yaw, target_pos, target_orientation)
            
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
    
    def _compute_and_send_control(self, current_pos: np.ndarray, current_yaw: float, 
                                 target_pos: np.ndarray, target_yaw: float):
        """计算并发送控制指令"""
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
        
        # 平滑速度变化
        linear_vel = self._smooth_velocity(float(linear_vel), self.prev_linear)
        angular_vel = self._smooth_velocity(float(angular_vel), self.prev_angular)
        
        self.prev_linear = linear_vel
        self.prev_angular = angular_vel
        
        # 发送控制指令
        self._send_velocity_command(linear_vel, angular_vel)
    
    def _mark_coverage_ultra_smooth(self, robot_position: np.ndarray):
        """超级平滑覆盖标记 - 每步都尝试标记"""
        if self.coverage_visualizer and hasattr(self.coverage_visualizer, 'fluent_coverage_visualizer'):
            self.coverage_visualizer.fluent_coverage_visualizer.mark_coverage_realtime(robot_position)
    
    def _smooth_velocity(self, new_vel: float, prev_vel: float) -> float:
        """平滑速度变化"""
        return self.smooth_factor * prev_vel + (1 - self.smooth_factor) * new_vel
    
    def _send_velocity_command(self, linear_vel: float, angular_vel: float):
        """发送速度指令到轮子 - 增强调试版本"""
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
        left_idx = right_idx = None
        if "left_wheel_joint" in self.mobile_base.dof_names:
            left_idx = self.mobile_base.dof_names.index("left_wheel_joint")
            joint_velocities[left_idx] = float(left_wheel_vel)
        
        if "right_wheel_joint" in self.mobile_base.dof_names:
            right_idx = self.mobile_base.dof_names.index("right_wheel_joint")
            joint_velocities[right_idx] = float(right_wheel_vel)
        
        # 应用控制指令
        action = self.ArticulationAction(joint_velocities=joint_velocities)
        articulation_controller.apply_action(action)
        
        # 调试输出 - 只在非零速度时输出
        if abs(linear_vel) > 0.01 or abs(angular_vel) > 0.01:
            print(f"Speed cmd -> Isaac: lin={linear_vel:.3f}, ang={angular_vel:.3f}, "
                f"left_wheel={left_wheel_vel:.2f}, right_wheel={right_wheel_vel:.2f}")
            if left_idx is None or right_idx is None:
                print(f"Warning: 轮子关节未找到. left_idx={left_idx}, right_idx={right_idx}")
                print(f"Available joints: {self.mobile_base.dof_names}")
    
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
    
    def get_slam_status(self):
        """获取SLAM状态"""
        return {
            'mapex_active': self.mapex_active,
            'exploration_complete': self.mapex_interface.is_exploration_complete(),
            'map_available': self.cartographer_interface.get_slam_map() is not None,
            'robot_pose': self._get_robot_pose()
        }
    
    def emergency_stop(self):
        """紧急停止"""
        self._send_zero_velocity()
        self.mapex_active = False
        print("紧急停止执行")
    
    def publish_robot_status_to_ros(self):
        """发布机器人状态到ROS（供MapEx使用）"""
        try:
            # 获取当前位姿
            position, yaw = self._get_robot_pose()
            
            # 可以在这里发布tf变换或其他ROS消息
            # 让MapEx知道机器人的当前状态
            
            pass  # 具体实现取决于MapEx的要求
            
        except Exception as e:
            rospy.logwarn(f"发布机器人状态失败: {e}")