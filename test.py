#!/usr/bin/env python3
"""
Franka机械臂抓取调试器
简化版本，只包含地板、方块、task区域，专门用于调试抓取流程
"""

import numpy as np
import math
import torch
import time
from typing import List, Dict

print("启动Franka机械臂抓取调试器...")

from isaacsim import SimulationApp
simulation_app = SimulationApp({
    "headless": False,
    "enable_livestream": False, 
    "enable_cameras": True,
    "enable_rtx": True,
    "physics_dt": 1.0/60.0,
    "rendering_dt": 1.0/30.0,
})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdLux, UsdPhysics, Gf, UsdGeom
import isaacsim.core.utils.prims as prim_utils
from scipy.spatial.transform import Rotation as R

class FrankaGraspDebugger:
    """Franka机械臂抓取调试器"""
    
    def __init__(self):
        self.world = None
        self.mobile_base = None
        
        # 关键调试参数 - 方便修改
        self.grasp_approach_distance = 0.37  # 抓取接近距离 - 可调试
        self.arm_stabilization_time = 1.0   # 机械臂稳定时间 - 可调试
        self.gripper_stabilization_time = 1.0  # 夹爪稳定时间 - 可调试
        
        # Franka机械臂姿态 - 关键调试参数
        self.arm_poses = {
            "home": [0.0, 0.524, 0.0, -0.785, 0.0, 1.571, 0.785],     # 30°, -170°, 90°, 45°
            "grasp_approach": [0.0, 1.676, 0.0, -0.646, 0.0, 2.234, 0.785],  # 96°, -37°, 128°, 45°
            "grasp_lift": [0.0, 1.047, 0.0, -0.646, 0.0, 2.234, 0.785],     # 60°, -37°, 128°, 45°
        }
        
        # 夹爪位置 - 可调试
        self.gripper_open_pos = 0.04   # 张开位置
        self.gripper_close_pos = 0.025  # 闭合位置
        
        # 场景对象
        self.grasp_object = None
        self.task_area = None
        
        print("调试参数配置:")
        print(f"  抓取接近距离: {self.grasp_approach_distance}m")
        print(f"  机械臂稳定时间: {self.arm_stabilization_time}s")
        print(f"  夹爪稳定时间: {self.gripper_stabilization_time}s")
        print(f"  夹爪张开位置: {self.gripper_open_pos}m")
        print(f"  夹爪闭合位置: {self.gripper_close_pos}m")
    
    def initialize_scene(self):
        """初始化简化场景"""
        print("初始化简化调试场景...")
        
        self.world = World(
            stage_units_in_meters=1.0,
            physics_dt=1.0/60.0,
            rendering_dt=1.0/30.0
        )
        self.world.scene.clear()
        
        # 物理设置
        physics_context = self.world.get_physics_context()
        physics_context.set_gravity(-9.81)
        physics_context.enable_gpu_dynamics(True)
        
        # 1. 创建地板
        ground = FixedCuboid(
            prim_path="/World/Ground",
            name="ground",
            position=np.array([0.0, 0.0, -0.5]),
            scale=np.array([20.0, 20.0, 1.0]),
            color=np.array([0.5, 0.5, 0.5])
        )
        self.world.scene.add(ground)
        
        # 2. 创建抓取方块（绿色）
        self.grasp_object = DynamicCuboid(
            prim_path="/World/GraspObject",
            name="grasp_object",
            position=np.array([1.5, 0.0, 0.08]),
            scale=np.array([0.05, 0.05, 0.05]),
            color=np.array([0.2, 0.8, 0.2])
        )
        self.world.scene.add(self.grasp_object)
        
        # 3. 创建task区域（蓝色）
        self.task_area = FixedCuboid(
            prim_path="/World/TaskArea",
            name="task_area",
            position=np.array([3.0, 0.0, 0.2]),
            scale=np.array([1.0, 1.0, 0.4]),
            color=np.array([0.2, 0.2, 0.8])
        )
        self.world.scene.add(self.task_area)
        
        # 4. 设置照明
        self._setup_lighting()
        
        print("简化场景创建完成:")
        print(f"  抓取方块位置: [1.5, 0.0, 0.08]")
        print(f"  任务区域位置: [3.0, 0.0, 0.2]")
    
    def _setup_lighting(self):
        """设置照明"""
        main_light = prim_utils.create_prim("/World/MainLight", "DistantLight")
        distant_light = UsdLux.DistantLight(main_light)
        distant_light.CreateIntensityAttr(3000)
        distant_light.CreateColorAttr((1.0, 1.0, 0.9))
        
        env_light = prim_utils.create_prim("/World/EnvLight", "DomeLight")
        dome_light = UsdLux.DomeLight(env_light)
        dome_light.CreateIntensityAttr(500)
        distant_light.CreateColorAttr((0.8, 0.9, 1.0))
    
    def initialize_robot(self):
        """初始化机器人"""
        print("初始化机器人...")
        
        robot_usd_path = "/home/lwb/isaacsim_assets/Assets/Isaac/4.5/Isaac/Robots/iRobot/create_3_with_arm4.usd"
        
        self.mobile_base = WheeledRobot(
            prim_path="/World/create3_robot",
            name="create3_robot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=robot_usd_path,
            position=np.array([0.0, 0.0, 0.0])
        )
        
        self.world.scene.add(self.mobile_base)
        print("机器人初始化完成")
    
    def setup_post_load(self):
        """后加载设置"""
        print("后加载设置...")
        
        # 重置世界
        self.world.reset()
        
        # 稳定物理
        for _ in range(60):
            self.world.step(render=False)
        
        # 获取机器人对象
        self.mobile_base = self.world.scene.get_object("create3_robot")
        
        # 设置控制增益
        self._setup_control_gains()
        
        # 机械臂回到home位置
        self._move_arm_to_pose("home")
        
        print("后加载设置完成")
    
    def _setup_control_gains(self):
        """设置控制增益"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        num_dofs = len(self.mobile_base.dof_names)
        kp = torch.zeros(num_dofs, dtype=torch.float32)
        kd = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 轮子控制
        wheel_names = ["left_wheel_joint", "right_wheel_joint"]
        for wheel_name in wheel_names:
            if wheel_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(wheel_name)
                kp[idx] = 0.0
                kd[idx] = 500.0
        
        # 机械臂控制
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        for joint_name in arm_joint_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 800.0
                kd[idx] = 40.0
        
        # 夹爪控制
        gripper_names = ["panda_finger_joint1", "panda_finger_joint2"]
        for joint_name in gripper_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                kp[idx] = 1e5
                kd[idx] = 1e3
        
        articulation_controller.set_gains(kps=kp, kds=kd)
    
    def _move_arm_to_pose(self, pose_name: str):
        """移动机械臂到指定姿态"""
        print(f"机械臂移动到: {pose_name}")
        
        target_positions = self.arm_poses[pose_name]
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        # 获取当前关节位置，保持夹爪状态
        current_joint_positions = self.mobile_base.get_joint_positions()
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 保持当前所有关节位置
        for i in range(num_dofs):
            joint_positions[i] = float(current_joint_positions[i])
        
        # 只设置机械臂关节位置，保持夹爪位置不变
        arm_joint_names = [f"panda_joint{i+1}" for i in range(7)]
        for i, joint_name in enumerate(arm_joint_names):
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                if i < len(target_positions):
                    joint_positions[idx] = target_positions[i]
        
        # 应用动作
        action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)
        
        # 等待稳定
        stabilization_steps = int(self.arm_stabilization_time * 60)
        for _ in range(stabilization_steps):
            self.world.step(render=True)
        
        print(f"  机械臂稳定完成: {self.arm_stabilization_time}s")
    
    def _control_gripper(self, action: str):
        """控制夹爪"""
        print(f"夹爪动作: {action}")
        
        gripper_names = ["panda_finger_joint1", "panda_finger_joint2"]
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        # 设置夹爪位置
        if action == "close":
            gripper_pos = self.gripper_close_pos
        else:  # "open"
            gripper_pos = self.gripper_open_pos
        
        # 获取当前关节位置，避免其他关节移动
        current_joint_positions = self.mobile_base.get_joint_positions()
        num_dofs = len(self.mobile_base.dof_names)
        joint_positions = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 保持当前所有关节位置 - 确保类型转换
        for i in range(num_dofs):
            joint_positions[i] = float(current_joint_positions[i])
        
        # 只修改夹爪关节位置
        for joint_name in gripper_names:
            if joint_name in self.mobile_base.dof_names:
                idx = self.mobile_base.dof_names.index(joint_name)
                joint_positions[idx] = gripper_pos
        
        action_obj = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action_obj)
        
        # 夹爪稳定时间
        stabilization_steps = int(self.gripper_stabilization_time * 60)
        for _ in range(stabilization_steps):
            self.world.step(render=True)
        
        print(f"  夹爪动作完成: {action} -> {gripper_pos}m")
    
    def _get_robot_pose(self):
        """获取机器人位姿"""
        position, orientation = self.mobile_base.get_world_pose()
        position = np.array(position)
        
        # 四元数转欧拉角
        quat = np.array([orientation[1], orientation[2], orientation[3], orientation[0]])
        r = R.from_quat(quat)
        yaw = r.as_euler('xyz')[2]
        
        return position, yaw
    
    def _move_to_position(self, target_pos: np.ndarray, target_yaw: float = 0.0):
        """移动到目标位置"""
        print(f"移动到位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}], 朝向: {math.degrees(target_yaw):.1f}°")
        
        max_steps = 3000
        position_tolerance = 0.1
        angle_tolerance = 0.1
        
        # 控制参数
        linear_kp = 2.0
        angular_kp = 3.0
        max_linear_vel = 0.2
        max_angular_vel = 1.0
        
        for step in range(max_steps):
            current_pos, current_yaw = self._get_robot_pose()
            
            # 检查是否到达
            pos_error = np.linalg.norm(current_pos[:2] - target_pos[:2])
            yaw_error = self._normalize_angle(target_yaw - current_yaw)
            
            if pos_error < position_tolerance and abs(yaw_error) < angle_tolerance:
                print(f"  到达目标! 误差: {pos_error:.3f}m, {math.degrees(yaw_error):.1f}°")
                self._send_velocity(0.0, 0.0)
                return True
            
            # 计算控制指令
            direction = target_pos[:2] - current_pos[:2]
            distance = np.linalg.norm(direction)
            
            if distance > 0.001:
                direction_norm = direction / distance
                desired_yaw = np.arctan2(direction_norm[1], direction_norm[0])
                yaw_to_target = self._normalize_angle(desired_yaw - current_yaw)
                
                # 先转向，后前进
                if abs(yaw_to_target) > angle_tolerance:
                    linear_vel = 0.0
                    angular_vel = np.clip(angular_kp * yaw_to_target, -max_angular_vel, max_angular_vel)
                else:
                    linear_vel = np.clip(linear_kp * distance, 0.0, max_linear_vel)
                    angular_vel = np.clip(angular_kp * yaw_error * 0.5, -max_angular_vel, max_angular_vel)
            else:
                linear_vel = 0.0
                angular_vel = np.clip(angular_kp * yaw_error, -max_angular_vel, max_angular_vel)
            
            self._send_velocity(linear_vel, angular_vel)
            self.world.step(render=True)
            
            if step % 100 == 0:
                print(f"    步骤 {step}: 距离 {pos_error:.3f}m, 角度 {math.degrees(yaw_error):.1f}°")
        
        print(f"  移动超时!")
        self._send_velocity(0.0, 0.0)
        return False
    
    def _send_velocity(self, linear_vel: float, angular_vel: float):
        """发送速度指令"""
        articulation_controller = self.mobile_base.get_articulation_controller()
        
        # 轮子参数
        wheel_radius = 0.036
        wheel_base = 0.235
        
        # 计算轮子速度
        left_wheel_vel = (linear_vel - angular_vel * wheel_base / 2.0) / wheel_radius
        right_wheel_vel = (linear_vel + angular_vel * wheel_base / 2.0) / wheel_radius
        
        num_dofs = len(self.mobile_base.dof_names)
        joint_velocities = torch.zeros(num_dofs, dtype=torch.float32)
        
        # 设置轮子速度
        if "left_wheel_joint" in self.mobile_base.dof_names:
            left_idx = self.mobile_base.dof_names.index("left_wheel_joint")
            joint_velocities[left_idx] = float(left_wheel_vel)
        
        if "right_wheel_joint" in self.mobile_base.dof_names:
            right_idx = self.mobile_base.dof_names.index("right_wheel_joint")
            joint_velocities[right_idx] = float(right_wheel_vel)
        
        action = ArticulationAction(joint_velocities=joint_velocities)
        articulation_controller.apply_action(action)
    
    def _normalize_angle(self, angle):
        """角度归一化"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def run_grasp_debug_sequence(self):
        """运行抓取调试序列"""
        print("\n" + "="*50)
        print("开始Franka机械臂抓取调试序列")
        print("="*50)
        
        # 获取对象位置
        grasp_pos, _ = self.grasp_object.get_world_pose()
        grasp_pos = np.array(grasp_pos)
        
        task_pos, _ = self.task_area.get_world_pose()
        task_pos = np.array(task_pos)
        
        print(f"抓取对象位置: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
        print(f"任务区域位置: [{task_pos[0]:.3f}, {task_pos[1]:.3f}, {task_pos[2]:.3f}]")
        
        # 第一步: 移动到抓取对象正前方
        print(f"\n=== 步骤1: 移动到抓取对象正前方 ===")
        approach_pos = grasp_pos.copy()
        approach_pos[0] -= self.grasp_approach_distance  # 在X轴负方向后退
        approach_pos[2] = 0.0
        
        # 计算面向对象的角度
        direction_to_object = grasp_pos[:2] - approach_pos[:2]
        target_yaw = np.arctan2(direction_to_object[1], direction_to_object[0])
        
        print(f"接近位置: [{approach_pos[0]:.3f}, {approach_pos[1]:.3f}]")
        print(f"面向角度: {math.degrees(target_yaw):.1f}°")
        
        success = self._move_to_position(approach_pos, target_yaw)
        if not success:
            print("步骤1失败!")
            return
        
        # 等待稳定
        time.sleep(1.0)
        
        # 第二步: 执行抓取
        print(f"\n=== 步骤2: 执行抓取动作 ===")
        print("2.1 闭合夹爪抓取")
        self._control_gripper("close")
        print("2.2 机械臂移动到抓取姿态")
        self._move_arm_to_pose("grasp_approach")

        print("2.3 抬起物体")
        self._move_arm_to_pose("grasp_lift")
        
        print("抓取动作完成!")
        
        # 第三步: 移动到任务区域
        print(f"\n=== 步骤3: 移动到任务区域 ===")
        task_approach_pos = task_pos.copy()
        task_approach_pos[0] -= 0.8  # 在任务区域前方停止
        task_approach_pos[2] = 0.0
        
        success = self._move_to_position(task_approach_pos)
        if not success:
            print("步骤3失败!")
            return
        
        # 等待稳定
        time.sleep(1.0)
        
        # 第四步: 释放物体
        print(f"\n=== 步骤4: 释放物体 ===")
        print("4.1 张开夹爪释放")
        self._control_gripper("open")
        
        # 等待物体自然掉落
        print("4.2 等待物体掉落")
        for _ in range(60):
            self.world.step(render=True)
            time.sleep(0.01)
        
        print("4.3 机械臂回到home位置")
        self._move_arm_to_pose("home")
        
        print(f"\n=== 抓取调试序列完成! ===")
        print("请检查以下调试参数是否合适:")
        print(f"  抓取接近距离: {self.grasp_approach_distance}m")
        print(f"  机械臂稳定时间: {self.arm_stabilization_time}s")
        print(f"  夹爪稳定时间: {self.gripper_stabilization_time}s")
        print(f"  夹爪张开/闭合位置: {self.gripper_open_pos}m / {self.gripper_close_pos}m")

def main():
    """主函数"""
    print("Franka机械臂抓取调试器")
    print("专门用于调试抓取流程参数")
    
    debugger = FrankaGraspDebugger()
    
    print("\n=== 场景初始化 ===")
    debugger.initialize_scene()
    
    print("\n=== 机器人初始化 ===")
    debugger.initialize_robot()
    
    print("\n=== 后加载设置 ===")
    debugger.setup_post_load()
    
    print("\n=== 场景稳定 ===")
    for i in range(120):
        debugger.world.step(render=True)
        time.sleep(0.02)
        if i % 40 == 0:
            print(f"  稳定进度: {i+1}/120")
    
    print("\n=== 开始调试序列 ===")
    debugger.run_grasp_debug_sequence()
    
    print("\n=== 观察结果 ===")
    for i in range(300):
        debugger.world.step(render=True)
        time.sleep(0.05)
        if i % 60 == 0:
            print(f"  观察进度: {i+1}/300")
    
    print("调试完成!")

if __name__ == "__main__":
    main()