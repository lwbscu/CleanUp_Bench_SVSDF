#!/usr/bin/env python3
"""
修复版MapEx桥接节点 - 自动启动探索并确保正确通信
"""

import rospy
import json
import socket
import threading
import time
import subprocess
import os
import signal
import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Bool, String, Float32MultiArray
from sensor_msgs.msg import PointCloud2

class MapExBridgeNode:
    """修复版MapEx桥接节点 - 自动探索启动"""
    
    def __init__(self):
        rospy.init_node('mapex_bridge_node', anonymous=True)
        
        # Socket服务器配置 - 作为服务器等待MapEx连接
        self.mapex_socket_host = 'localhost'
        self.mapex_socket_port = 9998  # MapEx专用端口
        self.mapex_server_socket = None  # 服务器socket
        self.mapex_client_socket = None  # 客户端连接
        self.mapex_connected = False
        
        # MapEx进程管理
        self.mapex_process = None
        self.mapex_script_path = "/home/lwb/MapEx/scripts/explore.py"
        self.mapex_config_path = "/home/lwb/MapEx/configs/base.yaml"
        
        # ROS发布器
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.exploration_status_pub = rospy.Publisher('/exploration_status', String, queue_size=10)
        self.exploration_done_pub = rospy.Publisher('/exploration_done', Bool, queue_size=10)
        self.mapex_goal_pub = rospy.Publisher('/mapex/goal', Float32MultiArray, queue_size=10)
        
        # ROS订阅器
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback, queue_size=1)
        self.lidar_sub = rospy.Subscriber('/robot_lidar_pointcloud', PointCloud2, self.lidar_callback, queue_size=1)
        self.robot_pose_sub = rospy.Subscriber('/robot_pose', Float32MultiArray, self.robot_pose_callback, queue_size=10)
        
        # 状态变量
        self.current_map = None
        self.robot_pose = [0.0, 0.0, 0.0]  # x, y, yaw
        self.exploration_active = False
        self.last_map_update = time.time()
        self.map_received_count = 0
        self.pose_received_count = 0
        
        # 关键修复：自动启动探索的条件
        self.auto_start_enabled = True
        self.auto_start_conditions = {
            'map_received': False,
            'pose_received': False,
            'mapex_connected': False,
            'exploration_started': False
        }
        self.min_map_updates = 3  # 至少收到3次地图更新
        self.min_pose_updates = 5  # 至少收到5次位姿更新
        
        # 线程管理
        self.server_thread = None
        self.mapex_monitor_thread = None
        self.running = True
        
        print("修复版MapEx桥接节点初始化完成")
        print("自动探索启动已启用")
    
    def start_socket_server(self):
        """启动Socket服务器 - 修复连接问题"""
        try:
            # 创建服务器socket
            self.mapex_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.mapex_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # 绑定端口并开始监听
            self.mapex_server_socket.bind((self.mapex_socket_host, self.mapex_socket_port))
            self.mapex_server_socket.listen(1)
            
            print(f"MapEx Socket服务器启动，监听 {self.mapex_socket_host}:{self.mapex_socket_port}")
            
            while self.running:
                try:
                    print("等待MapEx连接...")
                    
                    # 关键修复：使用更短的超时时间，避免长时间阻塞
                    self.mapex_server_socket.settimeout(1.0)  # 1秒超时
                    
                    try:
                        self.mapex_client_socket, addr = self.mapex_server_socket.accept()
                        
                        # 立即设置连接状态
                        self.mapex_connected = True
                        self.auto_start_conditions['mapex_connected'] = True
                        
                        print(f"✓ MapEx已连接: {addr}")
                        print(f"✓ 连接状态已更新: {self.mapex_connected}")
                        
                        # 设置客户端socket为非阻塞模式
                        self.mapex_client_socket.settimeout(0.1)
                        
                        # 发送连接确认
                        confirm_sent = self._send_to_mapex({
                            'type': 'bridge_connected',
                            'timestamp': time.time(),
                            'status': 'connected'
                        })
                        
                        if confirm_sent:
                            print("✓ 连接确认消息已发送")
                        else:
                            print("❌ 连接确认消息发送失败")
                        
                        # 启动通信处理 - 强制保持连接
                        print("开始MapEx通信处理...")
                        
                        # 关键修复：一旦连接就不再等待新连接，专注处理当前连接
                        while self.running and self.mapex_connected and self.mapex_client_socket:
                            try:
                                # 发送数据到MapEx
                                self._send_queued_data_to_mapex()
                                
                                # 接收MapEx的命令
                                self._receive_mapex_commands()
                                
                                # 检查自动启动条件
                                self._check_auto_start_conditions()
                                
                                # 强制发送探索命令给已连接的MapEx
                                self._force_exploration_if_needed()
                                
                                # 关键修复：增加适当延时，避免CPU占用过高
                                time.sleep(0.2)  # 5Hz，减缓通信频率
                                
                            except Exception as comm_error:
                                print(f"MapEx通信处理错误: {comm_error}")
                                # 不要因为小错误就断开连接，稍作延时继续
                                time.sleep(0.5)
                                continue
                        
                        print("MapEx通信循环结束，连接将关闭")
                        
                    except socket.timeout:
                        # 接受连接超时，继续循环
                        continue
                    
                except Exception as e:
                    print(f"接受连接时出错: {e}")
                    self.mapex_connected = False
                    self.auto_start_conditions['mapex_connected'] = False
                    if self.mapex_client_socket:
                        self.mapex_client_socket.close()
                        self.mapex_client_socket = None
                
                if not self.running:
                    break
                    
                if self.mapex_connected:
                    print("MapEx连接断开，等待重连...")
                    self.mapex_connected = False
                    self.auto_start_conditions['mapex_connected'] = False
                
                time.sleep(0.5)  # 短暂等待后继续监听
                
        except Exception as e:
            print(f"Socket服务器错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup_sockets()
    
    def _force_exploration_if_needed(self):
        """关键修复：强制触发探索如果MapEx空闲太久"""
        current_time = time.time()
        
        # 如果MapEx已连接但长时间没有速度命令，强制触发探索
        if (self.mapex_connected and 
            self.auto_start_conditions['exploration_started'] and
            not hasattr(self, 'last_velocity_command_time')):
            
            # 初始化时间戳
            self.last_velocity_command_time = current_time
            self.idle_start_time = current_time
        
        # 检查是否长时间空闲（超过10秒无速度命令）
        if (hasattr(self, 'last_velocity_command_time') and 
            current_time - self.last_velocity_command_time > 10.0):
            
            if not hasattr(self, 'idle_start_time'):
                self.idle_start_time = current_time
            
            idle_duration = current_time - self.idle_start_time
            
            # 每30秒强制发送一次探索命令
            if idle_duration > 30.0 and int(idle_duration) % 30 == 0:
                print(f"MapEx空闲{idle_duration:.0f}秒，强制触发探索...")
                
                # 发送强制探索命令
                self._send_to_mapex({
                    'type': 'force_exploration',
                    'timestamp': current_time,
                    'data': {
                        'reason': 'idle_timeout',
                        'current_pose': self.robot_pose,
                        'map_info': {
                            'width': self.current_map['width'] if self.current_map else 0,
                            'height': self.current_map['height'] if self.current_map else 0
                        }
                    }
                })
                
                # 重置空闲时间
                self.idle_start_time = current_time
    
    def _check_auto_start_conditions(self):
        """关键修复：检查并触发自动启动探索"""
        if not self.auto_start_enabled or self.auto_start_conditions['exploration_started']:
            return
        
        # 检查所有启动条件
        conditions_met = (
            self.auto_start_conditions['map_received'] and
            self.auto_start_conditions['pose_received'] and
            self.auto_start_conditions['mapex_connected'] and
            self.map_received_count >= self.min_map_updates and
            self.pose_received_count >= self.min_pose_updates
        )
        
        if conditions_met:
            print("自动启动条件满足，开始MapEx探索...")
            print(f"  地图更新: {self.map_received_count}/{self.min_map_updates}")
            print(f"  位姿更新: {self.pose_received_count}/{self.min_pose_updates}")
            print(f"  MapEx连接: {self.auto_start_conditions['mapex_connected']}")
            
            # 发送详细的探索开始命令到MapEx
            exploration_command = {
                'type': 'start_exploration',
                'timestamp': time.time(),
                'data': {
                    'mode': 'autonomous_slam',
                    'initial_pose': self.robot_pose,
                    'map_info': {
                        'width': self.current_map['width'] if self.current_map else 0,
                        'height': self.current_map['height'] if self.current_map else 0,
                        'resolution': self.current_map['resolution'] if self.current_map else 0.05
                    },
                    'exploration_params': {
                        'max_linear_velocity': 0.3,
                        'max_angular_velocity': 2.5,  # 大幅提高角速度从1.0到2.5
                        'exploration_radius': 5.0,
                        'frontier_threshold': 0.1
                    }
                }
            }
            
            success = self._send_to_mapex(exploration_command)
            
            if success:
                # 发布探索状态
                self.publish_exploration_status("EXPLORATION_AUTO_STARTED")
                
                # 标记探索已启动
                self.auto_start_conditions['exploration_started'] = True
                self.exploration_active = True
                
                print("MapEx自动探索已启动!")
                
                # 初始化速度命令监控
                self.last_velocity_command_time = time.time()
            else:
                print("发送探索命令失败，MapEx连接异常")
    
    def _send_to_mapex(self, message):
        """发送消息到MapEx - 彻底修复缓冲区问题"""
        if not self.mapex_connected or not self.mapex_client_socket:
            return False
        
        try:
            # 关键修复：根据消息类型采用不同策略
            msg_type = message.get('type', 'unknown')
            data = json.dumps(message).encode('utf-8')
            
            # 对于关键消息（速度命令、探索命令），使用阻塞发送确保送达
            if msg_type in ['velocity_command', 'start_exploration', 'force_exploration']:
                try:
                    # 设置较短的发送超时，避免长时间阻塞
                    self.mapex_client_socket.settimeout(0.5)  # 500ms超时
                    self.mapex_client_socket.send(data + b'\n')
                    
                    # 成功发送关键消息
                    if msg_type == 'velocity_command':
                        vel_data = message.get('data', {})
                        print(f"🚀 关键速度命令已发送: linear={vel_data.get('linear_x', 0):.3f}, angular={vel_data.get('angular_z', 0):.3f}")
                    else:
                        print(f"✓ 关键消息已发送: {msg_type}")
                    
                    return True
                    
                except socket.timeout:
                    print(f"❌ 关键消息发送超时: {msg_type}")
                    return False
                except Exception as e:
                    print(f"❌ 关键消息发送失败: {msg_type}, 错误: {e}")
                    return False
            
            # 对于非关键消息（地图、位姿），使用非阻塞发送，失败时直接跳过
            else:
                try:
                    self.mapex_client_socket.setblocking(False)
                    self.mapex_client_socket.send(data + b'\n')
                    return True
                except BlockingIOError:
                    # 非关键消息被跳过，这是正常的
                    if msg_type not in ['map_update', 'robot_pose']:
                        print(f"⚠️ 非关键消息跳过: {msg_type}")
                    return False
                except Exception as e:
                    print(f"❌ 非关键消息发送失败: {msg_type}")
                    return False
            
        except Exception as e:
            print(f"❌ 发送消息到MapEx时出现异常: {e}")
            self.mapex_connected = False
            self.auto_start_conditions['mapex_connected'] = False
            return False
        finally:
            # 恢复默认阻塞模式
            try:
                if self.mapex_client_socket:
                    self.mapex_client_socket.setblocking(True)
                    self.mapex_client_socket.settimeout(1.0)  # 恢复1秒默认超时
            except:
                pass
    
    def _receive_mapex_commands(self):
        """接收MapEx的命令 - 修复超时问题"""
        if not self.mapex_connected or not self.mapex_client_socket:
            return
        
        try:
            # 关键修复：使用更长的接收超时时间
            self.mapex_client_socket.settimeout(1.0)  # 1秒超时
            data = self.mapex_client_socket.recv(4096)
            
            if data:
                # 处理可能包含多个JSON对象的数据
                data_str = data.decode('utf-8')
                
                # 将接收到的数据添加到缓冲区
                if not hasattr(self, 'message_buffer'):
                    self.message_buffer = ""
                
                self.message_buffer += data_str
                
                # 按行分割处理多个JSON消息
                while '\n' in self.message_buffer:
                    line, self.message_buffer = self.message_buffer.split('\n', 1)
                    line = line.strip()
                    
                    if line:
                        try:
                            message = json.loads(line)
                            self._handle_mapex_command(message)
                        except json.JSONDecodeError as e:
                            print(f"MapEx JSON解析错误: {e}")
                            continue
                
        except socket.timeout:
            pass  # 正常超时，继续
        except ConnectionResetError:
            print("MapEx连接被重置")
            self.mapex_connected = False
            self.auto_start_conditions['mapex_connected'] = False
        except Exception as e:
            print(f"接收MapEx命令时出错: {e}")
            # 不要因为小错误就断开连接
            pass
    
    def _handle_mapex_command(self, command):
        """处理来自MapEx的命令 - 增强版本"""
        cmd_type = command.get('type')
        
        if cmd_type == 'velocity_command':
            # 发布速度命令
            data = command.get('data', {})
            twist = Twist()
            twist.linear.x = data.get('linear_x', 0.0)
            twist.angular.z = data.get('angular_z', 0.0)
            self.cmd_vel_pub.publish(twist)
            
            # 关键修复：更新速度命令时间戳
            self.last_velocity_command_time = time.time()
            
            # 强制输出所有速度命令（包括零速度）- 降低频率，渐进式角速度标识
            current_time = time.time()
            if not hasattr(self, 'last_velocity_debug_time') or current_time - self.last_velocity_debug_time > 2.0:
                # 关键修复：根据角速度大小分级显示
                abs_angular = abs(twist.angular.z)
                if abs_angular > 1.5:
                    print(f"🚀 MapEx高角速度命令: linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f} (⚡快速转弯)")
                elif abs_angular > 0.8:
                    print(f"🚀 MapEx中角速度命令: linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f} (🔄中速转弯)")
                elif abs_angular > 0.1:
                    print(f"🚀 MapEx低角速度命令: linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f} (🎯微调)")
                else:
                    print(f"🚀 MapEx速度命令: linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f}")
                self.last_velocity_debug_time = current_time
            
        elif cmd_type == 'exploration_status':
            # 发布探索状态
            status = command.get('data', 'UNKNOWN')
            self.publish_exploration_status(status)
            
            if status == 'EXPLORATION_COMPLETED':
                # 探索完成
                self.exploration_active = False
                done_msg = Bool()
                done_msg.data = True
                self.exploration_done_pub.publish(done_msg)
                print("MapEx探索完成!")
            elif status == 'EXPLORATION_STARTED':
                print("MapEx确认探索已开始")
                self.last_velocity_command_time = time.time()
            
        elif cmd_type == 'request_goal':
            # MapEx请求新的探索目标
            data = command.get('data', {})
            goal_msg = Float32MultiArray()
            goal_msg.data = [data.get('x', 0.0), data.get('y', 0.0), data.get('yaw', 0.0)]
            self.mapex_goal_pub.publish(goal_msg)
            print(f"MapEx请求目标: [{data.get('x', 0.0):.2f}, {data.get('y', 0.0):.2f}]")
            
        elif cmd_type == 'heartbeat':
            # 心跳响应
            self._send_to_mapex({
                'type': 'heartbeat_response',
                'timestamp': time.time()
            })
            
        elif cmd_type == 'frontier_info':
            # 前沿信息（调试用）
            data = command.get('data', {})
            frontier_count = data.get('frontier_count', 0)
            print(f"MapEx前沿检测: 发现{frontier_count}个前沿区域")
        
        else:
            print(f"⚠️ 未知MapEx命令类型: {cmd_type}")
    
    def _send_queued_data_to_mapex(self):
        """发送队列中的数据到MapEx - 优化频率"""
        if not self.mapex_connected:
            return
        
        current_time = time.time()
        
        # 关键修复：降低地图发送频率，避免Socket缓冲区溢出
        if (self.current_map and 
            current_time - getattr(self, 'last_map_send_time', 0) > 0.5):  # 2Hz发送地图
            
            map_message = {
                'type': 'map_update',
                'data': self.current_map
            }
            success = self._send_to_mapex(map_message)
            if success:
                self.last_map_send_time = current_time
        
        # 关键修复：降低位姿发送频率，但确保数据精度
        if current_time - getattr(self, 'last_pose_send_time', 0) > 0.1:  # 10Hz发送位姿，提高频率
            pose_message = {
                'type': 'robot_pose',
                'data': {
                    'x': float(self.robot_pose[0]),  # 确保精度
                    'y': float(self.robot_pose[1]),
                    'yaw': float(self.robot_pose[2])
                },
                'timestamp': current_time  # 添加时间戳
            }
            success = self._send_to_mapex(pose_message)
            if success:
                self.last_pose_send_time = current_time
    
    def map_callback(self, msg: OccupancyGrid):
        """地图更新回调 - 修复负值哈希问题"""
        # 转换地图数据为MapEx可用格式
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        
        # 关键修复：检查地图是否有实质性变化
        current_time = time.time()
        
        # 修复：正确处理包含负值的地图数据
        map_data_array = np.array(msg.data, dtype=np.int8)
        map_hash = hash(map_data_array.tobytes())  # 使用numpy数组的tobytes()方法
        
        # 避免发送相同的地图
        if (hasattr(self, 'last_map_hash') and 
            map_hash == self.last_map_hash and 
            current_time - getattr(self, 'last_map_send_time', 0) < 1.0):
            return  # 1秒内相同地图不重复发送
        
        # 转换占用栅格数据
        map_data = map_data_array.reshape((height, width))
        
        self.current_map = {
            'width': width,
            'height': height,
            'resolution': resolution,
            'origin': origin,
            'data': map_data.tolist()
        }
        
        self.last_map_update = current_time
        self.last_map_hash = map_hash
        self.map_received_count += 1
        
        # 关键修复：标记地图接收状态
        if not self.auto_start_conditions['map_received']:
            self.auto_start_conditions['map_received'] = True
            print(f"首次收到地图数据: {width}x{height}, 分辨率: {resolution:.3f}m/cell")
        
        # 定期状态报告 - 降低频率
        if self.map_received_count % 20 == 0:  # 每20次更新报告一次
            # 计算地图完成度
            unknown_count = np.sum(map_data == -1)
            total_cells = width * height
            known_ratio = (total_cells - unknown_count) / total_cells if total_cells > 0 else 0
            
            print(f"地图更新计数: {self.map_received_count}, 已知区域: {known_ratio:.1%}")
    
    def robot_pose_callback(self, msg: Float32MultiArray):
        """机器人位姿回调 - 增强调试版本"""
        if len(msg.data) >= 3:
            old_pose = self.robot_pose.copy()
            self.robot_pose = [msg.data[0], msg.data[1], msg.data[2]]
            self.pose_received_count += 1
            
            # 关键修复：标记位姿接收状态
            if not self.auto_start_conditions['pose_received']:
                self.auto_start_conditions['pose_received'] = True
                print(f"首次收到机器人位姿: [{self.robot_pose[0]:.3f}, {self.robot_pose[1]:.3f}], yaw: {np.degrees(self.robot_pose[2]):.1f}°")
            
            # 位姿变化检测和调试输出 - 降低频率
            current_time = time.time()
            if (abs(old_pose[0] - self.robot_pose[0]) > 0.01 or 
                abs(old_pose[1] - self.robot_pose[1]) > 0.01 or 
                abs(old_pose[2] - self.robot_pose[2]) > 0.1):
                
                # 限制输出频率：每3秒最多输出一次位姿变化
                if not hasattr(self, 'last_pose_debug_time') or current_time - self.last_pose_debug_time > 3.0:
                    print(f"🔄 桥接节点位姿变化: [{self.robot_pose[0]:.3f}, {self.robot_pose[1]:.3f}], yaw: {np.degrees(self.robot_pose[2]):.1f}°")
                    self.last_pose_debug_time = current_time
            
            # 定期状态报告 - 每100次更新报告一次，降低频率
            if self.pose_received_count % 100 == 0:  # 降低频率到每100次
                print(f"位姿更新计数: {self.pose_received_count}, 当前位姿: [{self.robot_pose[0]:.3f}, {self.robot_pose[1]:.3f}], yaw: {np.degrees(self.robot_pose[2]):.1f}°")
    
    def lidar_callback(self, msg: PointCloud2):
        """激光雷达数据回调"""
        # MapEx主要使用地图数据，激光雷达数据可以用于实时避障
        pass
    
    def publish_exploration_status(self, status: str):
        """发布探索状态"""
        msg = String()
        msg.data = status
        self.exploration_status_pub.publish(msg)
        print(f"发布探索状态: {status}")
    
    def _cleanup_sockets(self):
        """清理socket连接"""
        if self.mapex_client_socket:
            try:
                self.mapex_client_socket.close()
            except:
                pass
            self.mapex_client_socket = None
            
        if self.mapex_server_socket:
            try:
                self.mapex_server_socket.close()
            except:
                pass
            self.mapex_server_socket = None
        
        self.mapex_connected = False
        self.auto_start_conditions['mapex_connected'] = False
    
    def print_status_summary(self):
        """打印状态摘要 - 增强版本"""
        print("\n=== MapEx桥接状态摘要 ===")
        print(f"地图接收: {self.auto_start_conditions['map_received']} (计数: {self.map_received_count})")
        print(f"位姿接收: {self.auto_start_conditions['pose_received']} (计数: {self.pose_received_count})")
        print(f"MapEx连接: {self.auto_start_conditions['mapex_connected']}")
        print(f"探索已启动: {self.auto_start_conditions['exploration_started']}")
        print(f"当前位姿: [{self.robot_pose[0]:.3f}, {self.robot_pose[1]:.3f}], yaw: {np.degrees(self.robot_pose[2]):.1f}°")
        if self.current_map:
            print(f"当前地图: {self.current_map['width']}x{self.current_map['height']}")
        
        # 关键修复：显示速度命令状态
        if hasattr(self, 'last_velocity_command_time'):
            time_since_last_cmd = time.time() - self.last_velocity_command_time
            print(f"最后速度命令: {time_since_last_cmd:.1f}秒前")
        else:
            print(f"最后速度命令: 从未收到")
        
        print("========================\n")
    
    def run(self):
        """运行桥接节点"""
        print("MapEx桥接节点开始运行...")
        
        # 启动Socket服务器线程
        self.server_thread = threading.Thread(target=self.start_socket_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        print("Socket服务器线程已启动")
        
        # 等待其他节点启动
        rospy.sleep(2.0)
        
        # 主循环
        rate = rospy.Rate(10)  # 10Hz
        last_status_time = time.time()
        
        try:
            while not rospy.is_shutdown() and self.running:
                # 关键修复：检查服务器线程状态
                if not self.server_thread.is_alive():
                    print("❌ Socket服务器线程已停止，尝试重启...")
                    self.server_thread = threading.Thread(target=self.start_socket_server)
                    self.server_thread.daemon = True
                    self.server_thread.start()
                
                # 定期打印状态摘要
                if time.time() - last_status_time > 15.0:
                    self.print_status_summary()
                    last_status_time = time.time()
                
                rate.sleep()
                
        except rospy.ROSInterruptException:
            pass
        except KeyboardInterrupt:
            print("收到中断信号")
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """关闭桥接节点"""
        print("正在关闭MapEx桥接节点...")
        
        self.running = False
        
        # 清理socket连接
        self._cleanup_sockets()
        
        # 等待线程结束
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=3.0)
        
        print("MapEx桥接节点已关闭")

if __name__ == '__main__':
    try:
        bridge = MapExBridgeNode()
        bridge.run()
    except rospy.ROSInterruptException:
        print("MapEx桥接节点被中断")
    except Exception as e:
        print(f"MapEx桥接节点出错: {e}")
        import traceback
        traceback.print_exc()