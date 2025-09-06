#!/usr/bin/env python3
"""
MapEx桥接节点修复版 - 修复Socket服务器启动问题
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
    """MapEx桥接节点 - 修复Socket服务器问题"""
    
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
        
        # 线程管理
        self.server_thread = None
        self.mapex_monitor_thread = None
        self.running = True
        
        print("MapEx桥接节点初始化完成")
    
    def start_socket_server(self):
        """启动Socket服务器 - 修复版"""
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
                    self.mapex_client_socket, addr = self.mapex_server_socket.accept()
                    self.mapex_connected = True
                    
                    print(f"MapEx已连接: {addr}")
                    
                    # 发送连接确认
                    self._send_to_mapex({
                        'type': 'bridge_connected',
                        'timestamp': time.time()
                    })
                    
                    # 启动通信处理
                    self._handle_mapex_communication()
                    
                except Exception as e:
                    print(f"接受连接时出错: {e}")
                    self.mapex_connected = False
                    if self.mapex_client_socket:
                        self.mapex_client_socket.close()
                        self.mapex_client_socket = None
                
                if not self.running:
                    break
                    
                print("MapEx连接断开，等待重连...")
                time.sleep(1.0)
                
        except Exception as e:
            print(f"Socket服务器错误: {e}")
        finally:
            self._cleanup_sockets()
    
    def _handle_mapex_communication(self):
        """处理MapEx通信"""
        try:
            while self.running and self.mapex_connected and self.mapex_client_socket:
                # 发送数据到MapEx
                self._send_queued_data_to_mapex()
                
                # 接收MapEx的命令
                self._receive_mapex_commands()
                
                time.sleep(0.1)  # 10Hz
                
        except Exception as e:
            print(f"MapEx通信错误: {e}")
            self.mapex_connected = False
    
    def _send_to_mapex(self, message):
        """发送消息到MapEx"""
        if not self.mapex_connected or not self.mapex_client_socket:
            return False
        
        try:
            data = json.dumps(message).encode('utf-8')
            self.mapex_client_socket.send(data + b'\n')
            return True
        except Exception as e:
            print(f"发送消息到MapEx失败: {e}")
            self.mapex_connected = False
            return False
    
    def _receive_mapex_commands(self):
        """接收MapEx的命令"""
        if not self.mapex_connected or not self.mapex_client_socket:
            return
        
        try:
            # 设置非阻塞接收
            self.mapex_client_socket.settimeout(0.1)
            data = self.mapex_client_socket.recv(4096)
            
            if data:
                message = json.loads(data.decode('utf-8'))
                self._handle_mapex_command(message)
                
        except socket.timeout:
            pass  # 正常超时，继续
        except Exception as e:
            print(f"接收MapEx命令时出错: {e}")
            self.mapex_connected = False
    
    def _handle_mapex_command(self, command):
        """处理来自MapEx的命令"""
        cmd_type = command.get('type')
        
        if cmd_type == 'velocity_command':
            # 发布速度命令
            data = command.get('data', {})
            twist = Twist()
            twist.linear.x = data.get('linear_x', 0.0)
            twist.angular.z = data.get('angular_z', 0.0)
            self.cmd_vel_pub.publish(twist)
            print(f"发布速度命令: linear={twist.linear.x:.3f}, angular={twist.angular.z:.3f}")
            
        elif cmd_type == 'exploration_status':
            # 发布探索状态
            status = command.get('data', 'UNKNOWN')
            msg = String()
            msg.data = status
            self.exploration_status_pub.publish(msg)
            
            if status == 'EXPLORATION_COMPLETED':
                # 探索完成
                self.exploration_active = False
                done_msg = Bool()
                done_msg.data = True
                self.exploration_done_pub.publish(done_msg)
                print("MapEx探索完成!")
            
        elif cmd_type == 'request_goal':
            # MapEx请求新的探索目标
            data = command.get('data', {})
            goal_msg = Float32MultiArray()
            goal_msg.data = [data.get('x', 0.0), data.get('y', 0.0), data.get('yaw', 0.0)]
            self.mapex_goal_pub.publish(goal_msg)
            
        elif cmd_type == 'heartbeat':
            # 心跳响应
            self._send_to_mapex({
                'type': 'heartbeat_response',
                'timestamp': time.time()
            })
    
    def _send_queued_data_to_mapex(self):
        """发送队列中的数据到MapEx"""
        if not self.mapex_connected:
            return
        
        # 发送地图数据（如果有更新）
        if self.current_map and time.time() - self.last_map_update < 1.0:
            map_message = {
                'type': 'map_update',
                'data': self.current_map
            }
            self._send_to_mapex(map_message)
        
        # 发送机器人位姿
        pose_message = {
            'type': 'robot_pose',
            'data': {
                'x': self.robot_pose[0],
                'y': self.robot_pose[1],
                'yaw': self.robot_pose[2]
            }
        }
        self._send_to_mapex(pose_message)
    
    def map_callback(self, msg: OccupancyGrid):
        """地图更新回调"""
        # 转换地图数据为MapEx可用格式
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        
        # 转换占用栅格数据
        map_data = np.array(msg.data).reshape((height, width))
        
        self.current_map = {
            'width': width,
            'height': height,
            'resolution': resolution,
            'origin': origin,
            'data': map_data.tolist()
        }
        
        self.last_map_update = time.time()
        print(f"收到地图更新: {width}x{height}, 分辨率: {resolution:.3f}m/cell")
    
    def lidar_callback(self, msg: PointCloud2):
        """激光雷达数据回调"""
        # MapEx主要使用地图数据，激光雷达数据可以用于实时避障
        pass
    
    def robot_pose_callback(self, msg: Float32MultiArray):
        """机器人位姿回调"""
        if len(msg.data) >= 3:
            self.robot_pose = [msg.data[0], msg.data[1], msg.data[2]]
    
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
    
    def run(self):
        """运行桥接节点"""
        print("MapEx桥接节点开始运行...")
        
        # 启动Socket服务器线程
        self.server_thread = threading.Thread(target=self.start_socket_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # 等待其他节点启动
        rospy.sleep(2.0)
        
        # 主循环
        rate = rospy.Rate(10)  # 10Hz
        
        try:
            while not rospy.is_shutdown() and self.running:
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