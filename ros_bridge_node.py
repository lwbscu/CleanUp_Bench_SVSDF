#!/usr/bin/env python3
"""
独立ROS桥接节点 - 运行在Python 3.8环境中
与Isaac Sim通过socket通信，避免Python版本冲突
"""

import rospy
import json
import socket
import threading
import time
import numpy as np
from geometry_msgs.msg import Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan
import tf2_ros
import tf2_geometry_msgs
from tf_conversions import transformations

class IsaacSimROSBridge:
    """Isaac Sim ROS桥接节点"""
    
    def __init__(self):
        rospy.init_node('isaac_sim_ros_bridge', anonymous=True)
        
        # Socket服务器配置
        self.socket_host = 'localhost'
        self.socket_port = 9999
        self.server_socket = None
        self.client_socket = None
        
        # ROS发布器
        self.velocity_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10)
        self.exploration_status_pub = rospy.Publisher('/exploration_status', String, queue_size=10)
        self.isaac_status_pub = rospy.Publisher('/isaac_sim_status', String, queue_size=10)
        
        # ROS订阅器
        self.map_sub = rospy.Subscriber('/map', OccupancyGrid, self.map_callback, queue_size=1)
        self.mapex_cmd_sub = rospy.Subscriber('/cmd_vel', Twist, self.mapex_cmd_callback, queue_size=10)
        self.exploration_done_sub = rospy.Subscriber('/exploration_done', Bool, self.exploration_done_callback, queue_size=1)
        
        # TF监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # 状态变量
        self.current_map = None
        self.exploration_complete = False
        self.mapex_velocity = Twist()
        self.robot_pose = None
        
        # 消息队列
        self.message_queue = []
        self.queue_lock = threading.Lock()
        
        print("Isaac Sim ROS桥接节点初始化完成")
        
    def start_socket_server(self):
        """启动socket服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.socket_host, self.socket_port))
            self.server_socket.listen(1)
            
            print(f"ROS桥接服务器启动，监听 {self.socket_host}:{self.socket_port}")
            
            while not rospy.is_shutdown():
                print("等待Isaac Sim连接...")
                self.client_socket, addr = self.server_socket.accept()
                print(f"Isaac Sim已连接: {addr}")
                
                # 启动通信线程
                comm_thread = threading.Thread(target=self.handle_client_communication)
                comm_thread.daemon = True
                comm_thread.start()
                
                # 发送连接状态
                self.publish_isaac_status("CONNECTED")
                
                # 等待连接断开
                comm_thread.join()
                print("Isaac Sim连接断开")
                
        except Exception as e:
            print(f"Socket服务器错误: {e}")
        finally:
            if self.server_socket:
                self.server_socket.close()
    
    def handle_client_communication(self):
        """处理与Isaac Sim的通信"""
        try:
            while not rospy.is_shutdown() and self.client_socket:
                # 接收Isaac Sim的消息
                try:
                    data = self.client_socket.recv(4096)
                    if not data:
                        break
                    
                    # 解析消息
                    message = json.loads(data.decode('utf-8'))
                    self.handle_isaac_message(message)
                    
                except socket.timeout:
                    continue
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}")
                    continue
                
                # 发送排队的消息到Isaac Sim
                self.send_queued_messages()
                
                time.sleep(0.01)  # 100Hz
                
        except Exception as e:
            print(f"通信错误: {e}")
        finally:
            if self.client_socket:
                self.client_socket.close()
                self.client_socket = None
    
    def handle_isaac_message(self, message):
        """处理来自Isaac Sim的消息"""
        msg_type = message.get('type')
        
        if msg_type == 'robot_pose':
            # 机器人位姿更新
            self.robot_pose = message.get('data')
            
        elif msg_type == 'request_goal':
            # 请求导航目标
            goal_data = message.get('data')
            self.send_navigation_goal(goal_data['x'], goal_data['y'], goal_data.get('yaw', 0.0))
            
        elif msg_type == 'exploration_status':
            # 探索状态更新
            status = message.get('data')
            self.publish_exploration_status(status)
        
        elif msg_type == 'heartbeat':
            # 心跳消息
            self.queue_message({
                'type': 'heartbeat_response',
                'timestamp': time.time()
            })
    
    def queue_message(self, message):
        """将消息加入队列"""
        with self.queue_lock:
            self.message_queue.append(message)
    
    def send_queued_messages(self):
        """发送队列中的消息"""
        if not self.client_socket:
            return
            
        with self.queue_lock:
            while self.message_queue:
                message = self.message_queue.pop(0)
                try:
                    data = json.dumps(message).encode('utf-8')
                    self.client_socket.send(data + b'\n')
                except Exception as e:
                    print(f"发送消息失败: {e}")
                    break
    
    def map_callback(self, msg: OccupancyGrid):
        """地图更新回调"""
        self.current_map = msg
        
        # 转换地图数据
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        
        # 转换占用栅格数据
        map_data = np.array(msg.data).reshape((height, width)).tolist()
        
        # 发送地图更新到Isaac Sim
        map_message = {
            'type': 'map_update',
            'data': {
                'width': width,
                'height': height,
                'resolution': resolution,
                'origin': origin,
                'data': map_data
            }
        }
        self.queue_message(map_message)
    
    def mapex_cmd_callback(self, msg: Twist):
        """MapEx命令回调"""
        self.mapex_velocity = msg
        
        # 发送速度命令到Isaac Sim
        velocity_message = {
            'type': 'velocity_command',
            'data': {
                'linear_x': msg.linear.x,
                'angular_z': msg.angular.z
            }
        }
        self.queue_message(velocity_message)
    
    def exploration_done_callback(self, msg: Bool):
        """探索完成回调"""
        self.exploration_complete = msg.data
        
        # 发送探索完成状态到Isaac Sim
        done_message = {
            'type': 'exploration_done',
            'data': msg.data
        }
        self.queue_message(done_message)
    
    def get_robot_pose_from_tf(self):
        """从TF获取机器人位姿"""
        try:
            # 获取base_link到map的变换
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link', rospy.Time(), rospy.Duration(1.0))
            
            # 提取位置
            position = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            
            # 提取航向角
            quaternion = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ]
            _, _, yaw = transformations.euler_from_quaternion(quaternion)
            
            # 发送位姿到Isaac Sim
            pose_message = {
                'type': 'robot_pose_tf',
                'data': {
                    'position': position,
                    'yaw': yaw
                }
            }
            self.queue_message(pose_message)
            
        except Exception as e:
            rospy.logdebug(f"无法获取机器人位姿: {e}")
    
    def send_navigation_goal(self, x: float, y: float, yaw: float = 0.0):
        """发送导航目标"""
        goal = PoseStamped()
        goal.header.frame_id = "map"
        goal.header.stamp = rospy.Time.now()
        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.position.z = 0.0
        
        # 转换偏航角为四元数
        quaternion = transformations.quaternion_from_euler(0, 0, yaw)
        goal.pose.orientation.x = quaternion[0]
        goal.pose.orientation.y = quaternion[1]
        goal.pose.orientation.z = quaternion[2]
        goal.pose.orientation.w = quaternion[3]
        
        self.goal_pub.publish(goal)
        rospy.loginfo(f"发送导航目标: ({x:.2f}, {y:.2f}, {np.degrees(yaw):.1f}°)")
    
    def publish_exploration_status(self, status: str):
        """发布探索状态"""
        msg = String()
        msg.data = status
        self.exploration_status_pub.publish(msg)
    
    def publish_isaac_status(self, status: str):
        """发布Isaac Sim状态"""
        msg = String()
        msg.data = status
        self.isaac_status_pub.publish(msg)
    
    def save_cartographer_map(self, filename: str) -> bool:
        """保存Cartographer地图"""
        try:
            # 调用Cartographer保存状态服务
            rospy.wait_for_service('/write_state', timeout=5.0)
            from cartographer_ros_msgs.srv import WriteState
            
            write_state = rospy.ServiceProxy('/write_state', WriteState)
            response = write_state(filename, True)  # include_unfinished_submaps=True
            
            if response.status.code == 0:  # SUCCESS
                rospy.loginfo(f"成功保存SLAM地图到: {filename}")
                
                # 通知Isaac Sim
                save_message = {
                    'type': 'map_saved',
                    'data': {
                        'filename': filename,
                        'success': True
                    }
                }
                self.queue_message(save_message)
                return True
            else:
                rospy.logerr(f"保存地图失败: {response.status.message}")
                return False
                
        except Exception as e:
            rospy.logerr(f"保存地图时发生异常: {e}")
            return False
    
    def run(self):
        """运行ROS桥接节点"""
        # 启动socket服务器线程
        server_thread = threading.Thread(target=self.start_socket_server)
        server_thread.daemon = True
        server_thread.start()
        
        # ROS主循环
        rate = rospy.Rate(30)  # 30Hz
        
        while not rospy.is_shutdown():
            # 定期更新机器人位姿
            self.get_robot_pose_from_tf()
            
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                break
        
        print("ROS桥接节点关闭")

if __name__ == '__main__':
    try:
        bridge = IsaacSimROSBridge()
        bridge.run()
    except rospy.ROSInterruptException:
        print("ROS桥接节点被中断")