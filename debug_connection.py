#!/usr/bin/env python3
"""
调试脚本：测试MapEx和桥接节点之间的连接
"""

import socket
import json
import time
import threading

def test_connection():
    """测试连接到桥接节点"""
    try:
        print("尝试连接到桥接节点...")
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(10.0)
        client.connect(('localhost', 9998))
        
        print("✓ 连接成功!")
        
        # 设置接收线程
        def receive_messages():
            client.settimeout(1.0)
            while True:
                try:
                    data = client.recv(4096)
                    if data:
                        print(f"📥 收到桥接节点消息: {data.decode('utf-8').strip()}")
                    else:
                        break
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"接收消息错误: {e}")
                    break
        
        # 启动接收线程
        receive_thread = threading.Thread(target=receive_messages, daemon=True)
        receive_thread.start()
        
        # 发送测试消息
        test_messages = [
            {'type': 'test_connection', 'timestamp': time.time()},
            {'type': 'velocity_command', 'data': {'linear_x': 0.1, 'angular_z': 0.0}},
            {'type': 'velocity_command', 'data': {'linear_x': 0.0, 'angular_z': 0.2}},
            {'type': 'velocity_command', 'data': {'linear_x': 0.0, 'angular_z': 0.0}},
            {'type': 'exploration_status', 'data': 'EXPLORATION_STARTED'},
        ]
        
        for i, msg in enumerate(test_messages):
            try:
                data = json.dumps(msg).encode('utf-8')
                client.send(data + b'\n')
                print(f"📤 发送测试消息 {i+1}: {msg['type']}")
                time.sleep(2.0)  # 等待2秒观察反应
            except Exception as e:
                print(f"❌ 发送消息失败: {e}")
                break
        
        # 保持连接一段时间
        print("保持连接15秒，观察消息交换...")
        time.sleep(15.0)
        
        client.close()
        print("连接已关闭")
        
    except Exception as e:
        print(f"❌ 连接失败: {e}")

if __name__ == '__main__':
    test_connection()
