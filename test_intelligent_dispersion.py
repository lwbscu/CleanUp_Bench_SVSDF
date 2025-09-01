#!/usr/bin/env python3
"""
测试智能物理距离分散算法
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# 模拟数据结构
class CoveragePoint:
    def __init__(self, position, orientation=0.0, has_object=False):
        self.position = np.array(position)
        self.orientation = orientation
        self.has_object = has_object

def create_test_path_points() -> List[CoveragePoint]:
    """创建测试路径点 - 模拟机器人在某些区域停留较多的情况"""
    points = []
    
    # 区域1: 密集停留区域 (0,0) 附近
    for i in range(50):
        x = np.random.normal(0, 0.3)  # 在(0,0)附近集中
        y = np.random.normal(0, 0.3)
        points.append(CoveragePoint([x, y, 0]))
    
    # 区域2: 中等密度区域 (3,2) 附近
    for i in range(30):
        x = np.random.normal(3, 0.5)
        y = np.random.normal(2, 0.5)
        points.append(CoveragePoint([x, y, 0]))
    
    # 区域3: 正常分布区域
    for i in range(40):
        x = np.random.uniform(-2, 5)
        y = np.random.uniform(-2, 4)
        points.append(CoveragePoint([x, y, 0]))
    
    # 区域4: 另一个密集区域 (-1, 3) 附近
    for i in range(35):
        x = np.random.normal(-1, 0.4)
        y = np.random.normal(3, 0.4)
        points.append(CoveragePoint([x, y, 0]))
    
    return points

def select_ghost_positions_intelligent(all_path_points: List[CoveragePoint], max_ghosts: int = 12) -> List[int]:
    """智能选择虚影位置 - 基于物理距离分散算法"""
    total_points = len(all_path_points)
    
    if total_points <= max_ghosts:
        return list(range(total_points))
    
    print(f"开始智能物理距离分散算法: {total_points}个路径点，目标{max_ghosts}个虚影")
    
    selected_indices = []
    
    # 1. 必须包含起点和终点
    selected_indices.append(0)
    if total_points > 1:
        selected_indices.append(total_points - 1)
    
    # 2. 使用贪心算法选择物理距离最分散的点
    min_distance_threshold = 1.0  # 最小间距阈值(米)
    remaining_slots = max_ghosts - len(selected_indices)
    
    # 候选点池：排除已选择的起点和终点
    candidate_indices = [i for i in range(1, total_points - 1)]
    
    for slot in range(remaining_slots):
        if not candidate_indices:
            break
        
        best_candidate = None
        best_min_distance = 0
        
        # 对每个候选点，计算它到已选点的最小距离
        for candidate_idx in candidate_indices:
            candidate_pos = all_path_points[candidate_idx].position
            
            # 计算到所有已选点的最小距离
            min_distance_to_selected = float('inf')
            for selected_idx in selected_indices:
                selected_pos = all_path_points[selected_idx].position
                distance = np.linalg.norm(candidate_pos[:2] - selected_pos[:2])
                min_distance_to_selected = min(min_distance_to_selected, distance)
            
            # 选择与已选点距离最远的候选点
            if min_distance_to_selected > best_min_distance:
                best_min_distance = min_distance_to_selected
                best_candidate = candidate_idx
        
        if best_candidate is not None:
            selected_indices.append(best_candidate)
            candidate_indices.remove(best_candidate)
            print(f"选择虚影点 {len(selected_indices)}/{max_ghosts}: 索引{best_candidate}, 最小间距{best_min_distance:.2f}m")
        else:
            break
    
    # 验证分散效果
    selected_indices_sorted = sorted(selected_indices)
    print(f"物理距离分散验证:")
    for i, idx in enumerate(selected_indices_sorted):
        pos = all_path_points[idx].position
        print(f"  虚影{i+1}: 索引{idx}, 位置[{pos[0]:.2f}, {pos[1]:.2f}]")
    
    # 计算平均间距
    if len(selected_indices_sorted) > 1:
        total_distance = 0
        distance_count = 0
        for i in range(len(selected_indices_sorted)):
            for j in range(i+1, len(selected_indices_sorted)):
                pos1 = all_path_points[selected_indices_sorted[i]].position
                pos2 = all_path_points[selected_indices_sorted[j]].position
                distance = np.linalg.norm(pos1[:2] - pos2[:2])
                total_distance += distance
                distance_count += 1
        
        avg_distance = total_distance / distance_count if distance_count > 0 else 0
        print(f"平均虚影间距: {avg_distance:.2f}m")
        
        # 找出最小间距
        min_distance = float('inf')
        for i in range(len(selected_indices_sorted)):
            for j in range(i+1, len(selected_indices_sorted)):
                pos1 = all_path_points[selected_indices_sorted[i]].position
                pos2 = all_path_points[selected_indices_sorted[j]].position
                distance = np.linalg.norm(pos1[:2] - pos2[:2])
                min_distance = min(min_distance, distance)
        print(f"最小虚影间距: {min_distance:.2f}m")
    
    print(f"智能物理距离分散完成: 选择{len(selected_indices_sorted)}个虚影点")
    
    return selected_indices_sorted

def select_ghost_positions_old(all_path_points: List[CoveragePoint], max_ghosts: int = 12) -> List[int]:
    """旧版虚影选择算法 - 均匀分布"""
    total_points = len(all_path_points)
    
    if total_points <= max_ghosts:
        return list(range(total_points))
    
    selected_indices = []
    
    # 包含起点和终点
    selected_indices.append(0)
    if total_points > 1:
        selected_indices.append(total_points - 1)
    
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

def visualize_comparison():
    """可视化对比新旧算法"""
    # 创建测试数据
    test_points = create_test_path_points()
    
    # 获取新旧算法结果
    intelligent_indices = select_ghost_positions_intelligent(test_points, max_ghosts=12)
    old_indices = select_ghost_positions_old(test_points, max_ghosts=12)
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制所有路径点
    all_x = [p.position[0] for p in test_points]
    all_y = [p.position[1] for p in test_points]
    
    # 左图：旧算法
    ax1.scatter(all_x, all_y, c='lightgray', s=20, alpha=0.6, label='所有路径点')
    old_x = [test_points[i].position[0] for i in old_indices]
    old_y = [test_points[i].position[1] for i in old_indices]
    ax1.scatter(old_x, old_y, c='red', s=100, alpha=0.8, label='虚影位置(旧算法)')
    ax1.set_title('旧算法: 均匀分布选择')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：新算法
    ax2.scatter(all_x, all_y, c='lightgray', s=20, alpha=0.6, label='所有路径点')
    intelligent_x = [test_points[i].position[0] for i in intelligent_indices]
    intelligent_y = [test_points[i].position[1] for i in intelligent_indices]
    ax2.scatter(intelligent_x, intelligent_y, c='green', s=100, alpha=0.8, label='虚影位置(智能分散)')
    ax2.set_title('新算法: 智能物理距离分散')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/lwb/Project/CleanUp_Bench_SVSDF/ghost_dispersion_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n算法对比总结:")
    print(f"旧算法选择的虚影数量: {len(old_indices)}")
    print(f"新算法选择的虚影数量: {len(intelligent_indices)}")
    
    # 计算分散程度
    def calculate_dispersion_score(indices, points):
        if len(indices) < 2:
            return 0, 0, 0
        
        distances = []
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                pos1 = points[indices[i]].position
                pos2 = points[indices[j]].position
                distance = np.linalg.norm(pos1[:2] - pos2[:2])
                distances.append(distance)
        
        return np.mean(distances), np.min(distances), np.std(distances)
    
    old_avg, old_min, old_std = calculate_dispersion_score(old_indices, test_points)
    new_avg, new_min, new_std = calculate_dispersion_score(intelligent_indices, test_points)
    
    print(f"\n分散程度对比:")
    print(f"旧算法 - 平均间距: {old_avg:.2f}m, 最小间距: {old_min:.2f}m, 标准差: {old_std:.2f}m")
    print(f"新算法 - 平均间距: {new_avg:.2f}m, 最小间距: {new_min:.2f}m, 标准差: {new_std:.2f}m")
    
    improvement_avg = (new_avg - old_avg) / old_avg * 100 if old_avg > 0 else 0
    improvement_min = (new_min - old_min) / old_min * 100 if old_min > 0 else 0
    
    print(f"\n改进效果:")
    print(f"平均间距改进: {improvement_avg:+.1f}%")
    print(f"最小间距改进: {improvement_min:+.1f}%")

def main():
    print("智能物理距离分散算法测试")
    print("="*50)
    
    # 运行可视化对比
    visualize_comparison()
    
    print("\n测试完成！")
    print("结果图像已保存为: ghost_dispersion_comparison.png")

if __name__ == "__main__":
    main()
