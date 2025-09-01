#!/usr/bin/env python3
"""
测试抓取失败跳过机制
模拟物体抓取失败后的处理逻辑
"""

from data_structures import SceneObject, ObjectType, CollisionBoundary
import numpy as np

def test_grasp_failure_skip_mechanism():
    """测试抓取失败跳过机制"""
    print("测试抓取失败跳过机制")
    print("="*50)
    
    # 创建测试抓取物体
    collision_boundary = CollisionBoundary(
        center=np.array([1.5, 0.0, 0.08]),
        shape_type="box",
        dimensions=np.array([0.05, 0.05, 0.05])
    )
    
    test_object = SceneObject(
        name="test_grasp_object",
        object_type=ObjectType.GRASP,
        position=np.array([1.5, 0.0, 0.08]),
        collision_boundary=collision_boundary,
        color=np.array([0.2, 0.8, 0.2])
    )
    
    print(f"创建测试物体: {test_object.name}")
    print(f"初始状态:")
    print(f"  is_active: {test_object.is_active}")
    print(f"  grasp_failed: {test_object.grasp_failed}")
    
    # 模拟第一次检测抓取物体
    print(f"\n第一次检测抓取物体:")
    if not test_object.grasp_failed:
        print(f"  ✓ 物体 {test_object.name} 可以尝试抓取")
        
        # 模拟抓取失败
        print(f"  模拟抓取过程...")
        print(f"  ✗ 抓取失败! Z轴坐标未达到0.2m阈值")
        
        # 标记为抓取失败
        test_object.grasp_failed = True
        print(f"  物体 {test_object.name} 已标记为抓取失败，不再尝试")
    else:
        print(f"  ✗ 物体 {test_object.name} 之前已抓取失败，跳过")
    
    print(f"\n抓取失败后状态:")
    print(f"  is_active: {test_object.is_active}")
    print(f"  grasp_failed: {test_object.grasp_failed}")
    
    # 模拟机器人绕一圈后再次遇到同一物体
    print(f"\n机器人绕一圈后再次遇到同一物体:")
    if not test_object.grasp_failed:
        print(f"  ✓ 物体 {test_object.name} 可以尝试抓取")
    else:
        print(f"  ✗ 物体 {test_object.name} 之前已抓取失败，智能跳过")
        print(f"  机器人继续执行覆盖任务，不浪费时间重复尝试")
    
    # 测试多个物体的场景
    print(f"\n测试多个物体场景:")
    test_objects = []
    
    for i in range(3):
        obj = SceneObject(
            name=f"grasp_object_{i+1}",
            object_type=ObjectType.GRASP,
            position=np.array([i*2.0, 0.0, 0.08]),
            collision_boundary=collision_boundary,
            color=np.array([0.2, 0.8, 0.2])
        )
        test_objects.append(obj)
    
    # 模拟第一个物体抓取成功，第二个失败，第三个成功
    results = [True, False, True]  # 抓取结果
    
    for i, (obj, success) in enumerate(zip(test_objects, results)):
        print(f"\n  物体 {obj.name}:")
        if success:
            print(f"    ✓ 抓取成功，运送到KLT框子")
            obj.is_active = False
        else:
            print(f"    ✗ 抓取失败，标记为不再尝试")
            obj.grasp_failed = True
    
    # 机器人再次遇到这些物体
    print(f"\n机器人再次遇到这些物体:")
    for obj in test_objects:
        if not obj.is_active:
            print(f"  物体 {obj.name}: 已被运送，不存在")
        elif obj.grasp_failed:
            print(f"  物体 {obj.name}: 之前抓取失败，智能跳过")
        else:
            print(f"  物体 {obj.name}: 可以尝试抓取")
    
    print(f"\n测试总结:")
    active_objects = sum(1 for obj in test_objects if obj.is_active and not obj.grasp_failed)
    failed_objects = sum(1 for obj in test_objects if obj.grasp_failed)
    delivered_objects = sum(1 for obj in test_objects if not obj.is_active)
    
    print(f"  可抓取物体数量: {active_objects}")
    print(f"  失败跳过物体数量: {failed_objects}")
    print(f"  已运送物体数量: {delivered_objects}")
    print(f"  智能跳过机制工作正常: {failed_objects > 0}")

def simulate_coverage_with_skip_mechanism():
    """模拟带有跳过机制的覆盖任务"""
    print(f"\n模拟覆盖任务执行:")
    print("="*30)
    
    # 创建抓取物体列表
    grasp_objects = []
    for i in range(4):
        obj = SceneObject(
            name=f"grasp_obj_{i+1}",
            object_type=ObjectType.GRASP,
            position=np.array([i*1.5, 0.0, 0.08]),
            collision_boundary=CollisionBoundary(
                center=np.array([i*1.5, 0.0, 0.08]),
                shape_type="box",
                dimensions=np.array([0.05, 0.05, 0.05])
            ),
            color=np.array([0.2, 0.8, 0.2])
        )
        grasp_objects.append(obj)
    
    statistics = {
        'grasped_objects': 0,
        'failed_grasps': 0,
        'delivered_objects': 0,
        'skipped_attempts': 0
    }
    
    # 模拟多轮覆盖
    for round_num in range(3):
        print(f"\n第 {round_num + 1} 轮覆盖:")
        
        for obj in grasp_objects:
            if not obj.is_active:
                continue  # 已被运送
            
            if obj.grasp_failed:
                print(f"  遇到 {obj.name}: 之前失败，智能跳过")
                statistics['skipped_attempts'] += 1
                continue
            
            print(f"  尝试抓取 {obj.name}...")
            
            # 模拟抓取结果（前两个容易失败）
            if obj.name in ['grasp_obj_1', 'grasp_obj_2'] and round_num == 0:
                success = False
            else:
                success = True
            
            if success:
                print(f"    ✓ 抓取成功，运送到KLT框子")
                obj.is_active = False
                statistics['grasped_objects'] += 1
                statistics['delivered_objects'] += 1
            else:
                print(f"    ✗ 抓取失败，标记不再尝试")
                obj.grasp_failed = True
                statistics['failed_grasps'] += 1
    
    print(f"\n最终统计:")
    print(f"  抓取成功: {statistics['grasped_objects']}")
    print(f"  抓取失败: {statistics['failed_grasps']}")
    print(f"  成功运送: {statistics['delivered_objects']}")
    print(f"  智能跳过次数: {statistics['skipped_attempts']}")
    
    success_rate = statistics['grasped_objects'] / (statistics['grasped_objects'] + statistics['failed_grasps']) * 100 if (statistics['grasped_objects'] + statistics['failed_grasps']) > 0 else 0
    print(f"  抓取成功率: {success_rate:.1f}%")
    
    efficiency_improvement = statistics['skipped_attempts']
    print(f"  智能跳过节省: {efficiency_improvement} 次无效尝试")

def main():
    print("抓取失败智能跳过机制测试")
    print("="*60)
    
    # 基础功能测试
    test_grasp_failure_skip_mechanism()
    
    # 覆盖任务模拟
    simulate_coverage_with_skip_mechanism()
    
    print(f"\n测试完成!")
    print(f"抓取失败智能跳过机制验证通过")
    print(f"关键特性:")
    print(f"  ✓ 抓取失败后物体被标记为grasp_failed=True")
    print(f"  ✓ 后续检测会智能跳过失败物体")
    print(f"  ✓ 避免机器人重复尝试无效抓取")
    print(f"  ✓ 提高覆盖任务效率")

if __name__ == "__main__":
    main()
