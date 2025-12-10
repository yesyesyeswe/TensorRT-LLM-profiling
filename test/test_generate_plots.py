#!/usr/bin/env python3
"""
测试修改后的generate_bandwidth_plots函数
验证它能正确生成带宽、延迟、通信三种对比图
"""

import sys
import os
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from run_benchmark import generate_bandwidth_plots

def test_generate_plots():
    """测试生成所有三种对比图"""
    print("=== 测试生成所有三种对比图 ===")
    
    # 测试参数 - 使用实际可用的数据
    sequence_lengths = ['128', '256']  # 基于results/avg_bandwidth.csv中的实际数据
    batch_sizes = [1]  # 基于results/avg_bandwidth.csv中的实际数据
    tp_sizes = [2]
    algorithms = ['NCCL_LL', 'ONESHOT']  # 基于results/avg_bandwidth.csv中的实际数据
    
    # 测试所有三种指标和两种X轴类型
    comparison_metrics = ['bandwidth', 'latency', 'communication']
    x_axis_types = ['sequence_length', 'batch_size']
    
    print(f"测试参数:")
    print(f"  sequence_lengths: {sequence_lengths}")
    print(f"  batch_sizes: {batch_sizes}")
    print(f"  tp_sizes: {tp_sizes}")
    print(f"  algorithms: {algorithms}")
    print(f"  comparison_metrics: {comparison_metrics}")
    print(f"  x_axis_types: {x_axis_types}")
    
    try:
        # 调用函数
        generate_bandwidth_plots(
            sequence_lengths=sequence_lengths,
            batch_sizes=batch_sizes,
            tp_sizes=tp_sizes,
            algorithms=algorithms,
            comparison_metrics=comparison_metrics,
            x_axis_types=x_axis_types
        )
        print("\n✅ 测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_individual_metrics():
    """分别测试每种指标"""
    print("\n=== 分别测试每种指标 ===")
    
    test_cases = [
        {
            'name': '仅带宽对比图',
            'comparison_metrics': ['bandwidth'],
            'x_axis_types': ['sequence_length']
        },
        {
            'name': '仅延迟对比图', 
            'comparison_metrics': ['latency'],
            'x_axis_types': ['batch_size']
        },
        {
            'name': '仅通信对比图',
            'comparison_metrics': ['communication'],
            'x_axis_types': ['sequence_length']
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        try:
            generate_bandwidth_plots(
                sequence_lengths=['128', '256'],  # 使用实际可用的数据
                batch_sizes=[1],  # 使用实际可用的数据
                tp_sizes=[2],
                algorithms=['NCCL_LL'],  # 使用实际可用的数据
                comparison_metrics=test_case['comparison_metrics'],
                x_axis_types=test_case['x_axis_types']
            )
            print(f"✅ {test_case['name']} 测试通过")
        except Exception as e:
            print(f"❌ {test_case['name']} 测试失败: {e}")

if __name__ == "__main__":
    print("开始测试修改后的generate_bandwidth_plots函数...")
    
    # 检查数据文件是否存在
    results_dir = Path(__file__).parent / "results"
    avg_bandwidth_csv = results_dir / "avg_bandwidth.csv"
    
    if not avg_bandwidth_csv.exists():
        print(f"警告: {avg_bandwidth_csv} 不存在，将创建测试数据...")
        # 这里可以添加创建测试数据的代码
        print("请确保有测试数据文件，或者先运行基准测试生成数据")
        sys.exit(1)
    
    # 运行测试
    test_generate_plots()
    test_individual_metrics()
    
    print("\n=== 所有测试完成 ===")