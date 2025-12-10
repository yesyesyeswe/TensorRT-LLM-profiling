#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Tuple

class BandwidthPlotter:
    def __init__(self, data_file: str):
        """初始化绘图器"""
        self.df = pd.read_csv(data_file)
        self.setup_style()
        
    def setup_style(self):
        """设置绘图样式"""
        plt.style.use('seaborn-v0_8')
        #sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def filter_data(self, 
                   algorithm: Optional[str] = None,
                   batch_size: Optional[int] = None,
                   sequence_length: Optional[int] = None,
                   tp_size: Optional[int] = None,
                   cuda_device: Optional[int] = None) -> pd.DataFrame:
        """根据条件过滤数据"""
        filtered_df = self.df.copy()
        
        if algorithm:
            filtered_df = filtered_df[filtered_df['Algorithm'] == algorithm]
        if batch_size:
            filtered_df = filtered_df[filtered_df['Batch_size'] == batch_size]
        if sequence_length:
            filtered_df = filtered_df[filtered_df['Sequence_length'] == sequence_length]
        if tp_size:
            filtered_df = filtered_df[filtered_df['TP_Size'] == tp_size]
        if cuda_device is not None:
            filtered_df = filtered_df[filtered_df['CudaDevice'] == cuda_device]
            
        return filtered_df
    
    def plot_seq_length_vs_bandwidth(self, batch_size: int, tp_size: int = 2, 
                                   algorithms: Optional[List[str]] = None):
        """
        需求1：给定batchsize，画sequence_length不同时的带宽变化
        最多两列：第一列Prefill，第二列Decoder，使用对数坐标折线图
        每行代表一个GPU设备，如果有TP，则显示TP行
        """
        if algorithms is None:
            algorithms = self.df['Algorithm'].unique()
        
        # 过滤数据
        filtered_df = self.filter_data(batch_size=batch_size, tp_size=tp_size)
        if algorithms:
            filtered_df = filtered_df[filtered_df['Algorithm'].isin(algorithms)]
        
        # 获取唯一的算法和CudaDevice
        unique_algos = filtered_df['Algorithm'].unique()
        unique_devices = sorted(filtered_df['CudaDevice'].unique())
        
        # 检查是否有数据
        if len(unique_algos) == 0 or len(unique_devices) == 0:
            print(f"警告: 没有符合条件的数据。batch_size={batch_size}, tp_size={tp_size}")
            return None
        
        # 创建子图：每个算法一行，每行有两列（Prefill和Decoder）
        # 如果TP size > 1，则为每个设备创建单独的行
        if tp_size > 1:
            n_rows = len(unique_algos) * len(unique_devices)
            device_labels = {dev: f"GPU{dev}" for dev in unique_devices}
        else:
            n_rows = len(unique_algos)
            device_labels = {dev: "Single GPU" for dev in unique_devices}
            
        n_cols = 2  # 固定两列：Prefill和Decoder
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(12, 4*n_rows), 
                                squeeze=False)
        
        row_idx = 0
        for algo in unique_algos:
            # 获取该算法的数据
            algo_data = filtered_df[filtered_df['Algorithm'] == algo]
            
            if len(algo_data) == 0:
                # 为Prefill和Decoder子图都显示No Data
                for col in range(2):
                    axes[row_idx, col].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                                           transform=axes[row_idx, col].transAxes)
                row_idx += 1
                continue
            
            # 为每个设备创建单独的行
            for device in unique_devices:
                device_data = algo_data[algo_data['CudaDevice'] == device]
                
                if len(device_data) == 0:
                    # 为Prefill和Decoder子图都显示No Data
                    for col in range(2):
                        axes[row_idx, col].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                                               transform=axes[row_idx, col].transAxes)
                    row_idx += 1
                    continue
                
                # 按sequence_length分组
                seq_groups = device_data.groupby('Sequence_length').agg({
                    'Prefill': 'mean',
                    'Decoder': 'mean'
                }).reset_index()
                
                # 排序sequence_length
                seq_groups = self._sort_sequence_lengths(seq_groups)
                
                # 转换sequence_length为数值用于绘图
                seq_lengths_num = seq_groups['Sequence_length']
                
                # Prefill子图 (第0列)
                ax_prefill = axes[row_idx, 0]
                if len(seq_groups) > 0:
                    line_prefill = ax_prefill.loglog(seq_lengths_num, seq_groups['Prefill'], 
                                     marker='o', linewidth=2, markersize=6, 
                                     label=f'{algo} - {device_labels[device]}', 
                                     base=2)
                    # 在数据点处添加纵坐标值标注
                    for x, y in zip(seq_lengths_num, seq_groups['Prefill']):
                        if not pd.isna(y) and y > 0:
                            ax_prefill.annotate(f'{y:.1f}', (x, y), 
                                              textcoords="offset points", 
                                              xytext=(0,10), ha='center', fontsize=8)
                ax_prefill.set_xlabel('Sequence Length (log₂ scale)')
                ax_prefill.set_ylabel('Bandwidth (GB/s) (log₂ scale)')
                device_title = f"{device_labels[device]}" if tp_size > 1 else ""
                ax_prefill.set_title(f'{algo} {device_title} - Prefill Bandwidth (Log₂-Log₂ Scale)', pad=10)
                ax_prefill.grid(True, alpha=0.3)
                ax_prefill.legend()
                
                # Decoder子图 (第1列)
                ax_decoder = axes[row_idx, 1]
                if len(seq_groups) > 0:
                    line_decoder = ax_decoder.loglog(seq_lengths_num, seq_groups['Decoder'], 
                                     marker='s', linewidth=2, markersize=6, 
                                     label=f'{algo} - {device_labels[device]}', 
                                     base=2)
                    # 在数据点处添加纵坐标值标注
                    for x, y in zip(seq_lengths_num, seq_groups['Decoder']):
                        if not pd.isna(y) and y > 0:
                            ax_decoder.annotate(f'{y:.1f}', (x, y), 
                                                textcoords="offset points", 
                                                xytext=(0,10), ha='center', fontsize=8)
                ax_decoder.set_xlabel('Sequence Length (log₂ scale)')
                ax_decoder.set_ylabel('Bandwidth (GB/s) (log₂ scale)')
                ax_decoder.set_title(f'{algo} {device_title} - Decoder Bandwidth (Log₂-Log₂ Scale)', pad=10)
                ax_decoder.grid(True, alpha=0.3)
                ax_decoder.legend()
                
                row_idx += 1
        
        plt.subplots_adjust(top=1)
        plt.tight_layout()
        return fig
    
    def plot_batch_size_vs_bandwidth(self, sequence_length: int, tp_size: int = 2,
                                   algorithms: Optional[List[str]] = None):
        """
        需求2：给定sequence_length，画batch_size不同时的带宽变化
        最多两列：第一列Prefill，第二列Decoder，使用对数坐标折线图
        每行代表一个GPU设备，如果有TP，则显示TP行
        """
        if algorithms is None:
            algorithms = self.df['Algorithm'].unique()
        
        # 过滤数据
        filtered_df = self.filter_data(sequence_length=sequence_length, tp_size=tp_size)
        if algorithms:
            filtered_df = filtered_df[filtered_df['Algorithm'].isin(algorithms)]
        
        # 获取唯一的算法和CudaDevice
        unique_algos = filtered_df['Algorithm'].unique()
        unique_devices = sorted(filtered_df['CudaDevice'].unique())
        
        # 检查是否有数据
        if len(unique_algos) == 0 or len(unique_devices) == 0:
            print(f"警告: 没有符合条件的数据。sequence_length={sequence_length}, tp_size={tp_size}")
            return None
        
        # 创建子图：每个算法一行，每行有两列（Prefill和Decoder）
        # 如果TP size > 1，则为每个设备创建单独的行
        if tp_size > 1:
            n_rows = len(unique_algos) * len(unique_devices)
            device_labels = {dev: f"GPU{dev}" for dev in unique_devices}
        else:
            n_rows = len(unique_algos)
            device_labels = {dev: "Single GPU" for dev in unique_devices}
            
        n_cols = 2  # 固定两列：Prefill和Decoder
        
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(12, 4*n_rows), 
                                squeeze=False)
        
        row_idx = 0
        for algo in unique_algos:
            # 获取该算法的数据
            algo_data = filtered_df[filtered_df['Algorithm'] == algo]
            
            if len(algo_data) == 0:
                # 为Prefill和Decoder子图都显示No Data
                for col in range(2):
                    axes[row_idx, col].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                                           transform=axes[row_idx, col].transAxes)
                row_idx += 1
                continue
            
            # 为每个设备创建单独的行
            for device in unique_devices:
                device_data = algo_data[algo_data['CudaDevice'] == device]
                
                if len(device_data) == 0:
                    # 为Prefill和Decoder子图都显示No Data
                    for col in range(2):
                        axes[row_idx, col].text(0.5, 0.5, 'No Data', ha='center', va='center', 
                                               transform=axes[row_idx, col].transAxes)
                    row_idx += 1
                    continue
                
                # 按batch_size分组
                batch_groups = device_data.groupby('Batch_size').agg({
                    'Prefill': 'mean',
                    'Decoder': 'mean'
                }).reset_index()
                
                # 排序batch_size
                batch_groups = batch_groups.sort_values('Batch_size')
                
                # Prefill子图 (第0列)
                ax_prefill = axes[row_idx, 0]
                if len(batch_groups) > 0:
                    line_prefill = ax_prefill.loglog(batch_groups['Batch_size'], batch_groups['Prefill'], 
                                     marker='o', linewidth=2, markersize=6, 
                                     label=f'{algo} - {device_labels[device]}')
                    # 在数据点处添加纵坐标值标注
                    for x, y in zip(batch_groups['Batch_size'], batch_groups['Prefill']):
                        if not pd.isna(y) and y > 0:
                            ax_prefill.annotate(f'{y:.1f}', (x, y), 
                                                textcoords="offset points", 
                                                xytext=(0,10), ha='center', fontsize=8)
                ax_prefill.set_xlabel('Batch Size')
                ax_prefill.set_ylabel('Bandwidth (GB/s)')
                device_title = f"{device_labels[device]}" if tp_size > 1 else ""
                ax_prefill.set_title(f'{algo} {device_title} - Prefill Bandwidth (Log-Log Scale)', pad=10)
                ax_prefill.grid(True, alpha=0.3)
                ax_prefill.legend()
                
                # Decoder子图 (第1列)
                ax_decoder = axes[row_idx, 1]
                if len(batch_groups) > 0:
                    line_decoder = ax_decoder.loglog(batch_groups['Batch_size'], batch_groups['Decoder'], 
                                     marker='s', linewidth=2, markersize=6, 
                                     label=f'{algo} - {device_labels[device]}')
                    # 在数据点处添加纵坐标值标注
                    for x, y in zip(batch_groups['Batch_size'], batch_groups['Decoder']):
                        if not pd.isna(y) and y > 0:
                            ax_decoder.annotate(f'{y:.1f}', (x, y), 
                                                textcoords="offset points", 
                                                xytext=(0,10), ha='center', fontsize=8)
                ax_decoder.set_xlabel('Batch Size')
                ax_decoder.set_ylabel('Bandwidth (GB/s)')
                ax_decoder.set_title(f'{algo} {device_title} - Decoder Bandwidth (Log-Log Scale)', pad=10)
                ax_decoder.grid(True, alpha=0.3)
                ax_decoder.legend()
                
                row_idx += 1
        
        plt.subplots_adjust(top=1)
        plt.tight_layout()
        return fig
    
    def _seq_length_to_int(self, seq: str) -> int:
        """将sequence length字符串转换为整数"""
        mapping = {'128': 128, '256': 256, '512': 512, '1k': 1024, 
                  '2k': 2048, '4k': 4096, '8k': 8192, '16k': 16384, '32k': 32768}
        return mapping.get(seq, 0)
    
    def _sort_sequence_lengths(self, df: pd.DataFrame) -> pd.DataFrame:
        """按数值大小排序sequence length"""
        df = df.copy()
        df['seq_int'] = df['Sequence_length'].apply(self._seq_length_to_int)
        df = df.sort_values('seq_int')
        df = df.drop('seq_int', axis=1)
        return df
    
    def _add_value_labels(self, bars, ax):
        """在柱状图上添加数值标签"""
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
    
    def save_plot(self, fig, filename: str, dpi: int = 300):
        """保存图表"""
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Plot bandwidth analysis from avg_bandwidth.csv")
    parser.add_argument("--data-file", 
                       default="/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/avg_bandwidth.csv",
                       help="Input CSV file with average bandwidth data")
    parser.add_argument("--output-dir", 
                       default="/root/autodl-tmp/TensorRT-LLM/mybenchmark/figures",
                       help="Output directory for plots")
    
    # 绘图参数
    parser.add_argument("--plot-type", choices=['seq_vs_bandwidth', 'batch_vs_bandwidth'], required=True,
                       help="Type of plot to generate")
    parser.add_argument("--fixed-value", type=str, required=True,
                       help="Fixed value for plotting (batch_size or sequence_length)")
    parser.add_argument("--tp-size", type=int, default=2,
                       help="TP size to filter (default: 2)")
    parser.add_argument("--algorithms", nargs='*', default=None,
                       help="Algorithms to include (default: all)")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化绘图器
    plotter = BandwidthPlotter(args.data_file)
    
    # 生成图表
    if args.plot_type == 'seq_vs_bandwidth':
        # 给定batch size，画sequence length变化
        batch_size = int(args.fixed_value)
        fig = plotter.plot_seq_length_vs_bandwidth(
            batch_size=batch_size,
            tp_size=args.tp_size,
            algorithms=args.algorithms
        )
        if fig is None:
            print(f"跳过生成图表: batch_size={batch_size}, tp_size={args.tp_size}")
            return
        output_file = os.path.join(args.output_dir, 
                                  f"seq_vs_bandwidth_batch{batch_size}_tp{args.tp_size}.png")
        
    elif args.plot_type == 'batch_vs_bandwidth':
        # 给定sequence length，画batch size变化
        # 处理sequence length字符串标签（如"1k" -> 1024）
        try:
            seq_length = int(args.fixed_value)  # 尝试直接转换为整数
        except ValueError:
            # 如果是字符串标签，使用转换函数
            seq_length = plotter._seq_length_to_int(args.fixed_value)
            if seq_length == 0:
                print(f"错误: 无效的sequence length标签: {args.fixed_value}")
                return
        
        fig = plotter.plot_batch_size_vs_bandwidth(
            sequence_length=seq_length,
            tp_size=args.tp_size,
            algorithms=args.algorithms
        )
        if fig is None:
            print(f"跳过生成图表: sequence_length={seq_length}, tp_size={args.tp_size}")
            return
        output_file = os.path.join(args.output_dir, 
                                  f"batch_vs_bandwidth_seq{args.fixed_value}_tp{args.tp_size}.png")
    
    # 保存图表
    plotter.save_plot(fig, output_file)
    plt.close(fig)
    
    print(f"Analysis completed! Check results in: {args.output_dir}")

if __name__ == "__main__":
    main()