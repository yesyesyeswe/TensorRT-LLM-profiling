#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Optional, Tuple, Union

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
                   sequence_length: Optional[Union[int, str]] = None,
                   tp_size: Optional[int] = None,
                   cuda_device: Optional[int] = None) -> pd.DataFrame:
        """根据条件过滤数据"""
        filtered_df = self.df.copy()
        
        if algorithm:
            filtered_df = filtered_df[filtered_df['Algorithm'] == algorithm]
        if batch_size:
            filtered_df = filtered_df[filtered_df['Batch_size'] == batch_size]
        if sequence_length:
            # 处理sequence_length的多种格式
            if isinstance(sequence_length, int):
                # 如果是整数，直接使用整数比较（CSV中存储为整数）
                filtered_df = filtered_df[filtered_df['Sequence_length'] == sequence_length]
            else:
                # 如果是字符串，尝试转换为整数再比较
                try:
                    seq_int = int(sequence_length)
                    filtered_df = filtered_df[filtered_df['Sequence_length'] == seq_int]
                except ValueError:
                    # 如果是字符串标签如'1k'，使用转换函数
                    seq_int = self._seq_length_to_int(sequence_length)
                    if seq_int > 0:
                        filtered_df = filtered_df[filtered_df['Sequence_length'] == seq_int]
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
    
    def plot_bandwidth_comparison(self, batch_size: int, sequence_length: int, 
                                 tp_size: int = 2, algorithms: Optional[List[str]] = None,
                                 x_axis: str = 'sequence_length') -> plt.Figure:
        """带宽对比图的便捷方法"""
        return self.plot_algorithms_comparison(
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_size=tp_size,
            algorithms=algorithms,
            metric='bandwidth',
            x_axis=x_axis
        )
    
    def plot_latency_comparison(self, batch_size: int, sequence_length: int,
                               tp_size: int = 2, algorithms: Optional[List[str]] = None,
                               x_axis: str = 'sequence_length') -> plt.Figure:
        """延迟对比图的便捷方法"""
        return self.plot_algorithms_comparison(
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_size=tp_size,
            algorithms=algorithms,
            metric='latency',
            x_axis=x_axis
        )
    
    def plot_communication_comparison(self, batch_size: int, sequence_length: int,
                                     tp_size: int = 2, algorithms: Optional[List[str]] = None,
                                     x_axis: str = 'sequence_length') -> plt.Figure:
        """通信对比图的便捷方法"""
        return self.plot_algorithms_comparison(
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_size=tp_size,
            algorithms=algorithms,
            metric='communication',
            x_axis=x_axis
        )
    
    def get_available_metrics(self) -> Dict[str, List[str]]:
        """获取数据中可用的指标"""
        available = {}
        
        # 检查带宽相关列
        bandwidth_cols = []
        if 'Prefill' in self.df.columns:
            bandwidth_cols.append('Prefill')
        if 'Decoder' in self.df.columns:
            bandwidth_cols.append('Decoder')
        if bandwidth_cols:
            available['bandwidth'] = bandwidth_cols
        
        # 检查延迟相关列
        latency_cols = []
        if 'Prefill_Latency' in self.df.columns:
            latency_cols.append('Prefill_Latency')
        if 'Decoder_Latency' in self.df.columns:
            latency_cols.append('Decoder_Latency')
        if latency_cols:
            available['latency'] = latency_cols
        
        # 检查通信相关列 - 支持多种命名约定
        communication_cols = []
        if 'Communication_Prefill' in self.df.columns:
            communication_cols.append('Communication_Prefill')
        if 'Communication_Decoder' in self.df.columns:
            communication_cols.append('Communication_Decoder')
        if 'Prefill_Communication' in self.df.columns:
            communication_cols.append('Prefill_Communication')
        if 'Decoder_Communication' in self.df.columns:
            communication_cols.append('Decoder_Communication')
        if communication_cols:
            available['communication'] = communication_cols
        
        return available
    
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
    
    def _int_to_seq_length(self, seq_int: int) -> str:
        """将整数转换为sequence length字符串"""
        reverse_mapping = {128: '128', 256: '256', 512: '512', 1024: '1k', 
                          2048: '2k', 4096: '4k', 8192: '8k', 16384: '16k', 32768: '32k'}
        return reverse_mapping.get(seq_int, str(seq_int))
    
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
    
    def plot_algorithms_comparison(self, 
                                  batch_size: int, 
                                  sequence_length: int,
                                  tp_size: int = 2,
                                  algorithms: Optional[List[str]] = None,
                                  metric: str = 'bandwidth',
                                  x_axis: str = 'sequence_length') -> plt.Figure:
        """
        通用的多算法对比绘图函数
        
        布局：2x2 子图
        - 行：GPU0, GPU1
        - 列：Prefill, Decoder
        - 每个子图显示多个算法的对比
        
        Args:
            batch_size: 批大小
            sequence_length: 序列长度 (当x_axis='batch_size'时使用)
            tp_size: TP大小
            algorithms: 算法列表，None则使用所有算法
            metric: 要绘制的指标 ('bandwidth', 'latency', 'communication')
            x_axis: X轴变量 ('sequence_length' 或 'batch_size')
        """
        if algorithms is None:
            algorithms = sorted(self.df['Algorithm'].unique())
        
        # 根据x_axis类型过滤数据
        if x_axis == 'sequence_length':
            filtered_df = self.filter_data(batch_size=batch_size, tp_size=tp_size)
        else:  # x_axis == 'batch_size'
            filtered_df = self.filter_data(sequence_length=sequence_length, tp_size=tp_size)
        
        if algorithms:
            filtered_df = filtered_df[filtered_df['Algorithm'].isin(algorithms)]
        
        # 获取唯一的GPU设备
        unique_devices = sorted(filtered_df['CudaDevice'].unique())
        if len(unique_devices) == 0:
            print(f"警告: 没有符合条件的数据")
            return None
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), squeeze=False)
        
        # 设置指标对应的列名和标签
        metric_mapping = {
            'bandwidth': {'prefill': 'Prefill', 'decoder': 'Decoder', 'ylabel': 'Bandwidth (GB/s)', 'title': 'Bandwidth'},
            'latency': {'prefill': 'Prefill_Latency', 'decoder': 'Decoder_Latency', 'ylabel': 'Latency (us)', 'title': 'Latency'},
            'communication': {'prefill': 'Communication_Prefill', 'decoder': 'Communication_Decoder', 'ylabel': 'Communication (KB)', 'title': 'Communication'}
        }
        
        # 检查实际可用的列名
        available_cols = filtered_df.columns.tolist()
        if metric == 'communication':
            # 检查通信列名的变体
            if 'Communication_Prefill' in available_cols and 'Communication_Decoder' in available_cols:
                pass  # 使用默认映射
            elif 'Prefill_Communication' in available_cols and 'Decoder_Communication' in available_cols:
                metric_mapping['communication'] = {
                    'prefill': 'Prefill_Communication', 
                    'decoder': 'Decoder_Communication', 
                    'ylabel': 'Communication (KB)', 
                    'title': 'Communication'
                }
            else:
                print(f"警告: 未找到通信相关列。可用列: {[col for col in available_cols if 'ommunication' in col]}")
                return None
        elif metric == 'latency':
            # 检查延迟列名的变体
            if 'Prefill_Latency' not in available_cols or 'Decoder_Latency' not in available_cols:
                print(f"警告: 未找到延迟相关列。可用列: {[col for col in available_cols if 'atency' in col]}")
                return None
        
        if metric not in metric_mapping:
            raise ValueError(f"不支持的指标: {metric}. 支持的指标: {list(metric_mapping.keys())}")
        
        metric_info = metric_mapping[metric]
        
        # 为每个GPU设备和阶段绘制对比图
        for gpu_idx, device in enumerate(unique_devices[:2]):  # 只处理前两个GPU
            device_data = filtered_df[filtered_df['CudaDevice'] == device]
            
            if len(device_data) == 0:
                continue
            
            # Prefill子图
            ax_prefill = axes[gpu_idx, 0]
            # Decoder子图
            ax_decoder = axes[gpu_idx, 1]
            
            # 为每个算法绘制线条
            for algo in algorithms:
                algo_data = device_data[device_data['Algorithm'] == algo]
                
                if len(algo_data) == 0:
                    continue
                
                # 根据x_axis类型分组数据
                if x_axis == 'sequence_length':
                    # 按sequence_length分组
                    groups = algo_data.groupby('Sequence_length').agg({
                        metric_info['prefill']: 'mean',
                        metric_info['decoder']: 'mean'
                    }).reset_index()
                    groups = self._sort_sequence_lengths(groups)
                    x_values = groups['Sequence_length']
                    x_label = 'Sequence Length'
                else:  # x_axis == 'batch_size'
                    # 按batch_size分组
                    groups = algo_data.groupby('Batch_size').agg({
                        metric_info['prefill']: 'mean',
                        metric_info['decoder']: 'mean'
                    }).reset_index()
                    groups = groups.sort_values('Batch_size')
                    x_values = groups['Batch_size']
                    x_label = 'Batch Size'
                
                # 绘制Prefill线条
                if len(groups) > 0 and not groups[metric_info['prefill']].isna().all():
                    ax_prefill.plot(x_values, groups[metric_info['prefill']], 
                                  marker='o', linewidth=2, markersize=6, 
                                  label=f'{algo}')
                    # 添加数值标签
                    for x, y in zip(x_values, groups[metric_info['prefill']]):
                        if not pd.isna(y) and y > 0:
                            ax_prefill.annotate(f'{y:.1f}', (x, y), 
                                              textcoords="offset points", 
                                              xytext=(0, 10), ha='center', fontsize=8)
                
                # 绘制Decoder线条
                if len(groups) > 0 and not groups[metric_info['decoder']].isna().all():
                    ax_decoder.plot(x_values, groups[metric_info['decoder']], 
                                  marker='s', linewidth=2, markersize=6, 
                                  label=f'{algo}')
                    # 添加数值标签
                    for x, y in zip(x_values, groups[metric_info['decoder']]):
                        if not pd.isna(y) and y > 0:
                            ax_decoder.annotate(f'{y:.1f}', (x, y), 
                                                textcoords="offset points", 
                                                xytext=(0, 10), ha='center', fontsize=8)
            
            # 设置子图属性
            for ax, phase in [(ax_prefill, 'Prefill'), (ax_decoder, 'Decoder')]:
                ax.set_xlabel(x_label)
                ax.set_ylabel(metric_info['ylabel'])
                ax.set_title(f'GPU{device} {phase} {metric_info["title"]} Comparison', pad=10)
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # 根据x_axis类型设置坐标轴
                if x_axis == 'sequence_length':
                    # 对sequence_length使用对数坐标
                    ax.set_xscale('log', base=2)
                    if metric in ['bandwidth', 'latency']:
                        ax.set_yscale('log', base=2)
                else:
                    # 对batch_size使用线性坐标
                    if metric in ['bandwidth', 'latency']:
                        ax.set_yscale('log', base=2)
        
        # 如果只有一个GPU，隐藏第二行
        if len(unique_devices) < 2:
            for col in range(2):
                axes[1, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
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
    parser.add_argument("--plot-type", choices=['seq_vs_bandwidth', 'batch_vs_bandwidth', 'algorithms_comparison'], required=True,
                       help="Type of plot to generate")
    parser.add_argument("--fixed-value", type=str, required=True,
                       help="Fixed value for plotting (batch_size or sequence_length)")
    parser.add_argument("--comparison-metric", choices=['bandwidth', 'latency', 'communication'], 
                       default='bandwidth', help="Metric for algorithms comparison (default: bandwidth)")
    parser.add_argument("--x-axis", choices=['sequence_length', 'batch_size'], 
                       default='sequence_length', help="X-axis variable for comparison (default: sequence_length)")
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
    
    elif args.plot_type == 'algorithms_comparison':
        # 多算法对比图
        # 根据x_axis类型确定fixed_value的含义
        if args.x_axis == 'sequence_length':
            batch_size = int(args.fixed_value)
            seq_length = '1k'  # 默认值，实际不会用到
        else:  # x_axis == 'batch_size'
            # 处理sequence length字符串标签（如"1k" -> 1024）
            try:
                seq_length = int(args.fixed_value)  # 尝试直接转换为整数
            except ValueError:
                # 如果是字符串标签，使用转换函数
                seq_length = plotter._seq_length_to_int(args.fixed_value)
                if seq_length == 0:
                    print(f"错误: 无效的sequence length标签: {args.fixed_value}")
                    return
            batch_size = 8  # 默认值，实际不会用到
            
        fig = plotter.plot_algorithms_comparison(
            batch_size=batch_size,
            sequence_length=seq_length,
            tp_size=args.tp_size,
            algorithms=args.algorithms,
            metric=args.comparison_metric,
            x_axis=args.x_axis
        )
        if fig is None:
            print(f"跳过生成图表: batch_size={batch_size}, sequence_length={seq_length}, tp_size={args.tp_size}")
            return
        metric_name = args.comparison_metric
        x_name = args.x_axis.replace('_', '')
        fixed_name = args.fixed_value
        output_file = os.path.join(args.output_dir, 
                                  f"{metric_name}_comparison_{x_name}{fixed_name}_tp{args.tp_size}.png")
    
    # 保存图表
    plotter.save_plot(fig, output_file)
    plt.close(fig)
    
    print(f"Analysis completed! Check results in: {args.output_dir}")

if __name__ == "__main__":
    main()

# 使用示例
"""
# 1. 检查可用的指标
plotter = BandwidthPlotter("results/avg_bandwidth.csv")
available_metrics = plotter.get_available_metrics()
print("可用指标:", available_metrics)

# 2. 绘制多算法带宽对比图（固定batch_size，变化sequence_length）
fig = plotter.plot_bandwidth_comparison(
    batch_size=8,
    sequence_length=1024,  # 这个值不会用到，但需要提供
    tp_size=2,
    algorithms=['Megatron', 'DeepSpeed', 'FasterTransformer'],
    x_axis='sequence_length'
)
plotter.save_plot(fig, "figures/bandwidth_comparison_seq.png")

# 3. 绘制多算法延迟对比图（固定sequence_length，变化batch_size）
fig = plotter.plot_latency_comparison(
    batch_size=8,  # 这个值不会用到，但需要提供
    sequence_length=1024,
    tp_size=2,
    algorithms=['Megatron', 'DeepSpeed'],
    x_axis='batch_size'
)
plotter.save_plot(fig, "figures/latency_comparison_batch.png")

# 4. 使用通用函数绘制通信对比图
fig = plotter.plot_algorithms_comparison(
    batch_size=16,
    sequence_length=2048,
    tp_size=2,
    algorithms=['Megatron', 'DeepSpeed', 'FasterTransformer'],
    metric='communication',
    x_axis='sequence_length'
)
plotter.save_plot(fig, "figures/communication_comparison.png")

# 5. 命令行使用示例
# 生成多算法带宽对比图（固定batch_size=8，变化sequence_length）
# python plot_bandwidth_analysis.py --plot-type algorithms_comparison --fixed-value 8 --comparison-metric bandwidth --x-axis sequence_length --tp-size 2

# 生成多算法延迟对比图（固定sequence_length=1k，变化batch_size）
# python plot_bandwidth_analysis.py --plot-type algorithms_comparison --fixed-value 1k --comparison-metric latency --x-axis batch_size --tp-size 2
"""