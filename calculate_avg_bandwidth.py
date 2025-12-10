#!/usr/bin/env python3
import os
import csv
import glob
import pandas as pd
from pathlib import Path

def convert_sequence_length(seq_len_str):
    """
    将sequence length字符串转换为数值
    例如: "1k" -> 1024, "2k" -> 2048, "16k" -> 16384, "32k" -> 32768
    """
    if isinstance(seq_len_str, str):
        if seq_len_str.lower().endswith('k'):
            return int(seq_len_str[:-1]) * 1024
        else:
            return int(seq_len_str)
    return int(seq_len_str)

def calculate_avg_bandwidth(merged_dir, output_file):
    """
    计算平均带宽并分为Prefill和Decoder部分
    Prefill: 
    - sequence_length <= 8k: 前56行
    - sequence_length > 8k: 前(sequence_length/8k)*56行
    Decoder: 剩余的行
    """
    merged_files = glob.glob(os.path.join(merged_dir, "merge_*.csv"))
    
    if not merged_files:
        print(f"No merge_*.csv files found in {merged_dir}")
        return
    
    results = []
    
    for file_path in merged_files:
        print(f"Processing: {file_path}")
        
        # 读取CSV文件
        try:
            df = pd.read_csv(file_path)
            
            # 确保必要的列存在
            required_cols = ['Algorithm', 'Batch_size', 'Sequence_length', 'TP_Size', 'CudaDevice', 'MPI_Rank', 'Bandwidth']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns {missing_cols} in {file_path}")
                continue
            
            # 按 CudaDevice 和 MPI_Rank 分组
            grouped = df.groupby(['Algorithm', 'Batch_size', 'Sequence_length', 'TP_Size', 'CudaDevice', 'MPI_Rank'])
            
            for name, group in grouped:
                algorithm, batch_size, seq_len_str, tp_size, cuda_device, mpi_rank = name
                
                # 转换sequence length字符串为数值
                seq_len = convert_sequence_length(seq_len_str)
                
                # 按行号排序确保顺序
                group_sorted = group.sort_index()
                
                # 根据sequence length计算prefill行数
                # 对于NCCL Simple算法：
                # - sequence_length <= 8k: 56行
                # - sequence_length > 8k: (sequence_length/8k)*56行
                if seq_len <= 8192:
                #    prefill_row_count = 56
                    prefill_row_count = 80
                else:
                #    prefill_row_count = int((seq_len / 8192) * 56)
                    prefill_row_count = 80
                
                # Prefill: 计算出的前N行
                prefill_rows = group_sorted.head(prefill_row_count)
                #prefill_rows = group_sorted
                prefill_avg = prefill_rows['Bandwidth'].mean() if len(prefill_rows) > 0 else 0.0
                
                # Decoder: 剩余的行
                decoder_rows = group_sorted.iloc[prefill_row_count:]
                decoder_avg = decoder_rows['Bandwidth'].mean() if len(decoder_rows) > 0 else 0.0
                
                results.append({
                    'Algorithm': algorithm,
                    'Batch_size': batch_size,
                    'Sequence_length': seq_len,
                    'TP_Size': tp_size,
                    'CudaDevice': cuda_device,
                    'MPI_RANK': mpi_rank,
                    'Prefill': round(prefill_avg, 3),
                    'Decoder': round(decoder_avg, 3)
                })
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    if not results:
        print("No data processed successfully")
        return
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(results)
    
    # 按指定列排序
    result_df = result_df.sort_values([
        'Algorithm', 'Batch_size', 'Sequence_length', 'TP_Size', 'CudaDevice', 'MPI_RANK'
    ])
    
    # 保存结果
    result_df.to_csv(output_file, index=False)
    print(f"Average bandwidth results saved to: {output_file}")
    print(f"Total records: {len(result_df)}")
    
    # 显示统计信息
    print("\nSummary statistics:")
    print(f"TP=2 records: {len(result_df[result_df['TP_Size'] == 2])}")
    if len(result_df[result_df['TP_Size'] == 2]) > 0:
        tp2_data = result_df[result_df['TP_Size'] == 2]
        print(f"TP=2 Prefill avg: {tp2_data['Prefill'].mean()}")
        print(f"TP=2 Decoder avg: {tp2_data['Decoder'].mean()}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate average bandwidth from merged CSV files")
    parser.add_argument("--merged-dir", 
                         default="/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/merged",
                         help="Directory containing merged CSV files")
    parser.add_argument("--output-file", 
                         default="/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/avg_bandwidth.csv",
                         help="Output CSV file path")
    
    args = parser.parse_args()
    
    calculate_avg_bandwidth(args.merged_dir, args.output_file)

if __name__ == "__main__":
    main()