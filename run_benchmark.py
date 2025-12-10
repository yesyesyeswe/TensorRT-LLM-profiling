import os
import sys
import subprocess
import csv
from pathlib import Path
import argparse

# SEQ_LABELS = ["128", "256", "512", "1k", "2k", "4k", "8k", "16k", "32k", "64k"]
SEQ_LABELS = ["128", "256", "512", "1k", "2k"]
BATCH_SIZES = [1, 2, 3, 4, 5]
SEQ_MAP = {
    "128": 128,
    "256": 256,
    "512": 512,
    "1k": 1024,
    "2k": 2048,
    "4k": 4096,
    "8k": 8192,
    "16k": 16384,
    "32k": 32768,
    "64k": 65536,
}


BASE = Path("/root/autodl-tmp/TensorRT-LLM/mybenchmark")
DATA_DIR = BASE / "prepare_data_json"
RESULTS_DIR = BASE / "results"
SQLITES_DIR = RESULTS_DIR / "sqlites"
COMM_DIR = RESULTS_DIR / "communication"
FIGURES_DIR = BASE / "figures"
PREPARE_DATASET = Path("/root/autodl-tmp/TensorRT-LLM/benchmarks/cpp/prepare_dataset.py")
BENCH_BIN = Path("/root/autodl-tmp/TensorRT-LLM/cpp/build/benchmarks/gptManagerBenchmark")
TOKENIZER_DIR = Path("/root/autodl-tmp/tmp/Qwen/14B/14B")
ENGINE_BASE = Path("/root/autodl-tmp/tmp/qwen/14B/trt_engines/fp16")
EXTRACT_SCRIPT = BASE / "extract_nccl_latency.py"


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SQLITES_DIR.mkdir(parents=True, exist_ok=True)
    COMM_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def ensure_pydeps():
    try:
        import pandas  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas"], check=False)
    try:
        import matplotlib  # noqa: F401
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib"], check=False)


def generate_dataset(seq_label, batch_size, su_algo, tp, nccl_proto):
    input_mean = SEQ_MAP[str(seq_label)]
    
    out_file = DATA_DIR / f"token_norm_dist_sl{seq_label}.json"
    
    if out_file.exists():
        return True
    cmd = [
        sys.executable,
        str(PREPARE_DATASET),
        "--output",
        str(out_file),
        "--tokenizer",
        str(TOKENIZER_DIR),
        "token-norm-dist",
        "--num-requests",
        "1",
        "--input-mean",
        str(input_mean),
        "--input-stdev",
        "0",
        "--output-mean",
        "1",
        "--output-stdev",
        "0",
    ]
    return subprocess.run(cmd, check=False).returncode == 0


def run_benchmark(seq_label, batch_size, su_algo, tp, nccl_proto):
    dataset_path = DATA_DIR / f"token_norm_dist_sl{seq_label}.json"
    if su_algo == "NCCL" and nccl_proto:
        nsys_out = SQLITES_DIR / f"bs{batch_size}_sl{seq_label}_tp{tp}_algo{su_algo}_{nccl_proto}"
    else:
        nsys_out = SQLITES_DIR / f"bs{batch_size}_sl{seq_label}_tp{tp}_algo{su_algo}"

#"--trace=cuda,osrt",
    engine_dir = ENGINE_BASE / f"{tp}-gpu"
    cmd = [
        "nsys", "profile",
        "--export=sqlite",
        "--output", str(nsys_out),
        "mpirun", "-n", str(tp), "--allow-run-as-root",
        str(BENCH_BIN),
        "--engine_dir", str(engine_dir),
        "--request_rate", "-1",
        "--warm_up", "10",
        "--static_emulated_batch_size", "1",
        "--dataset", str(dataset_path),
    ]
    env = os.environ.copy()
    env["COMM_RESULTS_DIR"] = str(COMM_DIR)
    env["BATCH_SIZE"] = str(batch_size)
    env["SEQUENCE_LENGTH"] = str(seq_label)
    env["SU_ALGO"] = str(su_algo)
    env["TP"] = str(tp)
    if su_algo == "NCCL" and nccl_proto:
        env["NCCL_PROTO"] = str(nccl_proto)
    else:
        env.pop("NCCL_PROTO", None)
    return subprocess.run(cmd, check=False, env=env).returncode == 0

def sort_comm_csv(path):
    if not os.path.exists(path):
        return
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        rows = list(r)
    if not rows:
        return
    header = rows[0]
    body = rows[1:]
    try:
        idx = header.index("CudaDevice")
    except ValueError:
        return
    
    # 过滤掉无效行
    valid_body = []
    for row in body:
        if len(row) > idx and row[idx] != "CudaDevice":  # 跳过重复表头
            try:
                # 验证是否为有效整数
                int(row[idx])
                valid_body.append(row)
            except (ValueError, IndexError):
                # 跳过非数字行
                continue
    
    # 排序
    valid_body.sort(key=lambda x: int(x[idx]))
    
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(valid_body)


def merge_communication_csv(batch_size, seq_len, tp_size, su_algo, nccl_proto=None):
    """Merge multiple GPU-specific communication CSV files into a single file."""
    import glob
    import re
    
    # Build the base filename pattern (without gpu suffix)
    if nccl_proto and nccl_proto.strip():
        merged_filename = f"comm_bs{batch_size}_sl{seq_len}_tp{tp_size}_algo{su_algo}_{nccl_proto}.csv"
        pattern = f"comm_bs{batch_size}_sl{seq_len}_tp{tp_size}_algo{su_algo}_{nccl_proto}_gpu*.csv"
    else:
        merged_filename = f"comm_bs{batch_size}_sl{seq_len}_tp{tp_size}_algo{su_algo}.csv"
        pattern = f"comm_bs{batch_size}_sl{seq_len}_tp{tp_size}_algo{su_algo}_gpu*.csv"
    
    # Find all GPU-specific files
    search_pattern = str(COMM_DIR / pattern)
    gpu_files = glob.glob(search_pattern)
    
    if not gpu_files:
        print(f"No GPU-specific communication files found for pattern: {pattern}")
        return False
    
    print(f"Found {len(gpu_files)} GPU-specific files to merge")
    
    # Read and merge all files
    merged_rows = []
    header = None
    
    for gpu_file in sorted(gpu_files):
        print(f"Processing: {gpu_file}")
        try:
            with open(gpu_file, 'r', newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
                
                if not rows:
                    continue
                    
                if header is None:
                    header = rows[0]
                    merged_rows.append(header)
                
                # Add all data rows (skip header)
                for row in rows[1:]:
                    if row and row[0] != header[0]:  # Skip duplicate headers
                        merged_rows.append(row)
                        
        except Exception as e:
            print(f"Error reading {gpu_file}: {e}")
            continue
    
    if len(merged_rows) <= 1:  # Only header or no data
        print("No data to merge")
        return False
    
    # Write merged file
    merged_path = COMM_DIR / merged_filename
    try:
        with open(merged_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(merged_rows)
        print(f"Successfully merged {len(gpu_files)} files into: {merged_path}")
        
        # Sort the merged file by CudaDevice
        sort_comm_csv(str(merged_path))
        
        return True
        
    except Exception as e:
        print(f"Error writing merged file {merged_path}: {e}")
        return False


def run_extract_latency(batch_size, seq_label, su_algo, tp, nccl_proto, nsys_out_prefix):
    sqlite_path = Path(f"{nsys_out_prefix}.sqlite")
    if not sqlite_path.exists():
        return False
    env = os.environ.copy()
    env["BATCH_SIZE"] = str(batch_size)
    env["SEQUENCE_LENGTH"] = str(seq_label)
    env["SU_ALGO"] = str(su_algo)
    env["TP"] = str(tp)
    kernel = "ncclDevKernel_AllReduce_Sum_f16_RING_LL"
    if su_algo == "NCCL" and nccl_proto:
        env["NCCL_PROTO"] = str(nccl_proto)
    else:
        env.pop("NCCL_PROTO", None)
        if su_algo == "ONESHOT":
            kernel = "oneShotAllReduceKernel"
        elif su_algo == "TWOSHOT":
            kernel = "twoShotAllReduceKernel"
    cmd = [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--input", str(sqlite_path),
        "--output-dir", str(RESULTS_DIR),
        "--kernel-name", kernel,
    ]
    return subprocess.run(cmd, check=False, env=env).returncode == 0


def merge_csv():
    import pandas as pd
    frames = []
    for p in RESULTS_DIR.glob("result_*.csv"):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")] if hasattr(df.columns, "str") else df
        name = p.stem
        try:
            parts = name.split("_")
            b = int(parts[1])
            s = "_".join(parts[2:])
        except Exception:
            continue
        df.insert(0, "batch_size", b)
        df.insert(1, "sequence_length", s)
        df.insert(2, "sequence_length_num", SEQ_MAP.get(s, None))
        frames.append(df)
    if not frames:
        return None
    all_df = pd.concat(frames, ignore_index=True)
    out = RESULTS_DIR / "all_results.csv"
    all_df.to_csv(out, index=False)
    return out


def generate_bandwidth_plots(sequence_lengths=None, batch_sizes=None, tp_sizes=None, algorithms=None, 
                             comparison_metrics=None, x_axis_types=None):
    """生成带宽分析图表（增强版）
    
    Args:
        sequence_lengths: 要生成的 sequence length 列表，默认为 SEQ_LABELS
        batch_sizes: 要生成的 batch size 列表，默认为 BATCH_SIZES  
        tp_sizes: 要生成的 TP size 列表，默认为 [2]
        algorithms: 要包含的算法列表，默认为 None (使用所有算法)
        comparison_metrics: 要生成的对比指标列表，默认为 ['bandwidth']
        x_axis_types: 要生成的X轴类型列表，默认为 ['sequence_length']
    """
    avg_bandwidth_csv = RESULTS_DIR / "avg_bandwidth.csv"
    if not avg_bandwidth_csv.exists():
        print(f"[WARNING] {avg_bandwidth_csv} 不存在，跳过带宽图表生成")
        return
    
    # 使用默认参数
    if sequence_lengths is None:
        sequence_lengths = SEQ_LABELS
    if batch_sizes is None:
        batch_sizes = BATCH_SIZES
    if tp_sizes is None:
        tp_sizes = [2]  # 默认 TP=2
    if comparison_metrics is None:
        comparison_metrics = ['bandwidth']
    if x_axis_types is None:
        x_axis_types = ['sequence_length']
    
    print("[INFO] 开始生成带宽分析图表...")
    print(f"[INFO] Sequence lengths: {sequence_lengths}")
    print(f"[INFO] Batch sizes: {batch_sizes}")
    print(f"[INFO] TP sizes: {tp_sizes}")
    print(f"[INFO] Comparison metrics: {comparison_metrics}")
    print(f"[INFO] X-axis types: {x_axis_types}")
    
    if algorithms:
        try:
            import pandas as pd
            df_alg = pd.read_csv(avg_bandwidth_csv)
            available_algos = [str(a) for a in df_alg.get("Algorithm", []).unique().tolist()]
        except Exception:
            available_algos = []
        expanded = []
        for a in algorithms:
            a = a.strip()
            if not a:
                continue
            matches = [x for x in available_algos if x == a or x.startswith(a)]
            if matches:
                for m in matches:
                    if m not in expanded:
                        expanded.append(m)
            else:
                if a not in expanded:
                    expanded.append(a)
        algorithms = expanded if expanded else algorithms
        print(f"[INFO] Algorithms: {algorithms}")
    
    # 构建算法参数
    algo_args = []
    if algorithms:
        algo_args = ["--algorithms"] + [a.strip() for a in algorithms if a.strip()]
    
    # 计算总图表数量
    total_plots = (
        len(sequence_lengths) * len(tp_sizes) +  # batch_vs_bandwidth
        len(batch_sizes) * len(tp_sizes) +      # seq_vs_bandwidth
        len(comparison_metrics) * len(x_axis_types) * len(tp_sizes)  # algorithms_comparison
    )
    print(f"[INFO] 预计生成 {total_plots} 个图表")
    
    plot_count = 0
    
    # 1. 对每个 sequence_length 可取的值，固定，画 batch_vs_bandwidth
    for seq_length in sequence_lengths:
        for tp_size in tp_sizes:
            plot_count += 1
            print(f"[INFO] [{plot_count}/{total_plots}] 生成 batch_vs_bandwidth 图表: sequence_length={seq_length}, tp_size={tp_size}")
            cmd = [
                sys.executable,
                str(BASE / "plot_bandwidth_analysis.py"),
                "--data-file", str(avg_bandwidth_csv),
                "--output-dir", str(FIGURES_DIR),
                "--plot-type", "batch_vs_bandwidth",
                "--fixed-value", seq_length,
                "--tp-size", str(tp_size),
            ]
            if algo_args:
                cmd.extend(algo_args)
            subprocess.run(cmd, check=False)
    
    # 2. 对每个 batch_size 可取的值，固定，画 seq_vs_bandwidth
    for batch_size in batch_sizes:
        for tp_size in tp_sizes:
            plot_count += 1
            print(f"[INFO] [{plot_count}/{total_plots}] 生成 seq_vs_bandwidth 图表: batch_size={batch_size}, tp_size={tp_size}")
            cmd = [
                sys.executable,
                str(BASE / "plot_bandwidth_analysis.py"),
                "--data-file", str(avg_bandwidth_csv),
                "--output-dir", str(FIGURES_DIR),
                "--plot-type", "seq_vs_bandwidth",
                "--fixed-value", str(batch_size),
                "--tp-size", str(tp_size),
            ]
            if algo_args:
                cmd.extend(algo_args)
            subprocess.run(cmd, check=False)
    
    # 3. 生成多算法对比图（新增功能）
    for metric in comparison_metrics:
        for x_axis in x_axis_types:
            for tp_size in tp_sizes:
                plot_count += 1
                print(f"[INFO] [{plot_count}/{total_plots}] 生成 {metric} 对比图: x_axis={x_axis}, tp_size={tp_size}")
                
                # 根据x_axis类型确定fixed_value
                if x_axis == 'sequence_length':
                    # 使用batch_size作为固定值，使用第一个batch_size
                    fixed_value = str(batch_sizes[0]) if batch_sizes else '8'
                else:  # x_axis == 'batch_size'
                    # 使用sequence_length作为固定值，使用第一个sequence_length
                    fixed_value = sequence_lengths[0] if sequence_lengths else '1k'
                
                cmd = [
                    sys.executable,
                    str(BASE / "plot_bandwidth_analysis.py"),
                    "--data-file", str(avg_bandwidth_csv),
                    "--output-dir", str(FIGURES_DIR),
                    "--plot-type", "algorithms_comparison",
                    "--fixed-value", fixed_value,
                    "--comparison-metric", metric,
                    "--x-axis", x_axis,
                    "--tp-size", str(tp_size),
                ]
                if algo_args:
                    cmd.extend(algo_args)
                subprocess.run(cmd, check=False)
    
    print(f"[INFO] 带宽分析图表生成完成，结果保存在: {FIGURES_DIR}")
    print(f"[INFO] 共生成 {plot_count} 个图表")


def plot_graphs(all_csv_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    df = pd.read_csv(all_csv_path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")] if hasattr(df.columns, "str") else df
    df = df.dropna(subset=["sequence_length_num"]) if "sequence_length_num" in df.columns else df
    if "batch_size" not in df.columns or "sequence_length" not in df.columns:
        return
    df = df.sort_values(["batch_size", "sequence_length_num"]) if "sequence_length_num" in df.columns else df
    for b in sorted(df["batch_size"].unique()):
        g = df[df["batch_size"] == b]
        x = g["sequence_length_num"].tolist()
        y_comm = (g["token_throughput(token/sec)"] * (g["total_latency(ms)"] / 1000.0)).tolist() if "token_throughput(token/sec)" in g.columns and "total_latency(ms)" in g.columns else []
        y_total = g["total_latency(ms)"].tolist() if "total_latency(ms)" in g.columns else []
        y_ttft = g["avg_time_to_first_token(ms)"].tolist() if "avg_time_to_first_token(ms)" in g.columns else []
        y_inter = g["avg_inter_token_latency(ms)"].tolist() if "avg_inter_token_latency(ms)" in g.columns else []
        # 1) 通信量
        if x and y_comm:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, y_comm, marker="o")
            ax.set_xlabel("sequence_length")
            ax.set_ylabel("Total tokens (token_throughput × total_latency_s)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            fig.tight_layout(pad=2)          # 关键：多留边距
            fig.savefig(RESULTS_DIR / f"batch{b}_communication.png", dpi=200)
            plt.close(fig)

        # 2) 总延迟
        if x and y_total:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, y_total, marker="o", color="C1")
            ax.set_xlabel("sequence_length")
            ax.set_ylabel("total_latency (ms)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            fig.tight_layout(pad=2)
            fig.savefig(RESULTS_DIR / f"batch{b}_total_latency.png", dpi=200)
            plt.close(fig)

        # 3) 首 token
        if x and y_ttft:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, y_ttft, marker="o", color="C2")
            ax.set_xlabel("sequence_length")
            ax.set_ylabel("avg_time_to_first_token (ms)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            fig.tight_layout(pad=2)
            fig.savefig(RESULTS_DIR / f"batch{b}_ttft.png", dpi=200)
            plt.close(fig)

        # 4) inter-token
        if x and y_inter:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, y_inter, marker="o", color="C3")
            ax.set_xlabel("sequence_length")
            ax.set_ylabel("avg_inter_token_latency (ms)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="--", alpha=0.5)
            fig.tight_layout(pad=2)
            fig.savefig(RESULTS_DIR / f"batch{b}_inter_token.png", dpi=200)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM benchmark runner & plotter")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip dataset/benchmark generation; only (re)draw figures from all_results.csv")
    parser.add_argument("--skip-bandwidth-plots", action="store_true",
                        help="Skip generating bandwidth analysis plots")
    parser.add_argument("--batches", default="1-5", help="batch sizes, e.g. 1-5 or 1,3,5")
    parser.add_argument("--seqs", default=",".join(SEQ_LABELS), help="sequence labels list")
    parser.add_argument("--algos", default="NCCL_Simple", help="algorithms to try")
    parser.add_argument("--tp", default="2", help="tensor parallel sizes, e.g. 2-8 or 2,4,8")
    parser.add_argument("--nccl-protos", default="Simple,LL,LL128", help="NCCL protos when algo=NCCL")
    args = parser.parse_args()

    if args.plot_only:
        csv_path = RESULTS_DIR / "all_results.csv"
        if not csv_path.exists():
            print(f"[ERROR] --plot-only 需要 {csv_path} 已存在")
            sys.exit(1)
        print("[INFO] 仅绘图模式，跳过数据生成与 benchmark")
        plot_graphs(csv_path)
        return

    ensure_dirs()
    ensure_pydeps()

    def parse_range_list(spec, to_int=True):
        if "-" in spec:
            a, b = spec.split("-", 1)
            rng = list(range(int(a), int(b) + 1))
            return rng
        vals = [v.strip() for v in spec.split(",") if v.strip()]
        return [int(v) if to_int else v for v in vals]

    batches = parse_range_list(args.batches)
    seqs = parse_range_list(args.seqs, to_int=False)
    algos = [a.strip() for a in args.algos.split(",") if a.strip()]
    tps = parse_range_list(args.tp)
    protos = [p.strip() for p in args.nccl_protos.split(",") if p.strip()]

    jobs = []
    # 生成所有参数组合
    from itertools import product
    for b, s, algo, tp in product(batches, seqs, algos, tps):
        proto_list = protos if algo == "NCCL" else [None]
        for proto in proto_list:
            ok_gen = generate_dataset(s, b, algo, tp, proto)
            if not ok_gen:
                continue
            ok_bench = run_benchmark(s, b, algo, tp, proto)
            # name = f"comm_bs{b}_sl{s}_tp{tp}_algo{algo}{'_' + proto if proto else ''}.csv"
            # sort_comm_csv(str(COMM_DIR / name))
            
            # Merge GPU-specific communication CSV files
            merge_communication_csv(b, s, tp, algo, proto)
            
            if ok_bench:
                nsys_prefix = SQLITES_DIR / (f"bs{b}_sl{s}_tp{tp}_algo{algo}_{proto}" if proto else f"bs{b}_sl{s}_tp{tp}_algo{algo}")
                jobs.append((b, s, algo, tp, proto, nsys_prefix))
    for b, s, algo, tp, proto, nsys_prefix in jobs:
        pass
        run_extract_latency(b, s, algo, tp, proto, nsys_prefix)
    
    # 为每个文件对调用 merge_comm_latency.py
    for b, s, algo, tp, proto, nsys_prefix in jobs:
        comm_file = COMM_DIR / f"comm_bs{b}_sl{s}_tp{tp}_algo{algo}{'_' + proto if proto else ''}.csv"
        lat_file = RESULTS_DIR / "latency" / f"latency_bs{b}_sl{s}_tp{tp}_algo{algo}{'_' + proto if proto else ''}.csv"
        
        if comm_file.exists() and lat_file.exists():
            subprocess.run([
                sys.executable,
                str(BASE / "merge_comm_latency.py"),
                "--comm-file", str(comm_file),
                "--latency-file", str(lat_file),
                "--output-dir", str(RESULTS_DIR / "merged"),
            ], check=False)

    # Compute average bandwidth
    subprocess.run([
            sys.executable,
            str(BASE / "calculate_avg_bandwidth.py"),
            "--merged-dir", str(RESULTS_DIR / "merged"),
            "--output-file", str(RESULTS_DIR / "avg_bandwidth.csv"),
        ], check=False)

    # 生成带宽分析图表
    if not args.skip_bandwidth_plots:
        # 使用现有的参数来生成带宽图表
        # 解析序列长度、批次大小、TP大小和算法参数
        seq_list = args.seqs.split(",") if args.seqs else SEQ_LABELS
        batch_list = parse_range_list(args.batches) if args.batches else BATCH_SIZES
        tp_list = parse_range_list(args.tp) if args.tp else [2]
        algo_list = args.algos.split(",") if args.algos else None
        
        generate_bandwidth_plots(
            sequence_lengths=seq_list,
            batch_sizes=batch_list,
            tp_sizes=tp_list,
            algorithms=algo_list,
            comparison_metrics=['bandwidth', 'latency', 'communication'],
        )

    # 绘图逻辑暂保留原函数，如需兼容新 CSV 再迭代


if __name__ == "__main__":
    main()
