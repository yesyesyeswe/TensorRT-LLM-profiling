# TensorRT-LLM Benchmark Runner & Plotter

This tool is designed to automate the benchmarking process for TensorRT-LLM, including dataset generation, benchmark execution, data extraction, and result visualization. It supports various configurations for batch sizes, sequence lengths, algorithms, and tensor parallelism settings.

## Features

- **Automated Workflow**: Handles the entire pipeline from dataset generation to plotting.
- **Flexible Configuration**: Supports ranges and lists for batch sizes, sequence lengths, and tensor parallel sizes.
- **Visualization**: Automatically generates plots for analysis, including latency and bandwidth comparisons.
- **Resume Capability**: Can skip benchmarking and only re-plot results using existing data.

## Prerequisites: Source Code Modification

To generate the communication logs (`comm_*.csv`), you need to modify the TensorRT-LLM source code to enable detailed logging in the NCCL plugin.

**Reference Implementation:**
[allreducePlugin.cpp Modification Example](https://github.com/yesyesyeswe/TensorRT-LLM/blob/82239824ab19695c5459035d8e05ecdfadfa51f8/cpp/tensorrt_llm/plugins/ncclPlugin/allreducePlugin.cpp#L395-L473)

**Why is this needed?**
The standard TensorRT-LLM build does not output per-operation communication volumes. The modification injects logging logic into `allreducePlugin.cpp` to record Communication Data sizes

The benchmark tool relies on these logs (saved as `comm_bs{...}.csv`) to correlate communication events with other metrics. The csv files will be saved in the path defined by the macro COMM_RESULTS_DIR.

## Prerequisites: Build & Installation

To run this benchmark, you must have TensorRT-LLM compiled and installed from source.
Please refer to this guide for detailed compilation and installation instructions:
[Building from Source Code on Linux](https://nvidia.github.io/TensorRT-LLM/1.2.0rc3/installation/build-from-source-linux.html)
[TensorRT-LLM Source Compilation and Installation Guide](https://zhuanlan.zhihu.com/p/1978452911798371424)


## Profiling with Nsight Systems (nsys)

This tool automatically integrates NVIDIA Nsight Systems (`nsys`) to profile the benchmark execution.

- **Purpose**: To capture CUDA and OS runtime traces for performance analysis.
- **Trace Options**: `--trace=cuda,osrt`
- **Output**:
    - The profiling results are exported to SQLite format (`.sqlite`).
    - Files are saved in the `results/sqlites` directory.
    - Filenames follow the pattern: `bs{batch}_sl{seq}_tp{tp}_algo{algo}_{proto}.sqlite` (or similar).

These `.sqlite` files can be opened with NVIDIA Nsight Systems for detailed timeline analysis, or processed programmatically to extract specific metrics (as this tool does for latency extraction).

## Usage

Run the script using Python:

```bash
python run_benchmark.py [OPTIONS]
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--batches` | Batch sizes to benchmark. Supports range (e.g., `1-5`) or list (e.g., `1,3,5`). | `1-5` |
| `--seqs` | Sequence labels list (e.g., `128,256`). | (128,256,512,1k,2k) |
| `--algos` | Algorithms to evaluate (comma-separated). | `NCCL,ONWSHOT,TWOSHOT` |
| `--tp` | Tensor parallel sizes. Supports range or list. | `4` |
| `--nccl-protos` | NCCL protocols to use when algorithm is set to `NCCL`. | `Simple,LL,LL128` |
| `--plot-e2e-only` | Only generate end-to-end comparison plots from existing `e2e_results.csv`. | `False` |
| `--skip-bandwidth-plots` | Skip generating bandwidth analysis plots. | `False` |

## Examples

### 1. Basic Run
Before running the benchmark, set the directory path where the communication CSV result files will be saved by executing the following command:
```bash
export COMM_RESULTS_DIR=/root/autodl-tmp/TensorRT-LLM/mybenchmark/results
```
You can then run the benchmark with its default configuration (batch sizes 1-5, TP=2, Algorithm=NCCL) using:
```bash
python run_benchmark.py
```

### 2. Custom Configuration
Run with specific batch sizes (1, 2, 4, 8), sequence length (1k), tensor parallelism 4, and multiple algorithms:
```bash
python run_benchmark.py --batches 1,2,4,8 --seqs 1k --tp 4 --algos NCCL,ONESHOT --nccl-protos LL
```
For a specific NCCL strategy that uses only the ring algorithm for allreduce, configure the NCCL_ALGO environment variable as "allreduce:ring".

### 3. Plotting Only
If you have already run the benchmarks and want to regenerate the plots from the saved results:
```bash
python run_benchmark.py --plot-e2e-only
```

## Output

The tool generates the following outputs:
- **Results CSV**: Consolidated benchmark results.
- **Figures**: Plots showing performance comparisons (latency, bandwidth, etc.) are saved in the figures directory.
- **Logs**: Execution logs and intermediate data files.

