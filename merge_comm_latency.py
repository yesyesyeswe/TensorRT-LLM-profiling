#!/usr/bin/env python3
import os
import csv
import argparse
from collections import defaultdict


REQUIRED_COMM_COLS = {"Algorithm", "Batch_size", "Sequence_length", "TP_Size", "CudaDevice", "MPI_Rank", "Communication"}
REQUIRED_LAT_COLS = {"Algorithm", "Batch_size", "Sequence_length", "TP_Size", "CudaDevice", "Latency"}


def read_csv_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]
        idx = {h: i for i, h in enumerate(header)}
        for row in reader:
            rows.append((header, idx, row))
    return rows


def ensure_columns(header_set, required_set, label):
    missing = required_set - header_set
    if missing:
        raise RuntimeError(f"{label} CSV is missing required columns: {', '.join(sorted(missing))}")


def make_key(idx, row):
    return (
        row[idx["Algorithm"]].strip(),
        row[idx["Batch_size"]].strip(),
        row[idx["Sequence_length"]].strip(),
        row[idx["TP_Size"]].strip(),
        row[idx["CudaDevice"]].strip(),
    )


def load_comm(comm_file):
    comm_map = defaultdict(list)
    for header, idx, row in read_csv_rows(comm_file):
        ensure_columns(set(header), REQUIRED_COMM_COLS, "Communication")
        key = make_key(idx, row)
        comm_map[key].append({
            "Algorithm": row[idx["Algorithm"]].strip(),
            "Batch_size": row[idx["Batch_size"]].strip(),
            "Sequence_length": row[idx["Sequence_length"]].strip(),
            "TP_Size": row[idx["TP_Size"]].strip(),
            "CudaDevice": row[idx["CudaDevice"]].strip(),
            "MPI_Rank": row[idx["MPI_Rank"]].strip(),
            "Communication": row[idx["Communication"]].strip(),
        })
    return comm_map


def load_latency(lat_file):
    lat_map = defaultdict(list)
    for header, idx, row in read_csv_rows(lat_file):
        ensure_columns(set(header), REQUIRED_LAT_COLS, "Latency")
        key = make_key(idx, row)
        lat_map[key].append({
            "Algorithm": row[idx["Algorithm"]].strip(),
            "Batch_size": row[idx["Batch_size"]].strip(),
            "Sequence_length": row[idx["Sequence_length"]].strip(),
            "TP_Size": row[idx["TP_Size"]].strip(),
            "CudaDevice": row[idx["CudaDevice"]].strip(),
            "Latency": row[idx["Latency"]].strip(),
        })
    return lat_map


def write_merged(out_path, merged_rows):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Batch_size", "Sequence_length", "TP_Size", "CudaDevice", "MPI_Rank", "Latency", "Communication", "Bandwidth"])
        writer.writerows(merged_rows)


def merge_files(comm_file, lat_file, output_dir):
    """
    Merge communication and latency data files, compute bandwidth metrics.
    
    This function combines communication volume data with corresponding latency measurements
    to calculate effective bandwidth for different parallel communication algorithms.
    The calculations account for algorithm-specific communication patterns and scaling factors.
    
    Args:
        comm_file: Path to CSV file containing communication volume data
        lat_file: Path to CSV file containing latency measurements
        output_dir: Directory to save the merged results with computed bandwidth
    """
    comm_rows = read_csv_rows(comm_file)
    lat_rows = read_csv_rows(lat_file)
    
    merged_rows = []
    
    # Merge rows one-to-one by row number
    min_rows = min(len(comm_rows), len(lat_rows))
    for i in range(min_rows):
        comm_header, comm_idx, comm_row = comm_rows[i]
        lat_header, lat_idx, lat_row = lat_rows[i]
        
        ensure_columns(set(comm_header), REQUIRED_COMM_COLS, "Communication")
        ensure_columns(set(lat_header), REQUIRED_LAT_COLS, "Latency")
        
        tp_size = int(comm_row[comm_idx["TP_Size"]].strip())  
        algorithm = comm_row[comm_idx["Algorithm"]].strip()

        # -----------------------------------------------------------------
        # Communication volume adjustment based on algorithm type
        # Different NCCL algorithms have different communication patterns
        # -----------------------------------------------------------------
        if algorithm == "NCCL_Simple":
            comm_row[comm_idx["Communication"]] = float(comm_row[comm_idx["Communication"]].strip()) * 2 * (tp_size - 1)
        elif algorithm == "NCCL_LL":
            comm_row[comm_idx["Communication"]] = float(comm_row[comm_idx["Communication"]].strip()) * 2 * (tp_size - 1) * 2
        elif algorithm == "NCCL_LL128":
            comm_row[comm_idx["Communication"]] = float(comm_row[comm_idx["Communication"]].strip()) * 2 * (tp_size - 1) * (128 / 120)
        elif algorithm == "ONESHOT":
            comm_row[comm_idx["Communication"]] = float(comm_row[comm_idx["Communication"]].strip()) * tp_size * (tp_size - 1)
        elif algorithm == "TWOSHOT":
            comm_row[comm_idx["Communication"]] = float(comm_row[comm_idx["Communication"]].strip()) * 2 * (tp_size - 1)

        communication_bytes = comm_row[comm_idx["Communication"]]
        latency_us = float(lat_row[lat_idx["Latency"]].strip())

        # ------------------------------------------------------------------------------------
        # Bandwidth calculation
        # Convert communication volume (GB) and latency (μs) to bandwidth (GB/s)
        # Formula: (Communication(B)/1024/1024/1024) / (Latency(us)/1000/1000) = BandwidthGB/s
        # ------------------------------------------------------------------------------------
        if latency_us > 0: 
            Bandwidth = communication_bytes * (1000.0 / 1024.0) * (1000.0 / 1024.0) / latency_us / 1024.0 
        else:
            Bandwidth = 0.0

        merged_rows.append([
            comm_row[comm_idx["Algorithm"]].strip(),
            comm_row[comm_idx["Batch_size"]].strip(),
            comm_row[comm_idx["Sequence_length"]].strip(),
            comm_row[comm_idx["TP_Size"]].strip(),
            comm_row[comm_idx["CudaDevice"]].strip(),
            comm_row[comm_idx["MPI_Rank"]].strip(),
            lat_row[lat_idx["Latency"]].strip(),
            comm_row[comm_idx["Communication"]],
            f"{Bandwidth:.3f}",
        ])

    if merged_rows:
        first_row = merged_rows[0]
        algo, bs, sl, tp = first_row[0], first_row[1], first_row[2], first_row[3]
        out_name = f"merge_bs{bs}_sl{sl}_tp{tp}_algo{algo}.csv"
    else:
        out_name = "merge_empty.csv"
    
    out_path = os.path.join(output_dir, out_name)
    write_merged(out_path, merged_rows)
    return out_path, len(merged_rows)


def main():
    parser = argparse.ArgumentParser(description="Merge communication CSV with latency CSV by key columns")
    parser.add_argument("--comm-file", required=True, help="Path to communication CSV file")
    parser.add_argument("--latency-file", required=True, help="Path to latency CSV file")
    parser.add_argument("--output-dir", default=os.path.join(os.getcwd(), "merged"), help="Directory to write merged CSV")
    args = parser.parse_args()

    out_path, nrows = merge_files(args.comm_file, args.latency_file, args.output_dir)
    print(f"Merged {nrows} rows -> {out_path}")


if __name__ == "__main__":
    main()

