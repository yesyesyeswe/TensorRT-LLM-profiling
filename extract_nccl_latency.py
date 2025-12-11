#!/usr/bin/env python3
import os
import sys
import csv
import argparse
import sqlite3

#                 ncclDevKernel_AllReduce_Sum_bf16_RING_LL
DEFAULT_KERNEL = "ncclDevKernel_AllReduce_Sum_bf16_RING_LL"

def find_sqlites(input_path: str):
    paths = []
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.endswith(".sqlite"):
                    paths.append(os.path.join(root, f))
    else:
        paths.append(input_path)
    return paths

def pick_kernel_id(conn: sqlite3.Connection, kernel_name: str):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cur.fetchall()}
    if "StringIds" not in tables or "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        return None
    candidates = []
    cur.execute("SELECT id FROM StringIds WHERE value = ? LIMIT 1", (kernel_name,))
    row = cur.fetchone()
    if row:
        candidates.append(row[0])
    core = kernel_name.split("(")[0].strip()
    cur.execute("SELECT id FROM StringIds WHERE value = ? OR value LIKE ?", (core, f"%{core}%"))
    for r in cur.fetchall():
        if r[0] not in candidates:
            candidates.append(r[0])
    best = None
    best_cnt = 0
    for cid in candidates:
        cur.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE shortName=?", (cid,))
        cnt = cur.fetchone()[0]
        if cnt > best_cnt:
            best_cnt = cnt
            best = cid
    return best

def query_latencies(conn: sqlite3.Connection, kernel_name: str):
    cur = conn.cursor()
    name_id = pick_kernel_id(conn, kernel_name)
    if name_id is None:
        return []

    cur.execute(
        """
        SELECT deviceId, start, (end - start) / 1000.0 AS duration_us
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE shortName = ?
        ORDER BY deviceId, start
        """,
        (name_id,),
    )
    return cur.fetchall()

def build_algo(su_algo: str, nccl_proto: str | None) -> str:
    if nccl_proto:
        nccl_proto = nccl_proto.strip()
    return su_algo if not nccl_proto else f"{su_algo}_{nccl_proto}"

def main():
    parser = argparse.ArgumentParser(description="Extract NCCL kernel latencies from SQLite to CSV")
    parser.add_argument("--input", required=True, help="SQLite file or directory containing .sqlite files")
    parser.add_argument("--output-dir", default=os.getcwd(), help="Directory to write CSV outputs (default: cwd)")
    parser.add_argument("--kernel-name", default=DEFAULT_KERNEL, help="Kernel name to match in StringIds.value")
    args = parser.parse_args()

    batch_size = os.environ.get("BATCH_SIZE")
    seq_len = os.environ.get("SEQUENCE_LENGTH")
    su_algo = os.environ.get("SU_ALGO")
    tp_size = os.environ.get("TP")
    nccl_proto = os.environ.get("NCCL_PROTO")

    if not batch_size or not seq_len or not su_algo:
        print("Missing required env vars: BATCH_SIZE, SEQUENCE_LENGTH, SU_ALGO", file=sys.stderr)
        sys.exit(1)

    algo = build_algo(su_algo, nccl_proto)

    if nccl_proto and nccl_proto.strip():
        filename = f"latency_bs{batch_size}_sl{seq_len}_tp{tp_size}_algo{su_algo}_{nccl_proto}.csv"
    else:
        filename = f"latency_bs{batch_size}_sl{seq_len}_tp{tp_size}_algo{su_algo}.csv"

    latency_dir = os.path.join(args.output_dir, "latency")
    os.makedirs(latency_dir, exist_ok=True)
    out_path = os.path.join(latency_dir, filename)

    rows_out = []
    for db_path in find_sqlites(args.input):
        try:
            conn = sqlite3.connect(db_path)
        except Exception as e:
            print(f"Skip {db_path}: cannot open ({e})", file=sys.stderr)
            continue
        try:
            lat_rows = query_latencies(conn, args.kernel_name)
        finally:
            conn.close()

        for device_id, _start, duration_us in lat_rows:
            rows_out.append([
                algo,
                batch_size,
                seq_len,
                tp_size,
                device_id,
                duration_us,
            ])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Batch_size", "Sequence_length", "TP_Size", "CudaDevice", "Latency"])
        writer.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows -> {out_path}")

if __name__ == "__main__":
    main()
