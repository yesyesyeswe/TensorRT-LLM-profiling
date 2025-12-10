# AllReduce CSV Recording Feature

This feature adds CSV recording functionality to the AllReduce plugin to track communication data sizes during operations.

## Overview

The feature automatically records AllReduce operation details in CSV files when specific environment variables are set:
- **Algorithm**: Combined SU_ALGO and NCCL_PROTO (if available)
- **Batch_size**: From BATCH_SIZE environment variable
- **Sequence_length**: From SEQUENCE_LENGTH environment variable  
- **Communication**: Data size in bytes (size * getDTypeSize(mType))

## CSV File Format

Files are saved to: `/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication/`

Filename format: `comm_${BATCH_SIZE}_${SEQUENCE_LENGTH}_${SU_ALGO}${NCCL_PROTO}.csv`

CSV structure:
```
Algorithm,Batch_size,Sequence_length,Communication
NCCL_LL,32,512,4096
ONESHOT,64,256,8192
TWOSHOT_SIMPLE,16,1024,16384
```

## Required Environment Variables

- `BATCH_SIZE`: Batch size for the operation
- `SEQUENCE_LENGTH`: Sequence length for the operation
- `SU_ALGO`: Algorithm selection (NCCL, ONESHOT, TWOSHOT, etc.)
- `NCCL_PROTO`: Optional NCCL protocol (LL, SIMPLE, etc.)

## Usage Example

```bash
# Set environment variables
export BATCH_SIZE=32
export SEQUENCE_LENGTH=512
export SU_ALGO=NCCL
export NCCL_PROTO=LL

# Run your TensorRT-LLM application
# CSV file will be automatically generated
```

## Testing

A test suite is provided to verify the functionality:

```bash
cd /root/autodl-tmp/TensorRT-LLM/mybenchmark
./build_and_test.sh
```

The test will:
1. Build the test application
2. Run multiple test cases with different environment variable combinations
3. Verify CSV file generation and content
4. Display the generated CSV files

## Implementation Details

The CSV recording is implemented in `allreducePlugin.cpp` at line 385, right after the "Log runtime strategy" comment. The feature:

1. Calculates data size using `size * tensorrt_llm::common::getDTypeSize(mType)`
2. Reads environment variables
3. Creates the results directory if it doesn't exist
4. Generates CSV files with proper headers
5. Appends data rows for each AllReduce operation

## Notes

- If NCCL_PROTO is not set or empty, it will be omitted from the algorithm name
- Files are appended to, allowing multiple runs to accumulate data
- Headers are only written when creating new files
- The feature only activates when all required environment variables are set