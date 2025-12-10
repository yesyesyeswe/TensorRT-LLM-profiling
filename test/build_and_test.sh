#!/bin/bash

# Build script for CSV recording test with debug information

echo "=== Building CSV recording test ==="

# Clean previous builds
rm -f test_csv_recording test_csv_recording

# Compile the test
echo "Compiling test_csv_recording.cpp..."
g++ -std=c++17 -o test_csv_recording test_csv_recording.cpp

if [ $? -eq 0 ]; then
    echo "✓ Build successful!"
    echo ""
    echo "=== Running test ==="
    
    # Create results directory
    mkdir -p /root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication
    echo "Created results directory: /root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication"
    
    # Run the test
    ./test_csv_recording
    
    echo ""
    echo "=== Verifying CSV files ==="
    if [ -d "/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication" ]; then
        echo "Contents of results directory:"
        ls -la /root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication/
        echo ""
        echo "CSV file details:"
        for file in /root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication/*.csv; do
            if [ -f "$file" ]; then
                echo "--- $(basename "$file") ---"
                cat "$file"
                echo ""
            fi
        done
    else
        echo "✗ Results directory not found!"
    fi
else
    echo "✗ Build failed!"
    exit 1
fi