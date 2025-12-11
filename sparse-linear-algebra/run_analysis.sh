#!/bin/bash
# filepath: sparse-linear-algebra/run_analysis.sh

echo "====================================="
echo "SpMV Performance Analysis Pipeline"
echo "====================================="

# Navigate to SPMV directory
cd SPMV || { echo "Error: SPMV directory not found"; exit 1; }

echo ""
echo "[1/3] Cleaning previous build..."
make clean

echo ""
echo "[2/3] Building SpMV benchmarks..."
make || { echo "Error: Build failed"; exit 1; }

echo ""
echo "[3/3] Generating performance visualizations..."
cd ../output || { echo "Error: output directory not found"; exit 1; }

if [ ! -f "spmv_profile.csv" ]; then
    echo "Error: spmv_profile.csv not found"
    exit 1
fi

python3 visualize_spmv.py || { echo "Error: Visualization failed"; exit 1; }

echo ""
echo "====================================="
echo "Analysis complete!"
echo "Generated files:"
echo "  - spmv_performance.png"
echo "  - spmv_summary_table.png"
echo "====================================="