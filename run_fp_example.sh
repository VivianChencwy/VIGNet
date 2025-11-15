#!/bin/bash
# Quick start script for FP1/FP2-only VIGNet experiments

echo "=========================================="
echo "VIGNet FP1/FP2 Training Example"
echo "=========================================="

# Example 1: Run single trial (trial 2) with regression task
echo ""
echo "Example 1: Single trial regression"
echo "------------------------------------------"
echo "Command: python experiment_fp.py --trial 2 --task RGS"
echo ""
# Uncomment to run:
# python experiment_fp.py --trial 2 --task RGS

# Example 2: Run trials 1-3 (for comparison with analysis)
echo ""
echo "Example 2: Trials 1-3 for comparison"
echo "------------------------------------------"
for trial in 1 2 3; do
    echo "Running trial $trial..."
    # Uncomment to run:
    # python experiment_fp.py --trial $trial --task RGS
done

# Example 3: Run all trials
echo ""
echo "Example 3: All trials (1-23)"
echo "------------------------------------------"
echo "Command: python experiment_fp.py --task RGS"
echo ""
# Uncomment to run:
# python experiment_fp.py --task RGS

echo ""
echo "=========================================="
echo "To actually run the examples, uncomment"
echo "the python commands in this script."
echo "=========================================="

