#!/bin/bash
#
# VIGNet Single-Target Training - Run All Targets
# This script trains 4 separate models, one for each fatigue indicator:
#   1. PERCLOS - eye closure percentage
#   2. KSS - Karolinska Sleepiness Scale (normalized to 0-1)
#   3. miss_rate - target miss rate
#   4. false_alarm - false alarm rate
#
# Usage:
#   bash run_all_targets.sh
#   bash run_all_targets.sh --log-dir ./my_logs
#

set -e  # Exit on error

# Parse arguments
LOG_DIR="./logs_experiment"
DATA_DIR="data/experiment_20251124_140734"

while [[ $# -gt 0 ]]; do
    case $1 in
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "VIGNet Single-Target Training - All Targets"
echo "============================================================"
echo "Log directory: $LOG_DIR"
echo "Data directory: $DATA_DIR"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Change to VIGNet directory
cd "$(dirname "$0")"

# Create log directory
mkdir -p "$LOG_DIR"

# Define targets
TARGETS=("perclos" "kss" "miss_rate" "false_alarm")

# Train each target
for target in "${TARGETS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Training target: $target"
    echo "============================================================"
    
    python experiment_single_target.py \
        --target "$target" \
        --log-dir "$LOG_DIR" \
        --data-dir "$DATA_DIR"
    
    echo ""
    echo "Completed: $target"
done

echo ""
echo "============================================================"
echo "All training completed!"
echo "============================================================"
echo ""

# Generate analysis and report
echo "Generating analysis and visualizations..."
python analyze_results.py --log-dir "$LOG_DIR"

echo ""
echo "============================================================"
echo "All Done!"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Results saved to: $LOG_DIR"
echo ""
echo "Summary report: $LOG_DIR/summary/comprehensive_report.md"
echo "============================================================"


