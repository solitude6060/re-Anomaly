#!/bin/bash
# Monitor unified experiment and continue with next steps when complete

LOG_FILE="results/unified_full.log"
RESULTS_DIR="results/unified_full"
UNIFIED_PID=1861992

echo "=========================================="
echo "Monitoring Unified Experiment (PID: $UNIFIED_PID)"
echo "Started: $(date)"
echo "=========================================="

# Function to count completed categories
count_completed() {
    find "$RESULTS_DIR" -name "results.json" 2>/dev/null | wc -l
}

# Monitor loop
while ps -p $UNIFIED_PID > /dev/null 2>&1; do
    COMPLETED=$(count_completed)
    ELAPSED=$(ps -p $UNIFIED_PID -o etime= 2>/dev/null | tr -d ' ')
    echo "[$(date '+%H:%M:%S')] Running... Elapsed: $ELAPSED, Categories completed: $COMPLETED/15"
    sleep 60
done

echo ""
echo "=========================================="
echo "Unified Experiment COMPLETED at $(date)"
echo "=========================================="

# Check final results
echo ""
echo "Final Results:"
COMPLETED=$(count_completed)
echo "Categories completed: $COMPLETED/15"

if [ -f "$RESULTS_DIR/summary.json" ]; then
    echo ""
    echo "Summary:"
    cat "$RESULTS_DIR/summary.json"
fi

# Show last lines of log
echo ""
echo "Last 30 lines of log:"
tail -30 "$LOG_FILE"

# Check GPU availability
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv

# Start RectFlow experiment if GPU is free
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
if [ "$GPU_UTIL" -lt 50 ]; then
    echo ""
    echo "=========================================="
    echo "GPU available! Starting RectFlow experiment..."
    echo "=========================================="
    
    cd /home/ma/Research/side_project/re-fastflow
    PYTHONPATH=. nohup uv run python scripts/run_experiment_matrix.py \
        --backbones dinov3_vitl16 \
        --heads rectflow \
        --epochs 100 \
        --output_dir results/rectflow_fixed \
        > results/rectflow_fixed.log 2>&1 &
    
    RECTFLOW_PID=$!
    echo "RectFlow experiment started with PID: $RECTFLOW_PID"
    echo "Log: results/rectflow_fixed.log"
else
    echo ""
    echo "GPU still busy (utilization: $GPU_UTIL%). Manual start required."
fi

echo ""
echo "Done monitoring at $(date)"
