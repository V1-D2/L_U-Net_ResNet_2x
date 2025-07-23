#!/bin/bash
#SBATCH --job-name=cascaded_4x_inference
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --nodelist=r740-105-15
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=24:00:00

echo "============================================"
echo "CASCADED 4X INFERENCE FOR AMSR2 SR"
echo "Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 32GB per GPU"
echo "============================================"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1

# Change to project directory
cd /home/vdidur/L_U-Net_ResNet_2x

# Step 1: Verify environment and dependencies
echo "============================================"
echo "Verifying environment..."
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import torch
import numpy as np
import cv2
print('PyTorch:', torch.__version__)
print('NumPy:', np.__version__)
print('OpenCV:', cv2.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB')
"

# Step 2: Create output directories
echo "============================================"
echo "Creating output directories..."
echo "============================================"

mkdir -p ./cascaded_results ./cascaded_results/arrays ./cascaded_results/images ./cascaded_results/visualizations

# Step 3: Check for model file
echo "============================================"
echo "Checking for trained model..."
echo "============================================"

# Try different possible model locations
MODEL_PATH=""
if [ -f "./models_enhanced_2nd_generation/best_model.pth" ]; then
    MODEL_PATH="./models_enhanced_2nd_generation/best_model.pth"
    echo "Found model: $MODEL_PATH"
elif [ -f "./models_enhanced/best_model.pth" ]; then
    MODEL_PATH="./models_enhanced/best_model.pth"
    echo "Found model: $MODEL_PATH"
else
    echo "ERROR: No model found!"
    echo "Checked locations:"
    echo "  - ./models_enhanced_2nd_generation/best_model.pth"
    echo "  - ./models_enhanced/best_model.pth"
    exit 1
fi

# Show model info
ls -lh $MODEL_PATH

# Step 4: Run cascaded 4x inference
echo "============================================"
echo "Starting cascaded 4x inference..."
echo "Processing 5 samples from last NPZ file"
echo "Stage 1: Original → 2x SR"
echo "Stage 2: 2x SR → 4x SR"
echo "Using 75% overlap with Gaussian blending"
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/L_U-Net_ResNet_2x:/workspace \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --env OMP_NUM_THREADS=4 \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python patch_based_inference_4x_cascade.py \
    --npz-dir /data \
    --model-path $MODEL_PATH \
    --num-samples 5 \
    --save-dir ./cascaded_results \
    --overlap-ratio 0.75

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "Cascaded inference completed successfully!"
    echo "============================================"

    # Show results summary
    echo ""
    echo "Results summary:"
    echo "----------------"

    # Count output files
    echo "Arrays saved: $(ls -1 ./cascaded_results/arrays/*.npz 2>/dev/null | wc -l)"
    echo "Images saved: $(ls -1 ./cascaded_results/images/*.png 2>/dev/null | wc -l)"
    echo "Visualizations: $(ls -1 ./cascaded_results/visualizations/*.png 2>/dev/null | wc -l)"

    # Show summary statistics if available
    if [ -f "./cascaded_results/cascaded_summary_statistics.json" ]; then
        echo ""
        echo "Performance metrics:"
        apptainer exec --nv \
            --bind $HOME/local-python:$HOME/.local \
            /home/shared/containers/tensorflow-25.02-py3.sif \
            python -c "
import json
with open('./cascaded_results/cascaded_summary_statistics.json', 'r') as f:
    stats = json.load(f)
    print(f'  Average time for 2x stage: {stats.get(\"average_processing_time_2x\", \"N/A\"):.2f}s')
    print(f'  Average time for 4x stage: {stats.get(\"average_processing_time_4x\", \"N/A\"):.2f}s')
    print(f'  Total average time per image: {stats.get(\"total_average_time\", \"N/A\"):.2f}s')
    print(f'  Total samples processed: {stats.get(\"num_samples\", \"N/A\")}')
"
    fi

    # List output structure
    echo ""
    echo "Output directory structure:"
    echo "--------------------------"
    ls -la ./cascaded_results/

else
    echo "============================================"
    echo "ERROR: Cascaded inference failed!"
    echo "Check the error log above for details"
    echo "============================================"
    exit 1
fi

echo ""
echo "============================================"
echo "Cascaded 4x Inference Finished: $(date)"
echo "============================================"

# Optional: Create a compressed archive of results
echo ""
echo "Creating compressed archive of results..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf cascaded_results_${TIMESTAMP}.tar.gz ./cascaded_results/
echo "Archive created: cascaded_results_${TIMESTAMP}.tar.gz"
ls -lh cascaded_results_${TIMESTAMP}.tar.gz

# Show GPU status at the end
echo ""
echo "Final GPU status:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"