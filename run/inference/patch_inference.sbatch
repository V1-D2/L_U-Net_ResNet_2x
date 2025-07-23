#!/bin/bash
#SBATCH --job-name=patch_inference_amsr2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --nodelist=r740-105-15
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=24:00:00

echo "============================================"
echo "PATCH-BASED INFERENCE FOR AMSR2 SR"
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

mkdir -p ./patch_inference_results ./patch_inference_results/arrays ./patch_inference_results/images ./patch_inference_results/visualizations

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

# Step 4: Run patch-based inference
echo "============================================"
echo "Starting patch-based inference..."
echo "Processing 20 samples from last NPZ file"
echo "Using 75% overlap with Gaussian blending"
echo "============================================"

# Run with the larger dataset (500_split)
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/L_U-Net_ResNet_2x:/workspace \
#    --bind /scratch/tmp/vdidur/500_split:/data:ro \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --env OMP_NUM_THREADS=4 \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python patch_based_inference.py \
    --npz-dir /data \
    --model-path $MODEL_PATH \
    --num-samples 20 \
    --save-dir ./patch_inference_results \
    --overlap-ratio 0.75

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "Inference completed successfully!"
    echo "============================================"

    # Show results summary
    echo ""
    echo "Results summary:"
    echo "----------------"

    # Count output files
    echo "Arrays saved: $(ls -1 ./patch_inference_results/arrays/*.npz 2>/dev/null | wc -l)"
    echo "Images saved: $(ls -1 ./patch_inference_results/images/*.png 2>/dev/null | wc -l)"
    echo "Visualizations: $(ls -1 ./patch_inference_results/visualizations/*.png 2>/dev/null | wc -l)"

    # Show summary statistics if available
    if [ -f "./patch_inference_results/summary_statistics.json" ]; then
        echo ""
        echo "Performance metrics:"
        apptainer exec --nv \
            --bind $HOME/local-python:$HOME/.local \
            /home/shared/containers/tensorflow-25.02-py3.sif \
            python -c "
import json
with open('./patch_inference_results/summary_statistics.json', 'r') as f:
    stats = json.load(f)
    print(f'  Average PSNR: {stats.get(\"average_psnr\", \"N/A\"):.2f} dB')
    print(f'  Average SSIM: {stats.get(\"average_ssim\", \"N/A\"):.4f}')
    print(f'  Average processing time: {stats.get(\"average_processing_time\", \"N/A\"):.2f}s per image')
    print(f'  Total samples processed: {stats.get(\"num_samples\", \"N/A\")}')
"
    fi

    # List output structure
    echo ""
    echo "Output directory structure:"
    echo "--------------------------"
    ls -la ./patch_inference_results/

else
    echo "============================================"
    echo "ERROR: Inference failed!"
    echo "Check the error log above for details"
    echo "============================================"
    exit 1
fi

echo ""
echo "============================================"
echo "Patch-based Inference Finished: $(date)"
echo "============================================"

# Optional: Create a compressed archive of results
echo ""
echo "Creating compressed archive of results..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf patch_inference_results_${TIMESTAMP}.tar.gz ./patch_inference_results/
echo "Archive created: patch_inference_results_${TIMESTAMP}.tar.gz"
ls -lh patch_inference_results_${TIMESTAMP}.tar.gz

# Show GPU status at the end
echo ""
echo "Final GPU status:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

# Alternative: If you want to use the other data directory, uncomment below:
# echo ""
# echo "============================================"
# echo "Running inference on alternative dataset..."
# echo "============================================"
#
# apptainer exec --nv \
#     --bind $HOME/local-python:$HOME/.local \
#     --bind /home/vdidur/L_U-Net_ResNet_2x:/workspace \
#     --bind /home/vdidur/temperature_sr_project/data:/data:ro \
#     --env PYTHONPATH=/workspace:$PYTHONPATH \
#     --env OMP_NUM_THREADS=4 \
#     --workdir /workspace \
#     /home/shared/containers/tensorflow-25.02-py3.sif \
#     python patch_based_inference.py \
#     --npz-dir /data \
#     --model-path $MODEL_PATH \
#     --num-samples 20 \
#     --save-dir ./patch_inference_results_alt \
#     --overlap-ratio 0.75