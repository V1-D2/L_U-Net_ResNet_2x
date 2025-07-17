#!/bin/bash
#SBATCH --job-name=amsr2_enhanced_attention
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=64G
#SBATCH --time=72:00:00

echo "============================================"
echo "ENHANCED AMSR2 Training with Spatial Attention"
echo "Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/vdidur/L_U-Net_ResNet_2x

# Step 1: Install essential packages + opencv for metrics
echo "============================================"
echo "Installing packages for enhanced model..."
echo "============================================"

# Core packages
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user --no-deps \
    torch==2.1.0 \
    torchvision==0.16.0 \
    'numpy>=1.21.0,<1.24.0' \
    tqdm \
    psutil

# Additional packages for enhanced model
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user \
    matplotlib \
    scikit-learn \
    Pillow \
    opencv-python-headless

# Step 2: Verify installations
echo "============================================"
echo "Verifying installations for enhanced model..."
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
    torch.cuda.empty_cache()
print('Ready for enhanced training with spatial attention!')
"

# Step 3: Create directories for enhanced model
mkdir -p ./models_enhanced ./logs ./results ./visualizations

# Step 4: Run enhanced training
echo "============================================"
echo "Starting ENHANCED training with attention..."
echo "Target: 34+ PSNR with spatial attention"
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/L_U-Net_ResNet_2x:/workspace \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --env OMP_NUM_THREADS=4 \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python enhanced_amsr2_model.py \
    --npz-dir /data \
    --max-files 100 \
    --epochs 150 \
    --batch-size 8 \
    --num-workers 4 \
    --files-per-batch 5 \
    --max-swaths-per-file 1000 \
    --gradient-accumulation 4 \
    --lr 5e-5 \
    --save-dir ./models_enhanced \
    --validate-every 2 \
    --target-height 2048 \
    --target-width 208 \
    --use-amp

echo "============================================"
echo "Enhanced Training Finished: $(date)"
echo "============================================"

# Check results
echo "Checking training results..."

if [ -f "./models_enhanced/best_model.pth" ]; then
    echo "✅ Enhanced model saved successfully!"
    ls -lh ./models_enhanced/best_model.pth

    # Check for training artifacts
    echo ""
    echo "Training artifacts:"
    ls -la ./models_enhanced/

    if [ -f "./models_enhanced/training_progress.png" ]; then
        echo "📊 Training progress plot saved"
    fi

    if [ -f "./models_enhanced/training_history.json" ]; then
        echo "📈 Training history saved"
        # Show best metrics
        echo ""
        echo "Best metrics achieved:"
        apptainer exec --nv \
            --bind $HOME/local-python:$HOME/.local \
            /home/shared/containers/tensorflow-25.02-py3.sif \
            python -c "
import json
with open('./models_enhanced/training_history.json', 'r') as f:
    history = json.load(f)
    if 'best_metrics' in history:
        print(f'  PSNR: {history[\"best_metrics\"][\"psnr\"]:.2f} dB')
        print(f'  SSIM: {history[\"best_metrics\"][\"ssim\"]:.4f}')
        print(f'  Loss: {history[\"best_metrics\"][\"loss\"]:.4f}')
"
    fi
else
    echo "❌ Model file not found - training may have failed"
fi

echo ""
echo "============================================"
echo "GPU Memory Usage Summary:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

# Optional: Run quick test if model exists
if [ -f "./models_enhanced/best_model.pth" ]; then
    echo ""
    echo "============================================"
    echo "Running quick inference test..."
    echo "=================