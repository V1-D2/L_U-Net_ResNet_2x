#!/bin/bash
#SBATCH --job-name=amsr2_8x_minimal
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=64G
#SBATCH --time=72:00:00

echo "============================================"
echo "AMSR2 Minimal Training Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/vdidur/L_U-Net_ResNet_2x

# Step 1: Install ONLY essential packages
echo "============================================"
echo "Installing essential packages only..."
echo "============================================"

# Use Python 3.11 or 3.10 if available to avoid Python 3.12 issues
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user --no-deps \
    torch==2.1.0 \
    torchvision==0.16.0 \
    'numpy>=1.21.0,<1.24.0' \
    tqdm \
    psutil

# Install remaining packages without dependencies conflicts
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user \
    matplotlib \
    scikit-learn \
    Pillow

# Step 2: Quick verification
echo "============================================"
echo "Verifying core installations..."
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import torch
import numpy as np
print('PyTorch:', torch.__version__)
print('NumPy:', np.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    torch.cuda.empty_cache()
"

# Step 3: Create directories
mkdir -p ./models ./logs ./results

# Step 4: Run training
echo "============================================"
echo "Starting training..."
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/L_U-Net_ResNet_2x:/workspace \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --env OMP_NUM_THREADS=4 \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python gpu_sequential_amsr2_optimized.py \
    --npz-dir /data \
    --max-files 4 \
    --epochs 60 \
    --batch-size 32 \
    --num-workers 3 \
    --files-per-batch 8 \
    --max-swaths-per-file 3000 \
    --gradient-accumulation 4 \
    --lr 5e-5 \
    --save-path ./models/best_amsr2_2x_1st.pth

echo "============================================"
echo "Training Finished: $(date)"
echo "============================================"

# Check results
if [ -f "./models/best_amsr2_8x.pth" ]; then
    echo "✅ Model saved successfully!"
    ls -lh ./models/best_amsr2_8x.pth
fi

# Show GPU status
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"