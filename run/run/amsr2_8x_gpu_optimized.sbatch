#!/bin/bash
#SBATCH --job-name=amsr2_8x_22gb
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=12      # Немного меньше CPU
#SBATCH --mem-per-gpu=32G      # 32GB RAM достаточно
#SBATCH --time=48:00:00

echo "============================================"
echo "AMSR2 Training for 22GB GPU Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /home/vdidur/L_U-Net_ResNet

# Step 1: Install/Update all required packages
echo "============================================"
echo "Installing required packages..."
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user --upgrade pip

# Core dependencies with specific versions
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user \
    'torch>=2.0.0' \
    'torchvision>=0.15.0' \
    'numpy>=1.21.0,<1.24.0' \
    'matplotlib>=3.5.0' \
    'scikit-learn>=1.0.0' \
    'tqdm>=4.60.0' \
    'Pillow>=8.0.0' \
    'psutil>=5.8.0' \
    'h5py>=3.0.0' \
    'pandas>=1.3.0' \
    'tensorboard>=2.10.0'

# Additional dependencies for optimization
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user \
    'scipy>=1.7.0' \
    'tensorboardX>=2.2' \
    'wandb>=0.12.0' \
    'opencv-python-headless>=4.5.0' \
    'albumentations>=1.0.0'

# Step 2: Verify installations
echo "============================================"
echo "Verifying installations..."
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
import torch
import torchvision
import numpy as np
import matplotlib
import sklearn
import tqdm
import PIL
import psutil
import h5py
import pandas as pd

print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('Torchvision:', torchvision.__version__)
print('NumPy:', np.__version__)
print('Matplotlib:', matplotlib.__version__)
print('Scikit-learn:', sklearn.__version__)
print('TQDM:', tqdm.__version__)
print('Pillow:', PIL.__version__)
print('Psutil:', psutil.__version__)
print('H5py:', h5py.__version__)
print('Pandas:', pd.__version__)

# GPU check
print('\nGPU Information:')
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('GPU Memory:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')
    print('CUDA Version:', torch.version.cuda)
    print('cuDNN Version:', torch.backends.cudnn.version())

    # Test GPU computation
    x = torch.randn(10, 10).cuda()
    y = x @ x.T
    print('GPU computation test: PASSED')

    # Clear GPU cache
    torch.cuda.empty_cache()
    print('GPU cache cleared')
else:
    print('WARNING: No GPU detected!')
"

# Step 3: Create necessary directories
echo "============================================"
echo "Setting up directories..."
echo "============================================"

mkdir -p ./models
mkdir -p ./logs
mkdir -p ./results
mkdir -p ./checkpoints
mkdir -p ./tensorboard_logs

echo "Directories created:"
ls -la ./

# Step 4: Check data accessibility
echo "============================================"
echo "Checking data accessibility..."
echo "============================================"

if [ -d "/home/vdidur/temperature_sr_project/data" ]; then
    echo "Data directory found!"
    echo "Number of NPZ files: $(ls -1 /home/vdidur/temperature_sr_project/data/*.npz 2>/dev/null | wc -l)"
    echo "Total data size: $(du -sh /home/vdidur/temperature_sr_project/data 2>/dev/null | cut -f1)"
else
    echo "WARNING: Data directory not found!"
fi

# Step 5: Run the training
echo "============================================"
echo "Starting optimized training..."
echo "============================================"

# Оптимальные параметры для 22GB GPU
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/L_U-Net_ResNet:/workspace \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --env OMP_NUM_THREADS=4 \
    --env PYTHONUNBUFFERED=1 \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python gpu_sequential_amsr2_optimized.py \
    --npz-dir /data \
    --max-files 4 \
    --epochs 20 \
    --batch-size 12 \
    --num-workers 3 \
    --files-per-batch 2 \
    --max-swaths-per-file 200 \
    --gradient-accumulation 2 \
    --lr 1e-4 \
    --save-path ./models/best_amsr2_8x_22gb.pth

# Step 6: Post-training summary
echo "============================================"
echo "Training Completed: $(date)"
echo "============================================"

# Check if model was saved
if [ -f "./models/best_amsr2_8x_22gb.pth" ]; then
    echo "✅ Model saved successfully!"
    ls -lh ./models/best_amsr2_8x_22gb.pth
else
    echo "❌ Warning: Model file not found!"
fi

# Show log tail
echo "============================================"
echo "Last 20 lines of training log:"
echo "============================================"
tail -n 20 amsr2_optimized_training.log 2>/dev/null || echo "Log file not found"

# Show GPU memory usage at the end
echo "============================================"
echo "Final GPU status:"
echo "============================================"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"

echo "============================================"
echo "Job completed!"
echo "============================================"