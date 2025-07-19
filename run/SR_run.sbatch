#!/bin/bash
#SBATCH --job-name=C_amsr2_enhanced_attention
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=64G
#SBATCH --time=72:00:00

echo "============================================"
echo "ENHANCED AMSR2 Training with Spatial Attention"
echo "Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Memory allocated: 64GB per GPU"
echo "============================================"

# Set environment variables to suppress nvidia-smi warnings
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1

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

# Step 2: Create directories for enhanced model
mkdir -p ./models_enhanced_2nd_generation ./logs_2nd ./results_2nd ./visualizations_2nd

# Step 3: Run enhanced training
echo "============================================"
echo "Starting ENHANCED training with attention..."
echo "Target: 34+ PSNR with spatial attention"
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/L_U-Net_ResNet_2x:/workspace \
    --bind /scratch/tmp/data:/data:ro \
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
    --max-swaths-per-file 500 \
    --gradient-accumulation 4 \
    --lr 1e-4 \
    --save-dir ./models_enhanced \
    --validate-every 2 \
    --target-height 2048 \
    --target-width 208 \
    --use-amp

echo "============================================"
echo "Enhanced Training Finished: $(date)"
echo "============================================"