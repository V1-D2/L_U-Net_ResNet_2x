#!/bin/bash
#SBATCH --job-name=test_amsr2_8x
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=01:00:00

echo "============================================"
echo "AMSR2 Model Testing Started: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

# Change to project directory
cd /home/vdidur/L_U-Net_ResNet

# Create test results directory
mkdir -p test_results

# Run testing
echo "============================================"
echo "Testing model on single image..."
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/L_U-Net_ResNet:/workspace \
    --bind /home/vdidur/temperature_sr_project/test:/test_data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python test_amsr2_model.py \
    --model-path /workspace/models/best_amsr2_8x.pth \
    --test-file /test_data/single_amsr2_image.npz \
    --output-dir /workspace/test_results

echo "============================================"
echo "Testing Finished: $(date)"
echo "============================================"


# Check results
if [ -f "./test_results/comparison.png" ]; then
    echo "✅ Testing completed successfully!"
    echo "Results saved in: test_results/"
    ls -la models/
else
    echo "❌ Testing may have failed - no output files found"
fi

# Show GPU status
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"