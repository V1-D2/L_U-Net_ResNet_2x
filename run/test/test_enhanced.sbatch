#!/bin/bash
#SBATCH --job-name=test_enhanced_amsr2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=01:00:00

echo "============================================"
echo "Testing Enhanced AMSR2 Model: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

cd /home/vdidur/L_U-Net_ResNet_2x

# Create test results directory
mkdir -p test_enhanced_results

echo "Testing enhanced model..."

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/L_U-Net_ResNet_2x:/workspace \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python test_enhanced_model.py

echo "============================================"
echo "Enhanced Model Testing Finished: $(date)"
echo "============================================"

# Check results
if [ -f "./test_enhanced_results/enhanced_test_results.png" ]; then
    echo "✅ Testing completed successfully!"
    echo "Results saved in: test_enhanced_results/"
    ls -la test_enhanced_results/
else
    echo "❌ Testing may have failed - no output files found"
fi