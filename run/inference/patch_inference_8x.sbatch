#!/bin/bash
#SBATCH --job-name=cascaded_8x_inference
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --nodelist=r740-105-15
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=48:00:00

echo "============================================"
echo "CASCADED 8X INFERENCE FOR AMSR2 SR"
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

# Step 1: Verify environment
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
echo "Creating output directories for 8x results..."
echo "============================================"

mkdir -p ./cascaded_8x_results
mkdir -p ./cascaded_8x_results/variant1_bicubic_first
mkdir -p ./cascaded_8x_results/variant2_triple_model

# Step 3: Check for model file
echo "============================================"
echo "Checking for trained model..."
echo "============================================"

MODEL_PATH=""
if [ -f "./models_enhanced_2nd_generation/best_model.pth" ]; then
    MODEL_PATH="./models_enhanced_2nd_generation/best_model.pth"
    echo "Found model: $MODEL_PATH"
elif [ -f "./models_enhanced/best_model.pth" ]; then
    MODEL_PATH="./models_enhanced/best_model.pth"
    echo "Found model: $MODEL_PATH"
else
    echo "ERROR: No model found!"
    exit 1
fi

ls -lh $MODEL_PATH

# Step 4: Save the 8x script
echo "============================================"
echo "Creating cascaded_8x_inference.py script..."
echo "============================================"

# Note: In actual use, you would save the script from the artifact above
# For now, we assume it's already saved

# Step 5: Run cascaded 8x inference
echo "============================================"
echo "Starting cascaded 8x inference..."
echo "Processing 5 samples with two variants:"
echo "  Variant 1: Bicubic 2x → Model 2x → Model 2x = 8x"
echo "  Variant 2: Model 2x → Model 2x → Model 2x = 8x"
echo "============================================"

apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind /home/vdidur/L_U-Net_ResNet_2x:/workspace \
    --bind /home/vdidur/temperature_sr_project/data:/data:ro \
    --env PYTHONPATH=/workspace:$PYTHONPATH \
    --env OMP_NUM_THREADS=4 \
    --workdir /workspace \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python cascaded_8x_inference.py \
    --npz-dir /data \
    --model-path $MODEL_PATH \
    --num-samples 5 \
    --save-dir ./cascaded_8x_results \
    --overlap-ratio 0.75

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "Cascaded 8x inference completed successfully!"
    echo "============================================"

    # Show results summary
    echo ""
    echo "Results summary:"
    echo "----------------"
    echo "Variant 1 (Bicubic → Model → Model):"
    ls -la ./cascaded_8x_results/variant1_bicubic_first/ | head -5

    echo ""
    echo "Variant 2 (Model → Model → Model):"
    ls -la ./cascaded_8x_results/variant2_triple_model/ | head -5

    echo ""
    echo "Comparison visualizations:"
    ls -la ./cascaded_8x_results/*.png

else
    echo "============================================"
    echo "ERROR: Cascaded 8x inference failed!"
    echo "Check the error log above for details"
    echo "============================================"
    exit 1
fi

echo ""
echo "============================================"
echo "Cascaded 8x Inference Finished: $(date)"
echo "============================================"

# Create archive of results
echo ""
echo "Creating compressed archive of 8x results..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
tar -czf cascaded_8x_results_${TIMESTAMP}.tar.gz ./cascaded_8x_results/
echo "Archive created: cascaded_8x_results_${TIMESTAMP}.tar.gz"
ls -lh cascaded_8x_results_${TIMESTAMP}.tar.gz

# Show final GPU status
echo ""
echo "Final GPU status:"
nvidia-smi 2>/dev/null || echo "nvidia-smi not available"