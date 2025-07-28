#!/bin/bash
#SBATCH --job-name=test_enhanced_amsr2
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=salvador
#SBATCH --gres=gpu:turing:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32G
#SBATCH --time=02:00:00

echo "============================================"
echo "Testing Enhanced AMSR2 SR Model: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set environment variables to suppress warnings
export APPTAINER_QUIET=1
export SINGULARITY_QUIET=1

# Project directory
PROJECT_DIR="$HOME/L_U-Net_ResNet_2x"
cd $PROJECT_DIR

# Create test results directory
mkdir -p test_enhanced_results

echo "============================================"
echo "Checking environment..."
echo "============================================"

# Install required packages if needed
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user --no-deps \
    torch==2.1.0 \
    torchvision==0.16.0 \
    'numpy>=1.21.0,<1.24.0' \
    tqdm \
    psutil

# Additional packages
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    pip install --user \
    matplotlib \
    scikit-learn \
    Pillow \
    opencv-python-headless

# Test environment
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ PyTorch: {e}')
try:
    import cv2
    print(f'✅ OpenCV: {cv2.__version__}')
except Exception as e:
    print(f'❌ OpenCV: {e}')
"

echo "============================================"
echo "Looking for trained models..."
echo "============================================"

# Find the latest model checkpoint
MODEL_DIR="$PROJECT_DIR/models_enhanced"
if [ -d "$MODEL_DIR" ]; then
    echo "Checking model directory: $MODEL_DIR"
    ls -la $MODEL_DIR/best_model.pth 2>/dev/null || echo "No best_model.pth found yet"

    # Find the best model
    BEST_MODEL=$(ls -t $MODEL_DIR/best_model.pth 2>/dev/null | head -1)
    if [ -z "$BEST_MODEL" ]; then
        # Try alternative location
        MODEL_DIR="$PROJECT_DIR/models_enhanced"
        BEST_MODEL=$(ls -t $MODEL_DIR/best_model.pth 2>/dev/null | head -1)
    fi

    if [ -z "$BEST_MODEL" ]; then
        echo "❌ No trained model found in $MODEL_DIR"
        echo "Please ensure training has completed and saved checkpoints"
        exit 1
    else
        echo "✅ Found model: $BEST_MODEL"
    fi
else
    echo "❌ Model directory not found: $MODEL_DIR"
    echo "Please check if training has been run"
    exit 1
fi

echo "============================================"
echo "Testing Enhanced AMSR2 Model..."
echo "============================================"

# Run the test script
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    --bind $PROJECT_DIR:$PROJECT_DIR \
    --bind /home/vdidur/temperature_sr_project/data:/home/vdidur/temperature_sr_project/data:ro \
    --env PYTHONPATH=$PROJECT_DIR:$PYTHONPATH \
    --workdir $PROJECT_DIR \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python test_enhanced_amsr2.py

# Check results
if [ $? -eq 0 ]; then
    echo "============================================"
    echo "✅ Testing completed successfully!"
    echo "============================================"

    # Show test results summary
    echo ""
    echo "Test results have been displayed in the console output above."
    echo "Look for the ENHANCED AMSR2 TEST SUMMARY section for average metrics."

    # Check if visualization results were saved
    if [ -d "./test_enhanced_results" ] && [ "$(ls -A ./test_enhanced_results 2>/dev/null)" ]; then
        echo ""
        echo "Saved files in test_enhanced_results/:"
        ls -la test_enhanced_results/
    fi
else
    echo "============================================"
    echo "❌ Testing failed!"
    echo "Check the error messages above for details"
    echo "============================================"
fi

# Clean up GPU memory
echo ""
echo "Cleaning up GPU memory..."
apptainer exec --nv \
    --bind $HOME/local-python:$HOME/.local \
    /home/shared/containers/tensorflow-25.02-py3.sif \
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('✅ GPU memory cleared')
else:
    print('No GPU available for cleanup')
"

echo ""
echo "============================================"
echo "Enhanced AMSR2 Testing Finished: $(date)"
echo "============================================"