#!/usr/bin/env python3
"""
Test script for Enhanced AMSR2 Super-Resolution Model with Spatial Attention
Tests pretrained model on AMSR2 temperature data
"""

import os
import sys
import torch
import numpy as np
import glob
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Import model components
from enhanced_amsr2_model import EnhancedUNetWithAttention, AMSR2DataPreprocessor, MetricsCalculator
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim

# Flag to control file saving
SAVE_FILES = False  # Set to True to save images and NPZ files

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_enhanced_model(model_path: str, device: torch.device) -> EnhancedUNetWithAttention:
    """Load pretrained Enhanced AMSR2 Model"""

    # Create model
    model = EnhancedUNetWithAttention(
        in_channels=1,
        out_channels=1,
        scale_factor=2
    )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Determine checkpoint format and load weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_psnr' in checkpoint:
            logger.info(f"Model best PSNR: {checkpoint['best_psnr']:.2f} dB")
        if 'best_ssim' in checkpoint:
            logger.info(f"Model best SSIM: {checkpoint['best_ssim']:.4f}")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    logger.info(f"✓ Enhanced AMSR2 model loaded from {model_path}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✓ Model parameters: {total_params:,}")

    return model


def test_single_sample(model: EnhancedUNetWithAttention,
                       temperature: np.ndarray,
                       preprocessor: AMSR2DataPreprocessor,
                       device: torch.device) -> Dict:
    """Test model on single temperature sample"""

    # Preprocess temperature data
    temperature = preprocessor.crop_and_pad_to_target(temperature)

    # Save original min/max for denormalization
    temp_min = np.min(temperature)
    temp_max = np.max(temperature)

    # Normalize temperature
    temperature_norm = preprocessor.normalize_brightness_temperature(temperature)

    # Create LR version (same as training)
    h, w = temperature_norm.shape
    degradation_scale = 2
    new_h, new_w = h // degradation_scale, w // degradation_scale

    # Efficient numpy reshaping for downscaling (same as training)
    low_res = temperature_norm[:new_h * degradation_scale, :new_w * degradation_scale]
    low_res = low_res.reshape(new_h, degradation_scale,
                              new_w, degradation_scale).mean(axis=(1, 3))

    # Add noise (same as training)
    noise = np.random.randn(new_h, new_w).astype(np.float32) * 0.01
    low_res = low_res + noise

    # Convert to tensors
    low_res_tensor = torch.from_numpy(low_res).unsqueeze(0).unsqueeze(0).float().to(device)
    high_res_tensor = torch.from_numpy(temperature_norm).unsqueeze(0).unsqueeze(0).float().to(device)

    # Run model inference
    with torch.no_grad():
        start_time = time.time()
        pred_tensor = model(low_res_tensor)
        pred_tensor = torch.clamp(pred_tensor, -1.5, 1.5)  # Same clamp as training
        inference_time = time.time() - start_time

    # Calculate metrics using MetricsCalculator from training
    metrics_calc = MetricsCalculator()
    psnr = metrics_calc.calculate_psnr_batch(pred_tensor, high_res_tensor)
    ssim = metrics_calc.calculate_ssim_batch(pred_tensor, high_res_tensor)

    # Convert back to numpy
    low_res_np = low_res
    high_res_np = temperature_norm
    pred_np = pred_tensor.cpu().numpy()[0, 0]

    # Denormalize to get temperature values
    low_res_temp = low_res_np * 150 + 200
    high_res_temp = high_res_np * 150 + 200
    pred_temp = pred_np * 150 + 200

    # Calculate temperature errors
    temp_error_mean = np.mean(np.abs(pred_temp - high_res_temp))
    temp_error_max = np.max(np.abs(pred_temp - high_res_temp))

    return {
        'low_res_temp': low_res_temp,
        'high_res_temp': high_res_temp,
        'pred_temp': pred_temp,
        'low_res_norm': low_res_np,
        'high_res_norm': high_res_np,
        'pred_norm': pred_np,
        'psnr': psnr,
        'ssim': ssim,
        'temp_error_mean': temp_error_mean,
        'temp_error_max': temp_error_max,
        'inference_time': inference_time,
        'temp_min': temp_min,
        'temp_max': temp_max
    }


def create_visualization(results: List[Dict], save_dir: str, sample_idx: int):
    """Create visualization for test results"""

    if not SAVE_FILES:
        return

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    result = results[sample_idx]

    # Row 1: Temperature values
    vmin = min(result['low_res_temp'].min(), result['high_res_temp'].min(), result['pred_temp'].min())
    vmax = max(result['low_res_temp'].max(), result['high_res_temp'].max(), result['pred_temp'].max())

    im1 = axes[0, 0].imshow(result['low_res_temp'], cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 0].set_title(f'Low Resolution\n{result["low_res_temp"].shape}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    im2 = axes[0, 1].imshow(result['high_res_temp'], cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 1].set_title(f'High Resolution (GT)\n{result["high_res_temp"].shape}')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    im3 = axes[0, 2].imshow(result['pred_temp'], cmap='turbo', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0, 2].set_title(f'Enhanced SR\n{result["pred_temp"].shape}')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # Row 2: Error map and metrics
    error_map = np.abs(result['pred_temp'] - result['high_res_temp'])
    im4 = axes[1, 0].imshow(error_map, cmap='hot', aspect='auto')
    axes[1, 0].set_title('Absolute Error (K)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    # Hide middle subplot
    axes[1, 1].axis('off')

    # Metrics text
    metrics_text = f"PSNR: {result['psnr']:.2f} dB\n"
    metrics_text += f"SSIM: {result['ssim']:.4f}\n"
    metrics_text += f"Mean Temp Error: {result['temp_error_mean']:.2f} K\n"
    metrics_text += f"Max Temp Error: {result['temp_error_max']:.2f} K\n"
    metrics_text += f"Inference Time: {result['inference_time'] * 1000:.1f} ms"

    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=14,
                    verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
    axes[1, 2].axis('off')

    plt.suptitle(f'Enhanced AMSR2 SR Test - Sample {sample_idx + 1}', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'test_sample_{sample_idx + 1:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_enhanced_amsr2_model(npz_dir: str, model_path: str, num_samples: int = 500,
                              save_dir: str = "./test_enhanced_results") -> Dict:
    """Test Enhanced AMSR2 model on multiple samples"""

    # Create output directory only if saving files
    if SAVE_FILES:
        os.makedirs(save_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    logger.info(f"\nLoading Enhanced AMSR2 model from: {model_path}")
    model = load_enhanced_model(model_path, device)

    # Create preprocessor
    preprocessor = AMSR2DataPreprocessor(
        target_height=2048,
        target_width=208
    )

    # Find NPZ files
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {npz_dir}")

    # Use last file (same as training validation)
    last_file = npz_files[-1]
    logger.info(f"\nUsing last NPZ file: {os.path.basename(last_file)}")

    # Load and test samples
    results = []

    with np.load(last_file, allow_pickle=True) as data:
        # Check data format
        if 'swath_array' in data:
            swath_array = data['swath_array']
        elif 'swaths' in data:
            swath_array = data['swaths']
        else:
            # Single temperature array
            temperature = data['temperature'].astype(np.float32)
            metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
            swath_array = [{'temperature': temperature, 'metadata': metadata}]

        total_swaths = len(swath_array)
        logger.info(f"Total swaths in file: {total_swaths}")
        logger.info(f"Testing {num_samples} samples from the end of file")

        # Process from the end of file (same as validation)
        tested_count = 0
        for idx in range(total_swaths - 1, max(0, total_swaths - 1000), -1):
            if tested_count >= num_samples:
                break

            try:
                swath = swath_array[idx].item() if hasattr(swath_array[idx], 'item') else swath_array[idx]

                if 'temperature' not in swath:
                    continue

                temperature = swath['temperature'].astype(np.float32)
                metadata = swath.get('metadata', {})
                scale_factor = metadata.get('scale_factor', 1.0)
                temperature = temperature * scale_factor

                # Filter invalid values (same as training)
                temperature = np.where((temperature < 50) | (temperature > 350), np.nan, temperature)
                valid_ratio = np.sum(~np.isnan(temperature)) / temperature.size

                if valid_ratio < 0.5:
                    continue

                # Fill NaN values
                valid_mask = ~np.isnan(temperature)
                if np.sum(valid_mask) > 0:
                    mean_temp = np.mean(temperature[valid_mask])
                    temperature = np.where(np.isnan(temperature), mean_temp, temperature)
                else:
                    continue

                # Test sample
                if tested_count % 100 == 0:
                    logger.info(f"\nTesting sample {tested_count + 1}/{num_samples} (swath {idx})")

                result = test_single_sample(model, temperature, preprocessor, device)

                # Add metadata
                result['swath_index'] = idx
                result['metadata'] = metadata

                results.append(result)
                tested_count += 1

                # Log metrics periodically
                if tested_count % 100 == 0:
                    logger.info(f"  PSNR: {result['psnr']:.2f} dB")
                    logger.info(f"  SSIM: {result['ssim']:.4f}")
                    logger.info(f"  Mean temp error: {result['temp_error_mean']:.2f} K")
                    logger.info(f"  Max temp error: {result['temp_error_max']:.2f} K")
                    logger.info(f"  Inference time: {result['inference_time'] * 1000:.1f} ms")

            except Exception as e:
                logger.warning(f"Error processing swath {idx}: {e}")
                continue

    # Calculate average metrics
    avg_metrics = {
        'psnr': np.mean([r['psnr'] for r in results]),
        'ssim': np.mean([r['ssim'] for r in results]),
        'temp_error_mean': np.mean([r['temp_error_mean'] for r in results]),
        'temp_error_max': np.mean([r['temp_error_max'] for r in results]),
        'inference_time': np.mean([r['inference_time'] for r in results])
    }

    # Calculate std for metrics
    std_metrics = {
        'psnr_std': np.std([r['psnr'] for r in results]),
        'ssim_std': np.std([r['ssim'] for r in results]),
        'temp_error_mean_std': np.std([r['temp_error_mean'] for r in results]),
        'temp_error_max_std': np.std([r['temp_error_max'] for r in results])
    }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ENHANCED AMSR2 TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {os.path.basename(model_path)}")
    logger.info(f"Tested samples: {len(results)}")
    logger.info(f"Average PSNR: {avg_metrics['psnr']:.2f} ± {std_metrics['psnr_std']:.2f} dB")
    logger.info(f"Average SSIM: {avg_metrics['ssim']:.4f} ± {std_metrics['ssim_std']:.4f}")
    logger.info(
        f"Average mean temp error: {avg_metrics['temp_error_mean']:.2f} ± {std_metrics['temp_error_mean_std']:.2f} K")
    logger.info(
        f"Average max temp error: {avg_metrics['temp_error_max']:.2f} ± {std_metrics['temp_error_max_std']:.2f} K")
    logger.info(f"Average inference time: {avg_metrics['inference_time'] * 1000:.1f} ms")
    logger.info("=" * 60)

    # Save results if enabled
    if SAVE_FILES:
        # Create visualizations for first 3 samples
        for i in range(min(3, len(results))):
            create_visualization(results, save_dir, i)

        # Save NPZ with all results
        save_path = os.path.join(save_dir, 'test_results.npz')
        np.savez(save_path,
                 results=results,
                 avg_metrics=avg_metrics,
                 std_metrics=std_metrics,
                 model_path=model_path)
        logger.info(f"\nResults saved to: {save_dir}")

    return {
        'results': results,
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics
    }


def main():
    """Main function"""

    # Configuration - UPDATE THESE PATHS
    NPZ_DIR = "/home/vdidur/temperature_sr_project/data"
    MODEL_PATH = "./models_enhanced/best_model.pth"  # Update with actual path
    NUM_SAMPLES = 500
    SAVE_DIR = "./test_enhanced_results"

    logger.info("Enhanced AMSR2 Super-Resolution Model Testing")
    logger.info(f"Save files: {SAVE_FILES}")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        logger.info("Looking for available models...")

        # Check multiple possible locations
        possible_dirs = [
            "./models_enhanced_2nd_generation",
            "./models_enhanced",
            "./models",
            "./experiments"
        ]

        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        if file.startswith("best_model") and file.endswith(".pth"):
                            logger.info(f"Found: {os.path.join(root, file)}")
        sys.exit(1)

    # Run tests
    try:
        test_results = test_enhanced_amsr2_model(
            npz_dir=NPZ_DIR,
            model_path=MODEL_PATH,
            num_samples=NUM_SAMPLES,
            save_dir=SAVE_DIR
        )

        logger.info("\nTesting completed successfully!")

    except Exception as e:
        logger.error(f"Testing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()