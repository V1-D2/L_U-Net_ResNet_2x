#!/usr/bin/env python3
"""
Test Enhanced AMSR2 Model - Take 5 samples from end of last NPZ file
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
from pathlib import Path
import logging

# Import from enhanced model
from enhanced_amsr2_model import EnhancedUNetWithAttention, AMSR2DataPreprocessor, MetricsCalculator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def load_last_npz_samples(npz_dir: str, num_samples: int = 5):
    """Load samples from the END of the LAST NPZ file"""

    # Find all NPZ files and get the last one
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {npz_dir}")

    last_file = npz_files[-1]
    logger.info(f"Loading from last file: {os.path.basename(last_file)}")

    samples = []

    with np.load(last_file, allow_pickle=True) as data:
        swath_array = data['swath_array']
        total_swaths = len(swath_array)

        logger.info(f"Total swaths in file: {total_swaths}")

        # Take samples from the END
        start_idx = max(0, total_swaths - num_samples * 10)  # Buffer to find valid samples

        for idx in range(total_swaths - 1, start_idx - 1, -1):  # Go backwards
            if len(samples) >= num_samples:
                break

            try:
                swath_dict = swath_array[idx]
                swath = swath_dict.item() if isinstance(swath_dict, np.ndarray) else swath_dict

                if 'temperature' not in swath or 'metadata' not in swath:
                    continue

                temperature = swath['temperature'].astype(np.float32)
                metadata = swath['metadata']
                scale_factor = metadata.get('scale_factor', 1.0)
                temperature = temperature * scale_factor

                # Filter invalid values
                temperature = np.where((temperature < 50) | (temperature > 350), np.nan, temperature)
                valid_ratio = np.sum(~np.isnan(temperature)) / temperature.size

                if valid_ratio < 0.5:
                    continue

                samples.append({
                    'temperature': temperature,
                    'metadata': metadata,
                    'swath_idx': idx
                })

                logger.info(f"Loaded sample {len(samples)} from swath {idx}")

            except Exception as e:
                logger.debug(f"Error loading swath {idx}: {e}")
                continue

    logger.info(f"Successfully loaded {len(samples)} samples from end of file")
    return samples


def test_enhanced_model():
    """Test the enhanced model"""

    # Paths
    model_path = "/home/vdidur/L_U-Net_ResNet_2x/models_enhanced/best_model.pth"
    npz_dir = "/home/vdidur/temperature_sr_project/data"
    output_dir = "./test_enhanced_results"

    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading enhanced model...")
    model = EnhancedUNetWithAttention(in_channels=1, out_channels=1, scale_factor=2)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Best PSNR: {checkpoint.get('best_psnr', 'unknown'):.2f} dB")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Load preprocessor
    preprocessor = AMSR2DataPreprocessor(target_height=2048, target_width=208)
    metrics_calc = MetricsCalculator()

    # Load test samples
    logger.info("Loading test samples from end of last NPZ file...")
    samples = load_last_npz_samples(npz_dir, num_samples=5)

    if len(samples) == 0:
        logger.error("No valid samples found!")
        return

    # Process each sample
    results = []

    for i, sample in enumerate(samples):
        logger.info(f"\nProcessing sample {i + 1}/{len(samples)} (swath {sample['swath_idx']})")

        # Preprocess
        temperature = sample['temperature']
        temperature = preprocessor.crop_and_pad_to_target(temperature)
        temp_normalized = preprocessor.normalize_brightness_temperature(temperature)

        # Create low-res version (2x downscale)
        h, w = temp_normalized.shape
        low_res = temp_normalized[::2, ::2]  # Simple 2x downscale

        # Add noise
        noise = np.random.randn(*low_res.shape).astype(np.float32) * 0.01
        low_res = low_res + noise

        # Convert to tensors
        low_res_tensor = torch.from_numpy(low_res).unsqueeze(0).unsqueeze(0).float().to(device)
        high_res_tensor = torch.from_numpy(temp_normalized).unsqueeze(0).unsqueeze(0).float().to(device)

        # Run super-resolution
        with torch.no_grad():
            pred_tensor = model(low_res_tensor)
            pred_tensor = torch.clamp(pred_tensor, -1, 1)

        # Calculate metrics
        psnr = metrics_calc.calculate_psnr_batch(pred_tensor, high_res_tensor)
        ssim = metrics_calc.calculate_ssim_batch(pred_tensor, high_res_tensor)

        # Convert back to numpy and denormalize
        low_res_temp = low_res * 150 + 200
        pred_temp = pred_tensor.cpu().numpy()[0, 0] * 150 + 200
        high_res_temp = temp_normalized * 150 + 200

        results.append({
            'low_res': low_res_temp,
            'prediction': pred_temp,
            'ground_truth': high_res_temp,
            'psnr': psnr,
            'ssim': ssim,
            'swath_idx': sample['swath_idx']
        })

        logger.info(f"  PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

    # Create visualization
    logger.info("\nCreating visualizations...")

    # Create combined grid (keep the existing one)
    fig, axes = plt.subplots(3, len(results), figsize=(4 * len(results), 12))
    if len(results) == 1:
        axes = axes.reshape(-1, 1)

    for i, result in enumerate(results):
        # Low resolution
        im1 = axes[0, i].imshow(result['low_res'], cmap='turbo', aspect='auto')
        axes[0, i].set_title(f'Low Res {i + 1}\n({result["low_res"].shape[0]}×{result["low_res"].shape[1]})')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i], fraction=0.046)

        # Prediction
        im2 = axes[1, i].imshow(result['prediction'], cmap='turbo', aspect='auto')
        axes[1, i].set_title(f'Enhanced {i + 1}\nPSNR: {result["psnr"]:.1f} dB')
        axes[1, i].axis('off')
        plt.colorbar(im2, ax=axes[1, i], fraction=0.046)

        # Ground truth
        im3 = axes[2, i].imshow(result['ground_truth'], cmap='turbo', aspect='auto')
        axes[2, i].set_title(f'Ground Truth {i + 1}\nSSIM: {result["ssim"]:.3f}')
        axes[2, i].axis('off')
        plt.colorbar(im3, ax=axes[2, i], fraction=0.046)

    # Calculate average metrics
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])

    plt.suptitle(f'Enhanced AMSR2 Super-Resolution Test Results\n'
                 f'Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}',
                 fontsize=16)
    plt.tight_layout()

    # Save combined results
    comparison_path = os.path.join(output_dir, 'enhanced_test_results.png')
    plt.savefig(comparison_path, dpi=200, bbox_inches='tight')
    plt.close()

    # CREATE INDIVIDUAL 1:1 PIXEL IMAGES
    logger.info("Creating individual 1:1 pixel images...")

    def save_array_as_1to1_image(data_array, save_path, cmap='turbo'):
        """Save array as image with exact 1:1 pixel mapping"""
        # Use matplotlib's imsave for exact pixel mapping
        from matplotlib import cm
        import matplotlib.pyplot as plt

        # Normalize data to [0, 1] for colormap
        data_min, data_max = np.min(data_array), np.max(data_array)
        if data_max > data_min:
            normalized = (data_array - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data_array)

        # Apply colormap and save
        plt.imsave(save_path, normalized, cmap=cmap, origin='upper')

        height, width = data_array.shape
        logger.info(f"Saved 1:1 pixel image: {os.path.basename(save_path)} ({height}×{width} pixels)")

    # Save individual images for each test
    individual_dir = os.path.join(output_dir, 'individual_images')
    os.makedirs(individual_dir, exist_ok=True)

    for i, result in enumerate(results):
        # Create subdirectory for each test
        test_dir = os.path.join(individual_dir, f'test_{i + 1}_swath_{result["swath_idx"]}')
        os.makedirs(test_dir, exist_ok=True)

        # Save 1:1 pixel images
        save_array_as_1to1_image(
            result['low_res'],
            os.path.join(test_dir, f'low_res_{result["low_res"].shape[0]}x{result["low_res"].shape[1]}.png')
        )

        save_array_as_1to1_image(
            result['prediction'],
            os.path.join(test_dir, f'enhanced_{result["prediction"].shape[0]}x{result["prediction"].shape[1]}.png')
        )

        save_array_as_1to1_image(
            result['ground_truth'],
            os.path.join(test_dir,
                         f'ground_truth_{result["ground_truth"].shape[0]}x{result["ground_truth"].shape[1]}.png')
        )

        # Save difference map
        diff = np.abs(result['prediction'] - result['ground_truth'])
        save_array_as_1to1_image(
            diff,
            os.path.join(test_dir, f'difference_{diff.shape[0]}x{diff.shape[1]}.png'),
            cmap='hot'
        )

        # Save metadata text file
        with open(os.path.join(test_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Test {i + 1} Results\n")
            f.write(f"Swath Index: {result['swath_idx']}\n")
            f.write(f"PSNR: {result['psnr']:.3f} dB\n")
            f.write(f"SSIM: {result['ssim']:.6f}\n")
            f.write(f"Low Res Shape: {result['low_res'].shape}\n")
            f.write(f"Enhanced Shape: {result['prediction'].shape}\n")
            f.write(f"Ground Truth Shape: {result['ground_truth'].shape}\n")
            f.write(f"Temperature Range - Low: [{np.min(result['low_res']):.1f}, {np.max(result['low_res']):.1f}] K\n")
            f.write(
                f"Temperature Range - Enhanced: [{np.min(result['prediction']):.1f}, {np.max(result['prediction']):.1f}] K\n")
            f.write(
                f"Temperature Range - Ground Truth: [{np.min(result['ground_truth']):.1f}, {np.max(result['ground_truth']):.1f}] K\n")

        logger.info(f"Saved individual images for test {i + 1} in: {test_dir}")

    logger.info(f"\n=== ENHANCED MODEL TEST RESULTS ===")
    logger.info(f"Average PSNR: {avg_psnr:.2f} dB")
    logger.info(f"Average SSIM: {avg_ssim:.4f}")
    logger.info(f"Combined visualization: {comparison_path}")
    logger.info(f"Individual 1:1 pixel images: {individual_dir}")
    logger.info(f"Arrays: {arrays_path}")
    logger.info(f"\nEach test has its own folder with:")
    logger.info(f"  - low_res_HxW.png (exact pixel mapping)")
    logger.info(f"  - enhanced_HxW.png (exact pixel mapping)")
    logger.info(f"  - ground_truth_HxW.png (exact pixel mapping)")
    logger.info(f"  - difference_HxW.png (error map)")
    logger.info(f"  - metrics.txt (detailed info)")


if __name__ == "__main__":
    test_enhanced_model()