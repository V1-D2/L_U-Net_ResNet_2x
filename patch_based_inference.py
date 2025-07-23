#!/usr/bin/env python3
"""
Patch-based Inference for AMSR2 Super-Resolution
Processes large images using overlapping patches with advanced blending

Key features:
- Adaptive patch processing for any input size
- 75% overlap with Gaussian weighted blending
- Memory efficient batch processing
- Temperature statistics and quality metrics
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import glob
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
import json
from tqdm import tqdm
import cv2
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Import model components
from enhanced_amsr2_model import EnhancedUNetWithAttention, AMSR2DataPreprocessor, MetricsCalculator
from gpu_sequential_amsr2_optimized import OptimizedAMSR2Dataset, aggressive_cleanup
from basicsr.utils import tensor2img, imwrite

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('patch_inference.log')
    ]
)
logger = logging.getLogger(__name__)


class PatchBasedSuperResolution:
    """Patch-based super-resolution processor with advanced blending"""

    def __init__(self, model: nn.Module, preprocessor: AMSR2DataPreprocessor,
                 device: torch.device = torch.device('cuda')):
        self.model = model.to(device)
        self.model.eval()
        self.preprocessor = preprocessor
        self.device = device
        self.metrics_calc = MetricsCalculator()

    def create_gaussian_weight_map(self, shape: Tuple[int, int], sigma_ratio: float = 0.3) -> np.ndarray:
        """Create 2D Gaussian weight map for smooth blending"""
        h, w = shape

        # Create coordinate grids
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2

        # Calculate Gaussian weights
        sigma_y = h * sigma_ratio
        sigma_x = w * sigma_ratio

        gaussian = np.exp(-((y - center_y) ** 2 / (2 * sigma_y ** 2) +
                            (x - center_x) ** 2 / (2 * sigma_x ** 2)))

        # Normalize to [0, 1]
        gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())

        return gaussian.astype(np.float32)

    def calculate_patch_positions(self, image_shape: Tuple[int, int],
                                  patch_size: Tuple[int, int],
                                  overlap_ratio: float) -> List[Tuple[int, int, int, int]]:
        """Calculate optimal patch positions with adaptive overlap"""
        h, w = image_shape
        ph, pw = patch_size

        # Calculate stride based on overlap
        stride_h = int(ph * (1 - overlap_ratio))
        stride_w = int(pw * (1 - overlap_ratio))

        # Ensure minimum stride
        stride_h = max(1, stride_h)
        stride_w = max(1, stride_w)

        positions = []

        # Calculate positions
        y = 0
        while y < h:
            x = 0
            while x < w:
                # Calculate patch boundaries
                y_end = min(y + ph, h)
                x_end = min(x + pw, w)

                # Adjust start position for edge patches to maintain size
                y_start = max(0, y_end - ph) if y_end == h else y
                x_start = max(0, x_end - pw) if x_end == w else x

                positions.append((y_start, y_end, x_start, x_end))

                # Move to next position
                if x_end >= w:
                    break
                x += stride_w

            if y_end >= h:
                break
            y += stride_h

        logger.info(f"Created {len(positions)} patches for image size {h}×{w}")
        return positions

    def process_patch(self, patch: np.ndarray) -> np.ndarray:
        """Process single patch through model"""
        # Ensure patch is correct size by padding if necessary
        h, w = patch.shape
        target_h, target_w = 1024, 104

        if h < target_h or w < target_w:
            # Pad patch to target size
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            pad_top = pad_h // 2
            pad_left = pad_w // 2

            patch_padded = np.pad(patch,
                                  ((pad_top, pad_h - pad_top), (pad_left, pad_w - pad_left)),
                                  mode='reflect')
        else:
            patch_padded = patch
            pad_top = pad_left = 0

        # Convert to tensor
        patch_tensor = torch.from_numpy(patch_padded).unsqueeze(0).unsqueeze(0).float().to(self.device)

        # Run super-resolution
        with torch.no_grad():
            sr_tensor = self.model(patch_tensor)
            sr_tensor = torch.clamp(sr_tensor, -1.5, 1.5)  # Clamp as in model

        # Convert back to numpy
        sr_patch = sr_tensor.cpu().numpy()[0, 0]

        # Remove padding from output if we padded input
        if pad_top > 0 or pad_left > 0:
            out_h = h * 2  # Scale factor is 2
            out_w = w * 2
            pad_top_out = pad_top * 2
            pad_left_out = pad_left * 2
            sr_patch = sr_patch[pad_top_out:pad_top_out + out_h,
                       pad_left_out:pad_left_out + out_w]

        return sr_patch

    def patch_based_super_resolution(self, image: np.ndarray,
                                     patch_size: Tuple[int, int] = (1024, 104),
                                     overlap_ratio: float = 0.75) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply super-resolution using patch-based approach with weighted blending

        Args:
            image: Input image (normalized, ~2036×220 but can vary)
            patch_size: Size of patches for model
            overlap_ratio: Overlap ratio (0.75 = 75% overlap)

        Returns:
            sr_image: Super-resolution result
            stats: Temperature statistics
        """
        start_time = time.time()
        h, w = image.shape
        scale_factor = 2

        logger.info(f"Processing image of size {h}×{w} with {overlap_ratio * 100:.0f}% overlap")

        # Initialize output arrays
        output_h, output_w = h * scale_factor, w * scale_factor
        sr_accumulated = np.zeros((output_h, output_w), dtype=np.float64)
        weight_accumulated = np.zeros((output_h, output_w), dtype=np.float64)

        # Create Gaussian weight map for blending
        weight_map = self.create_gaussian_weight_map(
            (patch_size[0] * scale_factor, patch_size[1] * scale_factor)
        )

        # Calculate patch positions
        positions = self.calculate_patch_positions((h, w), patch_size, overlap_ratio)

        # Process patches with progress bar
        with tqdm(total=len(positions), desc="Processing patches") as pbar:
            for i, (y_start, y_end, x_start, x_end) in enumerate(positions):
                # Extract patch
                patch = image[y_start:y_end, x_start:x_end]

                # Process patch
                sr_patch = self.process_patch(patch)

                # Calculate output position
                out_y_start = y_start * scale_factor
                out_y_end = y_end * scale_factor
                out_x_start = x_start * scale_factor
                out_x_end = x_end * scale_factor

                # Get weight map for this patch size
                patch_h = out_y_end - out_y_start
                patch_w = out_x_end - out_x_start

                if patch_h != weight_map.shape[0] or patch_w != weight_map.shape[1]:
                    # Create custom weight map for edge patches
                    patch_weight = self.create_gaussian_weight_map((patch_h, patch_w))
                else:
                    patch_weight = weight_map[:patch_h, :patch_w]

                # Accumulate weighted result
                sr_accumulated[out_y_start:out_y_end, out_x_start:out_x_end] += sr_patch * patch_weight
                weight_accumulated[out_y_start:out_y_end, out_x_start:out_x_end] += patch_weight

                # Update progress
                pbar.update(1)
                pbar.set_postfix({'patches': f'{i + 1}/{len(positions)}'})

                # Periodic memory cleanup
                if i % 50 == 0:
                    torch.cuda.empty_cache()

        # Normalize by accumulated weights
        mask = weight_accumulated > 0
        sr_image = np.zeros_like(sr_accumulated)
        sr_image[mask] = sr_accumulated[mask] / weight_accumulated[mask]

        # Denormalize to get temperature in Kelvin
        sr_temperature = sr_image * 150 + 200

        # Calculate statistics
        stats = {
            'min_temp': float(np.min(sr_temperature)),
            'max_temp': float(np.max(sr_temperature)),
            'avg_temp': float(np.mean(sr_temperature)),
            'processing_time': time.time() - start_time,
            'num_patches': len(positions)
        }

        logger.info(f"Completed in {stats['processing_time']:.2f}s using {stats['num_patches']} patches")
        logger.info(f"Temperature range: [{stats['min_temp']:.1f}, {stats['max_temp']:.1f}] K")

        return sr_image, stats


def process_last_npz_file(npz_dir: str, model_path: str, num_samples: int = 20,
                          save_dir: str = "./patch_inference_results") -> List[Dict]:
    """
    Process samples from the last NPZ file using patch-based inference

    Args:
        npz_dir: Directory containing NPZ files
        model_path: Path to trained model
        num_samples: Number of samples to process
        save_dir: Directory to save results

    Returns:
        List of results with original, super-resolution images and stats
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = EnhancedUNetWithAttention(in_channels=1, out_channels=1, scale_factor=2)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Model best PSNR: {checkpoint.get('best_psnr', 'N/A'):.2f} dB")
    else:
        model.load_state_dict(checkpoint)

    # Create preprocessor and patch processor
    preprocessor = AMSR2DataPreprocessor(target_height=2048, target_width=208)
    patch_processor = PatchBasedSuperResolution(model, preprocessor, device)

    # Find NPZ files
    npz_files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))
    if not npz_files:
        raise ValueError(f"No NPZ files found in {npz_dir}")

    # Use last file
    last_file = npz_files[-1]
    logger.info(f"\nProcessing last NPZ file: {os.path.basename(last_file)}")

    # Load samples
    results = []
    processed_count = 0

    with np.load(last_file, allow_pickle=True) as data:
        swath_array = data['swath_array']
        total_swaths = len(swath_array)
        logger.info(f"Total swaths in file: {total_swaths}")

        # Process from the end of file
        for idx in range(total_swaths - 1, max(0, total_swaths - 100), -1):
            if processed_count >= num_samples:
                break

            try:
                swath = swath_array[idx].item() if hasattr(swath_array[idx], 'item') else swath_array[idx]

                if 'temperature' not in swath:
                    continue

                temperature = swath['temperature'].astype(np.float32)
                metadata = swath.get('metadata', {})
                scale_factor = metadata.get('scale_factor', 1.0)
                temperature = temperature * scale_factor

                # Filter invalid values
                temperature = np.where((temperature < 50) | (temperature > 350), np.nan, temperature)
                valid_ratio = np.sum(~np.isnan(temperature)) / temperature.size

                if valid_ratio < 0.5:
                    continue

                # Fill NaN values
                valid_mask = ~np.isnan(temperature)
                if np.sum(valid_mask) > 0:
                    mean_temp = np.mean(temperature[valid_mask])
                    temperature = np.where(np.isnan(temperature), mean_temp, temperature)

                # Store original shape and data
                original_shape = temperature.shape
                original_temp = temperature.copy()

                # Normalize (don't crop/pad - work with original size)
                normalized = (temperature - 200) / 150

                logger.info(f"\nProcessing sample {processed_count + 1}/{num_samples}")
                logger.info(f"Original shape: {original_shape}")

                # Apply patch-based super-resolution
                sr_normalized, stats = patch_processor.patch_based_super_resolution(
                    normalized,
                    patch_size=(1024, 104),
                    overlap_ratio=0.75
                )

                # Denormalize
                sr_temperature = sr_normalized * 150 + 200

                # Create low-res version for comparison (simple downscale)
                h, w = original_shape
                low_res = cv2.resize(original_temp, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

                result = {
                    'original': original_temp,
                    'super_resolution': sr_temperature,
                    'low_resolution': low_res,
                    'temperature_stats': stats,
                    'metadata': {
                        'original_shape': original_shape,
                        'sr_shape': sr_temperature.shape,
                        'swath_index': idx,
                        'scale_factor': metadata.get('scale_factor', 1.0)
                    }
                }

                results.append(result)
                processed_count += 1

                logger.info(f"  Shape: {original_shape} → {sr_temperature.shape}")
                logger.info(f"  Temperature range: [{stats['min_temp']:.1f}, {stats['max_temp']:.1f}] K")
                logger.info(f"  Processing time: {stats['processing_time']:.2f}s")


            except Exception as e:
                logger.warning(f"Error processing swath {idx}: {e}")
                continue

    logger.info(f"\nSuccessfully processed {len(results)} samples")

    # Save results
    save_results(results, save_dir)

    return results


def save_results(results: List[Dict], save_dir: str):
    """Save all results including arrays, visualizations, and statistics"""

    # Create subdirectories
    arrays_dir = os.path.join(save_dir, 'arrays')
    images_dir = os.path.join(save_dir, 'images')
    visualizations_dir = os.path.join(save_dir, 'visualizations')

    for dir_path in [arrays_dir, images_dir, visualizations_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Save individual results
    all_stats = []

    for i, result in enumerate(results):
        # Save arrays
        array_path = os.path.join(arrays_dir, f'sample_{i + 1:03d}.npz')
        np.savez_compressed(
            array_path,
            original=result['original'],
            super_resolution=result['super_resolution'],
            low_resolution=result['low_resolution'],
            temperature_stats=result['temperature_stats'],
            metadata=result['metadata']
        )

        # Save individual images
        save_temperature_image(result['original'],
                               os.path.join(images_dir, f'sample_{i + 1:03d}_original.png'))
        save_temperature_image(result['super_resolution'],
                               os.path.join(images_dir, f'sample_{i + 1:03d}_sr.png'))
        save_temperature_image(result['low_resolution'],
                               os.path.join(images_dir, f'sample_{i + 1:03d}_lr.png'))

        # Add bicubic baseline from ORIGINAL
        h, w = result['original'].shape
        bicubic = cv2.resize(result['original'], (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        save_temperature_image(bicubic,
                               os.path.join(images_dir, f'sample_{i + 1:03d}_bicubic.png'))

        # Save grayscale versions using BasicSR
        # Convert temperature to normalized tensors for BasicSR
        sr_tensor_basicsr = torch.from_numpy((result['super_resolution'] - 200) / 150).unsqueeze(0).unsqueeze(0).float()
        lr_tensor_basicsr = torch.from_numpy((result['low_resolution'] - 200) / 150).unsqueeze(0).unsqueeze(0).float()
        orig_tensor_basicsr = torch.from_numpy((result['original'] - 200) / 150).unsqueeze(0).unsqueeze(0).float()
        bicubic_tensor_basicsr = torch.from_numpy((bicubic - 200) / 150).unsqueeze(0).unsqueeze(0).float()

        # Convert to grayscale images
        sr_img_basicsr = tensor2img([sr_tensor_basicsr])
        lr_img_basicsr = tensor2img([lr_tensor_basicsr])
        orig_img_basicsr = tensor2img([orig_tensor_basicsr])
        bicubic_img_basicsr = tensor2img([bicubic_tensor_basicsr])

        # Save grayscale versions
        imwrite(sr_img_basicsr, os.path.join(images_dir, f'sample_{i + 1:03d}_sr_gray.png'))
        imwrite(lr_img_basicsr, os.path.join(images_dir, f'sample_{i + 1:03d}_lr_gray.png'))
        imwrite(orig_img_basicsr, os.path.join(images_dir, f'sample_{i + 1:03d}_original_gray.png'))
        imwrite(bicubic_img_basicsr, os.path.join(images_dir, f'sample_{i + 1:03d}_bicubic_gray.png'))

        # Create comparison visualization
        create_comparison_plot(result, i + 1, visualizations_dir)

        # Collect statistics
        all_stats.append({
            'sample_id': i + 1,
            'swath_index': result['metadata']['swath_index'],
            'original_shape': result['metadata']['original_shape'],
            'sr_shape': result['metadata']['sr_shape'],
            'temperature_stats': result['temperature_stats'],
        })

    # Save summary statistics
    summary_path = os.path.join(save_dir, 'summary_statistics.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'num_samples': len(results),
            'average_processing_time': np.mean([s['temperature_stats']['processing_time'] for s in all_stats]),
            'individual_stats': all_stats
        }, f, indent=2)

    # Create summary visualization
    create_summary_visualization(results, save_dir)

    logger.info(f"\nResults saved to: {save_dir}")
    logger.info(f"  - Arrays: {arrays_dir}")
    logger.info(f"  - Images: {images_dir}")
    logger.info(f"  - Visualizations: {visualizations_dir}")
    logger.info(f"  - Summary: {summary_path}")


def save_temperature_image(temperature: np.ndarray, save_path: str, dpi: int = 100):
    """Save temperature array as image with exact pixel mapping"""
    plt.imsave(save_path, temperature, cmap='turbo', origin='upper')
    logger.debug(f"Saved: {save_path} ({temperature.shape[0]}×{temperature.shape[1]} pixels)")


def create_comparison_plot(result: Dict, sample_id: int, save_dir: str):
    """Create detailed comparison visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original
    im1 = axes[0, 0].imshow(result['original'], cmap='turbo', aspect='auto')
    axes[0, 0].set_title(f'Original\n{result["original"].shape}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # Low resolution
    # Change to 2x4 grid to include bicubic
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Original
    im1 = axes[0, 0].imshow(result['original'], cmap='turbo', aspect='auto')
    axes[0, 0].set_title(f'Original\n{result["original"].shape}')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # Low resolution
    im2 = axes[0, 1].imshow(result['low_resolution'], cmap='turbo', aspect='auto')
    axes[0, 1].set_title(f'Low Resolution\n{result["low_resolution"].shape}')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # Bicubic baseline from ORIGINAL
    h, w = result['original'].shape
    bicubic = cv2.resize(result['original'], (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    im3 = axes[0, 2].imshow(bicubic, cmap='turbo', aspect='auto')
    axes[0, 2].set_title(f'Bicubic Baseline\n{bicubic.shape}')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # Super resolution
    im4 = axes[0, 3].imshow(result['super_resolution'], cmap='turbo', aspect='auto')
    axes[0, 3].set_title(f'Super Resolution\n{result["super_resolution"].shape}')
    axes[0, 3].axis('off')
    plt.colorbar(im4, ax=axes[0, 3], fraction=0.046)

    # Super resolution
    im3 = axes[0, 2].imshow(result['super_resolution'], cmap='turbo', aspect='auto')
    axes[0, 2].set_title(f'Super Resolution\n{result["super_resolution"].shape}')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # Difference map (if possible)
    if result['original'].shape[0] * 2 == result['super_resolution'].shape[0]:
        # Upscale original for comparison
        h, w = result['original'].shape
        original_upscaled = cv2.resize(result['original'], (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        diff = np.abs(result['super_resolution'] - original_upscaled)

        im4 = axes[1, 0].imshow(diff, cmap='hot', aspect='auto')
        axes[1, 0].set_title(f'Absolute Difference\nMax: {np.max(diff):.1f} K')
        axes[1, 0].axis('off')
        plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)
    else:
        axes[1, 0].text(0.5, 0.5, 'Size mismatch\nfor difference',
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].axis('off')

    # Statistics
    stats_text = f"Temperature Statistics:\n"
    stats_text += f"Min: {result['temperature_stats']['min_temp']:.1f} K\n"
    stats_text += f"Max: {result['temperature_stats']['max_temp']:.1f} K\n"
    stats_text += f"Avg: {result['temperature_stats']['avg_temp']:.1f} K\n"
    stats_text += f"\nProcessing:\n"
    stats_text += f"Time: {result['temperature_stats']['processing_time']:.2f}s\n"
    stats_text += f"Patches: {result['temperature_stats']['num_patches']}\n"


    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[1, 1].axis('off')

    # Zoomed region
    if result['super_resolution'].shape[0] >= 256:
        # Find interesting region
        sr = result['super_resolution']
        h, w = sr.shape
        center_y, center_x = h // 2, w // 2
        size = 128

        y1 = max(0, center_y - size)
        y2 = min(h, center_y + size)
        x1 = max(0, center_x - size)
        x2 = min(w, center_x + size)

        zoom = sr[y1:y2, x1:x2]
        im5 = axes[1, 2].imshow(zoom, cmap='turbo', aspect='auto')
        axes[1, 2].set_title(f'Zoomed Region\n[{y1}:{y2}, {x1}:{x2}]')
        axes[1, 2].axis('off')
        plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)
    else:
        axes[1, 2].axis('off')

    plt.suptitle(f'Sample {sample_id} - Patch-based Super-Resolution Results', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'sample_{sample_id:03d}_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_visualization(results: List[Dict], save_dir: str):
    """Create summary visualization of all results"""
    n_samples = min(len(results), 6)  # Show up to 6 samples

    fig, axes = plt.subplots(3, n_samples, figsize=(4 * n_samples, 12))
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_samples):
        result = results[i]

        # Original
        axes[0, i].imshow(result['original'], cmap='turbo', aspect='auto')
        axes[0, i].set_title(f'Original {i + 1}')
        axes[0, i].axis('off')

        # Low resolution
        axes[1, i].imshow(result['low_resolution'], cmap='turbo', aspect='auto')
        axes[1, i].set_title(f'Low Res {i + 1}')
        axes[1, i].axis('off')

        # Super resolution
        axes[2, i].imshow(result['super_resolution'], cmap='turbo', aspect='auto')
        title = f'SR {i + 1}'
        axes[2, i].set_title(title)
        axes[2, i].axis('off')

    # Calculate averages
    avg_time = np.mean([r['temperature_stats']['processing_time'] for r in results])

    title = f'Patch-based Super-Resolution Summary ({len(results)} samples)\n'
    title += f'Avg Time: {avg_time:.2f}s'

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'summary_visualization.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Created summary visualization: {save_path}")


def main():
    """Main function for patch-based inference"""
    import argparse

    parser = argparse.ArgumentParser(description='Patch-based Inference for AMSR2 Super-Resolution')
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--num-samples', type=int, default=20,
                        help='Number of samples to process (default: 20)')
    parser.add_argument('--save-dir', type=str, default='./patch_inference_results',
                        help='Directory to save results')
    parser.add_argument('--overlap-ratio', type=float, default=0.75,
                        help='Overlap ratio for patches (default: 0.75)')

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("PATCH-BASED INFERENCE FOR AMSR2 SUPER-RESOLUTION")
    logger.info("=" * 60)
    logger.info(f"NPZ directory: {args.npz_dir}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Overlap ratio: {args.overlap_ratio * 100:.0f}%")
    logger.info(f"Save directory: {args.save_dir}")
    logger.info("=" * 60)

    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    # Process samples
    try:
        results = process_last_npz_file(
            npz_dir=args.npz_dir,
            model_path=args.model_path,
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )

        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING COMPLETED SUCCESSFULLY!")
        logger.info(f"Processed {len(results)} samples")
        logger.info(f"Results saved to: {args.save_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            aggressive_cleanup()


if __name__ == "__main__":
    main()