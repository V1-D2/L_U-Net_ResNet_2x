#!/usr/bin/env python3
"""
Cascaded 8x Super-Resolution for AMSR2 with Multiple Variants
Implements two approaches:
1. Bicubic 2x → Model 2x → Model 2x = 8x total
2. Model 2x → Model 2x → Model 2x = 8x total

Key features:
- Two different cascading strategies for 8x upscaling
- Comparison with 8x bicubic baseline
- Processes 5 samples from last NPZ file
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
        logging.FileHandler('cascaded_8x_inference.log')
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
                                     overlap_ratio: float = 0.75,
                                     stage_name: str = "Stage") -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Apply super-resolution using patch-based approach with weighted blending

        Args:
            image: Input image (normalized)
            patch_size: Size of patches for model
            overlap_ratio: Overlap ratio (0.75 = 75% overlap)
            stage_name: Name for logging

        Returns:
            sr_image: Super-resolution result
            stats: Temperature statistics
        """
        start_time = time.time()
        h, w = image.shape
        scale_factor = 2

        logger.info(f"\n{stage_name}: Processing image of size {h}×{w}")

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
        logger.info(f"{stage_name}: Created {len(positions)} patches")

        # Process patches with progress bar
        with tqdm(total=len(positions), desc=f"{stage_name} patches") as pbar:
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

        logger.info(f"{stage_name} completed in {stats['processing_time']:.2f}s")

        return sr_image, stats


def cascaded_super_resolution_8x(npz_dir: str, model_path: str, num_samples: int = 5,
                                 save_dir: str = "./cascaded_8x_results") -> List[Dict]:
    """
    Process samples using two variants of cascaded 8x super-resolution

    Variant 1: Bicubic 2x → Model 2x → Model 2x = 8x
    Variant 2: Model 2x → Model 2x → Model 2x = 8x

    Args:
        npz_dir: Directory containing NPZ files
        model_path: Path to trained model
        num_samples: Number of samples to process (default: 5)
        save_dir: Directory to save results

    Returns:
        List of results with both variants
    """
    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    variant1_dir = os.path.join(save_dir, 'variant1_bicubic_first')
    variant2_dir = os.path.join(save_dir, 'variant2_triple_model')

    for dir_path in [variant1_dir, variant2_dir]:
        os.makedirs(dir_path, exist_ok=True)
        for subdir in ['arrays', 'images', 'visualizations']:
            os.makedirs(os.path.join(dir_path, subdir), exist_ok=True)

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
    logger.info(f"Processing {num_samples} samples with cascaded 8x super-resolution")

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

                logger.info(f"\n{'=' * 80}")
                logger.info(f"Processing sample {processed_count + 1}/{num_samples}")
                logger.info(f"Original shape: {original_shape}")

                # === VARIANT 1: Bicubic 2x → Model 2x → Model 2x ===
                logger.info(f"\n--- Variant 1: Bicubic → Model → Model ---")

                # Stage 1: Bicubic 2x upscaling
                h, w = normalized.shape
                bicubic_2x_normalized = cv2.resize(normalized, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
                logger.info(f"V1 Stage 1 (Bicubic 2x): {normalized.shape} → {bicubic_2x_normalized.shape}")

                # Stage 2: First model application
                v1_stage2_normalized, v1_stats2 = patch_processor.patch_based_super_resolution(
                    bicubic_2x_normalized,
                    patch_size=(1024, 104),
                    overlap_ratio=0.75,
                    stage_name="V1 Stage 2 (Model 2x)"
                )
                logger.info(f"V1 Stage 2: {bicubic_2x_normalized.shape} → {v1_stage2_normalized.shape}")

                # Stage 3: Second model application
                v1_stage3_normalized, v1_stats3 = patch_processor.patch_based_super_resolution(
                    v1_stage2_normalized,
                    patch_size=(1024, 104),
                    overlap_ratio=0.75,
                    stage_name="V1 Stage 3 (Model 2x)"
                )

                # Denormalize to temperature
                v1_8x_temperature = v1_stage3_normalized * 150 + 200
                logger.info(f"V1 Final: {original_shape} → {v1_8x_temperature.shape} (8x)")

                # === VARIANT 2: Model 2x → Model 2x → Model 2x ===
                logger.info(f"\n--- Variant 2: Model → Model → Model ---")

                # Stage 1: First model application
                v2_stage1_normalized, v2_stats1 = patch_processor.patch_based_super_resolution(
                    normalized,
                    patch_size=(1024, 104),
                    overlap_ratio=0.75,
                    stage_name="V2 Stage 1 (Model 2x)"
                )
                logger.info(f"V2 Stage 1: {normalized.shape} → {v2_stage1_normalized.shape}")

                # Stage 2: Second model application
                v2_stage2_normalized, v2_stats2 = patch_processor.patch_based_super_resolution(
                    v2_stage1_normalized,
                    patch_size=(1024, 104),
                    overlap_ratio=0.75,
                    stage_name="V2 Stage 2 (Model 2x)"
                )
                logger.info(f"V2 Stage 2: {v2_stage1_normalized.shape} → {v2_stage2_normalized.shape}")

                # Stage 3: Third model application
                v2_stage3_normalized, v2_stats3 = patch_processor.patch_based_super_resolution(
                    v2_stage2_normalized,
                    patch_size=(1024, 104),
                    overlap_ratio=0.75,
                    stage_name="V2 Stage 3 (Model 2x)"
                )

                # Denormalize to temperature
                v2_8x_temperature = v2_stage3_normalized * 150 + 200
                logger.info(f"V2 Final: {original_shape} → {v2_8x_temperature.shape} (8x)")

                # Create 8x bicubic baseline from ORIGINAL
                h, w = original_shape
                bicubic_8x = cv2.resize(original_temp, (w * 8, h * 8), interpolation=cv2.INTER_CUBIC)

                # Store results
                result = {
                    'original': original_temp,
                    'variant1_8x': v1_8x_temperature,
                    'variant2_8x': v2_8x_temperature,
                    'bicubic_8x': bicubic_8x,
                    'variant1_stats': {
                        'stage2': v1_stats2,
                        'stage3': v1_stats3,
                        'total_time': v1_stats2['processing_time'] + v1_stats3['processing_time']
                    },
                    'variant2_stats': {
                        'stage1': v2_stats1,
                        'stage2': v2_stats2,
                        'stage3': v2_stats3,
                        'total_time': v2_stats1['processing_time'] + v2_stats2['processing_time'] + v2_stats3[
                            'processing_time']
                    },
                    'metadata': {
                        'original_shape': original_shape,
                        'final_shape': v1_8x_temperature.shape,
                        'swath_index': idx,
                        'scale_factor': metadata.get('scale_factor', 1.0)
                    }
                }

                results.append(result)
                processed_count += 1

                # Log summary
                logger.info(f"\nProcessing Summary:")
                logger.info(f"  Variant 1 total time: {result['variant1_stats']['total_time']:.2f}s")
                logger.info(f"  Variant 2 total time: {result['variant2_stats']['total_time']:.2f}s")

            except Exception as e:
                logger.warning(f"Error processing swath {idx}: {e}")
                continue

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Successfully processed {len(results)} samples with 8x SR")

    # Save results for both variants
    save_variant_results(results, variant1_dir, 'variant1')
    save_variant_results(results, variant2_dir, 'variant2')

    # Create comparison visualization
    create_8x_comparison(results, save_dir)

    return results


def save_variant_results(results: List[Dict], save_dir: str, variant: str):
    """Save results for a specific variant"""

    arrays_dir = os.path.join(save_dir, 'arrays')
    images_dir = os.path.join(save_dir, 'images')

    for i, result in enumerate(results):
        # Determine which data to save based on variant
        if variant == 'variant1':
            sr_8x = result['variant1_8x']
            stats = result['variant1_stats']
        else:
            sr_8x = result['variant2_8x']
            stats = result['variant2_stats']

        # Save arrays
        array_path = os.path.join(arrays_dir, f'sample_{i + 1:03d}.npz')
        np.savez_compressed(
            array_path,
            original=result['original'],
            sr_8x=sr_8x,
            bicubic_8x=result['bicubic_8x'],
            stats=stats,
            metadata=result['metadata']
        )

        # Save images
        save_temperature_image(result['original'],
                               os.path.join(images_dir, f'sample_{i + 1:03d}_original.png'))
        save_temperature_image(sr_8x,
                               os.path.join(images_dir, f'sample_{i + 1:03d}_sr_8x.png'))
        save_temperature_image(result['bicubic_8x'],
                               os.path.join(images_dir, f'sample_{i + 1:03d}_bicubic_8x.png'))

        # Save grayscale versions
        # Percentile normalization для grayscale
        p_low, p_high = 1, 99
        temp_min, temp_max = np.percentile(result['original'], [p_low, p_high])

        # Нормализация с percentile
        def normalize_with_percentile(data):
            data_clipped = np.clip(data, temp_min, temp_max)
            return (data_clipped - temp_min) / (temp_max - temp_min)

        orig_norm = normalize_with_percentile(result['original'])
        sr_8x_norm = normalize_with_percentile(sr_8x)
        bicubic_8x_norm = normalize_with_percentile(result['bicubic_8x'])

        # Convert to tensors
        orig_tensor = torch.from_numpy(orig_norm).unsqueeze(0).unsqueeze(0).float()
        sr_8x_tensor = torch.from_numpy(sr_8x_norm).unsqueeze(0).unsqueeze(0).float()
        bicubic_8x_tensor = torch.from_numpy(bicubic_8x_norm).unsqueeze(0).unsqueeze(0).float()

        orig_img = tensor2img([orig_tensor])
        sr_8x_img = tensor2img([sr_8x_tensor])
        bicubic_8x_img = tensor2img([bicubic_8x_tensor])

        imwrite(orig_img, os.path.join(images_dir, f'sample_{i + 1:03d}_original_gray.png'))
        imwrite(sr_8x_img, os.path.join(images_dir, f'sample_{i + 1:03d}_sr_8x_gray.png'))
        imwrite(bicubic_8x_img, os.path.join(images_dir, f'sample_{i + 1:03d}_bicubic_8x_gray.png'))


def save_temperature_image(temperature: np.ndarray, save_path: str, dpi: int = 100):
    """Save temperature array as image with exact pixel mapping"""
    import matplotlib.cm as cm
    from PIL import Image

    # Используем percentile normalization
    p_low, p_high = 1, 99
    temp_min, temp_max = np.percentile(temperature, [p_low, p_high])
    temperature_clipped = np.clip(temperature, temp_min, temp_max)
    temp_norm = (temperature_clipped - temp_min) / (temp_max - temp_min)

    # Apply turbo colormap
    turbo_cmap = cm.get_cmap('turbo')
    turbo_rgb = (turbo_cmap(temp_norm)[:, :, :3] * 255).astype(np.uint8)

    # Save as PNG
    Image.fromarray(turbo_rgb).save(save_path)


def create_8x_comparison(results: List[Dict], save_dir: str):
    """Create comparison visualization between variants"""

    n_samples = len(results)
    fig, axes = plt.subplots(4, n_samples, figsize=(4 * n_samples, 16))

    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    for i, result in enumerate(results):
        # Original
        axes[0, i].imshow(result['original'], cmap='turbo', aspect='auto')
        axes[0, i].set_title(f'Original {i + 1}\n{result["original"].shape}')
        axes[0, i].axis('off')

        # Variant 1 (Bicubic → Model → Model)
        axes[1, i].imshow(result['variant1_8x'], cmap='turbo', aspect='auto')
        axes[1, i].set_title(f'V1: Bicubic→Model→Model\nTime: {result["variant1_stats"]["total_time"]:.1f}s')
        axes[1, i].axis('off')

        # Variant 2 (Model → Model → Model)
        axes[2, i].imshow(result['variant2_8x'], cmap='turbo', aspect='auto')
        axes[2, i].set_title(f'V2: Model→Model→Model\nTime: {result["variant2_stats"]["total_time"]:.1f}s')
        axes[2, i].axis('off')

        # 8x Bicubic baseline
        axes[3, i].imshow(result['bicubic_8x'], cmap='turbo', aspect='auto')
        axes[3, i].set_title(f'8x Bicubic Baseline\n{result["bicubic_8x"].shape}')
        axes[3, i].axis('off')

    # Calculate average times
    avg_v1_time = np.mean([r['variant1_stats']['total_time'] for r in results])
    avg_v2_time = np.mean([r['variant2_stats']['total_time'] for r in results])

    title = f'8x Super-Resolution Comparison ({n_samples} samples)\n'
    title += f'Variant 1 avg: {avg_v1_time:.1f}s | Variant 2 avg: {avg_v2_time:.1f}s'

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'variants_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Create detailed comparison for first sample
    if len(results) > 0:
        create_detailed_comparison(results[0], save_dir)


def create_detailed_comparison(result: Dict, save_dir: str):
    """Create detailed comparison with zoomed regions"""

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Row 1: Full images
    images = [
        (result['original'], 'Original'),
        (result['variant1_8x'], 'V1: Bicubic→Model→Model'),
        (result['variant2_8x'], 'V2: Model→Model→Model'),
        (result['bicubic_8x'], '8x Bicubic')
    ]

    for i, (img, title) in enumerate(images):
        im = axes[0, i].imshow(img, cmap='turbo', aspect='auto')
        axes[0, i].set_title(f'{title}\n{img.shape}')
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)

    # Row 2: Zoomed regions
    if result['variant1_8x'].shape[0] >= 1024:
        h, w = result['variant1_8x'].shape
        center_y, center_x = h // 2, w // 2
        size = 512

        y1 = max(0, center_y - size)
        y2 = min(h, center_y + size)
        x1 = max(0, center_x - size)
        x2 = min(w, center_x + size)

        # Original zoomed (scaled to match)
        orig_h, orig_w = result['original'].shape
        orig_y1, orig_y2 = y1 // 8, y2 // 8
        orig_x1, orig_x2 = x1 // 8, x2 // 8
        orig_zoom = cv2.resize(result['original'][orig_y1:orig_y2, orig_x1:orig_x2],
                               (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)

        zoomed_images = [
            (orig_zoom, 'Original (upscaled)'),
            (result['variant1_8x'][y1:y2, x1:x2], 'V1 Zoomed'),
            (result['variant2_8x'][y1:y2, x1:x2], 'V2 Zoomed'),
            (result['bicubic_8x'][y1:y2, x1:x2], 'Bicubic Zoomed')
        ]

        for i, (img, title) in enumerate(zoomed_images):
            axes[1, i].imshow(img, cmap='turbo', aspect='auto')
            axes[1, i].set_title(title)
            axes[1, i].axis('off')
    else:
        for i in range(4):
            axes[1, i].axis('off')

    plt.suptitle('Detailed 8x Super-Resolution Comparison', fontsize=16)
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'detailed_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function for cascaded 8x inference"""
    import argparse

    parser = argparse.ArgumentParser(description='Cascaded 8x Super-Resolution with Multiple Variants')
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Directory containing NPZ files')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='Number of samples to process (default: 5)')
    parser.add_argument('--save-dir', type=str, default='./cascaded_8x_results',
                        help='Directory to save results')
    parser.add_argument('--overlap-ratio', type=float, default=0.75,
                        help='Overlap ratio for patches (default: 0.75)')

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("CASCADED 8X SUPER-RESOLUTION WITH MULTIPLE VARIANTS")
    logger.info("=" * 80)
    logger.info(f"NPZ directory: {args.npz_dir}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Number of samples: {args.num_samples}")
    logger.info(f"Save directory: {args.save_dir}")
    logger.info("=" * 80)

    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model not found: {args.model_path}")
        sys.exit(1)

    # Process samples
    try:
        results = cascaded_super_resolution_8x(
            npz_dir=args.npz_dir,
            model_path=args.model_path,
            num_samples=args.num_samples,
            save_dir=args.save_dir
        )

        logger.info("\n" + "=" * 80)
        logger.info("8X CASCADED PROCESSING COMPLETED SUCCESSFULLY!")
        logger.info(f"Processed {len(results)} samples with 8x upscaling")
        logger.info(f"Results saved to: {args.save_dir}")
        logger.info("=" * 80)

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