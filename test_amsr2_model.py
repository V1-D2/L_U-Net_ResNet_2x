#!/usr/bin/env python3
"""
Test script for AMSR2 8x Super-Resolution Model
Simplified version that imports model components
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import logging

# Import model and preprocessor from training script
from gpu_sequential_amsr2_optimized import (
    UNetResNetSuperResolution,
    AMSR2DataPreprocessor
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_array_as_image(data_array, save_path, dpi=100):
    """
    Сохраняет numpy массив как PNG изображение 1:1 пиксель к элементу
    """
    # Получаем размеры массива
    height, width = data_array.shape

    # Создаем фигуру с точным размером
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Отображаем массив с точным соответствием пикселей
    ax.imshow(data_array, cmap="turbo", origin='upper', interpolation='nearest',
              extent=[0, width, height, 0])

    # Убираем все лишнее
    ax.axis('off')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)

    # Сохраняем с точным размером
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

    logger.info(f"Сохранено изображение: {save_path} ({height}x{width} пикселей)")


def create_low_res(high_res: np.ndarray, scale: int = 8) -> np.ndarray:
    """Create low resolution version by downsampling"""
    h, w = high_res.shape
    new_h, new_w = h // scale, w // scale

    # Efficient numpy reshaping for downscaling
    low_res = high_res[:new_h * scale, :new_w * scale]
    low_res = low_res.reshape(new_h, scale, new_w, scale).mean(axis=(1, 3))

    return low_res.astype(np.float32)


def test_model(model_path: str, test_file: str, output_dir: str, device: torch.device):
    """Main testing function"""

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    logger.info(f"Loading model from: {model_path}")
    model = UNetResNetSuperResolution(in_channels=1, out_channels=1, scale_factor=8)

    # Load checkpoint - use weights_only=False for compatibility
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    logger.info("Model loaded successfully!")

    # Create preprocessor
    preprocessor = AMSR2DataPreprocessor(target_height=2048, target_width=208)

    # Load test data
    logger.info(f"Loading test data from: {test_file}")
    with np.load(test_file, allow_pickle=True) as data:
        logger.info(f"Available keys: {list(data.keys())}")

        # Extract temperature data
        if 'temperature' in data:
            temperature = data['temperature'].astype(np.float32)
            # Get metadata for scale factor
            if 'metadata' in data:
                metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
                scale_factor = metadata.get('scale_factor', 1.0)
                temperature = temperature * scale_factor
            else:
                # If no metadata, assume temperature needs scaling from typical AMSR2 values
                if np.max(temperature) > 1000:  # Likely unscaled data
                    temperature = temperature * 0.01  # Common AMSR2 scale factor
                metadata = {'scale_factor': 0.01}
        elif 'swath_array' in data:
            swath = data['swath_array'][0]
            if hasattr(swath, 'item'):
                swath = swath.item()
            temperature = swath['temperature'].astype(np.float32)
            metadata = swath.get('metadata', {})
            scale_factor = metadata.get('scale_factor', 1.0)
            temperature = temperature * scale_factor
        else:
            raise ValueError("No temperature data found in file")

        # Filter invalid temperature values
        temperature = np.where((temperature < 50) | (temperature > 350), np.nan, temperature)

        # Fill NaN values with mean
        valid_mask = ~np.isnan(temperature)
        if np.sum(valid_mask) > 0:
            mean_temp = np.mean(temperature[valid_mask])
            temperature = np.where(np.isnan(temperature), mean_temp, temperature)
        else:
            temperature = np.full_like(temperature, 250.0)

    logger.info(f"Original temperature shape: {temperature.shape}")
    logger.info(f"Temperature range: [{np.nanmin(temperature):.1f}, {np.nanmax(temperature):.1f}] K")

    # Store original for comparison
    original_temperature = temperature.copy()

    # Preprocess
    temperature = preprocessor.crop_and_pad_to_target(temperature)
    temperature_normalized = preprocessor.normalize_brightness_temperature(temperature)

    # Create low-res version
    low_res = create_low_res(temperature_normalized, scale=8)
    logger.info(f"Low-res shape: {low_res.shape}")

    # Prepare for model
    low_res_tensor = torch.from_numpy(low_res).unsqueeze(0).unsqueeze(0).float().to(device)

    # Run inference
    logger.info("Running super-resolution...")
    with torch.no_grad():
        sr_tensor = model(low_res_tensor)
        sr_tensor = torch.clamp(sr_tensor, -1, 1)  # Clamp to normalized range

    # Convert back to numpy
    sr_normalized = sr_tensor.cpu().numpy()[0, 0]

    # Denormalize to get back temperature in Kelvin
    sr_temperature = sr_normalized * 150 + 200  # Reverse normalization
    low_res_temperature = low_res * 150 + 200
    temperature_kelvin = temperature * 150 + 200  # The cropped/padded version

    # Calculate metrics
    mse = np.mean((sr_temperature - temperature_kelvin) ** 2)
    mae = np.mean(np.abs(sr_temperature - temperature_kelvin))

    logger.info(f"Results:")
    logger.info(f"  MSE: {mse:.4f}")
    logger.info(f"  MAE: {mae:.4f} K")
    logger.info(f"  SR temperature range: [{np.min(sr_temperature):.1f}, {np.max(sr_temperature):.1f}] K")

    # ====== Save arrays as .npz ======
    arrays_file = os.path.join(output_dir, 'sr_arrays.npz')
    np.savez(arrays_file,
             original=temperature_kelvin,
             low_res=low_res_temperature,
             super_res=sr_temperature,
             mse=mse,
             mae=mae,
             metadata=metadata)
    logger.info(f"Arrays saved to: {arrays_file}")

    # ====== Save 1:1 pixel images ======
    # Low resolution image
    save_array_as_image(low_res_temperature,
                        os.path.join(output_dir, 'low_res.png'))

    # Super resolution image
    save_array_as_image(sr_temperature,
                        os.path.join(output_dir, 'super_res.png'))

    # Original image
    save_array_as_image(temperature_kelvin,
                        os.path.join(output_dir, 'original.png'))

    # Difference map
    diff = np.abs(sr_temperature - temperature_kelvin)
    save_array_as_image(diff,
                        os.path.join(output_dir, 'difference.png'))

    # ====== Create comparison visualization ======
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Low resolution
    im1 = axes[0, 0].imshow(low_res_temperature, cmap='turbo', aspect='auto')
    axes[0, 0].set_title(f'Low Resolution ({low_res_temperature.shape[0]}×{low_res_temperature.shape[1]})')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # Super resolution
    im2 = axes[0, 1].imshow(sr_temperature, cmap='turbo', aspect='auto')
    axes[0, 1].set_title(f'Super Resolution ({sr_temperature.shape[0]}×{sr_temperature.shape[1]})')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # Original
    im3 = axes[1, 0].imshow(temperature_kelvin, cmap='turbo', aspect='auto')
    axes[1, 0].set_title(f'Original ({temperature_kelvin.shape[0]}×{temperature_kelvin.shape[1]})')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # Difference
    im4 = axes[1, 1].imshow(diff, cmap='hot', aspect='auto')
    axes[1, 1].set_title(f'Absolute Difference\nMax: {np.max(diff):.2f} K, Mean: {mae:.2f} K')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    plt.suptitle(f'AMSR2 8x Super-Resolution Results\nMSE: {mse:.4f}, MAE: {mae:.2f} K')
    plt.tight_layout()

    comparison_file = os.path.join(output_dir, 'comparison_grid.png')
    plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
    logger.info(f"Comparison grid saved to: {comparison_file}")
    plt.close()

    # ====== Create zoomed comparison ======
    # Find interesting region (with high variance)
    h, w = temperature_kelvin.shape
    window_size = min(128, h // 4, w // 4)

    # Calculate variance in sliding windows
    max_var = 0
    best_y, best_x = h // 2 - window_size // 2, w // 2 - window_size // 2

    for y in range(0, h - window_size, window_size // 2):
        for x in range(0, w - window_size, window_size // 2):
            window = temperature_kelvin[y:y + window_size, x:x + window_size]
            var = np.var(window)
            if var > max_var:
                max_var = var
                best_y, best_x = y, x

    # Extract regions
    lr_zoom = low_res_temperature[best_y // 8:(best_y + window_size) // 8,
              best_x // 8:(best_x + window_size) // 8]
    sr_zoom = sr_temperature[best_y:best_y + window_size,
              best_x:best_x + window_size]
    orig_zoom = temperature_kelvin[best_y:best_y + window_size,
                best_x:best_x + window_size]

    # Create zoomed comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(lr_zoom, cmap='turbo', aspect='auto')
    axes[0].set_title('Low Resolution (Zoomed)')
    axes[0].axis('off')

    im2 = axes[1].imshow(sr_zoom, cmap='turbo', aspect='auto')
    axes[1].set_title('Super Resolution (Zoomed)')
    axes[1].axis('off')

    im3 = axes[2].imshow(orig_zoom, cmap='turbo', aspect='auto')
    axes[2].set_title('Original (Zoomed)')
    axes[2].axis('off')

    plt.tight_layout()
    zoom_file = os.path.join(output_dir, 'comparison_zoom.png')
    plt.savefig(zoom_file, dpi=150, bbox_inches='tight')
    logger.info(f"Zoomed comparison saved to: {zoom_file}")
    plt.close()

    logger.info("\n=== Summary of outputs ===")
    logger.info(f"1. Arrays (.npz): {arrays_file}")
    logger.info(f"2. Individual 1:1 pixel images: low_res.png, super_res.png, original.png, difference.png")
    logger.info(f"3. Comparison grid: {comparison_file}")
    logger.info(f"4. Zoomed comparison: {zoom_file}")


def main():
    parser = argparse.ArgumentParser(description='Test AMSR2 8x Super-Resolution Model')
    parser.add_argument('--model-path', type=str, default='models/best_amsr2_8x.pth',
                        help='Path to trained model')
    parser.add_argument('--test-file', type=str,
                        default='/home/vdidur/temperature_sr_project/test/single_amsr2_image.npz',
                        help='Path to test NPZ file')
    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Output directory for results')

    args = parser.parse_args()

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()

    # Run testing
    try:
        test_model(args.model_path, args.test_file, args.output_dir, device)
        logger.info("\nTesting completed successfully!")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()