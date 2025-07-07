#!/usr/bin/env python3
"""
AMSR2 Sequential Trainer - OPTIMIZED All-in-One Version
Fast 8x super-resolution training with in-memory caching

This file combines all optimizations in a single script.
No need for separate files!
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import glob
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import time
import json
import argparse
from pathlib import Path
import psutil
import sys
from tqdm import tqdm
import gc
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

# GPU-optimized thread settings
if torch.cuda.is_available():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()
else:
    torch.set_num_threads(min(8, os.cpu_count()))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('amsr2_optimized_training.log')
    ]
)
logger = logging.getLogger(__name__)


# ====== MEMORY MANAGEMENT ======
def aggressive_cleanup():
    """Aggressive memory cleanup for GPU and CPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for _ in range(3):
            torch.cuda.empty_cache()
            gc.collect()


def log_memory_usage(prefix=""):
    """Log current memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        logger.info(f"{prefix} GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    memory = psutil.virtual_memory()
    logger.info(
        f"{prefix} RAM: {memory.percent:.1f}% used ({memory.used / 1024 ** 3:.1f}GB / {memory.total / 1024 ** 3:.1f}GB)")


# ====== OPTIMIZED DATASET WITH CACHING ======
class OptimizedAMSR2Dataset(Dataset):
    """Optimized dataset that caches all data in memory"""

    def __init__(self, npz_path: str, preprocessor,
                 degradation_scale: int = 2,
                 augment: bool = True,
                 filter_orbit_type: Optional[str] = None,
                 max_swaths_in_memory: int = 300):

        self.preprocessor = preprocessor
        self.degradation_scale = degradation_scale
        self.augment = augment

        # Pre-load all data
        self.data_cache = []
        self._load_and_cache_data(npz_path, filter_orbit_type, max_swaths_in_memory)

    def _load_and_cache_data(self, npz_path: str, filter_orbit_type: Optional[str], max_swaths: int):
        """Load all data into memory once"""
        logger.info(f"📂 Loading and caching: {os.path.basename(npz_path)}")

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if 'swath_array' not in data:
                    logger.error(f"❌ Invalid file structure")
                    return

                swath_array = data['swath_array']
                total_swaths = len(swath_array)
                logger.info(f"   Total swaths in file: {total_swaths}")

                valid_count = 0

                for idx in range(min(total_swaths, max_swaths)):
                    if idx % 50 == 0 and idx > 0:
                        logger.debug(f"   Processed {idx}/{total_swaths} swaths...")

                    try:
                        swath_dict = swath_array[idx]
                        swath = swath_dict.item() if isinstance(swath_dict, np.ndarray) else swath_dict

                        if 'temperature' not in swath or 'metadata' not in swath:
                            continue

                        metadata = swath['metadata']

                        # Filter by orbit type
                        if filter_orbit_type and metadata.get('orbit_type', 'U') != filter_orbit_type:
                            continue

                        # Load and process temperature
                        temperature = swath['temperature'].astype(np.float32)
                        scale_factor = metadata.get('scale_factor', 1.0)
                        temperature = temperature * scale_factor

                        # Filter invalid values
                        temperature = np.where((temperature < 50) | (temperature > 350), np.nan, temperature)

                        # Quick validity check
                        valid_ratio = np.sum(~np.isnan(temperature)) / temperature.size
                        if valid_ratio < 0.5:
                            continue

                        # Make contiguous copy
                        temperature = np.ascontiguousarray(temperature)
                        temperature = self.preprocessor.crop_and_pad_to_target(temperature)
                        temperature = self.preprocessor.normalize_brightness_temperature(temperature)
                        temperature = np.ascontiguousarray(temperature)  # This ensures it's contiguous

                        # Cache preprocessed data
                        self.data_cache.append(temperature)
                        valid_count += 1

                        if valid_count >= max_swaths:
                            break

                    except Exception as e:
                        logger.debug(f"Swath {idx}: error {e}")
                        continue

                logger.info(f"✅ Cached {len(self.data_cache)} valid swaths")

        except Exception as e:
            logger.error(f"❌ Error loading file: {e}")

        gc.collect()

    def __len__(self):
        return len(self.data_cache)

    def __getitem__(self, idx):
        """Get item from cache - much faster!"""
        temperature = self.data_cache[idx].copy()

        # Augmentation
        if self.augment and np.random.rand() > 0.7:
            if np.random.rand() > 0.5:
                temperature = np.fliplr(temperature).copy()  # Add .copy()
            if np.random.rand() > 0.5:
                temperature = np.flipud(temperature).copy()  # Add .copy()

        # Fast downscaling
        high_res = temperature
        low_res = self._fast_downscale(high_res)

        # Make sure arrays are contiguous before converting to tensors
        high_res = np.ascontiguousarray(high_res)  # Add this line
        low_res = np.ascontiguousarray(low_res)  # Add this line

        # Convert to tensors
        high_res_tensor = torch.from_numpy(high_res).unsqueeze(0).float()
        low_res_tensor = torch.from_numpy(low_res).unsqueeze(0).float()

        return low_res_tensor, high_res_tensor

    def _fast_downscale(self, high_res: np.ndarray) -> np.ndarray:
        """Fast CPU-based downscaling"""
        h, w = high_res.shape
        new_h, new_w = h // self.degradation_scale, w // self.degradation_scale

        # Efficient numpy reshaping for downscaling
        low_res = high_res[:new_h * self.degradation_scale, :new_w * self.degradation_scale]
        low_res = low_res.reshape(new_h, self.degradation_scale,
                                  new_w, self.degradation_scale).mean(axis=(1, 3))

        # Add noise
        noise = np.random.randn(new_h, new_w).astype(np.float32) * 0.01
        low_res = low_res + noise

        return low_res.astype(np.float32)


# ====== PREPROCESSOR ======
class AMSR2DataPreprocessor:
    """Preprocessor for AMSR2 data"""

    def __init__(self, target_height: int = 2048, target_width: int = 208):
        self.target_height = target_height
        self.target_width = target_width
        logger.info(f"📏 Preprocessor configured for size: {target_height}x{target_width}")

    def crop_and_pad_to_target(self, temperature: np.ndarray) -> np.ndarray:
        """Crop or pad to target size"""
        h, w = temperature.shape

        # Crop if larger
        if h > self.target_height:
            start_h = (h - self.target_height) // 2
            temperature = temperature[start_h:start_h + self.target_height]

        if w > self.target_width:
            start_w = (w - self.target_width) // 2
            temperature = temperature[:, start_w:start_w + self.target_width]

        current_h, current_w = temperature.shape

        # Pad if smaller
        if current_h < self.target_height or current_w < self.target_width:
            pad_h = max(0, self.target_height - current_h)
            pad_w = max(0, self.target_width - current_w)

            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left

            temperature = np.pad(temperature,
                                 ((pad_top, pad_bottom), (pad_left, pad_right)),
                                 mode='reflect')

        return temperature

    def normalize_brightness_temperature(self, temperature: np.ndarray) -> np.ndarray:
        """Normalize brightness temperature"""
        valid_mask = ~np.isnan(temperature)
        if np.sum(valid_mask) > 0:
            mean_temp = np.mean(temperature[valid_mask])
            temperature = np.where(np.isnan(temperature), mean_temp, temperature)
        else:
            temperature = np.full_like(temperature, 250.0)

        temperature = np.clip(temperature, 50, 350)
        normalized = (temperature - 200) / 150

        return normalized.astype(np.float32)


# ====== MODEL ARCHITECTURE ======
class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class UNetResNetEncoder(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []
        x = F.relu(self.bn1(self.conv1(x)))
        features.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        return x, features


class UNetDecoder(nn.Module):
    def __init__(self, out_channels: int = 1):
        super().__init__()
        self.up4 = self._make_upconv_block(512, 256)
        self.up3 = self._make_upconv_block(256 + 256, 128)
        self.up2 = self._make_upconv_block(128 + 128, 64)
        self.up1 = self._make_upconv_block(64 + 64, 64)
        self.final_up = nn.ConvTranspose2d(64 + 64, 32, 2, 2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1)
        )

    def _make_upconv_block(self, in_channels: int, out_channels: int):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_features):
        # Up4
        x = self.up4(x)
        if x.shape[2] != skip_features[3].shape[2] or x.shape[3] != skip_features[3].shape[3]:
            diff_h = skip_features[3].shape[2] - x.shape[2]
            diff_w = skip_features[3].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[3]], dim=1)

        # Up3
        x = self.up3(x)
        if x.shape[2] != skip_features[2].shape[2] or x.shape[3] != skip_features[2].shape[3]:
            diff_h = skip_features[2].shape[2] - x.shape[2]
            diff_w = skip_features[2].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[2]], dim=1)

        # Up2
        x = self.up2(x)
        if x.shape[2] != skip_features[1].shape[2] or x.shape[3] != skip_features[1].shape[3]:
            diff_h = skip_features[1].shape[2] - x.shape[2]
            diff_w = skip_features[1].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[1]], dim=1)

        # Up1
        x = self.up1(x)
        if x.shape[2] != skip_features[0].shape[2] or x.shape[3] != skip_features[0].shape[3]:
            diff_h = skip_features[0].shape[2] - x.shape[2]
            diff_w = skip_features[0].shape[3] - x.shape[3]
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip_features[0]], dim=1)

        x = self.final_up(x)
        x = self.final_conv(x)
        return x


class UNetResNetSuperResolution(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.encoder = UNetResNetEncoder(in_channels)
        self.decoder = UNetDecoder(out_channels)

        # Progressive upsampling for 8x
        if scale_factor == 2:
            self.upsampling = nn.Sequential(
                nn.ConvTranspose2d(out_channels, 64, 4, 2, 1),  # 2x upsampling
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels, 1)
            )
        elif scale_factor > 1:
            upsampling_layers = []
            current_scale = 1

            while current_scale < scale_factor:
                if scale_factor // current_scale >= 4:
                    factor = 4
                elif scale_factor // current_scale >= 2:
                    factor = 2
                else:
                    factor = scale_factor // current_scale

                upsampling_layers.extend([
                    nn.ConvTranspose2d(out_channels if len(upsampling_layers) == 0 else 32, 32,
                                       factor, factor),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.ReLU(inplace=True)
                ])
                current_scale *= factor

            upsampling_layers.append(nn.Conv2d(32, out_channels, 1))
            self.upsampling = nn.Sequential(*upsampling_layers)
        else:
            self.upsampling = nn.Identity()

    def forward(self, x):
        encoded, skip_features = self.encoder(x)
        decoded = self.decoder(encoded, skip_features)
        output = self.upsampling(decoded)
        return output


# ====== LOSS FUNCTION ======
class AMSR2SpecificLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.15, gamma: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def compute_gradients(x):
            grad_x = x[:, :, :-1, :] - x[:, :, 1:, :]
            grad_y = x[:, :, :, :-1] - x[:, :, :, 1:]
            return grad_x, grad_y

        pred_grad_x, pred_grad_y = compute_gradients(pred)
        target_grad_x, target_grad_y = compute_gradients(target)

        loss_x = self.l1_loss(pred_grad_x, target_grad_x)
        loss_y = self.l1_loss(pred_grad_y, target_grad_y)

        return loss_x + loss_y

    def brightness_temperature_consistency(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = torch.mean(pred, dim=[2, 3])
        target_mean = torch.mean(target, dim=[2, 3])
        energy_loss = self.mse_loss(pred_mean, target_mean)

        pred_std = torch.std(pred, dim=[2, 3])
        target_std = torch.std(target, dim=[2, 3])
        distribution_loss = self.mse_loss(pred_std, target_std)

        range_penalty = torch.mean(torch.relu(torch.abs(pred) - 1.0))

        return energy_loss + 0.5 * distribution_loss + 0.1 * range_penalty

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        l1_loss = self.l1_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        phys_loss = self.brightness_temperature_consistency(pred, target)

        total_loss = (self.alpha * l1_loss +
                      self.beta * grad_loss +
                      self.gamma * phys_loss)

        return total_loss, {
            'l1_loss': l1_loss.item(),
            'gradient_loss': grad_loss.item(),
            'physical_loss': phys_loss.item(),
            'total_loss': total_loss.item()
        }


# ====== OPTIMIZED TRAINER ======
class OptimizedTrainer:
    """Optimized trainer with batched file loading"""

    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 1e-4, weight_decay: float = 1e-5,
                 use_amp: bool = True, gradient_accumulation_steps: int = 1):

        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.criterion = AMSR2SpecificLoss()
        self.best_loss = float('inf')

    def train_on_files(self, npz_files: List[str], preprocessor,
                       epochs: int = 10,
                       batch_size: int = 16,
                       num_workers: int = 4,
                       files_per_batch: int = 3,
                       max_swaths_per_file: int = 4000,
                       save_path: str = "best_model.pth"):
        """Optimized training with batched file loading"""

        logger.info(f"🚀 Starting optimized training:")
        logger.info(f"   Files: {len(npz_files)}")
        logger.info(f"   Epochs: {epochs}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Workers: {num_workers}")
        logger.info(f"   Files per batch: {files_per_batch}")

        total_batches = (len(npz_files) + files_per_batch - 1) // files_per_batch

        for epoch in range(epochs):
            logger.info(f"\n📈 Epoch {epoch + 1}/{epochs}")
            epoch_losses = []

            # Process files in batches
            for batch_idx in range(total_batches):
                start_idx = batch_idx * files_per_batch
                end_idx = min(start_idx + files_per_batch, len(npz_files))
                batch_files = npz_files[start_idx:end_idx]

                logger.info(f"\n📦 Batch {batch_idx + 1}/{total_batches} - Loading {len(batch_files)} files")

                # Load multiple files
                datasets = []
                for file_path in batch_files:
                    dataset = OptimizedAMSR2Dataset(
                        file_path,
                        preprocessor,
                        degradation_scale=2,
                        augment=True,
                        max_swaths_in_memory=max_swaths_per_file
                    )
                    if len(dataset) > 0:
                        datasets.append(dataset)

                if not datasets:
                    continue

                # Create combined dataloader
                combined_dataset = ConcatDataset(datasets)
                dataloader = DataLoader(
                    combined_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    prefetch_factor=2,
                    persistent_workers=True if num_workers > 0 else False,
                    drop_last=True
                )

                # Train on this batch
                self.model.train()
                batch_losses = []

                progress_bar = tqdm(dataloader, desc=f"Batch {batch_idx + 1}")
                for data_idx, (low_res, high_res) in enumerate(progress_bar):
                    low_res = low_res.to(self.device, non_blocking=True)
                    high_res = high_res.to(self.device, non_blocking=True)

                    # Mixed precision training
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        pred = self.model(low_res)
                        loss, loss_components = self.criterion(pred, high_res)
                        loss = loss / self.gradient_accumulation_steps

                    # Backward pass
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    # Update weights
                    if (data_idx + 1) % self.gradient_accumulation_steps == 0:
                        if self.use_amp:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                            self.optimizer.step()

                        self.optimizer.zero_grad()

                    # Track loss
                    current_loss = loss.item() * self.gradient_accumulation_steps
                    batch_losses.append(current_loss)
                    progress_bar.set_postfix({'loss': f'{current_loss:.4f}'})

                    # Periodic cleanup
                    if data_idx % 100 == 0:
                        torch.cuda.empty_cache()

                epoch_losses.extend(batch_losses)

                # Clean up after batch
                del dataloader, combined_dataset, datasets
                aggressive_cleanup()

            # Epoch statistics
            if epoch_losses:
                avg_epoch_loss = np.mean(epoch_losses)
                logger.info(f"📊 Epoch {epoch + 1} - Average loss: {avg_epoch_loss:.4f}")

                # Save best model
                if avg_epoch_loss < self.best_loss:
                    self.best_loss = avg_epoch_loss
                    checkpoint = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': self.best_loss,
                        'epoch': epoch,
                        'scale_factor': 2
                    }
                    torch.save(checkpoint, save_path)
                    logger.info(f"💾 Saved best model: loss={self.best_loss:.4f}")

                # Scheduler step
                self.scheduler.step(avg_epoch_loss)

        logger.info(f"\n🎉 Training completed! Best loss: {self.best_loss:.4f}")


# ====== UTILITY FUNCTIONS ======
def find_npz_files(directory: str, max_files: Optional[int] = None) -> List[str]:
    """Find NPZ files"""
    if not os.path.exists(directory):
        logger.error(f"❌ Directory does not exist: {directory}")
        return []

    pattern = os.path.join(directory, "*.npz")
    all_files = glob.glob(pattern)

    if not all_files:
        logger.error(f"❌ No NPZ files found")
        return []

    all_files.sort()

    if max_files is not None and max_files > 0:
        selected_files = all_files[:max_files]
    else:
        selected_files = all_files

    logger.info(f"📁 Found {len(selected_files)} NPZ files")

    # Check total size
    total_size_gb = sum(os.path.getsize(f) / 1024 ** 3 for f in selected_files)
    logger.info(f"📊 Total data size: {total_size_gb:.2f} GB")

    return selected_files


# ====== MAIN FUNCTION ======
def main():
    """Main function for optimized training"""

    parser = argparse.ArgumentParser(
        description='AMSR2 Optimized 8x Super-Resolution Training'
    )

    # Required
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Path to directory with NPZ files')

    # Data parameters
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to train on')
    parser.add_argument('--max-swaths-per-file', type=int, default=300,
                        help='Maximum swaths to load per file (default: 300)')
    parser.add_argument('--orbit-filter', choices=['A', 'D', 'U'],
                        help='Filter by orbit type')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--gradient-accumulation', type=int, default=2,
                        help='Gradient accumulation steps (default: 2)')

    # Optimization parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--files-per-batch', type=int, default=3,
                        help='Files to load per batch (default: 3)')

    # System parameters
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--target-height', type=int, default=2048,
                        help='Target height (default: 2048)')
    parser.add_argument('--target-width', type=int, default=208,
                        help='Target width (default: 208)')

    # Output
    parser.add_argument('--save-path', type=str, default='best_amsr2_8x_optimized.pth',
                        help='Path to save best model')

    args = parser.parse_args()

    print("🛰️  AMSR2 OPTIMIZED 8x SUPER-RESOLUTION TRAINER")
    print("=" * 60)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️  Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
        aggressive_cleanup()
    else:
        logger.warning("⚠️  No GPU available, using CPU")
        args.use_amp = False

    # Memory check
    memory_info = psutil.virtual_memory()
    logger.info(f"💾 RAM: {memory_info.available / 1024 ** 3:.1f} GB available")

    # Find files
    npz_files = find_npz_files(args.npz_dir, args.max_files)
    if not npz_files:
        logger.error("❌ No NPZ files found")
        sys.exit(1)

    # Create model
    logger.info("🧠 Creating 8x super-resolution model...")
    model = UNetResNetSuperResolution(
        in_channels=1,
        out_channels=1,
        scale_factor=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Parameters: {total_params:,}")
    logger.info(f"   Model size: {total_params * 4 / 1024 ** 2:.1f} MB")

    # Create preprocessor
    preprocessor = AMSR2DataPreprocessor(
        target_height=args.target_height,
        target_width=args.target_width
    )

    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation
    )

    # Training configuration
    logger.info(f"\n⚙️  Configuration:")
    logger.info(f"   Files: {len(npz_files)}")
    logger.info(f"   Epochs: {args.epochs}")
    logger.info(f"   Batch size: {args.batch_size}")
    logger.info(f"   Workers: {args.num_workers}")
    logger.info(f"   Files per batch: {args.files_per_batch}")
    logger.info(f"   Learning rate: {args.lr}")
    logger.info(f"   Mixed precision: {args.use_amp}")

    # Start training
    logger.info("\n🚀 Starting optimized training...")
    start_time = time.time()

    try:
        trainer.train_on_files(
            npz_files=npz_files,
            preprocessor=preprocessor,
            epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            files_per_batch=args.files_per_batch,
            max_swaths_per_file=args.max_swaths_per_file,
            save_path=args.save_path
        )

        training_time = time.time() - start_time
        logger.info(f"\n⏱️  Training time: {training_time / 3600:.2f} hours")
        logger.info(f"📁 Model saved to: {args.save_path}")

    except KeyboardInterrupt:
        logger.info("\n⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if device.type == 'cuda':
            aggressive_cleanup()
        logger.info("🛑 Program finished")


if __name__ == "__main__":
    main()