#!/usr/bin/env python3
"""
Enhanced AMSR2 Sequential Trainer with Spatial Attention
Includes PSNR/SSIM tracking and improved monitoring

Key improvements:
1. Spatial Attention Module (CBAM-style)
2. Real-time PSNR/SSIM calculation
3. Rich progress bars with metrics
4. Optimized hyperparameters
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
from collections import defaultdict
import cv2
from datetime import datetime
from gpu_sequential_amsr2_optimized import OptimizedAMSR2Dataset, AMSR2DataPreprocessor, aggressive_cleanup
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim

warnings.filterwarnings('ignore', category=UserWarning)

# GPU-optimized thread settings
if torch.cuda.is_available():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.cuda.empty_cache()
else:
    torch.set_num_threads(min(8, os.cpu_count()))


# Enhanced logging with color support
class ColoredFormatter(logging.Formatter):
    """Colored log formatter"""

    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    green = "\x1b[32;21m"
    blue = "\x1b[34;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey,
        logging.INFO: green,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red
    }

    def format(self, record):
        log_color = self.FORMATS.get(record.levelno, self.grey)
        record.msg = f"{log_color}{record.msg}{self.reset}"
        return super().format(record)


# Setup enhanced logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Console handler with colors
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))

# File handler without colors
file_handler = logging.FileHandler('amsr2_enhanced_training.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ====== SPATIAL ATTENTION MODULE ======
class SpatialAttention(nn.Module):
    """Spatial attention module - focuses on WHERE is important"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and convolve
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(x_cat)
        attention = self.sigmoid(attention)

        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention module - focuses on WHAT is important"""

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ====== ENHANCED RESNET BLOCK WITH ATTENTION ======
class AttentionResNetBlock(nn.Module):
    """ResNet block with integrated CBAM attention"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, use_attention: bool = True):
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

        # Add attention module
        self.use_attention = use_attention
        if use_attention:
            self.cbam = CBAM(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        if self.use_attention:
            out = self.cbam(out)

        return out


# ====== PSNR/SSIM CALCULATION MODULE ======
class MetricsCalculator:
    """Calculate PSNR and SSIM metrics"""

    @staticmethod
    def calculate_psnr_torch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        """Calculate PSNR using PyTorch tensors"""
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(float('inf'))
        return 20 * torch.log10(max_val / torch.sqrt(mse))

    @staticmethod
    def calculate_ssim_torch(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Calculate SSIM using torchmetrics if available, else simplified version"""
        try:
            from torchmetrics.functional import structural_similarity_index_measure
            # Convert from [-1, 1] to [0, 1]
            pred_01 = (pred + 1.0) / 2.0
            target_01 = (target + 1.0) / 2.0
            return structural_similarity_index_measure(pred_01, target_01, data_range=1.0)
        except ImportError:
            # Fallback to fixed implementation
            # First convert from [-1, 1] to [0, 1]
            pred = (pred + 1.0) / 2.0
            target = (target + 1.0) / 2.0

            # Constants for SSIM (now correct for [0,1] range)
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            # Create Gaussian window
            def gaussian_window(size, sigma=1.5):
                coords = torch.arange(size, dtype=torch.float32, device=pred.device) - size // 2
                g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
                g = g / g.sum()
                window = g.view(1, 1, size, 1) * g.view(1, 1, 1, size)
                # Normalize the window
                window = window / window.sum()
                return window

            window = gaussian_window(window_size)
            channel = pred.shape[1]

            # Apply window using grouped convolution
            mu1 = F.conv2d(pred, window.expand(channel, 1, -1, -1),
                           padding=window_size // 2, groups=channel)
            mu2 = F.conv2d(target, window.expand(channel, 1, -1, -1),
                           padding=window_size // 2, groups=channel)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.conv2d(pred * pred, window.expand(channel, 1, -1, -1),
                                 padding=window_size // 2, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(target * target, window.expand(channel, 1, -1, -1),
                                 padding=window_size // 2, groups=channel) - mu2_sq
            sigma12 = F.conv2d(pred * target, window.expand(channel, 1, -1, -1),
                               padding=window_size // 2, groups=channel) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                       ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

            # Return mean SSIM value
            return torch.clamp(ssim_map.mean(), 0.0, 1.0)

    @staticmethod
    def denormalize_for_metrics(tensor: torch.Tensor) -> torch.Tensor:
        """Convert from normalized [-1, 1] to [0, 1] range for metrics"""
        return (tensor + 1.0) / 2.0


'''class MetricsCalculator:
    """Calculate PSNR and SSIM metrics using standard implementations"""
    
    @staticmethod
    def calculate_psnr_torch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
        """Calculate PSNR using standard implementation"""
        # Convert from [-1, 1] to [0, 255]
        pred_255 = ((pred + 1.0) * 127.5).clamp(0, 255).cpu().numpy()
        target_255 = ((target + 1.0) * 127.5).clamp(0, 255).cpu().numpy()
        
        # Calculate PSNR for each image in batch
        psnr_values = []
        for i in range(pred_255.shape[0]):
            psnr = calculate_psnr(
                pred_255[i, 0],  # Remove channel dimension
                target_255[i, 0],
                crop_border=0,
                input_order='HWC' if pred_255[i, 0].ndim == 2 else 'CHW',
                test_y_channel=False
            )
            psnr_values.append(psnr)
        
        return torch.tensor(np.mean(psnr_values))
    
    @staticmethod
    def calculate_ssim_torch(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
        """Calculate SSIM using standard implementation"""
        # Convert from [-1, 1] to [0, 255]
        pred_255 = ((pred + 1.0) * 127.5).clamp(0, 255).cpu().numpy()
        target_255 = ((target + 1.0) * 127.5).clamp(0, 255).cpu().numpy()
        
        # Calculate SSIM for each image in batch
        ssim_values = []
        for i in range(pred_255.shape[0]):
            # Add channel dimension if needed
            pred_img = pred_255[i, 0, :, :, np.newaxis]
            target_img = target_255[i, 0, :, :, np.newaxis]
            
            ssim = calculate_ssim(
                pred_img,
                target_img,
                crop_border=0,
                input_order='HWC',
                test_y_channel=False
            )
            ssim_values.append(ssim)
        
        return torch.tensor(np.mean(ssim_values))
    
    @staticmethod
    def denormalize_for_metrics(tensor: torch.Tensor) -> torch.Tensor:
        """Convert from normalized [-1, 1] to [0, 1] range for metrics"""
        return (tensor + 1.0) / 2.0'''


# ====== ENHANCED MODEL ARCHITECTURE ======
class EnhancedUNetResNetEncoder(nn.Module):
    """Enhanced encoder with attention mechanisms"""

    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # Enhanced layers with attention
        self.layer1 = self._make_layer(64, 64, 3, stride=1, use_attention=True)
        self.layer2 = self._make_layer(64, 128, 4, stride=2, use_attention=True)
        self.layer3 = self._make_layer(128, 256, 6, stride=2, use_attention=True)
        self.layer4 = self._make_layer(256, 512, 3, stride=2, use_attention=True)

        # Global attention at the end
        self.global_attention = CBAM(512)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int, use_attention: bool = True):
        layers = []
        layers.append(AttentionResNetBlock(in_channels, out_channels, stride, use_attention))
        for _ in range(1, blocks):
            layers.append(AttentionResNetBlock(out_channels, out_channels, use_attention=use_attention))
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
        x = self.global_attention(x)
        features.append(x)

        return x, features


class EnhancedUNetDecoder(nn.Module):
    """Enhanced decoder with attention in skip connections"""

    def __init__(self, out_channels: int = 1):
        super().__init__()

        # Attention modules for skip connections
        self.skip_attention4 = SpatialAttention()
        self.skip_attention3 = SpatialAttention()
        self.skip_attention2 = SpatialAttention()
        self.skip_attention1 = SpatialAttention()

        # Upsampling blocks
        self.up4 = self._make_upconv_block(512, 256)
        self.up3 = self._make_upconv_block(256 + 256, 128)
        self.up2 = self._make_upconv_block(128 + 128, 64)
        self.up1 = self._make_upconv_block(64 + 64, 64)

        self.final_up = nn.ConvTranspose2d(64 + 64, 32, 2, 2)
        self.final_attention = CBAM(32, reduction=8)

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
        # Process skip connections with attention
        skip_features[3] = self.skip_attention4(skip_features[3])
        skip_features[2] = self.skip_attention3(skip_features[2])
        skip_features[1] = self.skip_attention2(skip_features[1])
        skip_features[0] = self.skip_attention1(skip_features[0])

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
        x = self.final_attention(x)
        x = self.final_conv(x)

        return x


class EnhancedUNetWithAttention(nn.Module):
    """Complete enhanced U-Net with spatial attention"""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        self.encoder = EnhancedUNetResNetEncoder(in_channels)
        self.decoder = EnhancedUNetDecoder(out_channels)

        # Enhanced upsampling with residual learning
        if scale_factor == 2:
            self.upsampling = nn.Sequential(
                nn.ConvTranspose2d(out_channels, 64, 4, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels, 1)
            )
        else:
            self.upsampling = nn.Identity()

    def forward(self, x):
        # Store input for residual connection
        input_upsampled = F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)

        # Main forward pass
        encoded, skip_features = self.encoder(x)
        decoded = self.decoder(encoded, skip_features)
        output = self.upsampling(decoded)

        # Residual connection
        output = output + input_upsampled

        # CLAMP OUTPUT to prevent explosion
        output = torch.clamp(output, -1.5, 1.5)

        return output


# ====== ENHANCED LOSS FUNCTION ======
class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features (adapted for single channel)"""

    def __init__(self):
        super().__init__()
        # Simple feature extractor for single channel
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1, bias=False),  # Remove bias
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # Remove bias
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),  # Remove bias
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, bias=False),  # Remove bias
            nn.ReLU(inplace=True),
        )

        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        pred_features = self.features(pred)
        target_features = self.features(target)
        return F.l1_loss(pred_features, target_features)


class EnhancedAMSR2Loss(nn.Module):
    """Enhanced loss with perceptual component and SSIM"""

    def __init__(self, alpha: float = 1.0, beta: float = 0.15,
                 gamma: float = 0.05, delta: float = 0.0, epsilon: float = 0.1):  # Set delta=0.0
        super().__init__()
        self.alpha = alpha  # L1 loss weight
        self.beta = beta  # Gradient loss weight
        self.gamma = gamma  # Physical consistency weight
        self.delta = delta  # Perceptual loss weight (disabled)
        self.epsilon = epsilon  # SSIM loss weight

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        # self.perceptual_loss = PerceptualLoss()  # Comment out
        self.metrics_calc = MetricsCalculator()

    def gradient_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Edge-aware gradient loss"""

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
        """Physical consistency for brightness temperature"""
        pred_mean = torch.mean(pred, dim=[2, 3])
        target_mean = torch.mean(target, dim=[2, 3])
        energy_loss = self.mse_loss(pred_mean, target_mean)

        pred_std = torch.std(pred, dim=[2, 3])
        target_std = torch.std(target, dim=[2, 3])
        distribution_loss = self.mse_loss(pred_std, target_std)

        range_penalty = torch.mean(torch.relu(torch.abs(pred) - 1.0))

        return energy_loss + 0.5 * distribution_loss + 0.1 * range_penalty

    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # CLAMP predictions to valid range FIRST
        pred_clamped = torch.clamp(pred, -1.0, 1.0)
        target_clamped = torch.clamp(target, -1.0, 1.0)

        # Convert to [0, 1] range for SSIM calculation
        pred_01 = self.metrics_calc.denormalize_for_metrics(pred_clamped)
        target_01 = self.metrics_calc.denormalize_for_metrics(target_clamped)

        ssim_val = self.metrics_calc.calculate_ssim_torch(pred_01, target_01)
        # Add small epsilon to prevent log(0) issues
        return 1.0 - torch.clamp(ssim_val, 0.0, 1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        # Calculate individual losses
        l1_loss = self.l1_loss(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        phys_loss = self.brightness_temperature_consistency(pred, target)
        #percep_loss = self.perceptual_loss(pred, target)
        percep_loss = torch.tensor(0.0, device=pred.device)
        ssim_loss = self.ssim_loss(pred, target)

        # Combined loss
        total_loss = (self.alpha * l1_loss +
                      self.beta * grad_loss +
                      self.gamma * phys_loss +
                      self.delta * percep_loss +
                      self.epsilon * ssim_loss)

        # Calculate metrics for monitoring
        with torch.no_grad():
            pred_01 = self.metrics_calc.denormalize_for_metrics(pred)
            target_01 = self.metrics_calc.denormalize_for_metrics(target)
            psnr = self.metrics_calc.calculate_psnr_torch(pred_01, target_01)
            ssim = self.metrics_calc.calculate_ssim_torch(pred_01, target_01)

        return total_loss, {
            'l1_loss': l1_loss.item(),
            'gradient_loss': grad_loss.item(),
            'physical_loss': phys_loss.item(),
            'perceptual_loss': percep_loss.item(),
            'ssim_loss': ssim_loss.item(),
            'total_loss': total_loss.item(),
            'psnr': psnr.item(),
            'ssim': ssim.item()
        }


# ====== TRAINING HISTORY TRACKER ======
class TrainingHistory:
    """Track and visualize training progress"""

    def __init__(self):
        self.history = defaultdict(list)
        self.best_metrics = {
            'psnr': 0,
            'ssim': 0,
            'loss': float('inf')
        }

    def update(self, metrics: dict, epoch: int):
        """Update history with new metrics"""
        for key, value in metrics.items():
            self.history[key].append(value)

        # Update best metrics
        if 'psnr' in metrics and metrics['psnr'] > self.best_metrics['psnr']:
            self.best_metrics['psnr'] = metrics['psnr']

        if 'ssim' in metrics and metrics['ssim'] > self.best_metrics['ssim']:
            self.best_metrics['ssim'] = metrics['ssim']

        if 'total_loss' in metrics and metrics['total_loss'] < self.best_metrics['loss']:
            self.best_metrics['loss'] = metrics['total_loss']

    def plot_progress(self, save_path: str):
        """Create training progress plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16)

        # Loss components
        ax = axes[0, 0]
        for loss_type in ['l1_loss', 'gradient_loss', 'physical_loss', 'perceptual_loss', 'ssim_loss']:
            if loss_type in self.history:
                ax.plot(self.history[loss_type], label=loss_type)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True)

        # Total loss
        ax = axes[0, 1]
        if 'total_loss' in self.history:
            ax.plot(self.history['total_loss'], 'b-', linewidth=2)
            ax.axhline(y=self.best_metrics['loss'], color='r', linestyle='--',
                       label=f'Best: {self.best_metrics["loss"]:.4f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss')
        ax.legend()
        ax.grid(True)

        # PSNR
        ax = axes[0, 2]
        if 'psnr' in self.history:
            ax.plot(self.history['psnr'], 'g-', linewidth=2)
            ax.axhline(y=self.best_metrics['psnr'], color='r', linestyle='--',
                       label=f'Best: {self.best_metrics["psnr"]:.2f} dB')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR (dB)')
        ax.set_title('Peak Signal-to-Noise Ratio')
        ax.legend()
        ax.grid(True)

        # SSIM
        ax = axes[1, 0]
        if 'ssim' in self.history:
            ax.plot(self.history['ssim'], 'm-', linewidth=2)
            ax.axhline(y=self.best_metrics['ssim'], color='r', linestyle='--',
                       label=f'Best: {self.best_metrics["ssim"]:.4f}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SSIM')
        ax.set_title('Structural Similarity Index')
        ax.legend()
        ax.grid(True)

        # Learning rate
        ax = axes[1, 1]
        if 'learning_rate' in self.history:
            ax.semilogy(self.history['learning_rate'], 'c-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True)

        # Best metrics summary
        ax = axes[1, 2]
        ax.text(0.1, 0.8, f"Best PSNR: {self.best_metrics['psnr']:.2f} dB", fontsize=12)
        ax.text(0.1, 0.6, f"Best SSIM: {self.best_metrics['ssim']:.4f}", fontsize=12)
        ax.text(0.1, 0.4, f"Best Loss: {self.best_metrics['loss']:.4f}", fontsize=12)
        ax.text(0.1, 0.2, f"Total Epochs: {len(self.history.get('total_loss', []))}", fontsize=12)
        ax.axis('off')
        ax.set_title('Best Metrics')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def save_history(self, save_path: str):
        """Save training history to JSON"""
        history_dict = dict(self.history)
        history_dict['best_metrics'] = self.best_metrics

        with open(save_path, 'w') as f:
            json.dump(history_dict, f, indent=2)


# ====== ENHANCED TRAINER WITH RICH PROGRESS ======
class EnhancedTrainer:
    """Enhanced trainer with detailed progress monitoring"""

    def __init__(self, model: nn.Module, device: torch.device,
                 learning_rate: float = 2e-4, weight_decay: float = 1e-5,
                 use_amp: bool = True, gradient_accumulation_steps: int = 1):

        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Optimized optimizer settings
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Cosine annealing with warm restarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Restart every 10 epochs
            T_mult=2,  # Double the period after each restart
            eta_min=1e-6
        )

        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.criterion = EnhancedAMSR2Loss()
        self.history = TrainingHistory()
        self.best_psnr = 0
        self.best_ssim = 0

    def train_on_files(self, npz_files: List[str], preprocessor,
                       epochs: int = 50,
                       batch_size: int = 8,
                       num_workers: int = 4,
                       files_per_batch: int = 5,
                       max_swaths_per_file: int = 500,
                       save_dir: str = "./models",
                       validate_every: int = 5):
        """Enhanced training with rich progress monitoring"""

        os.makedirs(save_dir, exist_ok=True)

        # Create training summary
        logger.info("🚀 Enhanced AMSR2 Super-Resolution Training")
        logger.info("=" * 60)
        logger.info(f"📊 Model: Enhanced U-Net with Spatial Attention")
        logger.info(f"📁 Total files: {len(npz_files)}")
        logger.info(f"🔄 Epochs: {epochs}")
        logger.info(f"📦 Batch size: {batch_size}")
        logger.info(f"🧵 Workers: {num_workers}")
        logger.info(f"🎯 Target PSNR: 34+ dB")
        logger.info("=" * 60)

        total_batches = (len(npz_files) + files_per_batch - 1) // files_per_batch

        # Main training loop
        for epoch in range(epochs):
            epoch_start_time = time.time()
            logger.info(f"\n{'=' * 60}")
            logger.info(f"📈 EPOCH {epoch + 1}/{epochs}")
            logger.info(f"{'=' * 60}")

            epoch_metrics = defaultdict(list)

            # Process files in batches
            for batch_idx in range(total_batches):
                start_idx = batch_idx * files_per_batch
                end_idx = min(start_idx + files_per_batch, len(npz_files))
                batch_files = npz_files[start_idx:end_idx]

                logger.info(f"\n📦 Batch {batch_idx + 1}/{total_batches}")
                logger.info(f"   Loading {len(batch_files)} files...")

                # Load datasets
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

                # Create dataloader
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

                # Training on batch
                self.model.train()

                # Rich progress bar
                progress_bar = tqdm(
                    dataloader,
                    desc=f"Batch {batch_idx + 1}/{total_batches}",
                    ncols=120,
                    leave=False,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
                )

                for data_idx, (low_res, high_res) in enumerate(progress_bar):
                    low_res = low_res.to(self.device, non_blocking=True)
                    high_res = high_res.to(self.device, non_blocking=True)

                    # Forward pass with AMP
                    with torch.amp.autocast('cuda', enabled=self.use_amp):
                        pred = self.model(low_res)
                        loss, metrics = self.criterion(pred, high_res)
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
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                            self.optimizer.step()

                        self.optimizer.zero_grad()

                    # Update metrics
                    for key, value in metrics.items():
                        epoch_metrics[key].append(value)

                    # Update progress bar
                    progress_bar.set_postfix({
                        'Loss': f"{metrics['total_loss']:.4f}",
                        'PSNR': f"{metrics['psnr']:.1f}",
                        'SSIM': f"{metrics['ssim']:.3f}",
                        'LR': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                    })

                    # Periodic memory cleanup
                    if data_idx % 50 == 0:
                        torch.cuda.empty_cache()

                # Clean up after batch
                del dataloader, combined_dataset, datasets
                aggressive_cleanup()

            # Epoch statistics
            epoch_avg_metrics = {}
            for key, values in epoch_metrics.items():
                epoch_avg_metrics[key] = np.mean(values)

            # Add learning rate to metrics
            epoch_avg_metrics['learning_rate'] = self.optimizer.param_groups[0]['lr']

            # Update history
            self.history.update(epoch_avg_metrics, epoch)

            # Step scheduler
            self.scheduler.step()

            # Log epoch summary
            epoch_time = time.time() - epoch_start_time
            logger.info(f"\n📊 Epoch {epoch + 1} Summary:")
            logger.info(f"   ⏱️  Time: {epoch_time:.1f}s")
            logger.info(f"   📉 Loss: {epoch_avg_metrics['total_loss']:.4f}")
            logger.info(f"   📈 PSNR: {epoch_avg_metrics['psnr']:.2f} dB")
            logger.info(f"   🎯 SSIM: {epoch_avg_metrics['ssim']:.4f}")
            logger.info(f"   🔧 LR: {epoch_avg_metrics['learning_rate']:.2e}")

            # Save best model
            if epoch_avg_metrics['psnr'] > self.best_psnr:
                self.best_psnr = epoch_avg_metrics['psnr']
                self.best_ssim = epoch_avg_metrics['ssim']

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_psnr': self.best_psnr,
                    'best_ssim': self.best_ssim,
                    'metrics': epoch_avg_metrics,
                    'scale_factor': 2
                }

                save_path = os.path.join(save_dir, 'best_model.pth')
                torch.save(checkpoint, save_path)
                logger.info(f"   💾 Saved best model! PSNR: {self.best_psnr:.2f} dB")

            # Periodic validation and visualization
            if (epoch + 1) % validate_every == 0:
                self._validate_and_visualize(epoch, save_dir)

            # Save training progress plots
            if (epoch + 1) % 5 == 0:
                plot_path = os.path.join(save_dir, 'training_progress.png')
                self.history.plot_progress(plot_path)

                history_path = os.path.join(save_dir, 'training_history.json')
                self.history.save_history(history_path)

        # Final summary
        logger.info(f"\n{'=' * 60}")
        logger.info(f"🎉 Training Completed!")
        logger.info(f"   🏆 Best PSNR: {self.best_psnr:.2f} dB")
        logger.info(f"   🎯 Best SSIM: {self.best_ssim:.4f}")
        logger.info(f"   💾 Model saved to: {save_dir}")
        logger.info(f"{'=' * 60}")

    def _validate_and_visualize(self, epoch: int, save_dir: str):
        """Create validation visualizations"""
        self.model.eval()

        # Create a simple test pattern
        test_size = (1, 1, 256, 26)  # Scaled down for 2x SR
        test_low = torch.randn(test_size).to(self.device)

        with torch.no_grad():
            test_high = self.model(test_low)

        # Save visualization
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(test_low[0, 0].cpu().numpy(), cmap='turbo')
        axes[0].set_title('Low Resolution Input')
        axes[0].axis('off')

        axes[1].imshow(test_high[0, 0].cpu().numpy(), cmap='turbo')
        axes[1].set_title('Super Resolution Output')
        axes[1].axis('off')

        plt.suptitle(f'Epoch {epoch + 1} Validation')
        plt.tight_layout()

        viz_path = os.path.join(save_dir, f'validation_epoch_{epoch + 1}.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.model.train()


# ====== KEEP EXISTING DATASET AND PREPROCESSOR ======
# (Using the OptimizedAMSR2Dataset and AMSR2DataPreprocessor from your original code)
from gpu_sequential_amsr2_optimized import OptimizedAMSR2Dataset, AMSR2DataPreprocessor, aggressive_cleanup


# ====== MAIN FUNCTION ======
def main():
    """Enhanced main function with better parameter defaults"""

    parser = argparse.ArgumentParser(
        description='Enhanced AMSR2 Super-Resolution with Spatial Attention'
    )

    # Required
    parser.add_argument('--npz-dir', type=str, required=True,
                        help='Path to directory with NPZ files')

    # Data parameters
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to train on')
    parser.add_argument('--max-swaths-per-file', type=int, default=500,
                        help='Maximum swaths per file (default: 500)')

    # Training parameters - OPTIMIZED
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate (default: 2e-4)')
    parser.add_argument('--gradient-accumulation', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')

    # System parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--files-per-batch', type=int, default=5,
                        help='Files to load per batch (default: 5)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision')

    # Model parameters
    parser.add_argument('--target-height', type=int, default=2048,
                        help='Target height (default: 2048)')
    parser.add_argument('--target-width', type=int, default=208,
                        help='Target width (default: 208)')

    # Output
    parser.add_argument('--save-dir', type=str, default='./models_enhanced',
                        help='Directory to save models and results')
    parser.add_argument('--validate-every', type=int, default=5,
                        help='Validate every N epochs')

    args = parser.parse_args()

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("🛰️  ENHANCED AMSR2 SUPER-RESOLUTION WITH SPATIAL ATTENTION")
    print("=" * 60)

    logger.info(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        logger.info(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    # Find files
    npz_files = find_npz_files(args.npz_dir, args.max_files)
    if not npz_files:
        logger.error("❌ No NPZ files found")
        sys.exit(1)

    # Estimate data requirements
    estimated_examples = len(npz_files) * args.max_swaths_per_file
    logger.info(f"\n📊 Data Estimation:")
    logger.info(f"   Files: {len(npz_files)}")
    logger.info(f"   Estimated examples: ~{estimated_examples:,}")

    if estimated_examples < 100000:
        logger.warning("   ⚠️  For SOTA results (34+ PSNR), consider using 100k+ examples")
        logger.warning("   Current estimate may achieve 30-32 PSNR")

    # Create enhanced model
    logger.info("\n🧠 Creating Enhanced Model with Spatial Attention...")
    model = EnhancedUNetWithAttention(
        in_channels=1,
        out_channels=1,
        scale_factor=2
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"   Total parameters: {total_params:,}")
    logger.info(f"   Model size: {total_params * 4 / 1024 ** 2:.1f} MB")

    # Create preprocessor
    preprocessor = AMSR2DataPreprocessor(
        target_height=args.target_height,
        target_width=args.target_width
    )

    # Create enhanced trainer
    trainer = EnhancedTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        use_amp=args.use_amp,
        gradient_accumulation_steps=args.gradient_accumulation
    )

    # Start training
    logger.info("\n🚀 Starting Enhanced Training...")
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
            save_dir=args.save_dir,
            validate_every=args.validate_every
        )

        training_time = time.time() - start_time
        logger.info(f"\n⏱️  Total training time: {training_time / 3600:.2f} hours")

    except KeyboardInterrupt:
        logger.info("\n⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if device.type == 'cuda':
            aggressive_cleanup()
        logger.info("✅ Program finished")


# ====== HELPER FUNCTION ======
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


if __name__ == "__main__":
    main()