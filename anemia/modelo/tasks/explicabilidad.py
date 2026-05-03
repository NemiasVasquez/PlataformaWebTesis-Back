import torch
import numpy as np
import cv2
import os
from ..indicadores.nivel_detalle import calcular_nivel_detalle, calcular_exactitud_areas

def generate_smoothgrad(model, device, img_tensor, original_img, class_idx, num_samples=30, stdev_spread=0.15):
    """
    Generate SmoothGrad explanation for a given PyTorch model and image tensor.
    
    Args:
        model: PyTorch model (already in eval mode)
        device: CPU or CUDA device
        img_tensor: NCHW tensor (normalized 0-1) requiring grad
        original_img: original BGR image for overlay (from cv2.imread)
        class_idx: target class index for gradients
        num_samples: number of noisy samples
        stdev_spread: standard deviation for noise (relative to data range)
        
    Returns:
        heatmap: colormapped heatmap image (BGR)
        overlay: original image overlaid with heatmap (BGR)
    """
    model.eval()
    
    # Calculate noise std based on input range (which is 0-1)
    stdev = stdev_spread * (img_tensor.max() - img_tensor.min()).item()
    
    total_gradients = torch.zeros_like(img_tensor)
    
    for _ in range(num_samples):
        noise = torch.randn_like(img_tensor) * stdev
        noisy_img = img_tensor + noise
        noisy_img.requires_grad = True
        
        outputs = model(noisy_img)
        score = outputs[0, class_idx]
        
        model.zero_grad()
        score.backward()
        
        # Add absolute gradients
        total_gradients += noisy_img.grad.abs()
        
    avg_gradients = total_gradients / num_samples
    
    # Process gradients to create heatmap
    grad_np = avg_gradients[0].cpu().numpy()  # CHW
    
    # For RGB/RGBA, take the maximum gradient across channels
    grad_np = np.max(grad_np, axis=0)  # HW
    
    # Normalize to 0-255
    grad_np = grad_np - np.min(grad_np)
    if np.max(grad_np) > 0:
        grad_np = grad_np / np.max(grad_np)
    grad_np = np.uint8(grad_np * 255)
    
    # Apply colormap
    heatmap = cv2.applyColorMap(grad_np, cv2.COLORMAP_JET)
    
    # Resize original_img to match heatmap if necessary (should match since tensor is made from it)
    if original_img.shape[:2] != heatmap.shape[:2]:
        original_img_resized = cv2.resize(original_img, (heatmap.shape[1], heatmap.shape[0]))
    else:
        original_img_resized = original_img
        
    # Remove alpha channel from original if present
    if original_img_resized.shape[2] == 4:
        original_img_resized = cv2.cvtColor(original_img_resized, cv2.COLOR_BGRA2BGR)
        
    # Overlay
    overlay = cv2.addWeighted(original_img_resized, 0.6, heatmap, 0.4, 0)
    
    return heatmap, overlay
