import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
from skimage import morphology
import tempfile
import os
import logging

# Configuración básica de logging
logger = logging.getLogger(__name__)



class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR(3, 64), CBR(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR(64, 128), CBR(128, 128))
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(CBR(128, 256), CBR(256, 256))

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(CBR(256, 128), CBR(128, 64))
        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(CBR(128, 64), CBR(64, 64))

        self.final = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return torch.sigmoid(self.final(d1))

def is_blurry(image, threshold=20):
    """
    Detects if the image is blurry using Fast Fourier Transform (FFT).
    It checks the presence of high-frequency components.
    """
    img_uint8 = image.astype(np.uint8)
    if len(img_uint8.shape) == 3:
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_uint8

    # Fast Fourier Transform
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Magnitud spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    
    # If the average magnitude is too low, it's blurry
    mean_val = np.mean(magnitude_spectrum)
    
    # Fallback to Laplacian for extra safety
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return (mean_val < threshold) or (variance < 40)


def improve_image_quality(image):
    """
    Applies Bilateral filter to reduce noise while keeping edges,
    and CLAHE to improve localized contrast in the conjunctiva.
    """
    img_uint8 = image.astype(np.uint8)
    
    # 1. Denoise (Bilateral Filter preserves edges better than Gaussian)
    denoised = cv2.bilateralFilter(img_uint8, 9, 75, 75)
    
    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Work on LAB color space to avoid shifting colors significantly
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    improved = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    
    return improved


def normalize_white_background(image):
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("La imagen debe ser RGB con 3 canales")
    
    img_rgb = image.astype(np.uint8)
    
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask_hsv = cv2.inRange(hsv, lower_white, upper_white)

    lower_white_rgb = np.array([200, 200, 200])
    upper_white_rgb = np.array([255, 255, 255])
    mask_rgb = cv2.inRange(img_rgb, lower_white_rgb, upper_white_rgb)

    mask = cv2.bitwise_or(mask_hsv, mask_rgb)
    mask = cv2.bitwise_not(mask)

    img_with_alpha = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2RGBA)
    img_with_alpha[:, :, 3] = mask
    
    background = np.zeros_like(img_rgb)
    alpha = img_with_alpha[:, :, 3:4] / 255.0
    normalized_image = img_with_alpha[:, :, :3] * alpha + background * (1 - alpha)
    
    return normalized_image.astype(np.uint8)

def remove_eyelashes(image):
    """
    Detects and removes eyelashes using morphological black-hat and inpainting.
    Useful for medical images of the eye to avoid occlusion during segmentation.
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
        
    img_uint8 = image.astype(np.uint8)
    
    # 1. Grayscale
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    
    # 2. Black-Hat morphology to find dark linear structures (lashes)
    # Much wider kernel to catch long horizontal-ish lashes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Thresholding to create a mask
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # 4. Aggressive dilation to cover shadows of lashes
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # 5. Inpaint using the mask (TELEA is better for larger areas)
    inpainted = cv2.inpaint(img_uint8, mask, 5, cv2.INPAINT_TELEA)
    
    return inpainted



def preprocess_image_for_unet(image_input, target_size=(256, 256), normalize_background=True, remove_lashes=True, improve_quality=True):
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_input}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = np.array(image_input)
        if len(image.shape) == 3 and image.shape[2] == 3:
            pass
        else:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 1. Blur Detection
    if is_blurry(image):
        logger.warning("Imagen detectada como borrosa. Los resultados pueden ser imprecisos.")

    # 2. Quality Improvement (Denoise + CLAHE)
    if improve_quality:
        image = improve_image_quality(image)

    # 3. Remove eyelashes before normalization
    if remove_lashes:
        image = remove_eyelashes(image)

    # 4. Background normalization
    if normalize_background:
        image = normalize_white_background(image)
    
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image_tensor

def find_iris_anchor(image):
    """
    Finds the center of the iris to use as a spatial anchor.
    The conjunctiva is always below and relatively centered to the iris.
    """
    img_uint8 = image.astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    
    # Iris is dark (Low V)
    mask_iris = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_iris = cv2.morphologyEx(mask_iris, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(mask_iris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_iris = None
    max_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            perim = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perim * perim) if perim > 0 else 0
            if circularity > 0.3 and area > max_area:
                max_area = area
                best_iris = cnt
                
    if best_iris is not None:
        M = cv2.moments(best_iris)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return cx, cy, int(np.sqrt(max_area/np.pi))
    return None

def post_process_mask(mask, original_image=None, min_area_ratio=0.001, kernel_size=3, remove_small_components=True, expand_borders=0):
    mask = (mask * 255).astype(np.uint8)
    h, w = mask.shape
    
    # 1. Fill holes
    mask_filled = np.zeros((h+2, w+2), dtype=np.uint8)
    mask_filled[1:h+1, 1:w+1] = mask
    cv2.floodFill(mask_filled, None, (0, 0), 255)
    holes = cv2.bitwise_not(mask_filled)
    holes = holes[1:h+1, 1:w+1]
    mask = cv2.bitwise_or(mask, holes)
    
    # 2. Color-based Validation (LAB Color Space)
    h_anchor, w_anchor = None, None
    if original_image is not None:
        img_res = cv2.resize(original_image, (w, h))
        
        # Try to find iris anchor for spatial logic
        anchor_data = find_iris_anchor(img_res)
        if anchor_data:
            w_anchor, h_anchor, r_anchor = anchor_data
            
        lab = cv2.cvtColor(img_res, cv2.COLOR_RGB2LAB)
        a_channel = lab[:, :, 1]
        
        # Use Otsu's thresholding to find the 'reddest' parts dynamically
        # This is better than a fixed threshold for different skin tones
        blur_a = cv2.GaussianBlur(a_channel, (5, 5), 0)
        _, mask_red_lab = cv2.threshold(blur_a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ensure only strong reds are kept (raise to 145 to exclude warm skin tones better)
        mask_red_lab = cv2.bitwise_and(mask_red_lab, cv2.threshold(a_channel, 145, 255, cv2.THRESH_BINARY)[1])
        
        mask = cv2.bitwise_and(mask, mask_red_lab)



    # 3. Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 4. Spatial Windowing (IRIS ANCHOR LOGIC)
    if h_anchor is not None:
        # FIXED: `:` after index to slice all rows below that point
        mask[:int(h_anchor + (h*0.08)), :] = 0 
        mask[int(h_anchor + (h*0.32)):, :] = 0


        # Lateral constraint based on iris center
        x_min = max(0, w_anchor - int(w*0.35))
        x_max = min(w, w_anchor + int(w*0.35))
        mask[:, :x_min] = 0
        mask[:, x_max:] = 0
    else:
        # Fallback to fixed sandwich if no iris found
        h_start = int(h * 0.55)
        h_end = int(h * 0.88)
        mask[:h_start, :] = 0
        mask[h_end:, :] = 0
        width_margin = int(w * 0.15)
        mask[:, :width_margin] = 0
        mask[:, w-width_margin:] = 0
    
    if remove_small_components:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            best_component = -1
            best_score = -1
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                w_comp = stats[i, cv2.CC_STAT_WIDTH]
                h_comp = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Aspect ratio: Conjunctiva is thin and wide
                ratio = w_comp / (h_comp + 1e-5)
                
                # Score based on area and shape (Prefer horizontal ellipses)
                if ratio > 2.5 and area > (h * w * 0.0005):
                    # Favor components in the center of the window
                    score = area * ratio
                    if score > best_score:
                        best_score = score
                        best_component = i
            
            if best_component != -1:
                clean_mask = np.zeros_like(mask)
                clean_mask[labels == best_component] = 255
                
                # Apply convex hull to clean jagged/irregular edges of the selected component
                cnts, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    # Take the largest contour from the chosen component
                    cnt = max(cnts, key=cv2.contourArea)
                    hull = cv2.convexHull(cnt)
                    hull_mask = np.zeros_like(clean_mask)
                    cv2.drawContours(hull_mask, [hull], -1, 255, -1)
                    # Intersect hull with the actual red pixels to not re-add non-red area
                    mask = cv2.bitwise_and(hull_mask, clean_mask)
                else:
                    mask = clean_mask
            else:
                mask = np.zeros_like(mask)



    if expand_borders > 0:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_borders*2+1, expand_borders*2+1))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    return mask.astype(np.float32) / 255.0



def segment_image_with_unet(image_input, model_path, device='cpu', target_size=(256, 256), 
                           post_process=True, min_area_ratio=0.001, kernel_size=3, 
                           remove_small_components=True, expand_borders=5, 
                           normalize_background=True, remove_lashes=True, improve_quality=True):

    # 1. Get original image for reference
    if isinstance(image_input, str):
        original_image = cv2.imread(image_input)
        if original_image is None:
            return None, None
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        original_image = np.array(image_input)
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            pass
        else:
            if len(original_image.shape) == 3:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    model = UNet().to(device)
    if not os.path.exists(model_path):
        print(f"⚠️ Modelo UNet no encontrado en {model_path}. Se requiere para segmentación precisa.")
        return None, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Preprocess (Pass original_image directly to avoid reloading)
    image_tensor = preprocess_image_for_unet(original_image, target_size, normalize_background, remove_lashes, improve_quality)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        mask = model(image_tensor)
        binary_mask = (mask > 0.5).float()
    
    mask_np = binary_mask.cpu().squeeze().numpy()
    
    # 3. Post-process with original_image for color validation
    if post_process:
        mask_np = post_process_mask(mask_np, original_image, min_area_ratio, kernel_size, remove_small_components, expand_borders)
    
    original_size = (original_image.shape[1], original_image.shape[0])
    
    if original_size != target_size:
        mask_np = cv2.resize(mask_np.astype(np.float32), original_size)
        mask_np = (mask_np > 0.5).astype(np.float32)
    
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    mask_np = mask_closed.astype(np.float32) / 255.0
    
    segmented_image = cv2.bitwise_and(original_image, original_image, mask=mask_closed)
    
    return segmented_image, mask_np


def crop_segmented_image(image, mask, padding=10):
    coords = np.where(mask > 0.5)
    if len(coords[0]) == 0:
        return image, (0, 0, image.shape[1], image.shape[0])
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    
    cropped_image = image[y_min:y_max, x_min:x_max]
    bbox = (x_min, y_min, x_max, y_max)
    
    return cropped_image, bbox
