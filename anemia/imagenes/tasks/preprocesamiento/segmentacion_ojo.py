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
    # A rect kernel works well for capturing lash lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # 3. Thresholding to create a mask
    _, mask = cv2.threshold(blackhat, 8, 255, cv2.THRESH_BINARY)
    
    # 4. Smooth mask to cover slightly more area
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    
    # 5. Inpaint using the mask
    inpainted = cv2.inpaint(img_uint8, mask, 3, cv2.INPAINT_TELEA)
    
    return inpainted

def preprocess_image_for_unet(image_input, target_size=(256, 256), normalize_background=True, remove_lashes=True):
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
    
    # Remove eyelashes before normalization to avoid them being treated as part of the structure
    if remove_lashes:
        image = remove_eyelashes(image)

    if normalize_background:
        image = normalize_white_background(image)
    
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image_tensor

def post_process_mask(mask, min_area_ratio=0.001, kernel_size=3, remove_small_components=True, expand_borders=0):
    mask = (mask * 255).astype(np.uint8)
    h, w = mask.shape
    mask_filled = np.zeros((h+2, w+2), dtype=np.uint8)
    mask_filled[1:h+1, 1:w+1] = mask
    cv2.floodFill(mask_filled, None, (0, 0), 255)
    holes = cv2.bitwise_not(mask_filled)
    holes = holes[1:h+1, 1:w+1]
    mask = cv2.bitwise_or(mask, holes)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    if remove_small_components:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            if len(areas) > 0:
                largest_component = np.argmax(areas) + 1
                total_area = mask.shape[0] * mask.shape[1]
                min_area = total_area * min_area_ratio
                if areas[largest_component - 1] >= min_area:
                    clean_mask = np.zeros_like(mask)
                    clean_mask[labels == largest_component] = 255
                    mask = clean_mask

    if expand_borders > 0:
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (expand_borders*2+1, expand_borders*2+1))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    return mask.astype(np.float32) / 255.0

def segment_image_with_unet(image_input, model_path, device='cpu', target_size=(256, 256), 
                           post_process=True, min_area_ratio=0.001, kernel_size=3, 
                           remove_small_components=True, expand_borders=5, 
                           normalize_background=True, remove_lashes=True):
    model = UNet().to(device)
    if not os.path.exists(model_path):
        print(f"⚠️ Modelo UNet no encontrado en {model_path}. Se requiere para segmentación precisa.")
        # Fallback a segmentación básica o retornar error? Retornaremos None para que el llamador decida
        return None, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    image_tensor = preprocess_image_for_unet(image_input, target_size, normalize_background, remove_lashes)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        mask = model(image_tensor)
        binary_mask = (mask > 0.5).float()
    
    mask_np = binary_mask.cpu().squeeze().numpy()
    
    if post_process:
        mask_np = post_process_mask(mask_np, min_area_ratio, kernel_size, remove_small_components, expand_borders)
    
    if isinstance(image_input, str):
        original_image = cv2.imread(image_input)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    else:
        original_image = np.array(image_input)

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
