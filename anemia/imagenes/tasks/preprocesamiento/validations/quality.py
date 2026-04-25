import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

def es_nitida(img, mask_conjuntiva=None):
    """
    NITIDEZ COMPLETA: Implementa las 5 métricas originales.
    1. Laplacian Global, 2. Tenengrad, 3. FFT High-Freq, 
    4. VMLAP en ROI Central, 5. VMLAP en Máscara de Conjuntiva.
    """
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gris.shape

    # 1. Laplacian global
    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()

    # 2. Tenengrad global
    sobelx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.sqrt(sobelx**2 + sobely**2).mean()

    # 3. FFT altas frecuencias
    fshift = np.fft.fftshift(np.fft.fft2(gris))
    mag = np.abs(fshift)
    cy, cx = h // 2, w // 2
    r_frac = float(os.getenv("NITIDEZ_FFT_RADIO_FRACCION", 0.10))
    r = int(min(h, w) * r_frac)
    y_grid, x_grid = np.ogrid[:h, :w]
    mask_low = (y_grid - cy)**2 + (x_grid - cx)**2 <= r**2
    hf_ratio = np.sum(mag[~mask_low]) / (np.sum(mag) + 1e-8)

    # 4. VMLAP ROI central
    margen_f = float(os.getenv("NITIDEZ_ROI_MARGEN", 0.30))
    margen_y = int(h * margen_f)
    margen_x = int(w * margen_f)
    roi = gris[margen_y: h - margen_y, margen_x: w - margen_x]
    if roi.size > 0:
        roi_sobelx = cv2.Sobel(roi, cv2.CV_64F, 2, 0, ksize=3)
        roi_sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 2, ksize=3)
        vmlap_centro = (np.abs(roi_sobelx) + np.abs(roi_sobely)).var()
    else:
        vmlap_centro = 0

    # Umbrales
    u_lap = float(os.getenv("NITIDEZ_UMBRAL_LAP", 3))
    u_ten = float(os.getenv("NITIDEZ_UMBRAL_TENENGRAD", 3))
    u_hfr = float(os.getenv("NITIDEZ_UMBRAL_HF_RATIO", 0.005))
    u_vmlap_c = float(os.getenv("NITIDEZ_UMBRAL_VMLAP_CENTRO", 3))

    pasa = (lap_var > u_lap and tenengrad > u_ten and hf_ratio > u_hfr and vmlap_centro > u_vmlap_c)

    # 5. VMLAP sobre la máscara de conjuntiva (Nitidez local)
    if pasa and mask_conjuntiva is not None and np.count_nonzero(mask_conjuntiva) > 0:
        min_pixels = int(os.getenv("NITIDEZ_MIN_PIXELES_MASCARA", 200))
        if np.count_nonzero(mask_conjuntiva) > min_pixels:
            ys, xs = np.where(mask_conjuntiva > 0)
            patch = gris[ys.min():ys.max()+1, xs.min():xs.max()+1]
            p_sx = cv2.Sobel(patch, cv2.CV_64F, 2, 0, ksize=3)
            p_sy = cv2.Sobel(patch, cv2.CV_64F, 0, 2, ksize=3)
            vmlap_mask = (np.abs(p_sx) + np.abs(p_sy)).var()
            u_vmlap_m = float(os.getenv("NITIDEZ_UMBRAL_VMLAP_MASCARA", 3))
            pasa = pasa and (vmlap_mask > u_vmlap_m)

    return pasa

def tiene_tamano_suficiente(img):
    min_px = int(os.getenv("TAMANO_MIN_PX", 200))
    alto, ancho = img.shape[:2]
    return alto >= min_px and ancho >= min_px
