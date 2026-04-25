import os
import cv2
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from imagenes.tasks.preprocesamiento.extraccionConjuntiva import ConjuntivaExtractor

extractor = ConjuntivaExtractor()

def obtener_metricas(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    area_img = h * w
    
    # 1. Ojo
    anchor, radius = extractor.detect_eye_anchor(img)
    area_iris = np.pi * radius**2
    iris_frac = area_iris / area_img
    
    # 2. Conjuntiva
    win_mask, _, _ = extractor.get_search_window(img, anchor, radius)
    raw_mask = extractor.find_medialuna_by_contrast(img, win_mask, anchor, radius)
    final_mask = extractor.polish_final(raw_mask)
    final_mask = extractor.cerrar_forma_medialuna(final_mask)
    
    pixeles = np.count_nonzero(final_mask)
    area_pct = pixeles / area_img
    
    aspect = 0
    ancho_frac = 0
    if pixeles > 0:
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt_mayor = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(cnt_mayor)
            aspect = bw / max(bh, 1)
            ancho_frac = bw / w

    # 3. Nitidez (sobre mascara)
    vmlap_mask = 0
    if pixeles > 200:
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(final_mask > 0)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        patch = gris[y0:y1+1, x0:x1+1]
        patch_sx = cv2.Sobel(patch, cv2.CV_64F, 2, 0, ksize=3)
        patch_sy = cv2.Sobel(patch, cv2.CV_64F, 0, 2, ksize=3)
        vmlap_mask = (np.abs(patch_sx) + np.abs(patch_sy)).var()

    return {
        'area_pct': area_pct,
        'aspect': aspect,
        'ancho_frac': ancho_frac,
        'vmlap_mask': vmlap_mask
    }

def main():
    import glob
    buenas_imgs = glob.glob('media/procesadas/area/CON ANEMIA/*.jpeg')
    print(f"Analizando {len(buenas_imgs)} imágenes BUENAS...")
    
    metricas_todas = {
        'area_pct': [], 'aspect': [], 'ancho_frac': [], 'vmlap_mask': []
    }
    
    for ruta in buenas_imgs:
        m = obtener_metricas(ruta)
        if m:
            for k, v in m.items():
                if v > 0: # only where it actually found something
                    metricas_todas[k].append(v)
                
    print("\n--- ESTADÍSTICAS DE IMÁGENES BUENAS ---")
    for k, v in metricas_todas.items():
        if not v: continue
        arr = np.array(v)
        print(f"{k}:")
        print(f"  Min: {np.min(arr):.4f}")
        print(f"  P05: {np.percentile(arr, 5):.4f}")
        print(f"  P10: {np.percentile(arr, 10):.4f}")
        print(f"  P25: {np.percentile(arr, 25):.4f}")
        print(f"  P50 (mediana): {np.percentile(arr, 50):.4f}")
        print(f"  Media: {np.mean(arr):.4f}")

if __name__ == '__main__':
    main()
