import os
import cv2
import numpy as np
import sys

# Añadir el path del backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagenes.tasks.preprocesamiento.extraccionConjuntiva import ConjuntivaExtractor
import re

extractor = ConjuntivaExtractor()

def obtener_metricas(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    area_img = h * w
    
    # 1. Ojo
    anchor, radius = extractor.detect_eye_anchor(img)
    if anchor is None: return None

    # Recorte para métricas reales
    eye_img, new_anchor, (x0, y0) = extractor.crop_to_eye(img, anchor, radius)
    h_e, w_e = eye_img.shape[:2]
    area_crop = h_e * w_e

    area_iris = np.pi * radius**2
    iris_frac = area_iris / area_img
    
    # 2. Conjuntiva
    win_mask, _, _ = extractor.get_search_window(eye_img, new_anchor, radius)
    raw_mask = extractor.find_medialuna_by_contrast(eye_img, win_mask, new_anchor, radius)
    final_mask = extractor.polish_final(raw_mask)
    final_mask = extractor.cerrar_forma_medialuna(final_mask, new_anchor, radius)
    
    pixeles = np.count_nonzero(final_mask)
    area_pct = pixeles / area_crop if area_crop > 0 else 0
    
    aspect = 0
    ancho_frac = 0
    if pixeles > 0:
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt_mayor = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(cnt_mayor)
            aspect = bw / max(bh, 1)
            ancho_frac = bw / w_e

    # 3. Nitidez (sobre mascara)
    vmlap_mask = 0
    if pixeles > 100:
        gris = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(final_mask > 0)
        y0_p, y1_p = ys.min(), ys.max()
        x0_p, x1_p = xs.min(), xs.max()
        patch = gris[y0_p:y1_p+1, x0_p:x1_p+1]
        patch_sx = cv2.Sobel(patch, cv2.CV_64F, 2, 0, ksize=3)
        patch_sy = cv2.Sobel(patch, cv2.CV_64F, 0, 2, ksize=3)
        vmlap_mask = (np.abs(patch_sx) + np.abs(patch_sy)).var()

    return {
        'iris_frac': iris_frac,
        'area_pct': area_pct,
        'aspect': aspect,
        'ancho_frac': ancho_frac,
        'vmlap_mask': vmlap_mask
    }

def salvar_buenas():
    # Lista de fotos que DEBEN pasar
    imagenes_buenas = [
        "1X1DCA4M-F.jpeg"
        # Añadir aquí más si es necesario
    ]
    
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'media', 'originales', 'CON ANEMIA')
    
    min_metrics = {
        'iris_frac': 9999, 'area_pct': 9999, 'aspect': 9999, 'ancho_frac': 9999, 'vmlap_mask': 9999
    }
    
    print("Analizando imágenes que DEBEN pasar...")
    for nombre in imagenes_buenas:
        ruta = os.path.join(base_dir, nombre)
        if not os.path.exists(ruta):
            print(f"No se encontró: {ruta}")
            continue
            
        m = obtener_metricas(ruta)
        if m:
            print(f"Métricas {nombre}:")
            for k, v in m.items():
                print(f"  {k}: {v:.5f}")
                if v > 0 and v < min_metrics[k]:
                    min_metrics[k] = v

    print("\nUmbrales mínimos requeridos para salvar estas fotos:")
    for k, v in min_metrics.items():
        if v != 9999:
            print(f"  {k}: {v:.5f}")
            
    # Ajustar a la baja (margen del 10%)
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Mapeo a nombres del env
        mapping = {
            'OJO_MIN_AREA_FRACCION': ('iris_frac', 0.9),
            'CONJUNTIVA_MIN_AREA_PCT': ('area_pct', 0.8),
            'CONJUNTIVA_MIN_ASPECT_RATIO': ('aspect', 0.8),
            'CONJUNTIVA_MIN_ANCHO_FRACCION': ('ancho_frac', 0.8),
            'NITIDEZ_UMBRAL_VMLAP_MASCARA': ('vmlap_mask', 0.8),
        }
        
        for env_key, (metric_key, margin) in mapping.items():
            if min_metrics[metric_key] != 9999:
                val = min_metrics[metric_key] * margin # aplicamos margen
                if val < 1: val = round(val, 4)
                else: val = int(val)
                
                content = re.sub(rf"^({env_key})=.*", rf"\1={val}", content, flags=re.MULTILINE)
                
        with open(env_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("\n[OK] .env relajado para permitir el paso de estas imágenes.")

if __name__ == '__main__':
    salvar_buenas()
