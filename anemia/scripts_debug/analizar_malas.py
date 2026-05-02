import os
import cv2
import numpy as np

# Añadir el path para importar
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagenes.tasks.preprocesamiento.extraccionConjuntiva import ConjuntivaExtractor

extractor = ConjuntivaExtractor()

def leer_malas(txt_path):
    malas = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            l = line.strip()
            if l and not l.endswith(':'):
                malas.append(l)
    return malas

def obtener_metricas(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    area_img = h * w
    
    # 1. Ojo
    anchor, radius = extractor.detect_eye_anchor(img)
    if anchor is None: return None

    # Recorte para que las métricas sean reales
    eye_img, new_anchor, (x0, y0) = extractor.crop_to_eye(img, anchor, radius)
    h_e, w_e = eye_img.shape[:2]
    area_crop = h_e * w_e

    area_iris = np.pi * radius**2
    iris_frac = area_iris / area_img
    
    # 2. Conjuntiva (sobre el recorte)
    win_mask, _, _ = extractor.get_search_window(eye_img, new_anchor, radius)
    raw_mask = extractor.find_medialuna_by_contrast(eye_img, win_mask, new_anchor, radius)
    final_mask = extractor.polish_final(raw_mask)
    final_mask = extractor.cerrar_forma_medialuna(final_mask, new_anchor, radius)
    
    pixeles = np.count_nonzero(final_mask)
    area_pct = pixeles / area_crop
    
    aspect = 0
    ancho_frac = 0
    if pixeles > 0:
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt_mayor = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(cnt_mayor)
            aspect = bw / max(bh, 1)
            ancho_frac = bw / w_e # Relative to crop width
    
    # 3. Nitidez (sobre mascara)
    vmlap_mask = 0
    if pixeles > 200:
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

def actualizar_env_automatico(metricas):
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if not os.path.exists(env_path): return
    
    with open(env_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    mapping = {
        'OJO_MIN_AREA_FRACCION': ('iris_frac', 95),
        'CONJUNTIVA_MIN_AREA_PCT': ('area_pct', 95),
        'CONJUNTIVA_MIN_ASPECT_RATIO': ('aspect', 95),
        'CONJUNTIVA_MIN_ANCHO_FRACCION': ('ancho_frac', 95),
        'NITIDEZ_UMBRAL_VMLAP_MASCARA': ('vmlap_mask', 95),
    }
    
    import re
    for env_key, (metric_key, p) in mapping.items():
        if metric_key in metricas and metricas[metric_key]:
            val = np.percentile(metricas[metric_key], p)
            # Redondear para que sea bonito
            if val < 1: val = round(val, 4)
            else: val = int(val)
            
            patron = rf"^({env_key})=.*"
            content = re.sub(patron, rf"\1={val}", content, flags=re.MULTILINE)
            
    with open(env_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("\n[OK] .env actualizado con umbrales protectores.")

def main():
    txt_path = 'media/notomar.txt'
    base_dir = 'media/originales/CON ANEMIA'
    
    malas = leer_malas(txt_path)
    print(f"Analizando {len(malas)} imágenes malas...")
    
    metricas_todas = {
        'iris_frac': [], 'area_pct': [], 'aspect': [], 'ancho_frac': [], 'vmlap_mask': []
    }
    
    for nombre in malas:
        ruta = os.path.join(base_dir, nombre)
        m = obtener_metricas(ruta)
        if m:
            for k, v in m.items():
                metricas_todas[k].append(v)
                
    # Calcular percentiles para ver dónde cortar
    print("\n--- ESTADÍSTICAS DE IMÁGENES MALAS ---")
    for k, v in metricas_todas.items():
        if not v: continue
        arr = np.array(v)
        print(f"{k}:")
        print(f"  Min: {np.min(arr):.4f}")
        print(f"  Max: {np.max(arr):.4f}")
        print(f"  Media: {np.mean(arr):.4f}")
        print(f"  P50 (mediana): {np.percentile(arr, 50):.4f}")
        print(f"  P75: {np.percentile(arr, 75):.4f}")
        print(f"  P90: {np.percentile(arr, 90):.4f}")
        print(f"  P95: {np.percentile(arr, 95):.4f}")

    actualizar_env_automatico(metricas_todas)

if __name__ == '__main__':
    main()
