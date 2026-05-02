import os
import cv2
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imagenes.tasks.preprocesamiento.core.extractor import ConjuntivaExtractor

def get_image_paths(txt_path, base_dir):
    paths = []
    if not os.path.exists(txt_path): return paths
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.endswith(':'): continue
            p1 = os.path.join(base_dir, 'CON ANEMIA', line)
            p2 = os.path.join(base_dir, 'SIN ANEMIA', line)
            if os.path.exists(p1): paths.append((line, p1))
            elif os.path.exists(p2): paths.append((line, p2))
    return paths

def get_metrics(img, extractor):
    h, w = img.shape[:2]
    anchor, radius = extractor.detect_eye_anchor(img)
    if anchor is None: return None
        
    eye_img, new_anchor, (x0, y0) = extractor.crop_to_eye(img, anchor, radius)
    h_e, w_e = eye_img.shape[:2]
    win_mask, _, _ = extractor.get_search_window(eye_img, new_anchor, radius)
    raw_mask = extractor.find_medialuna_by_contrast(eye_img, win_mask, new_anchor, radius)
    final_mask = extractor.polish_final(raw_mask)
    final_mask = extractor.cerrar_forma_medialuna(final_mask, new_anchor, radius)
    
    pixeles_conj = np.count_nonzero(final_mask)
    conj_area_pct = pixeles_conj / (h_e * w_e) if h_e * w_e > 0 else 0
    
    aspect = 0
    if pixeles_conj > 0:
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt_mayor = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(cnt_mayor)
            aspect = bw / max(bh, 1)

    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()

    return {"area": conj_area_pct, "aspect": aspect, "lap": lap_var}

def analyze():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../media/originales'))
    sitomar = get_image_paths(os.path.join(os.path.dirname(__file__), '../media/sitomar.txt'), base_dir)
    notomar = get_image_paths(os.path.join(os.path.dirname(__file__), '../media/notomar.txt'), base_dir)
    
    extractor = ConjuntivaExtractor()
    
    print("--- SI TOMAR ---")
    for name, p in sitomar:
        img = cv2.imread(p)
        if img is None: continue
        m = get_metrics(img, extractor)
        if m:
            print(f"SI {name} -> Area: {m['area']:.5f}, Aspect: {m['aspect']:.3f}, Lap: {m['lap']:.1f}")
            
    print("\n--- NO TOMAR ---")
    for name, p in notomar:
        img = cv2.imread(p)
        if img is None: continue
        m = get_metrics(img, extractor)
        if m:
            print(f"NO {name} -> Area: {m['area']:.5f}, Aspect: {m['aspect']:.3f}, Lap: {m['lap']:.1f}")

if __name__ == "__main__":
    analyze()
