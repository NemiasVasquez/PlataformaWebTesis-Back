import os
import cv2
import numpy as np
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

def simular_filtros(img_path, vmlap_thresh, area_thresh, ancho_thresh):
    img = cv2.imread(img_path)
    if img is None: return "ERROR_LECTURA"
    
    h, w = img.shape[:2]
    area_img = h * w
    
    anchor, radius = extractor.detect_eye_anchor(img)
    win_mask, _, _ = extractor.get_search_window(img, anchor, radius)
    raw_mask = extractor.find_medialuna_by_contrast(img, win_mask, anchor, radius)
    final_mask = extractor.polish_final(raw_mask)
    final_mask = extractor.cerrar_forma_medialuna(final_mask)
    
    pixeles = np.count_nonzero(final_mask)
    area_pct = pixeles / area_img
    
    if area_pct < area_thresh: return "RECHAZO_AREA"
    
    cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return "RECHAZO_CONJUNTIVA"
    
    cnt_mayor = max(cnts, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(cnt_mayor)
    ancho_frac = bw / w
    
    if ancho_frac < ancho_thresh: return "RECHAZO_ANCHO"
    
    if pixeles > 200:
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ys, xs = np.where(final_mask > 0)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        patch = gris[y0:y1+1, x0:x1+1]
        patch_sx = cv2.Sobel(patch, cv2.CV_64F, 2, 0, ksize=3)
        patch_sy = cv2.Sobel(patch, cv2.CV_64F, 0, 2, ksize=3)
        vmlap_mask = (np.abs(patch_sx) + np.abs(patch_sy)).var()
        if vmlap_mask < vmlap_thresh: return "RECHAZO_NITIDEZ"
    else:
        return "RECHAZO_AREA_PIXELES"
        
    return "ACEPTADA"

def main():
    malas = leer_malas('media/notomar.txt')
    base_dir = 'media/originales/CON ANEMIA'
    
    vmlap_t = 150
    area_t = 0.015
    ancho_t = 0.20
    
    resultados = {"ACEPTADA": 0}
    for m in malas:
        res = simular_filtros(os.path.join(base_dir, m), vmlap_t, area_t, ancho_t)
        resultados[res] = resultados.get(res, 0) + 1
        
    print(f"Total malas probadas: {len(malas)}")
    for k, v in resultados.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
