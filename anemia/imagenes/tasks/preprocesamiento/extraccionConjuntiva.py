import os
import cv2
import glob
import numpy as np
from .core.extractor import ConjuntivaExtractor

def _toca_borde(mask, margen=5):
    """Detecta si la máscara toca los bordes de la imagen (línea recta = recorte insuficiente)."""
    h, w = mask.shape[:2]
    toca_izq = np.count_nonzero(mask[:, :margen]) > margen
    toca_der = np.count_nonzero(mask[:, w-margen:]) > margen
    toca_abajo = np.count_nonzero(mask[h-margen:, :]) > margen
    return toca_izq or toca_der or toca_abajo

def segmentar_y_recortar_conjuntiva(ruta_entrada, ruta_salida_segmentadas, ruta_salida_recortadas, ruta_salida_png, ruta_salida_area=None):
    """
    PROCESO DE SEGMENTACIÓN CON LOGS.
    """
    extractor = ConjuntivaExtractor()
    categorias = ['SIN ANEMIA', 'CON ANEMIA']
    
    if ruta_salida_area is None:
        ruta_salida_area = os.path.join(os.path.dirname(ruta_salida_segmentadas), 'area')

    print(f"===== INICIANDO SEGMENTACIÓN DE CONJUNTIVA =====")
    for cat in categorias:
        os.makedirs(os.path.join(ruta_salida_segmentadas, cat), exist_ok=True)
        os.makedirs(os.path.join(ruta_salida_recortadas, cat), exist_ok=True)
        os.makedirs(os.path.join(ruta_salida_png, cat), exist_ok=True)
        ruta_area = os.path.join(ruta_salida_area, cat)
        os.makedirs(ruta_area, exist_ok=True)

        archivos = glob.glob(os.path.join(ruta_entrada, cat, '*.[jJ][pP][gG]')) + \
                   glob.glob(os.path.join(ruta_entrada, cat, '*.[jJ][pP][eE][gG]'))
        print(f"--- Categoría {cat}: {len(archivos)} imágenes ---")

        for f in archivos:
            img = cv2.imread(f)
            if img is None: continue
            
            nombre = os.path.basename(f)
            
            # 1. DETECTAR ANCLA
            anchor, radius = extractor.detect_eye_anchor(img)
            
            # 2. RECORTE DE OJO (Centrado)
            eye_img, new_anchor, _ = extractor.crop_to_eye(img, anchor, radius)
            
            # 3. SEGMENTAR
            win_mask, _, _ = extractor.get_search_window(eye_img, new_anchor, radius)
            raw_mask = extractor.find_medialuna_by_contrast(eye_img, win_mask, new_anchor, radius)
            final_mask = extractor.polish_final(raw_mask)
            final_mask = extractor.cerrar_forma_medialuna(final_mask, new_anchor, radius)

            # 4. CAVERNÍCOLA: Si la conjuntiva toca el borde = recorte la cortó
            # Re-recortar con factor 1.5x más amplio y re-segmentar
            if np.count_nonzero(final_mask) > 0 and _toca_borde(final_mask):
                print(f"  [RE] {nombre}: Conjuntiva toca borde, ampliando recorte...")
                eye_img2, new_anchor2, _ = extractor.crop_to_eye(img, anchor, radius, factor_extra=1.5)
                win_mask2, _, _ = extractor.get_search_window(eye_img2, new_anchor2, radius)
                raw_mask2 = extractor.find_medialuna_by_contrast(eye_img2, win_mask2, new_anchor2, radius)
                final_mask2 = extractor.polish_final(raw_mask2)
                final_mask2 = extractor.cerrar_forma_medialuna(final_mask2, new_anchor2, radius)
                
                if np.count_nonzero(final_mask2) > 0:
                    eye_img = eye_img2
                    new_anchor = new_anchor2
                    final_mask = final_mask2

            if np.count_nonzero(final_mask) == 0:
                print(f"  [X] {nombre}: No se encontró conjuntiva roja")
                continue
            
            print(f"  [OK] {nombre}: Conjuntiva extraída")
            
            # 5. GUARDAR
            cv2.imwrite(os.path.join(ruta_salida_segmentadas, cat, nombre), final_mask)
            cv2.imwrite(os.path.join(ruta_salida_recortadas, cat, nombre), eye_img)
            
            rgba = cv2.cvtColor(eye_img, cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = final_mask
            cv2.imwrite(os.path.join(ruta_salida_png, cat, os.path.splitext(nombre)[0] + '.png'), rgba)
            
            img_area = eye_img.copy()
            cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_area, cnts, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(ruta_area, nombre), img_area)

    print(f"===== SEGMENTACIÓN FINALIZADA =====\n")

