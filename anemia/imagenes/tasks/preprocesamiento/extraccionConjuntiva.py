import os
import cv2
import glob
import numpy as np
from .core.extractor import ConjuntivaExtractor

def segmentar_y_recortar_conjuntiva(ruta_entrada, ruta_salida_segmentadas, ruta_salida_recortadas, ruta_salida_png, ruta_salida_area=None):
    """
    PROCESO DE SEGMENTACIÓN ORIGINAL: Restaurado al estado funcional pre-modularización.
    """
    extractor = ConjuntivaExtractor()
    categorias = ['SIN ANEMIA', 'CON ANEMIA']
    
    if ruta_salida_area is None:
        ruta_salida_area = os.path.join(os.path.dirname(ruta_salida_segmentadas), 'area')

    for cat in categorias:
        os.makedirs(os.path.join(ruta_salida_segmentadas, cat), exist_ok=True)
        os.makedirs(os.path.join(ruta_salida_recortadas, cat), exist_ok=True)
        os.makedirs(os.path.join(ruta_salida_png, cat), exist_ok=True)
        ruta_area = os.path.join(ruta_salida_area, cat)
        os.makedirs(ruta_area, exist_ok=True)

        for f in glob.glob(os.path.join(ruta_entrada, cat, '*.[jJ][pP][gG]')) + \
                 glob.glob(os.path.join(ruta_entrada, cat, '*.[jJ][pP][eE][gG]')):
            img = cv2.imread(f)
            if img is None: continue
            
            # Lógica original: Procesamiento directo sobre la imagen de entrada
            anchor, radius = extractor.detect_eye_anchor(img)
            win_mask, _, _ = extractor.get_search_window(img, anchor, radius)
            
            raw_mask = extractor.find_medialuna_by_contrast(img, win_mask, anchor, radius)
            final_mask = extractor.polish_final(raw_mask)
            final_mask = extractor.cerrar_forma_medialuna(final_mask, anchor, radius)

            if np.count_nonzero(final_mask) == 0:
                print(f"DEBUG: Mascara vacia para {f}.")
                continue
            
            name = os.path.basename(f)
            
            # Dibujo de contorno para validación
            img_area = img.copy()
            cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_area, cnts, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(ruta_area, name), img_area)
            
            # Segmentados y Recortes
            x, y, w, h = cv2.boundingRect(final_mask)
            cv2.imwrite(os.path.join(ruta_salida_segmentadas, cat, name), final_mask)
            cv2.imwrite(os.path.join(ruta_salida_recortadas, cat, name), img[y:y+h, x:x+w])
            
            # PNG con transparencia
            rgba = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = final_mask[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(ruta_salida_png, cat, os.path.splitext(name)[0] + '.png'), rgba)

    print(f"Proceso de segmentación restaurado y completado.")
