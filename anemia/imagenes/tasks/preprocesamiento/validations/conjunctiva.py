import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

def validar_conjuntiva(img, extractor):
    """
    CONJUNTIVA VALIDA: Recorta al ojo primero para que el cálculo de área
    sea relativo a la zona de interés y no a toda la foto.
    """
    anchor, radius = extractor.detect_eye_anchor(img)
    if anchor is None:
        return False, 0.0, np.zeros(img.shape[:2], dtype=np.uint8), False, False, True

    # RECORTE DE OJO para validación precisa
    eye_img, new_anchor, (x0, y0) = extractor.crop_to_eye(img, anchor, radius)
    h_e, w_e = eye_img.shape[:2]

    # Buscar la conjuntiva en el recorte
    win_mask, _, _ = extractor.get_search_window(eye_img, new_anchor, radius)
    raw_mask = extractor.find_medialuna_by_contrast(eye_img, win_mask, new_anchor, radius)
    final_mask = extractor.polish_final(raw_mask)
    final_mask = extractor.cerrar_forma_medialuna(final_mask, new_anchor, radius)

    pixeles_conjuntiva = np.count_nonzero(final_mask)
    encontrada = pixeles_conjuntiva > 0

    # Máscara completa para compatibilidad con el resto del pipeline (debug/nitidez)
    h_o, w_o = img.shape[:2]
    full_mask = np.zeros((h_o, w_o), dtype=np.uint8)
    if encontrada:
        h_m, w_m = final_mask.shape[:2]
        full_mask[y0:y0+h_m, x0:x0+w_m] = final_mask

    forma_valida = False
    posicion_valida = True 
    pestanas_ok = True
    
    # Porcentaje de área sobre el RECORTE DEL OJO (más justo)
    porcentaje = pixeles_conjuntiva / (h_e * w_e)
    
    if encontrada:
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt_mayor = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(cnt_mayor)
            aspect = bw / max(bh, 1)
            
            # --- VALIDACIÓN DE FORMA ---
            min_aspect = float(os.getenv("CONJUNTIVA_MIN_ASPECT_RATIO", 1.2))
            forma_valida = (aspect >= min_aspect) and (bw >= w_e * 0.20)
            
            # --- PESTAÑAS (sobre el recorte) ---
            lab_eye = cv2.cvtColor(eye_img, cv2.COLOR_BGR2LAB)
            l_mask = lab_eye[:, :, 0][final_mask > 0]
            if len(l_mask) > 0:
                pct_oscuro = np.sum(l_mask < 50) / len(l_mask)
                max_pestanas = float(os.getenv("CONJUNTIVA_MAX_PESTANAS_PCT", 0.15))
                pestanas_ok = pct_oscuro <= max_pestanas

    return encontrada, porcentaje, full_mask, forma_valida, posicion_valida, pestanas_ok
