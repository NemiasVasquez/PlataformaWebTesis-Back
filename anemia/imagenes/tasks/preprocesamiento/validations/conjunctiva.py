import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

def validar_conjuntiva(img, extractor):
    """
    CONJUNTIVA VALIDA (ORIGINAL): Restaurado al 100% siguiendo old_filtrarImagenes.py.
    Asegura que el área, forma y posición se midan sobre el recorte del ojo.
    """
    anchor, radius = extractor.detect_eye_anchor(img)
    
    # Recorte enfocado (Igual que en la versión original)
    img_ojo, anchor_ojo, (off_x, off_y) = extractor.crop_to_eye(img, anchor, radius)
    h_o, w_o = img_ojo.shape[:2]

    win_mask, _, _ = extractor.get_search_window(img_ojo, anchor_ojo, radius)
    raw_mask = extractor.find_medialuna_by_contrast(img_ojo, win_mask, anchor_ojo, radius)
    final_mask = extractor.polish_final(raw_mask)
    final_mask = extractor.cerrar_forma_medialuna(final_mask, anchor_ojo, radius)

    pixeles_conjuntiva = np.count_nonzero(final_mask)
    encontrada = pixeles_conjuntiva > 0

    # Máscara mapeada al tamaño original
    full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    full_mask[off_y:off_y+h_o, off_x:off_x+w_o] = final_mask

    forma_valida = False
    posicion_valida = False
    pestanas_ok = True
    
    # Porcentaje de área sobre el RECORTE (como en la versión original)
    porcentaje = pixeles_conjuntiva / (h_o * w_o)
    
    if encontrada:
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cnt_mayor = max(cnts, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(cnt_mayor)
            aspect = bw / max(bh, 1)
            
            # --- VALIDACIÓN DE FORMA (HARDCODE 0.15 COINCIDE CON ORIGINAL) ---
            min_aspect = float(os.getenv("CONJUNTIVA_MIN_ASPECT_RATIO", 1.5))
            #bw >= w_o * 0.15 era el fallback hardcodificado en el original
            forma_valida = (aspect >= min_aspect) and (bw >= w_o * 0.15)
            
            # --- POSICIÓN ---
            cx_ojo, cy_ojo = anchor_ojo
            cx_conj = bx + bw / 2
            esta_debajo = (by + bh / 2) > (cy_ojo - radius * 0.5)
            alineada = abs(cx_ojo - cx_conj) < (radius * 1.8) 
            
            # Toca piso (limite 5% inferior del recorte)
            margen_piso = int(h_o * 0.05)
            toca_piso = (by + bh) > (h_o - margen_piso)
            posicion_valida = esta_debajo and alineada and not toca_piso

            # --- PESTAÑAS ---
            lab = cv2.cvtColor(img_ojo, cv2.COLOR_BGR2LAB)
            l_mask = lab[:, :, 0][final_mask > 0]
            if len(l_mask) > 0:
                pct_oscuro = np.sum(l_mask < 50) / len(l_mask)
                max_pestanas = float(os.getenv("CONJUNTIVA_MAX_PESTANAS_PCT", 0.08))
                pestanas_ok = pct_oscuro <= max_pestanas

    return encontrada, porcentaje, full_mask, forma_valida, posicion_valida, pestanas_ok
