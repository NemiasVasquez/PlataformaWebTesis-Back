# imagenes/tasks/preprocesamiento/filtrarImagenes.py

import os
import cv2
import numpy as np
import glob
import shutil
from dotenv import load_dotenv
from .extraccionConjuntiva import ConjuntivaExtractor

load_dotenv()

def filtrar_conjuntiva(ruta_entrada, ruta_salida, ruta_no_filtrados, ruta_reporte_txt):
    categorias = ['SIN ANEMIA', 'CON ANEMIA']
    razones_rechazo = {
        "Ojo no está abierto (no se detecta iris adecuado)": "Ojo cerrado",
        "No se detecta conjuntiva": "Sin conjuntiva",
        "Efecto Blur detectado": "Efecto Blur",
        "Tamaño insuficiente": "Tamaño insuficiente",
        "No se detecta esclerótica": "Sin esclerótica",
        "Area conjuntiva insuficiente (muy pequeña o mala forma)": "Area insuficiente",
        "Posición conjuntiva incorrecta (no debajo del ojo)": "Posicion incorrecta",
        "Pestañas tapando la conjuntiva": "Bloqueo pestanas",
    }

    for categoria in categorias:
        os.makedirs(os.path.join(ruta_salida, categoria), exist_ok=True)
        for razon in razones_rechazo.values():
            os.makedirs(os.path.join(ruta_no_filtrados, categoria, razon), exist_ok=True)

    with open(ruta_reporte_txt, 'w', encoding='utf-8') as f:
        f.write("Reporte de imagenes rechazadas y razones:\n\n")

    # Instancia reutilizable del extractor real
    extractor = ConjuntivaExtractor()

    # Metricas
    contador_total = contador_filtradas = contador_sin_iris = 0
    contador_sin_conjuntiva = contador_desenfoque = 0
    contador_tamano_insuficiente = contador_sin_esclerotica = 0
    contador_area_insuficiente = 0
    contador_mala_posicion = contador_pestanas = 0

    # -------------------------------------------------------------------------
    # Leer umbrales desde .env (con valores actuales como fallback)
    # -------------------------------------------------------------------------
    # --- Nitidez ---
    NITIDEZ_UMBRAL_LAP           = float(os.getenv("NITIDEZ_UMBRAL_LAP",           3))
    NITIDEZ_UMBRAL_TENENGRAD     = float(os.getenv("NITIDEZ_UMBRAL_TENENGRAD",     3))
    NITIDEZ_UMBRAL_HF_RATIO      = float(os.getenv("NITIDEZ_UMBRAL_HF_RATIO",      0.005))
    NITIDEZ_UMBRAL_VMLAP_CENTRO  = float(os.getenv("NITIDEZ_UMBRAL_VMLAP_CENTRO",  3))
    NITIDEZ_UMBRAL_VMLAP_MASCARA = float(os.getenv("NITIDEZ_UMBRAL_VMLAP_MASCARA", 3))
    NITIDEZ_FFT_RADIO_FRACCION   = float(os.getenv("NITIDEZ_FFT_RADIO_FRACCION",   0.10))
    NITIDEZ_ROI_MARGEN           = float(os.getenv("NITIDEZ_ROI_MARGEN",           0.30))
    NITIDEZ_MIN_PIXELES_MASCARA  = int(os.getenv("NITIDEZ_MIN_PIXELES_MASCARA",    200))

    # --- Ojo abierto ---
    OJO_MIN_AREA_FRACCION        = float(os.getenv("OJO_MIN_AREA_FRACCION",        0.005))

    # --- Esclerótica ---
    ESCLEROTICA_UMBRAL_AREA      = float(os.getenv("ESCLEROTICA_UMBRAL_AREA",      0.001))

    # --- Tamaño mínimo imagen ---
    TAMANO_MIN_PX                = int(os.getenv("TAMANO_MIN_PX",                  200))

    # --- Conjuntiva (forma y área) ---
    CONJUNTIVA_MIN_AREA_PCT      = float(os.getenv("CONJUNTIVA_MIN_AREA_PCT",      0.01))
    CONJUNTIVA_MIN_ASPECT_RATIO  = float(os.getenv("CONJUNTIVA_MIN_ASPECT_RATIO",  1.5))
    CONJUNTIVA_MIN_ANCHO_FRACCION= float(os.getenv("CONJUNTIVA_MIN_ANCHO_FRACCION",0.08))

    # -------------------------------------------------------------------------
    # Funciones auxiliares
    # -------------------------------------------------------------------------

    def es_nitida(img, mask_conjuntiva=None):
        """
        5 metricas de nitidez:
        1. Laplacian variance global.
        2. Tenengrad global.
        3. Ratio altas frecuencias FFT.
        4. VMLAP en ROI central.
        5. VMLAP dentro de la mascara de conjuntiva (si se provee).
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
        r = int(min(h, w) * NITIDEZ_FFT_RADIO_FRACCION)
        y_grid, x_grid = np.ogrid[:h, :w]
        mask_low = (y_grid - cy)**2 + (x_grid - cx)**2 <= r**2
        hf_ratio = np.sum(mag[~mask_low]) / (np.sum(mag) + 1e-8)

        # 4. VMLAP ROI central
        margen_y = int(h * NITIDEZ_ROI_MARGEN)
        margen_x = int(w * NITIDEZ_ROI_MARGEN)
        roi = gris[margen_y: h - margen_y, margen_x: w - margen_x]
        roi_sobelx = cv2.Sobel(roi, cv2.CV_64F, 2, 0, ksize=3)
        roi_sobely = cv2.Sobel(roi, cv2.CV_64F, 0, 2, ksize=3)
        vmlap_centro = (np.abs(roi_sobelx) + np.abs(roi_sobely)).var()

        pasa = (lap_var > NITIDEZ_UMBRAL_LAP and tenengrad > NITIDEZ_UMBRAL_TENENGRAD
                and hf_ratio > NITIDEZ_UMBRAL_HF_RATIO and vmlap_centro > NITIDEZ_UMBRAL_VMLAP_CENTRO)

        # 5. VMLAP sobre la mascara de conjuntiva
        if pasa and mask_conjuntiva is not None and np.count_nonzero(mask_conjuntiva) > 0:
            pixeles_conjuntiva = gris[mask_conjuntiva > 0]
            if len(pixeles_conjuntiva) > NITIDEZ_MIN_PIXELES_MASCARA:
                ys, xs = np.where(mask_conjuntiva > 0)
                y0, y1 = ys.min(), ys.max()
                x0, x1 = xs.min(), xs.max()
                patch = gris[y0:y1+1, x0:x1+1]
                patch_sx = cv2.Sobel(patch, cv2.CV_64F, 2, 0, ksize=3)
                patch_sy = cv2.Sobel(patch, cv2.CV_64F, 0, 2, ksize=3)
                vmlap_mask = (np.abs(patch_sx) + np.abs(patch_sy)).var()
                pasa = pasa and (vmlap_mask > NITIDEZ_UMBRAL_VMLAP_MASCARA)

        return pasa


    def ojo_abierto(img):
        anchor, radius = extractor.detect_eye_anchor(img)
        h, w = img.shape[:2]
        area_img = h * w
        area_iris = np.pi * radius**2
        return area_iris > area_img * OJO_MIN_AREA_FRACCION


    def contiene_esclerotica(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        return np.count_nonzero(mask_clean) / (img.shape[0] * img.shape[1]) > ESCLEROTICA_UMBRAL_AREA


    def tiene_tamano_suficiente(img):
        alto, ancho = img.shape[:2]
        return alto >= TAMANO_MIN_PX and ancho >= TAMANO_MIN_PX

    def conjuntiva_valida(img):
        anchor, radius = extractor.detect_eye_anchor(img)
        
        # CAVERNÍCOLA CENTRALIZA: Recortamos para ignorar basura de los bordes
        img_ojo, anchor_ojo, (off_x, off_y) = extractor.crop_to_eye(img, anchor, radius)
        
        h, w = img_ojo.shape[:2]
        area_ojo = h * w

        win_mask, _, _ = extractor.get_search_window(img_ojo, anchor_ojo, radius)
        raw_mask = extractor.find_medialuna_by_contrast(img_ojo, win_mask, anchor_ojo, radius)
        final_mask = extractor.polish_final(raw_mask)
        final_mask = extractor.cerrar_forma_medialuna(final_mask, anchor_ojo, radius)

        pixeles_conjuntiva = np.count_nonzero(final_mask)
        # El porcentaje se mide sobre el recorte del ojo, mas realista
        porcentaje = pixeles_conjuntiva / area_ojo
        encontrada = pixeles_conjuntiva > 0

        # Crear mascara del tamaño original para que el resto del codigo no rompa
        full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        full_mask[off_y:off_y+h, off_x:off_x+w] = final_mask

        forma_valida = False
        posicion_valida = False
        pestanas_ok = True
        
        if encontrada:
            cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                cnt_mayor = max(cnts, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(cnt_mayor)
                aspect = bw / max(bh, 1)
                
                # Forma en el recorte
                CONJ_MIN_ANCHO_FRAC = float(os.getenv("CONJUNTIVA_MIN_ANCHO_FRACCION", 0.08))
                forma_valida = (aspect >= CONJUNTIVA_MIN_ASPECT_RATIO) and (bw >= w * CON_MIN_ANCHO_FRAC if 'CON_MIN_ANCHO_FRAC' in locals() else bw >= w * 0.1)
                # Re-check variable names from .env
                forma_valida = (aspect >= CONJUNTIVA_MIN_ASPECT_RATIO) and (bw >= w * 0.15)
                
                # Validar posición (en el recorte, está centrado)
                cx_ojo, cy_ojo = anchor_ojo
                cx_conj = bx + bw / 2
                
                esta_debajo = (by + bh / 2) > (cy_ojo - radius * 0.5)
                # Ojo puede mirar a los lados. Tolerancia 1.8 radios.
                alineada = abs(cx_ojo - cx_conj) < (radius * 1.8) 
                
                # Solo falla si toca piso de forma extrema (cuero de cachete)
                toca_piso = (by + bh) > (h - 1)
                posicion_valida = esta_debajo and alineada and not toca_piso

                # Validar pestañas
                lab = cv2.cvtColor(img_ojo, cv2.COLOR_BGR2LAB)
                l_mask = lab[:, :, 0][final_mask > 0]
                if len(l_mask) > 0:
                    pct_oscuro = np.sum(l_mask < 50) / len(l_mask)
                    max_pestanas = float(os.getenv("CONJUNTIVA_MAX_PESTANAS_PCT", 0.08))
                    pestanas_ok = pct_oscuro <= max_pestanas

        return encontrada, porcentaje, full_mask, forma_valida, posicion_valida, pestanas_ok

    # -------------------------------------------------------------------------
    # Procesamiento principal
    # -------------------------------------------------------------------------
    for categoria in categorias:
        imagenes = glob.glob(os.path.join(ruta_entrada, categoria, '*.jpeg'))

        for ruta_img in imagenes:
            img = cv2.imread(ruta_img)
            if img is None:
                print(f"No se pudo leer: {ruta_img}")
                continue

            contador_total += 1
            nombre_img = os.path.basename(ruta_img)

            # --- Evaluacion de filtros ---
            pasa_iris        = ojo_abierto(img)
            pasa_tamano      = tiene_tamano_suficiente(img)
            pasa_esclerotica = contiene_esclerotica(img)

            # conjuntiva_valida primero: necesitamos la mascara para es_nitida
            conjuntiva_encontrada, pct_area, mascara, forma_valida, posicion_valida, pestanas_ok = conjuntiva_valida(img)
            pasa_conjuntiva = conjuntiva_encontrada
            pasa_area       = pct_area >= CONJUNTIVA_MIN_AREA_PCT
            pasa_forma      = forma_valida
            pasa_posicion   = posicion_valida
            pasa_pestanas   = pestanas_ok

            # Nitidez evaluada tambien sobre la mascara real de conjuntiva
            pasa_desenfoque = es_nitida(img, mask_conjuntiva=mascara)

            razones = []
            if not pasa_iris:
                razones.append("Ojo no está abierto (no se detecta iris adecuado)")
                contador_sin_iris += 1
            if not pasa_conjuntiva:
                razones.append("No se detecta conjuntiva")
                contador_sin_conjuntiva += 1
            if not pasa_desenfoque:
                razones.append("Efecto Blur detectado")
                contador_desenfoque += 1
            if not pasa_tamano:
                razones.append("Tamaño insuficiente")
                contador_tamano_insuficiente += 1
            if not pasa_esclerotica:
                razones.append("No se detecta esclerótica")
                contador_sin_esclerotica += 1
            if pasa_conjuntiva and not pasa_area:
                razones.append("Area conjuntiva insuficiente (muy pequeña o mala forma)")
                contador_area_insuficiente += 1
            if pasa_conjuntiva and pasa_area and not pasa_forma:
                razones.append("Area conjuntiva insuficiente (muy pequeña o mala forma)")
                contador_area_insuficiente += 1
            if pasa_conjuntiva and pasa_area and pasa_forma and not pasa_posicion:
                razones.append("Posición conjuntiva incorrecta (no debajo del ojo)")
                contador_mala_posicion += 1
            if pasa_conjuntiva and pasa_area and pasa_forma and pasa_posicion and not pasa_pestanas:
                razones.append("Pestañas tapando la conjuntiva")
                contador_pestanas += 1

            if all([pasa_iris, pasa_conjuntiva, pasa_area, pasa_forma, pasa_posicion, pasa_pestanas,
                    pasa_desenfoque, pasa_tamano, pasa_esclerotica]):
                shutil.copy(ruta_img, os.path.join(ruta_salida, categoria, nombre_img))
                contador_filtradas += 1
            else:
                # Si hay mascara (aunque sea rechazada), dibujar contorno verde
                img_rechazo = img.copy()
                if conjuntiva_encontrada:
                    cnts, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_rechazo, cnts, -1, (0, 255, 0), 2)
                
                for razon in razones:
                    destino = os.path.join(
                        ruta_no_filtrados, categoria, razones_rechazo[razon], nombre_img
                    )
                    cv2.imwrite(destino, img_rechazo)
                
                with open(ruta_reporte_txt, 'a', encoding='utf-8') as f:
                    f.write(f"{nombre_img} ({categoria}): {', '.join(razones)}\n")

    print("\nMETRICAS DE FILTRADO FINAL:")
    print(f"Total imagenes procesadas : {contador_total}")
    print(f"Aceptadas                 : {contador_filtradas}")
    print(f"Rechazadas ojo cerrado    : {contador_sin_iris}")
    print(f"Rechazadas sin conjuntiva : {contador_sin_conjuntiva}")
    print(f"Rechazadas area <10%      : {contador_area_insuficiente}")
    print(f"Rechazadas por desenfoque : {contador_desenfoque}")
    print(f"Rechazadas por tamano     : {contador_tamano_insuficiente}")
    print(f"Rechazadas sin esclerotica: {contador_sin_esclerotica}")
    print(f"Reporte guardado en       : {ruta_reporte_txt}")
