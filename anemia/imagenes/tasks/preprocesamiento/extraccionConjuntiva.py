# imagenes/tasks/preprocesamiento/extraccionConjuntiva.py

import os
import cv2
import numpy as np
import glob
from datetime import datetime
from .segmentacion_ojo import segment_image_with_unet, crop_segmented_image
# from modelo.tasks.config import MODELO_UNET_PATH  # Removido por instrucción del usuario

def segmentar_y_recortar_conjuntiva(ruta_entrada, ruta_salida_segmentadas, ruta_salida_recortadas, ruta_salida_png):
    categorias = ['SIN ANEMIA', 'CON ANEMIA']

    for categoria in categorias:
        os.makedirs(os.path.join(ruta_salida_segmentadas, categoria), exist_ok=True)
        os.makedirs(os.path.join(ruta_salida_recortadas, categoria), exist_ok=True)
        os.makedirs(os.path.join(ruta_salida_png, categoria), exist_ok=True)
        # Nueva carpeta para la imagen de área sobre el original
        ruta_salida_area = os.path.join(os.path.dirname(ruta_salida_segmentadas), 'area')
        os.makedirs(os.path.join(ruta_salida_area, categoria), exist_ok=True)

    start = datetime.now()

    def detect_iris(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 11)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,
                                  param1=100, param2=30, minRadius=30, maxRadius=150)
        if circles is not None:
            return circles[0][0]
        return None

    def region_grow_lab(img, seed_mask, tolerance=20, max_iter=15):
        """
        Desde la máscara semilla, expande píxel a píxel hacia vecinos cuyo
        color LAB esté dentro de 'tolerance' de la media LAB de la semilla.
        Triple guardia: techo espacial + veto absoluto de blanco + distancia LAB.
        """
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        current = seed_mask.copy()
        h_img, w_img = img.shape[:2]

        # ── 1. TECHO ESPACIAL: la conjuntiva NO está encima de la semilla ──────
        # El growing no puede subir más arriba del borde superior del seed.
        seed_rows = np.argwhere(seed_mask == 255)[:, 0]
        seed_top  = int(seed_rows.min()) if len(seed_rows) else 0
        # Pequeño margen hacia arriba (10px) para no cortar bordes reales
        y_ceiling = max(0, seed_top - 10)

        # Máscara espacial: solo filas >= y_ceiling
        spatial_ok = np.zeros((h_img, w_img), dtype=np.uint8)
        spatial_ok[y_ceiling:, :] = 255

        # ── 2. VETO ABSOLUTO: esclerótica blanca ──────────────────────────────
        # L > 175 (brillante) Y canal-a < 145 (sin rojo) = blanco esclerótica
        L_ch = lab[:, :, 0]
        a_ch = lab[:, :, 1]
        sclera_veto = ((L_ch > 175) & (a_ch < 145)).astype(np.uint8) * 255

        # Máscara combinada de zonas bloqueadas
        blocked = cv2.bitwise_or(sclera_veto, cv2.bitwise_not(spatial_ok))

        for _ in range(max_iter):
            dilated = cv2.dilate(current, np.ones((3, 3), np.uint8), iterations=1)
            border  = cv2.bitwise_and(dilated, cv2.bitwise_not(current))

            # Aplicar ambas restricciones al borde candidato
            border = cv2.bitwise_and(border, cv2.bitwise_not(blocked))

            if np.count_nonzero(border) == 0:
                break

            pixels_cur = lab[current == 255]
            if len(pixels_cur) == 0:
                break
            mean_lab = pixels_cur.mean(axis=0)

            border_coords = np.argwhere(border == 255)
            if len(border_coords) == 0:
                break

            border_lab = lab[border_coords[:, 0], border_coords[:, 1]]
            dist       = np.linalg.norm(border_lab - mean_lab, axis=1)
            accepted   = border_coords[dist <= tolerance]

            if len(accepted) == 0:
                break

            current[accepted[:, 0], accepted[:, 1]] = 255

        # ── 3. LIMPIEZA FINAL: eliminar residuos blancos ───────────────────────
        current = cv2.bitwise_and(current, cv2.bitwise_not(blocked))
        return current

    def heuristic_segmentation(img):
        high, wide = img.shape[:2]
        iris = detect_iris(img)
        iris_y = high * 0.4 if iris is None else iris[1]
        iris_r = 60 if iris is None else iris[2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask_not_black = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ── SEMILLA: rojo saturado seguro ─────────────────────────────────────
        # Alta saturación → solo tejido rojo real, sin piel ni esclerótica
        l1, u1 = np.array([0,   120, 40]), np.array([10,  255, 255])
        l2, u2 = np.array([168, 120, 40]), np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, l1, u1) | cv2.inRange(hsv, l2, u2)
        mask_seed_raw = cv2.bitwise_and(mask_red, mask_not_black)

        # Limpiar semilla (quitar ruido pequeño, preservar forma)
        k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_seed = cv2.morphologyEx(mask_seed_raw, cv2.MORPH_OPEN,  k_open)
        mask_seed = cv2.morphologyEx(mask_seed,     cv2.MORPH_CLOSE, k_close)

        # ── ELEGIR BLOB SEMILLA correcto espacialmente ─────────────────────────
        contours, _ = cv2.findContours(mask_seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_blobs = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800: continue

            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            cx, cy   = x + w_cnt / 2, y + h_cnt / 2
            aspect   = float(w_cnt) / max(h_cnt, 1)
            dist_v   = cy - iris_y
            dist_h   = abs(cx - wide / 2)

            in_v = iris_r * 0.4 < dist_v < iris_r * 2.8
            in_h = dist_h < wide * 0.38
            horiz = aspect > 1.1

            if in_v and in_h and horiz:
                score = area * (1.0 / (dist_v + 1))
                valid_blobs.append((cnt, score))

        if not valid_blobs:
            return np.zeros(img.shape[:2], dtype=np.uint8)

        best_cnt = max(valid_blobs, key=lambda b: b[1])[0]

        # Máscara semilla = solo el blob ganador
        seed_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(seed_mask, [best_cnt], -1, 255, -1)

        # ── REGION GROWING desde semilla ───────────────────────────────────────
        # tolerance=20: captura rosado/pálido anemia sin saltar a esclerótica
        grown_mask = region_grow_lab(img, seed_mask, tolerance=20, max_iter=15)

        # Limpieza final: cerrar huecos residuales del crecimiento
        k_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        grown_mask = cv2.morphologyEx(grown_mask, cv2.MORPH_CLOSE, k_final)

        # ── FILTRO SUAVIZADO EXTREMO: quitar picos y huecos ──────────────────
        # 1. CLOSE muy grande para rellenar ese hueco/mordida superior
        k_close_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        grown_mask = cv2.morphologyEx(grown_mask, cv2.MORPH_CLOSE, k_close_big)

        # 2. OPEN grande para cortar picos sobresalientes
        k_open_big = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        grown_mask = cv2.morphologyEx(grown_mask, cv2.MORPH_OPEN, k_open_big)
        
        # 3. Difuminar fuerte y binarizar para curva perfecta
        grown_mask = cv2.GaussianBlur(grown_mask, (25, 25), 0)
        _, grown_mask = cv2.threshold(grown_mask, 127, 255, cv2.THRESH_BINARY)

        return grown_mask


    for categoria in categorias:
        ruta_imgs = glob.glob(os.path.join(ruta_entrada, categoria, '*.jpeg')) + \
                    glob.glob(os.path.join(ruta_entrada, categoria, '*.jpg'))

        for ruta_img in ruta_imgs:
            img_original = cv2.imread(ruta_img)
            if img_original is None: continue

            mask_conjuntiva = heuristic_segmentation(img_original)
            x, y, w, h = cv2.boundingRect(mask_conjuntiva)
            if np.count_nonzero(mask_conjuntiva) == 0: continue
            
            img_cropped = img_original[y:y+h, x:x+w]
            nombre_img = os.path.basename(ruta_img)
            
            # Guardar Área (Original + Contorno verde)
            img_area = img_original.copy()
            cnts, _ = cv2.findContours(mask_conjuntiva, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_area, cnts, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(ruta_salida_area, categoria, nombre_img), img_area)

            cv2.imwrite(os.path.join(ruta_salida_segmentadas, categoria, nombre_img), mask_conjuntiva)
            cv2.imwrite(os.path.join(ruta_salida_recortadas, categoria, nombre_img), img_cropped)

            # PNG Transparente
            x, y, w, h = cv2.boundingRect(mask_conjuntiva)
            if w > 0 and h > 0:
                cropped_rgba = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2BGRA)
                alpha_mask = mask_conjuntiva[y:y+h, x:x+w]
                if alpha_mask.shape != (h, w):
                    alpha_mask = cv2.resize(alpha_mask, (w, h))
                
                # Asegurar que el recorte y la máscara tengan el mismo tamaño
                cropped_rgba = cropped_rgba[:h, :w]
                cropped_rgba[:, :, 3] = alpha_mask
                
                nombre_sin_ext = os.path.splitext(nombre_img)[0]
                ruta_png = os.path.join(ruta_salida_png, categoria, f"{nombre_sin_ext}.png")
                cv2.imwrite(ruta_png, cropped_rgba)

    print(f"✅ Segmentación completada en: {datetime.now() - start}")
