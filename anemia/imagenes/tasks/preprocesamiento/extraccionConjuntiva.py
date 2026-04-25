# imagenes/tasks/preprocesamiento/extraccionConjuntiva.py

import os
import cv2
import numpy as np
import glob
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class ConjuntivaExtractor:
    def __init__(self):
        # CLAHE para mejorar contraste local del canal A (rojo)
        clip = float(os.getenv("SEG_CLAHE_CLIP", 3.0))
        grid = int(os.getenv("SEG_CLAHE_GRID", 8))
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))

    def crop_to_eye(self, img, center, radius):
        """
        CAVERNÍCOLA RECORTA: Quita lo que no es ojo.
        Deja margen para que la conjuntiva respire y no toque la pared de piedra.
        """
        h, w = img.shape[:2]
        cx, cy = center
        
        # Margen generoso: 4 radios a cada lado, 2 arriba, 4 abajo
        x0 = max(0, cx - int(radius * 4.5))
        x1 = min(w, cx + int(radius * 4.5))
        y0 = max(0, cy - int(radius * 2.5))
        y1 = min(h, cy + int(radius * 4.5))
        
        cropped = img[y0:y1, x0:x1]
        new_center = (cx - x0, cy - y0)
        return cropped, new_center, (x0, y0)

    def detect_eye_anchor(self, img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Suavizado para reducir ruido en detección de círculos
        blurred = cv2.medianBlur(gray, 11)
        
        # Intentamos detectar el iris (círculo)
        dp = int(os.getenv("IRIS_HOUGH_DP", 1))
        min_dist = int(os.getenv("IRIS_HOUGH_DIST", 50))
        p1 = int(os.getenv("IRIS_HOUGH_P1", 100))
        p2 = int(os.getenv("IRIS_HOUGH_P2", 30))
        min_r = int(os.getenv("IRIS_HOUGH_MIN_R", 30))
        max_r = int(os.getenv("IRIS_HOUGH_MAX_R", 150))
        
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, min_dist,
                                  param1=p1, param2=p2, minRadius=min_r, maxRadius=max_r)
        if circles is not None:
            c = circles[0][0]
            return (int(c[0]), int(c[1])), int(c[2])
        
        # Si falla Hough, buscamos el centro de masa de la zona más oscura (Pupila/Iris)
        dark_thres = int(os.getenv("IRIS_DARK_THRES", 55))
        _, dark_mask = cv2.threshold(gray, dark_thres, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            best_dark = max(cnts, key=cv2.contourArea)
            min_area = int(os.getenv("IRIS_MIN_AREA", 400))
            if cv2.contourArea(best_dark) > min_area:
                M = cv2.moments(best_dark)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    return (cx, cy), int(w * 0.15)
                    
        return (w // 2, int(h * 0.45)), int(w * 0.15)

    def get_search_window(self, img, center, radius):
        h, w = img.shape[:2]
        cx, cy = center
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Ojo está aquí. No buscar en orejas ni pómulo bajo.
        w_box = int(radius * 10)
        
        y_start_factor = float(os.getenv("SEG_SEARCH_Y_START_FACTOR", 0.5))
        y_start = max(0, cy - int(radius * y_start_factor)) 
        # La conjuntiva no baja más de 3 radios del centro del iris
        y_end = min(h, cy + int(radius * 3.5))
        x_start = max(0, cx - w_box // 2)
        x_end = min(w, cx + w_box // 2)
        
        cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 255, -1)
        return mask, y_start, y_end

    def find_medialuna_by_contrast(self, img, window_mask, anchor, radius, rojo_offset=0):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a_ch, b_ch = cv2.split(lab)
        (ax, ay) = anchor 

        # 1. QUITAR (Piel y Pelo)
        pelo_l = int(os.getenv("SEG_PELO_L_THRES", 75))
        _, mask_no_pelo = cv2.threshold(L, pelo_l, 255, cv2.THRESH_BINARY)
        mask_no_piel = cv2.compare(a_ch, b_ch, cv2.CMP_GT) # Rojo gana a Amarillo

        # 2. BUSCAR BLANCO (Esclerótica)
        blanco_l = int(os.getenv("SEG_BLANCO_L_THRES", 185))
        _, mask_blanco = cv2.threshold(L, blanco_l, 255, cv2.THRESH_BINARY)
        mask_blanco = cv2.morphologyEx(mask_blanco, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
        
        # 3. BUSCAR CARNE ROJA (Mucosa) con offset adaptativo
        a_fuerte = self.clahe.apply(a_ch)
        rojo_a = int(os.getenv("SEG_ROJO_A_THRES", 138)) - rojo_offset
        _, mask_rojo = cv2.threshold(a_fuerte, rojo_a, 255, cv2.THRESH_BINARY)

        # 4. LIMPIAR MUCHO
        combined = cv2.bitwise_and(mask_rojo, mask_no_pelo)
        combined = cv2.bitwise_and(combined, mask_no_piel)
        combined = cv2.bitwise_and(combined, window_mask)
        
        # Blanco no es Carne (pero ojo con la palidez extrema)
        # Solo restamos blanco que esté MUY arriba o sea muy grande
        combined = cv2.subtract(combined, mask_blanco) 

        # 5. LÍMITE PIEDRA (No subir)
        y_wall_factor = float(os.getenv("SEG_Y_WALL_FACTOR", 0.8))
        y_pared = ay + int(radius * y_wall_factor)
        combined[0:y_pared, :] = 0
        
        # 6. VETO DE PIEL PROFUNDA (No bajar demasiado)
        # La conjuntiva no suele estar a más de 2.5 radios del CENTRO del iris
        # CAVERNÍCOLA CORTA PIEL: Si muy abajo, es cachete de mamut, no ojo.
        y_piso = ay + int(radius * 2.5)
        combined[y_piso:, :] = 0
        
        # --- MORFOLOGÍA ORIENTADA A MEDIALUNA ---
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_clean)

        # Cierre horizontal para unir pedazos de mucosa pálida
        kernel_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3)) 
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close_h)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if rojo_offset == 0:
                return self.find_medialuna_by_contrast(img, window_mask, anchor, radius, rojo_offset=15)
            # Reintento desesperado
            return np.zeros_like(L)

        best_mask = np.zeros_like(L)
        best_score = -1
        best_cnt = None

        min_area_seg = int(os.getenv("SEG_MIN_AREA", 500))
        min_aspect_pre = float(os.getenv("SEG_MIN_ASPECT_PRE", 1.1))
        min_aspect_cave = float(os.getenv("SEG_MIN_ASPECT_CAVE", 1.3))
        y_target_factor = float(os.getenv("SEG_Y_TARGET_FACTOR", 1.25))
        solidity_target = float(os.getenv("SEG_SOLIDITY_TARGET", 0.65))

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_seg: 
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(h, 1)
            cx_cnt = x + w // 2
            cy_cnt = y + h // 2
            
            # Penalizar fuertemente si no es horizontal
            if aspect_ratio < min_aspect_pre: 
                continue

            # Priorizar la "cercanía radial": la conjuntiva envuelve el iris
            dist_centro = np.sqrt((cx_cnt - ax)**2 + (cy_cnt - ay)**2)
            dist_factor = np.exp(-0.5 * ((dist_centro - radius * 1.6) / (radius * 0.5))**2)
            
            mask_cnt = np.zeros_like(L)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_vals = cv2.mean(lab, mask=mask_cnt)
            avg_a = mean_vals[1] 
            avg_b = mean_vals[2]

            # Skin bias detection: si tiene mucho amarillo (b) relativo al rojo (a), es probable piel
            skin_bias = 1.0
            if avg_b > avg_a - 2:
                skin_bias = 0.5

            dist_y = abs(cy_cnt - (ay + radius * y_target_factor)) / radius
            
            # SCORE MEJORADO
            score = area * (avg_a ** 2) * (aspect_ratio ** 1.5) * dist_factor * skin_bias * np.exp(-2.0 * dist_y)
            
            hull = cv2.convexHull(cnt)
            solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            score *= (1.0 - abs(solidity - solidity_target))
            
            # Penalización extra si el contorno es exageradamente ancho
            if w > radius * 5.5:
                score *= 0.01
            
            # Penalización si está muy a los lados
            dist_x = abs(cx_cnt - ax) / radius
            score *= np.exp(-0.5 * dist_x)

            if score > best_score:
                best_score = score
                best_cnt = cnt

        if best_cnt is not None:
            cv2.drawContours(best_mask, [best_cnt], -1, 255, -1)
            
            # --- SEGUNDO PASO: PULIDO DE PRECISIÓN ---
            x, y, w, h = cv2.boundingRect(best_cnt)
            roi_lab = lab[y:y+h, x:x+w]
            roi_mask = best_mask[y:y+h, x:x+w]
            
            a_roi = roi_lab[:, :, 1]
            b_roi = roi_lab[:, :, 2]
            
            refine_offset = int(os.getenv("SEG_REFINE_A_OFFSET", 5))
            _, local_red = cv2.threshold(a_roi, np.mean(a_roi) + refine_offset, 255, cv2.THRESH_BINARY)
            local_no_skin = cv2.compare(a_roi, b_roi, cv2.CMP_GT)
            
            refined_roi = cv2.bitwise_and(local_red, local_no_skin)
            refined_roi = cv2.bitwise_and(refined_roi, roi_mask)
            
            kernel_fino = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            refined_roi = cv2.morphologyEx(refined_roi, cv2.MORPH_OPEN, kernel_fino)
            
            best_mask = np.zeros_like(L)
            best_mask[y:y+h, x:x+w] = refined_roi
            
            # --- PASO TRES: SOLO LA GRAN CARNE ---
            nuevos_cnts, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if nuevos_cnts:
                ganador = max(nuevos_cnts, key=cv2.contourArea)
                best_mask = np.zeros_like(L)
                cv2.drawContours(best_mask, [ganador], -1, 255, -1)

        # REINTENTO SI EL AREA FINAL ES MUY POBRE (Pálida o Cortada)
        area_final = np.count_nonzero(best_mask)
        if area_final < int(os.getenv("SEG_MIN_AREA_PALIDA", 4500)) and rojo_offset == 0:
            return self.find_medialuna_by_contrast(img, window_mask, anchor, radius, rojo_offset=15)

        return best_mask

    def polish_final(self, mask):
        if np.count_nonzero(mask) == 0:
            return mask
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def cerrar_forma_medialuna(self, mask, anchor=None, radius=None):
        """
        CAVERNÍCOLA ANATÓMICO: Sigue la forma real pero aniquila picos.
        Conecta los pedazos sueltos llenando huecos blancos con líneas rectas, 
        y luego pasa rodillo (Gaussian Blur) para que fluya con el ojo.
        """
        if np.count_nonzero(mask) == 0:
            return mask

        h, w = mask.shape[:2]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return mask
            
        cnt_mayor = max(cnts, key=cv2.contourArea)
        x_bb, y_bb, bw, bh = cv2.boundingRect(cnt_mayor)
        
        # 1. Encontrar perfiles reales (techo y piso)
        xs = []
        y_tops = []
        y_bots = []
        
        for cx in range(x_bb, x_bb + bw):
            col = mask[:, cx]
            ys = np.where(col > 0)[0]
            if len(ys) > 0:
                xs.append(cx)
                y_tops.append(ys[0])
                y_bots.append(ys[-1])

        if len(xs) < 10:
            return mask

        # 2. Llenar huecos pálidos por donde se perdió el color
        x_min, x_max = min(xs), max(xs)
        full_xs = np.arange(x_min, x_max + 1)
        y_tops_interp = np.interp(full_xs, xs, y_tops).astype(np.float32)
        y_bots_interp = np.interp(full_xs, xs, y_bots).astype(np.float32)
        
        # 3. Suavizar picos para que sea un arco natural sin formas inventadas
        ventana = max(5, int((x_max - x_min) * 0.15))
        if ventana % 2 == 0: ventana += 1
        
        y_tops_smooth = cv2.GaussianBlur(y_tops_interp.reshape(1, -1), (ventana, 1), 0).flatten()
        y_bots_smooth = cv2.GaussianBlur(y_bots_interp.reshape(1, -1), (ventana, 1), 0).flatten()
        
        nueva_mask = np.zeros_like(mask)
        
        # 4. Pintar forma limpia
        for i, cx in enumerate(full_xs):
            yt = int(y_tops_smooth[i])
            yb = int(y_bots_smooth[i])
            
            yt = max(0, min(h - 1, yt))
            yb = max(0, min(h - 1, yb))
            
            if yb >= yt:
                cv2.line(nueva_mask, (cx, yt), (cx, yb), 255, 1)
                
        # 5. Cortar rebalses al iris (Atenuado: dejamos que el color natural defina, sin cortes duros bruscos)
        # El recorte elíptico brusco fue removido para no cortar esclerótida ni conjuntiva válida.
        # Si el usuario quiere seguir fiel al ojo, el suavizado de y_tops_smooth ya hizo el trabajo guiado por el contraste.
            
        # 6. Eliminar bordes de sierra residuales
        kernel_suave = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        nueva_mask = cv2.morphologyEx(nueva_mask, cv2.MORPH_OPEN, kernel_suave)
        
        return nueva_mask

def segmentar_y_recortar_conjuntiva(ruta_entrada, ruta_salida_segmentadas, ruta_salida_recortadas, ruta_salida_png, ruta_salida_area=None):
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
            
            anchor, radius = extractor.detect_eye_anchor(img)
            win_mask, _, _ = extractor.get_search_window(img, anchor, radius)
            
            raw_mask = extractor.find_medialuna_by_contrast(img, win_mask, anchor, radius)
            final_mask = extractor.polish_final(raw_mask)
            
            final_mask = extractor.cerrar_forma_medialuna(final_mask, anchor, radius)

            if np.count_nonzero(final_mask) == 0:
                print(f"DEBUG: Mascara vacia para {f}. No se encontro 'medialuna' con criterios actuales.")
                continue
            
            name = os.path.basename(f)
            img_area = img.copy()
            cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img_area, cnts, -1, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(ruta_area, name), img_area)
            
            x, y, w, h = cv2.boundingRect(final_mask)
            cv2.imwrite(os.path.join(ruta_salida_segmentadas, cat, name), final_mask)
            cv2.imwrite(os.path.join(ruta_salida_recortadas, cat, name), img[y:y+h, x:x+w])
            
            rgba = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2BGRA)
            rgba[:, :, 3] = final_mask[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(ruta_salida_png, cat, os.path.splitext(name)[0] + '.png'), rgba)

    print(f"Proceso contrastado terminado")
