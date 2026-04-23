# imagenes/tasks/preprocesamiento/extraccionConjuntiva.py

import os
import cv2
import numpy as np
import glob
from datetime import datetime

class ConjuntivaExtractor:
    def __init__(self):
        # CLAHE para mejorar contraste local del canal A (rojo)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def detect_eye_anchor(self, img):
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Suavizado para reducir ruido en detección de círculos
        blurred = cv2.medianBlur(gray, 11)
        
        # Intentamos detectar el iris (círculo)
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,
                                  param1=100, param2=30, minRadius=30, maxRadius=150)
        if circles is not None:
            c = circles[0][0]
            return (int(c[0]), int(c[1])), int(c[2])
        
        # Si falla Hough, buscamos el centro de masa de la zona más oscura (Pupila/Iris)
        _, dark_mask = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            best_dark = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(best_dark) > 400:
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
        
        # Ojo está aquí. No buscar en orejas.
        w_box = int(radius * 10)
        h_box = int(radius * 8)
        
        y_start = max(0, cy - int(radius * 0.5)) 
        y_end = min(h, cy + h_box)
        x_start = max(0, cx - w_box // 2)
        x_end = min(w, cx + w_box // 2)
        
        cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), 255, -1)
        return mask, y_start, y_end

    def find_medialuna_by_contrast(self, img, window_mask, anchor, radius):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a_ch, b_ch = cv2.split(lab)
        (ax, ay) = anchor 

        # 1. QUITAR (Piel y Pelo)
        # Pelo es tiniebla (L < 75)
        # Piel es arena (B >= A)
        _, mask_no_pelo = cv2.threshold(L, 75, 255, cv2.THRESH_BINARY)
        mask_no_piel = cv2.compare(a_ch, b_ch, cv2.CMP_GT) # Rojo gana a Amarillo

        # 2. BUSCAR BLANCO (Esclerótica)
        _, mask_blanco = cv2.threshold(L, 185, 255, cv2.THRESH_BINARY)
        mask_blanco = cv2.morphologyEx(mask_blanco, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
        
        # 3. BUSCAR CARNE ROJA (Mucosa)
        a_fuerte = self.clahe.apply(a_ch)
        _, mask_rojo = cv2.threshold(a_fuerte, 138, 255, cv2.THRESH_BINARY)

        # 4. LIMPIAR MUCHO
        combined = cv2.bitwise_and(mask_rojo, mask_no_pelo)
        combined = cv2.bitwise_and(combined, mask_no_piel)
        combined = cv2.bitwise_and(combined, window_mask)
        combined = cv2.subtract(combined, mask_blanco) # Blanco no es Carne

        # 5. LÍMITE PIEDRA (No subir)
        # Carne roja solo abajo del iris. Arriba es prohibido.
        y_pared = ay + int(radius * 0.8)
        combined[0:y_pared, :] = 0
        
        # --- MORFOLOGÍA ORIENTADA A MEDIALUNA ---
        # Limpieza de ruido vertical (pestañas) con kernel horizontal
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_clean)

        # Unir fragmentos de la medialuna (horizontalmente)
        # Usamos un kernel rectangular ancho para no unir la conjuntiva con la piel de abajo
        kernel_close_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3)) 
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close_h)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros_like(L)

        best_mask = np.zeros_like(L)
        best_score = -1
        best_cnt = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500: 
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(h, 1)
            cx_cnt = x + w // 2
            cy_cnt = y + h // 2
            
            # La conjuntiva es obligatoriamente horizontal y delgada verticalmente
            if aspect_ratio < 1.2: 
                continue

            # --- PUNTAJE CAVERNÍCOLA ---
            # Medialuna debe ser acostada
            if aspect_ratio < 1.4: continue
            
            # Métrica de Rojicidad Media
            mask_cnt = np.zeros_like(L)
            cv2.drawContours(mask_cnt, [cnt], -1, 255, -1)
            mean_vals = cv2.mean(lab, mask=mask_cnt)
            avg_a = mean_vals[1] 

            # Carne roja debe estar centradita abajo
            dist_y = abs(cy_cnt - (ay + radius * 1.25)) / radius
            score = area * (avg_a ** 2) * (aspect_ratio ** 1.5) * np.exp(-2.0 * dist_y)
            
            # Ver si es curva (Solidez 0.7 es truco bueno)
            hull = cv2.convexHull(cnt)
            solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            score *= (1.0 - abs(solidity - 0.70))
            
            if score > best_score:
                best_score = score
                best_cnt = cnt

        if best_cnt is not None:
            cv2.drawContours(best_mask, [best_cnt], -1, 255, -1)
            
            # --- SEGUNDO PASO: PULIDO DE PRECISIÓN (LA LUPA) ---
            # Vamos a ignorar lo que no sea el "corazón" rojo de esta área
            x, y, w, h = cv2.boundingRect(best_cnt)
            roi_lab = lab[y:y+h, x:x+w]
            roi_mask = best_mask[y:y+h, x:x+w]
            
            # Buscamos el rojo más puro en esta zona pequeña
            a_roi = roi_lab[:, :, 1]
            b_roi = roi_lab[:, :, 2]
            
            # Umbral adaptativo local para esta zona
            _, local_red = cv2.threshold(a_roi, np.mean(a_roi) + 5, 255, cv2.THRESH_BINARY)
            local_no_skin = cv2.compare(a_roi, b_roi, cv2.CMP_GT)
            
            refined_roi = cv2.bitwise_and(local_red, local_no_skin)
            refined_roi = cv2.bitwise_and(refined_roi, roi_mask)
            
            # Limpieza morfológica fina
            kernel_fino = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            refined_roi = cv2.morphologyEx(refined_roi, cv2.MORPH_OPEN, kernel_fino)
            
            # Reconstruir la máscara global
            best_mask = np.zeros_like(L)
            best_mask[y:y+h, x:x+w] = refined_roi
            
            # --- PASO TRES: SOLO LA GRAN CARNE (Single Component) ---
            # Si hay pedazos sueltos (como pestañas), los tiramos.
            nuevos_cnts, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if nuevos_cnts:
                ganador = max(nuevos_cnts, key=cv2.contourArea)
                best_mask = np.zeros_like(L)
                cv2.drawContours(best_mask, [ganador], -1, 255, -1)

        return best_mask

    def polish_final(self, mask):
        if np.count_nonzero(mask) == 0:
            return mask
        
        # Último pulido: cerrar huecos pequeños y suavizar bordes sin perder masa
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Difuminado muy suave para no separar componentes
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def cerrar_forma_medialuna(self, mask):
        """
        CAVERNICOLA ESCANEAR COLUMNA POR COLUMNA cada 20px en X.
        Construir perfil de la forma (tope y fondo de la luna en cada X).
        Si hay hueco (columna sin blanco entre dos activas), interpolar borde
        y dibujar el puente siguiendo la curva natural de la medialuna.
        Si la forma se estrecha y vuelve a abrirse (separacion), tambien cerrar.
        """
        if np.count_nonzero(mask) == 0:
            return mask

        h, w = mask.shape[:2]
        paso = 20  # Cada cuanto picar en X

        # --- PASO 1: Construir perfil (top, bottom) para cada X activo ---
        perfil = {}  # x -> (y_top, y_bot)
        for x in range(0, w, paso):
            col = mask[:, x]
            ys = np.where(col > 0)[0]
            if len(ys) > 0:
                perfil[x] = (int(ys[0]), int(ys[-1]))

        if len(perfil) < 2:
            return mask

        xs_activos = sorted(perfil.keys())
        nueva_mask = mask.copy()

        # --- PASO 2: Detectar huecos (columnas sin blanco entre dos activas) ---
        for k in range(len(xs_activos) - 1):
            x1 = xs_activos[k]
            x2 = xs_activos[k + 1]

            gap = x2 - x1
            if gap <= paso:
                continue  # Sin hueco, columnas consecutivas normales

            # Hay salto: columnas intermedias sin blanco
            top1, bot1 = perfil[x1]
            top2, bot2 = perfil[x2]

            print(f"DEBUG LUNA: hueco en X={x1}->{x2} (gap={gap}px), interpolando borde")

            # Interpolar linealmente top y bottom para cubrir el hueco
            for xi in range(x1, x2 + 1):
                t = (xi - x1) / max(gap, 1)
                top_i = int(top1 + t * (top2 - top1))
                bot_i = int(bot1 + t * (bot2 - bot1))
                top_i = max(0, min(h - 1, top_i))
                bot_i = max(0, min(h - 1, bot_i))
                # Dibujar solo los bordes (trazo de cierre, no rellenar todo)
                nueva_mask[top_i, xi] = 255
                nueva_mask[bot_i, xi] = 255

        # --- PASO 3: Cerrar morfologicamente para suavizar el trazo unido ---
        kernel_cierre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
        nueva_mask = cv2.morphologyEx(nueva_mask, cv2.MORPH_CLOSE, kernel_cierre)

        return nueva_mask

def segmentar_y_recortar_conjuntiva(ruta_entrada, ruta_salida_segmentadas, ruta_salida_recortadas, ruta_salida_png):
    extractor = ConjuntivaExtractor()
    categorias = ['SIN ANEMIA', 'CON ANEMIA']
    
    for cat in categorias:
        os.makedirs(os.path.join(ruta_salida_segmentadas, cat), exist_ok=True)
        os.makedirs(os.path.join(ruta_salida_recortadas, cat), exist_ok=True)
        os.makedirs(os.path.join(ruta_salida_png, cat), exist_ok=True)
        ruta_area = os.path.join(os.path.dirname(ruta_salida_segmentadas), 'area', cat)
        os.makedirs(ruta_area, exist_ok=True)

        for f in glob.glob(os.path.join(ruta_entrada, cat, '*.[jJ][pP][gG]')) + \
                 glob.glob(os.path.join(ruta_entrada, cat, '*.[jJ][pP][eE][gG]')):
            img = cv2.imread(f)
            if img is None: continue
            
            anchor, radius = extractor.detect_eye_anchor(img)
            win_mask, _, _ = extractor.get_search_window(img, anchor, radius)
            
            # Identificar medialuna por contraste superior usando el iris como anclaje
            raw_mask = extractor.find_medialuna_by_contrast(img, win_mask, anchor, radius)
            final_mask = extractor.polish_final(raw_mask)
            
            # --- CIERRE MEDIALUNA: escanear perfil X/Y y cerrar huecos naturalmente ---
            final_mask = extractor.cerrar_forma_medialuna(final_mask)

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
