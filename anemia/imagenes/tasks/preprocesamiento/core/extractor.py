import os
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class ConjuntivaExtractor:
    """
    MOTOR DE EXTRACCIÓN ORIGINAL: Restaurado al 100% siguiendo la lógica 
    pre-modularización. Incluye pulido de precisión y filtrado anatómico.
    """
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
        blurred = cv2.medianBlur(gray, 11)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 
            int(os.getenv("IRIS_HOUGH_DP", 1)), 
            int(os.getenv("IRIS_HOUGH_DIST", 50)),
            param1=int(os.getenv("IRIS_HOUGH_P1", 100)), 
            param2=int(os.getenv("IRIS_HOUGH_P2", 30)), 
            minRadius=int(os.getenv("IRIS_HOUGH_MIN_R", 30)), 
            maxRadius=int(os.getenv("IRIS_HOUGH_MAX_R", 150))
        )
        if circles is not None:
            c = circles[0][0]
            return (int(c[0]), int(c[1])), int(c[2])
        
        dark_thres = int(os.getenv("IRIS_DARK_THRES", 55))
        _, dark_mask = cv2.threshold(gray, dark_thres, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            best_dark = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(best_dark) > int(os.getenv("IRIS_MIN_AREA", 400)):
                M = cv2.moments(best_dark)
                if M["m00"] != 0:
                    return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])), int(w * 0.15)
        return (w // 2, int(h * 0.45)), int(w * 0.15)

    def get_search_window(self, img, center, radius):
        h, w = img.shape[:2]
        cx, cy = center
        mask = np.zeros((h, w), dtype=np.uint8)
        w_box = int(radius * 10)
        y_start = max(0, cy - int(radius * float(os.getenv("SEG_SEARCH_Y_START_FACTOR", 0.5)))) 
        y_end = min(h, cy + int(radius * 3.5))
        cv2.rectangle(mask, (max(0, cx - w_box // 2), y_start), (min(w, cx + w_box // 2), y_end), 255, -1)
        return mask, y_start, y_end

    def find_medialuna_by_contrast(self, img, window_mask, anchor, radius, rojo_offset=0):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a_ch, b_ch = cv2.split(lab)
        ax, ay = anchor 

        # 1. QUITAR (Piel y Pelo)
        _, mask_no_pelo = cv2.threshold(L, int(os.getenv("SEG_PELO_L_THRES", 75)), 255, cv2.THRESH_BINARY)
        mask_no_piel = cv2.compare(a_ch, b_ch, cv2.CMP_GT)

        # 2. BUSCAR BLANCO
        _, mask_blanco = cv2.threshold(L, int(os.getenv("SEG_BLANCO_L_THRES", 185)), 255, cv2.THRESH_BINARY)
        mask_blanco = cv2.morphologyEx(mask_blanco, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
        
        # 3. BUSCAR CARNE ROJA
        a_fuerte = self.clahe.apply(a_ch)
        rojo_a = int(os.getenv("SEG_ROJO_A_THRES", 138)) - rojo_offset
        _, mask_rojo = cv2.threshold(a_fuerte, rojo_a, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_and(mask_rojo, mask_no_pelo)
        combined = cv2.bitwise_and(combined, mask_no_piel)
        combined = cv2.bitwise_and(combined, window_mask)
        combined = cv2.subtract(combined, mask_blanco) 

        combined[0:ay + int(radius * float(os.getenv("SEG_Y_WALL_FACTOR", 0.8))), :] = 0
        combined[ay + int(radius * 2.5):, :] = 0
        
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3)))

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self.find_medialuna_by_contrast(img, window_mask, anchor, radius, 15) if rojo_offset == 0 else np.zeros_like(L)

        best_mask, best_score, best_cnt = np.zeros_like(L), -1, None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < int(os.getenv("SEG_MIN_AREA", 500)): continue
            x, y, w, h_cnt = cv2.boundingRect(cnt)
            aspect = float(w) / max(h_cnt, 1)
            if aspect < float(os.getenv("SEG_MIN_ASPECT_PRE", 1.1)): continue
            
            cx_c, cy_c = x + w // 2, y + h_cnt // 2
            dist_centro = np.sqrt((cx_c - ax)**2 + (cy_c - ay)**2)
            dist_f = np.exp(-0.5 * ((dist_centro - radius * 1.6) / (radius * 0.5))**2)
            
            m_tmp = np.zeros_like(L)
            cv2.drawContours(m_tmp, [cnt], -1, 255, -1)
            mean_vals = cv2.mean(lab, mask=m_tmp)
            avg_a, avg_b = mean_vals[1], mean_vals[2]
            skin_bias = 0.5 if avg_b > avg_a - 2 else 1.0
            
            dist_y = abs(cy_c - (ay + radius * float(os.getenv("SEG_Y_TARGET_FACTOR", 1.25)))) / radius
            score = area * (avg_a ** 2) * (aspect ** 1.5) * dist_f * skin_bias * np.exp(-2.0 * dist_y)
            
            hull = cv2.convexHull(cnt)
            solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            score *= (1.0 - abs(solidity - float(os.getenv("SEG_SOLIDITY_TARGET", 0.65))))
            
            if w > radius * 5.5: score *= 0.01
            score *= np.exp(-0.5 * (abs(cx_c - ax) / radius))

            if score > best_score:
                best_score, best_cnt = score, cnt

        if best_cnt is not None:
            cv2.drawContours(best_mask, [best_cnt], -1, 255, -1)
            
            # --- SEGUNDO PASO: PULIDO DE PRECISIÓN ---
            x, y, w, h_cnt = cv2.boundingRect(best_cnt)
            roi_lab = lab[y:y+h_cnt, x:x+w]
            roi_mask = best_mask[y:y+h_cnt, x:x+w]
            a_roi, b_roi = roi_lab[:, :, 1], roi_lab[:, :, 2]
            
            refine_o = int(os.getenv("SEG_REFINE_A_OFFSET", 5))
            _, local_red = cv2.threshold(a_roi, np.mean(a_roi) + refine_o, 255, cv2.THRESH_BINARY)
            local_no_skin = cv2.compare(a_roi, b_roi, cv2.CMP_GT)
            
            refined_roi = cv2.bitwise_and(cv2.bitwise_and(local_red, local_no_skin), roi_mask)
            refined_roi = cv2.morphologyEx(refined_roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            
            best_mask = np.zeros_like(L)
            best_mask[y:y+h_cnt, x:x+w] = refined_roi
            
            # --- PASO TRES: SOLO LA GRAN CARNE ---
            nuevos, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if nuevos:
                ganador = max(nuevos, key=cv2.contourArea)
                best_mask = np.zeros_like(L)
                cv2.drawContours(best_mask, [ganador], -1, 255, -1)

        area_f = np.count_nonzero(best_mask)
        if area_f < int(os.getenv("SEG_MIN_AREA_PALIDA", 4500)) and rojo_offset == 0:
            return self.find_medialuna_by_contrast(img, window_mask, anchor, radius, 15)

        return best_mask

    def polish_final(self, mask):
        if np.count_nonzero(mask) == 0: return mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def cerrar_forma_medialuna(self, mask, anchor=None, radius=None):
        if np.count_nonzero(mask) == 0: return mask
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return mask
        cnt = max(cnts, key=cv2.contourArea)
        xb, yb, bw, bh = cv2.boundingRect(cnt)
        h, w = mask.shape[:2]
        
        xs, y_t, y_b = [], [], []
        for x in range(xb, xb + bw):
            ys = np.where(mask[:, x] > 0)[0]
            if len(ys) > 0:
                xs.append(x); y_t.append(ys[0]); y_b.append(ys[-1])
        if len(xs) < 10: return mask

        full_xs = np.arange(min(xs), max(xs) + 1)
        y_t_i = np.interp(full_xs, xs, y_t).astype(np.float32)
        y_b_i = np.interp(full_xs, xs, y_b).astype(np.float32)
        
        ventana = max(5, int(len(full_xs) * 0.15))
        if ventana % 2 == 0: ventana += 1
        y_t_s = cv2.GaussianBlur(y_t_i.reshape(1, -1), (ventana, 1), 0).flatten()
        y_b_s = cv2.GaussianBlur(y_b_i.reshape(1, -1), (ventana, 1), 0).flatten()
        
        nueva = np.zeros_like(mask)
        for i, x in enumerate(full_xs):
            yt, yb = int(y_t_s[i]), int(y_b_s[i])
            if yb >= yt:
                cv2.line(nueva, (int(x), max(0, min(h-1, yt))), (int(x), max(0, min(h-1, yb))), 255, 1)
        return cv2.morphologyEx(nueva, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
