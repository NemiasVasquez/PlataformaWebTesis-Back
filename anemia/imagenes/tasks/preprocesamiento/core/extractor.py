import os
import cv2
import numpy as np
from dotenv import load_dotenv

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

load_dotenv()

class ConjuntivaExtractor:
    """
    MOTOR DE EXTRACCIÓN: Usa Mediapipe (IA) para detectar ojos con precisión.
    Si falla, usa Cascadas Haar mejoradas.
    """
    def __init__(self):
        # CLAHE para mejorar contraste local del canal A (rojo)
        clip = float(os.getenv("SEG_CLAHE_CLIP", 3.0))
        grid = int(os.getenv("SEG_CLAHE_GRID", 8))
        self.clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        
        # Cascadas para detección de rostro y ojos (Fallback)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        if MEDIAPIPE_AVAILABLE:
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5
            )

    def check_esclerotica(self, img, cx, cy, r):
        h, w = img.shape[:2]
        x0, x1 = max(0, int(cx - r*2.0)), min(w, int(cx + r*2.0))
        y0, y1 = max(0, int(cy - r*1.0)), min(h, int(cy + r*1.0))
        roi = img[y0:y1, x0:x1]
        if roi.size == 0: return 0.1
        
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        # Bajar exigencia de "blanco" a "gris claro" (130) para fotos con sombra
        _, mask_b = cv2.threshold(lab[:,:,0], 130, 255, cv2.THRESH_BINARY)
        # Cavernícola: Las uñas son blancas, pero la conjuntiva es ROJA (A > 140)
        _, mask_r = cv2.threshold(lab[:,:,1], 140, 255, cv2.THRESH_BINARY)
        
        hay_blanco = cv2.countNonZero(mask_b) > int(r*r * 0.05)
        hay_rojo = cv2.countNonZero(mask_r) > int(r*r * 0.02)
        
        if hay_blanco and hay_rojo:
            return 5.0  # Es un ojo real (blanco + rojo)
        elif hay_blanco:
            return 0.5  # Uña o frente con luz (blanco sin rojo)
        return 0.1  # Piel, pelo (nada de nada)

    def crop_to_eye(self, img, center, radius, align=False, factor_extra=1.0):
        """
        CAVERNÍCOLA RECORTA: Quita lo que no es ojo.
        Si la detección falló, usa el RECORTE FIJO: 15% arriba/abajo, 5% lados.
        factor_extra: Multiplicador para ampliar el recorte (1.0 = normal, 1.5 = 50% más amplio)
        """
        h, w = img.shape[:2]
        cx, cy = center
        
        # SI EL ANCLA ES EL CENTRO EXACTO (FALLBACK), USAR RECORTE FIJO DEL JEFE
        if cx == w // 2 and cy == h // 2 and radius == int(w * 0.15):
            margen = 0.15 / factor_extra
            y0, y1 = int(h * margen), int(h * (1 - margen))
            x0, x1 = int(w * 0.05), int(w * 0.95)
        else:
            # --- ALINEACIÓN DESACTIVADA POR JEFE ---
            if align:
                img, center = self.align_eye(img, center, radius)
            
            f_x = float(os.getenv("OJO_CROP_FACTOR_X", 6.5)) * factor_extra
            f_y_up = float(os.getenv("OJO_CROP_FACTOR_Y_UP", 3.0)) * factor_extra
            f_y_down = float(os.getenv("OJO_CROP_FACTOR_Y_DOWN", 7.0)) * factor_extra

            # CAVERNÍCOLA: Si el detector encontró solo la pupila, el radio es microscópico.
            # Asegurar un radio mínimo del 12% del ancho de la imagen.
            radio_seguro = max(radius, int(w * 0.12))

            x0 = max(0, cx - int(radio_seguro * f_x))
            x1 = min(w, cx + int(radio_seguro * f_x))
            y0 = max(0, cy - int(radio_seguro * f_y_up))
            y1 = min(h, cy + int(radio_seguro * f_y_down))
        
        cropped = img[y0:y1, x0:x1]
        new_center = (int(cx - x0), int(cy - y0))
        return cropped, new_center, (x0, y0)

    def align_eye(self, img, center, radius):
        """
        GIRA EL MUNDO: Alinea el ojo. 
        Si no hay blanco (esclerótica), busca la forma oscura del "hueco" del ojo.
        """
        h, w = img.shape[:2]
        cx, cy = center
        
        x_start = max(0, cx - int(radius * 6))
        x_end = min(w, cx + int(radius * 6))
        y_start = max(0, cy - int(radius * 3))
        y_end = min(h, cy + int(radius * 3))
        
        roi = img[y_start:y_end, x_start:x_end]
        if roi.size == 0: return img, center
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 1. INTENTO CON BLANCO (Esclerótica)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        _, mask_blanco = cv2.threshold(lab[:,:,0], int(os.getenv("SEG_BLANCO_L_THRES", 180)), 255, cv2.THRESH_BINARY)
        
        # 2. INTENTO CON OSCURO (Hueco del ojo/pestañas) si blanco falla
        _, mask_oscuro = cv2.threshold(gray_roi, int(os.getenv("IRIS_DARK_THRES", 60)), 255, cv2.THRESH_BINARY_INV)
        
        mask_usar = mask_blanco if cv2.countNonZero(mask_blanco) > 200 else mask_oscuro
        
        # Encontrar eje principal de la forma del ojo
        cnts, _ = cv2.findContours(mask_usar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            best_cnt = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(best_cnt) > 500:
                # Ajustar elipse para ver inclinación
                if len(best_cnt) >= 5:
                    (x_e, y_e), (MA, ma), angle = cv2.fitEllipse(best_cnt)
                    # El ángulo de fitEllipse es relativo a la vertical, lo queremos horizontal
                    rot_angle = angle - 90
                    if abs(rot_angle) < 40:
                        M = cv2.getRotationMatrix2D((float(cx), float(cy)), float(rot_angle), 1.0)
                        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        return img, center

    def detect_eye_anchor(self, img):
        h, w = img.shape[:2]
        
        # 1. INTENTO: MEDIAPIPE (IA SUPERIOR)
        if MEDIAPIPE_AVAILABLE:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.mp_face_detection.process(img_rgb)
            
            if results.detections:
                # Tomamos la cara con mejor confianza
                detection = results.detections[0]
                
                if detection.score[0] >= 0.75:
                    best_ojo = None
                    best_dist = float('inf')
                    
                    for idx in [0, 1]:
                        kp = detection.location_data.relative_keypoints[idx]
                        cx, cy = int(kp.x * w), int(kp.y * h)
                        
                        # Si el keypoint está fuera de la imagen o muy al borde, suele ser alucinación de MediaPipe
                        if cx < w*0.05 or cx > w*0.95 or cy < h*0.05 or cy > h*0.95:
                            continue
                            
                        dist = (cx - w//2)**2 + (cy - h//2)**2
                        if dist < best_dist:
                            best_dist = dist
                            best_ojo = (cx, cy)
                            
                    if best_ojo is not None:
                        bbox = detection.location_data.relative_bounding_box
                        radius = min(int(bbox.width * w * 0.12), int(w * 0.15))
                        
                        # CAVERNÍCOLA: Si MediaPipe dice que es ojo, PERO no tiene blanco de esclerótica... es la frente!
                        if self.check_esclerotica(img, best_ojo[0], best_ojo[1], radius) > 1.0:
                            return best_ojo, radius

        # 2. INTENTO: ROSTRO Y OJOS (Cascadas Haar)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        for (xf, yf, wf, hf) in faces:
            roi_gray = gray[yf:yf+hf, xf:xf+wf]
            roi_gray = cv2.equalizeHist(roi_gray)
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
            
            best_eye = None
            best_score = -1
            best_radius = 0
            
            for (ex, ey, ew, eh) in eyes:
                cx = xf + ex + ew // 2
                cy = yf + ey + eh // 2
                area = ew * eh
                dist = ((cx - w//2)**2 + (cy - h//2)**2) ** 0.5
                
                y_penalty = 1.0 if cy > h * 0.35 else 0.05
                white_mult = self.check_esclerotica(img, cx, cy, ew // 4)
                
                score = (area * y_penalty * white_mult) / (dist + 1.0)
                if score > best_score:
                    best_score = score
                    best_eye = (cx, cy)
                    best_radius = ew // 4
            
            # Cavernícola: Si la cascada detectó algo, verificamos que tenga rojo o blanco (score > 0.5)
            if best_eye and best_score >= 0.5:
                return best_eye, best_radius
            
            # Si no encontró ojos reales (o solo encontró pelo), NO retorna la cara a ciegas.
            # Deja que pasen a los Fallbacks 2.5 y 3.

        # 2.5 INTENTO: SOLO OJO CON CASCADA (Para fotos de muy cerca donde no hay cara)
        eyes_only = self.eye_cascade.detectMultiScale(gray, 1.1, 15, minSize=(80, 80))
        if len(eyes_only) > 0:
            best_ojo = None
            best_score = -1
            best_radius = 0
            
            for (ex, ey, ew, eh) in eyes_only:
                cx, cy = ex + ew // 2, ey + eh // 2
                area = ew * eh
                dist = ((cx - w//2)**2 + (cy - h//2)**2) ** 0.5
                
                # Castigar cejas (suelen estar arriba del 35% de la foto)
                y_penalty = 1.0 if cy > h * 0.35 else 0.05
                white_mult = self.check_esclerotica(img, cx, cy, ew // 4)
                score = (area * y_penalty * white_mult) / (dist + 1.0)
                
                if score > best_score:
                    best_score = score
                    best_ojo = (cx, cy)
                    best_radius = ew // 4
            
            if best_ojo:
                return best_ojo, best_radius

        # 3. INTENTO: HOUGH CIRCLES (Fallback 1)
        scale = 1.0
        if w > 1000:
            scale = 1000.0 / w
            proc_img = cv2.resize(img, (1000, int(h * scale)))
        else:
            proc_img = img.copy()
            
        gray_p = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray_p, 11)
        
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 
            int(os.getenv("IRIS_HOUGH_DP", 1)), 
            int(os.getenv("IRIS_HOUGH_DIST", 50)),
            param1=int(os.getenv("IRIS_HOUGH_P1", 100)), 
            param2=int(os.getenv("IRIS_HOUGH_P2", 20)), 
            minRadius=int(int(os.getenv("IRIS_HOUGH_MIN_R", 30)) * scale), 
            maxRadius=int(int(os.getenv("IRIS_HOUGH_MAX_R", 350)) * scale)
        )
        
        if circles is not None:
            best_c = None
            best_score_c = -1
            w_proc = 1000 if w > 1000 else w
            h_proc = int(h * scale) if w > 1000 else h
            
            for c in circles[0]:
                cx, cy, r = c[0], c[1], c[2]
                area = r * r
                dist = ((cx - w_proc//2)**2 + (cy - h_proc//2)**2) ** 0.5
                y_penalty = 1.0 if cy > h_proc * 0.35 else 0.05
                white_mult = self.check_esclerotica(img, cx * scale, cy * scale, r * scale)
                score = (area * y_penalty * white_mult) / (dist + 1.0)
                
                if score > best_score_c:
                    best_score_c = score
                    best_c = c
                    
            if best_c is not None:
                return (int(best_c[0] / scale), int(best_c[1] / scale)), int(best_c[2] / scale)
        
        # 3. INTENTO: MÁSCARA OSCURA (Fallback 2)
        _, dark_mask = cv2.threshold(gray_p, int(os.getenv("IRIS_DARK_THRES", 55)), 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            best_dark = None
            best_dark_score = -1
            w_proc = 1000 if w > 1000 else w
            h_proc = int(h * scale) if w > 1000 else h
            
            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if area > int(int(os.getenv("IRIS_MIN_AREA", 200)) * (scale**2)):
                    x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
                    aspect = min(w_b, h_b) / max(w_b, h_b)
                    
                    if aspect < 0.4: continue
                    
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx_p = int(M["m10"] / M["m00"])
                        cy_p = int(M["m01"] / M["m00"])
                        
                        dist = ((cx_p - w_proc//2)**2 + (cy_p - h_proc//2)**2) ** 0.5
                        # Castigar duro a la ceja/frente: no aceptar nada del 35% superior
                        y_penalty = 1.0 if cy_p > h_proc * 0.35 else 0.05
                        white_mult = self.check_esclerotica(img, cx_p * scale, cy_p * scale, w_b * scale / 2)
                        
                        score = (area * aspect * y_penalty * white_mult) / (dist + 1.0)
                        if score > best_dark_score:
                            best_dark_score = score
                            best_dark = (cx_p, cy_p, w_b)

            if best_dark:
                return (int(best_dark[0] / scale), int(best_dark[1] / scale)), int(best_dark[2] / (2*scale))
        
        # FALLBACK: RECORTE FIJO (15% ARRIBA/ABAJO, 5% LADOS)
        # Si todo falla, devolvemos el centro para que el sistema use el recorte estándar
        return (w // 2, h // 2), int(w * 0.15)

    def get_search_window(self, img, center, radius):
        h, w = img.shape[:2]
        cx, cy = center
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # VENTANA SIMÉTRICA: Busca arriba y abajo del iris
        axes = (int(radius * 8), int(radius * 8))
        cv2.ellipse(mask, (int(cx), int(cy)), axes, 0, 0, 360, 255, -1)
        
        # Rango vertical amplio para captar conjuntiva superior o inferior
        y_start = max(0, cy - int(radius * 6))
        y_end = min(h, cy + int(radius * 6))
        
        return mask, y_start, y_end

    def find_medialuna_by_contrast(self, img, window_mask, anchor, radius, rojo_offset=0):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, a_ch, b_ch = cv2.split(lab)
        ax, ay = anchor 

        # 1. QUITAR (Piel y Pelo + PESTAÑAS)
        # Subir umbral a 90 para atrapar pestañas grises
        _, mask_no_pelo = cv2.threshold(L, int(os.getenv("SEG_PELO_L_THRES", 75)), 255, cv2.THRESH_BINARY)
        mask_no_piel = cv2.compare(a_ch, b_ch, cv2.CMP_GT)

        # 2. BUSCAR BLANCO (esclerótica) - Expandir agresivamente para excluirla
        _, mask_blanco = cv2.threshold(L, int(os.getenv("SEG_BLANCO_L_THRES", 165)), 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(mask_blanco) < 30:
            return np.zeros_like(L)
        
        # Expandir zona blanca con dilatación GRANDE para cubrir toda la esclerótica
        mask_blanco = cv2.morphologyEx(mask_blanco, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
        mask_blanco = cv2.dilate(mask_blanco, np.ones((7,7), np.uint8), iterations=2)
        
        # 3. BUSCAR CARNE ROJA
        a_fuerte = self.clahe.apply(a_ch)
        rojo_a = int(os.getenv("SEG_ROJO_A_THRES", 138)) - rojo_offset
        _, mask_rojo = cv2.threshold(a_fuerte, rojo_a, 255, cv2.THRESH_BINARY)

        combined = cv2.bitwise_and(mask_rojo, mask_no_pelo)
        combined = cv2.bitwise_and(combined, mask_no_piel)
        combined = cv2.bitwise_and(combined, window_mask)
        
        # Restar esclerótica expandida
        combined = cv2.subtract(combined, mask_blanco) 

        # Pared circular superior (excluir zona del iris)
        h, w = img.shape[:2]
        wall_mask = np.ones((h, w), dtype=np.uint8) * 255
        cv2.circle(wall_mask, (int(ax), int(ay)), int(radius * float(os.getenv("SEG_Y_WALL_FACTOR", 0.8))), 0, -1)
        combined = cv2.bitwise_and(combined, wall_mask)
        
        # CAVERNÍCOLA: Conjuntiva vive DEBAJO del centro del iris, no arriba
        # Cortar todo lo que esté más de 0.5*radius por encima del centro
        y_limite_sup = max(0, int(ay - radius * 0.5))
        combined[:y_limite_sup, :] = 0

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
            if aspect < float(os.getenv("SEG_MIN_ASPECT_PRE", 0.8)): continue
            
            cx_c, cy_c = x + w // 2, y + h_cnt // 2
            dist_centro = np.sqrt((cx_c - ax)**2 + (cy_c - ay)**2)
            dist_f = np.exp(-0.5 * ((dist_centro - radius * 1.6) / (radius * 0.5))**2)
            
            m_tmp = np.zeros_like(L)
            cv2.drawContours(m_tmp, [cnt], -1, 255, -1)
            mean_vals = cv2.mean(lab, mask=m_tmp)
            avg_a, avg_b = mean_vals[1], mean_vals[2]
            skin_bias = 0.5 if avg_b > avg_a - 2 else 1.0
            
            # PUNTUACIÓN CAVERNÍCOLA: Prioridad ROJO (avg_a^3) y MEDIALUNA (aspect^2)
            # dist_f asegura que esté a una distancia razonable del centro del ojo
            score = area * (avg_a ** 3) * (aspect ** 2.0) * dist_f * skin_bias
            
            hull = cv2.convexHull(cnt)
            solidity = float(area) / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
            # Castigamos si no es sólido (huecos)
            score *= (1.0 - abs(solidity - float(os.getenv("SEG_SOLIDITY_TARGET", 0.65))))
            
            if w > radius * 4.5: score *= 0.01 # Muy ancho es probablemente piel o rostro
            # if cy_c < ay: score *= 0.01 # Castigar duramente si está arriba del centro del ojo (no es conjuntiva inferior)
            score *= np.exp(-0.5 * (abs(cx_c - ax) / radius)) # Centrado horizontalmente con el ojo

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
        if area_f < int(os.getenv("SEG_MIN_AREA_PALIDA", 1500)) and rojo_offset == 0:
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
