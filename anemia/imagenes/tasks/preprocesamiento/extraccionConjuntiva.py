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

    start = datetime.now()
    
    # El modelo UNet ya no se usa por instrucción del usuario.
    # Se utiliza segmentación por color (HEURÍSTICA).
    unet_path = None

    for categoria in categorias:
        ruta_imgs = glob.glob(os.path.join(ruta_entrada, categoria, '*.jpeg')) + \
                    glob.glob(os.path.join(ruta_entrada, categoria, '*.jpg'))

        for ruta_img in ruta_imgs:
            img_original = cv2.imread(ruta_img)
            if img_original is None:
                print(f"No se pudo leer la imagen {ruta_img}")
                continue

            # Selección de método de segmentación
            if unet_path and os.path.exists(unet_path):
                # Método UNet (Proporcionado por el usuario)
                try:
                    seg_img, mask = segment_image_with_unet(img_original, unet_path)
                    if mask is None:
                        raise ValueError("UNet falló al generar máscara")
                    img_cropped, bbox = crop_segmented_image(img_original, mask)
                    mask_conjuntiva = (mask * 255).astype(np.uint8)
                except Exception as e:
                    print(f"❌ Error en UNet para {ruta_img}: {e}. Usando fallback.")
                    # Fallback a segmentación básica aquí si es necesario
                    continue
            else:
                # Método Heurístico (Original del proyecto)
                def detect_iris(img):
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.medianBlur(gray, 11)
                    # El iris suele ser la zona más oscura y circular
                    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50,
                                              param1=100, param2=30, minRadius=30, maxRadius=150)
                    if circles is not None:
                        return circles[0][0] # x, y, r
                    return None

                def remove_eyelashes(img):
                    # Máscara para píxeles muy oscuros (pestañas)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, mask_dark = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
                    return mask_not_black # Omitiendo por brevedad, se integra abajo

                def heuristic_segmentation(img):
                    high, wide = img.shape[:2]
                    
                    # 1. Localizar Iris (Ancla espacial)
                    iris = detect_iris(img)
                    iris_y = high * 0.4 if iris is None else iris[1]
                    iris_r = 60 if iris is None else iris[2]
                    
                    # 2. Filtrar Pestañas (Negros/Marrones oscuros)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    _, mask_not_black = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
                    
                    # 3. Detectar Tejido Conuntival (Rojo/Rosa Saturado)
                    # Estrechamos Hue y subimos Saturation para no tomar piel (que es más pálida/amarilla)
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    # Rango 1 (Rojos)
                    l1, u1 = np.array([0, 110, 50]), np.array([10, 255, 255])
                    # Rango 2 (Rojos altos)
                    l2, u2 = np.array([170, 110, 50]), np.array([180, 255, 255])
                    
                    mask_red = cv2.inRange(hsv, l1, u1) | cv2.inRange(hsv, l2, u2)
                    
                    # Combinación: Rojo Saturado + No Pestaña
                    mask_combined = cv2.bitwise_and(mask_red, mask_not_black)
                    
                    # Limpieza morfológica para unir la "media luna"
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
                    mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
                    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
                    
                    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    final_mask = np.zeros_like(mask_red)
                    
                    valid_blobs = []
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if area < 700: continue
                        
                        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
                        cx, cy = x + w_cnt/2, y + h_cnt/2
                        aspect_ratio = float(w_cnt) / h_cnt
                        
                        # FILTRADO ESPACIAL DINÁMICO
                        # La conjuntiva DEBE estar cerca del iris, no perdida abajo
                        dist_vertical = cy - iris_y
                        dist_horizontal = abs(cx - wide/2)
                        
                        # Ventana estricta: mas de 0.5 radios pero menos de 2.5 radios
                        if dist_vertical > iris_r * 0.4 and dist_vertical < iris_r * 2.5 and \
                           dist_horizontal < wide * 0.35 and aspect_ratio > 1.3:
                            
                            # Puntuación basada en cercanía al iris y solidez
                            score = area / (dist_vertical + 1)
                            valid_blobs.append((cnt, score))
                    
                    if valid_blobs:
                        # Seleccionamos la mancha más "conectada" al iris (superior en la ventana)
                        best_cnt = max(valid_blobs, key=lambda b: b[1])[0]
                        cv2.drawContours(final_mask, [best_cnt], -1, 255, -1)
                        
                    return final_mask

                mask_conjuntiva = heuristic_segmentation(img_original)
                x, y, w, h = cv2.boundingRect(mask_conjuntiva)
                img_cropped = img_original[y:y+h, x:x+w]
                if np.count_nonzero(mask_conjuntiva) == 0:
                    continue

            # Guardar resultados
            nombre_img = os.path.basename(ruta_img)
            ruta_seg = os.path.join(ruta_salida_segmentadas, categoria, nombre_img)
            ruta_rec = os.path.join(ruta_salida_recortadas, categoria, nombre_img)
            
            cv2.imwrite(ruta_seg, mask_conjuntiva)
            cv2.imwrite(ruta_rec, img_cropped)

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
