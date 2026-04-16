# imagenes/tasks/preprocesamiento/extraccionConjuntiva.py

import os
import cv2
import numpy as np
import glob
from datetime import datetime
from .segmentacion_ojo import segment_image_with_unet, crop_segmented_image
from modelo.tasks.config import MODELO_UNET_PATH

def segmentar_y_recortar_conjuntiva(ruta_entrada, ruta_salida_segmentadas, ruta_salida_recortadas, ruta_salida_png):
    categorias = ['SIN ANEMIA', 'CON ANEMIA']

    for categoria in categorias:
        os.makedirs(os.path.join(ruta_salida_segmentadas, categoria), exist_ok=True)
        os.makedirs(os.path.join(ruta_salida_recortadas, categoria), exist_ok=True)
        os.makedirs(os.path.join(ruta_salida_png, categoria), exist_ok=True)

    start = datetime.now()
    
    # Verificar si el modelo UNet existe
    unet_path = MODELO_UNET_PATH
    if not os.path.exists(unet_path):
        print(f"⚠️ MODELO UNET NO ENCONTRADO en {unet_path}. Usando segmentación por color (HEURÍSTICA).")

    for categoria in categorias:
        ruta_imgs = glob.glob(os.path.join(ruta_entrada, categoria, '*.jpeg')) + \
                    glob.glob(os.path.join(ruta_entrada, categoria, '*.jpg'))

        for ruta_img in ruta_imgs:
            img_original = cv2.imread(ruta_img)
            if img_original is None:
                print(f"No se pudo leer la imagen {ruta_img}")
                continue

            # Selección de método de segmentación
            if os.path.exists(unet_path):
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
                def heuristic_segmentation(img):
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    lower_red1 = np.array([0, 100, 50]); upper_red1 = np.array([10, 255, 255])
                    lower_red2 = np.array([160, 100, 50]); upper_red2 = np.array([180, 255, 255])
                    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    mask_clean = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
                    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    final_mask = np.zeros_like(mask_red)
                    for cnt in contours:
                        if cv2.contourArea(cnt) > 200:
                            cv2.drawContours(final_mask, [cnt], -1, 255, -1)
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
