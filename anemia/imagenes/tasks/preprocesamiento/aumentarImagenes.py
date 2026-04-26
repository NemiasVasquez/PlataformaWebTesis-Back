import os
import cv2
import numpy as np
import random
import glob

def aumentar_dataset(ruta_entrada, ruta_salida, min_imagenes=None):
    """
    DATA AUGMENTATION: Incrementa la cantidad de imágenes aplicando métodos
    como espejado, rotación leve, y ajustes de brillo/contraste.
    Asegura un mínimo de imágenes por clase para mejorar el entrenamiento.
    """
    if min_imagenes is None:
        min_imagenes = int(os.getenv("MIN_IMAGES_AUGMENTATION", 300))
    categorias = ['SIN ANEMIA', 'CON ANEMIA']
    
    for cat in categorias:
        input_dir = os.path.join(ruta_entrada, cat)
        output_dir = os.path.join(ruta_salida, cat)
        os.makedirs(output_dir, exist_ok=True)
        
        # Obtener imágenes (asumiendo que son .png por el paso de resize)
        imagenes = glob.glob(os.path.join(input_dir, "*.png"))
        count_original = len(imagenes)
        
        if count_original == 0:
            print(f"⚠️ No hay imágenes en {cat} para aumentar.")
            continue
            
        # Copiar originales a la nueva carpeta
        for img_path in imagenes:
            name = os.path.basename(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(os.path.join(output_dir, name), img)
            
        print(f"📊 Categoría {cat}: {count_original} originales copiadas.")
        
        # Generar aumentadas hasta alcanzar el mínimo
        current_count = count_original
        idx = 0
        
        # Si ya hay más del mínimo, el usuario dijo "el mínimo... debería ser 300", 
        # así que si ya tiene 400, no aumentamos más a menos que el balanceo 
        # previo haya fallado o queramos más variabilidad.
        # Pero el usuario dice "aplicar esos métodos que hagan aumentar las imágenes obtenidas".
        
        while current_count < min_imagenes:
            img_path = random.choice(imagenes)
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            name_base = os.path.splitext(os.path.basename(img_path))[0]
            
            # Elegir una transformación
            metodo = random.choice(['flip_h', 'rotate', 'brightness', 'contrast', 'blur'])
            
            if metodo == 'flip_h':
                aug_img = cv2.flip(img, 1)
            elif metodo == 'rotate':
                angle = random.uniform(-10, 10)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
                aug_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            elif metodo == 'brightness':
                factor = random.uniform(0.8, 1.2)
                # Solo procesar canales color si hay transparencia
                if img.shape[2] == 4:
                    bgr = img[:, :, :3]
                    alpha = img[:, :, 3]
                    bgr = cv2.convertScaleAbs(bgr, alpha=factor, beta=0)
                    aug_img = cv2.merge([bgr[:,:,0], bgr[:,:,1], bgr[:,:,2], alpha])
                else:
                    aug_img = cv2.convertScaleAbs(img, alpha=factor, beta=0)
            elif metodo == 'contrast':
                alpha_val = random.uniform(0.9, 1.1)
                beta_val = random.uniform(-10, 10)
                if img.shape[2] == 4:
                    bgr = img[:, :, :3]
                    al = img[:, :, 3]
                    bgr = cv2.convertScaleAbs(bgr, alpha=alpha_val, beta=beta_val)
                    aug_img = cv2.merge([bgr[:,:,0], bgr[:,:,1], bgr[:,:,2], al])
                else:
                    aug_img = cv2.convertScaleAbs(img, alpha=alpha_val, beta=beta_val)
            elif metodo == 'blur':
                aug_img = cv2.GaussianBlur(img, (3, 3), 0)
            else:
                aug_img = img.copy()

            new_name = f"aug_{idx}_{name_base}.png"
            cv2.imwrite(os.path.join(output_dir, new_name), aug_img)
            current_count += 1
            idx += 1
            
        print(f"✅ Categoría {cat} finalizada con {current_count} imágenes.")

if __name__ == "__main__":
    # Para pruebas rápidas
    min_img = int(os.getenv("MIN_IMAGES_AUGMENTATION", 300))
    aumentar_dataset("media/procesadas/resize", "media/procesadas/aumentation", min_imagenes=min_img)
