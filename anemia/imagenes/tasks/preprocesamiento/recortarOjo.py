import os
import cv2
import glob
from .core.extractor import ConjuntivaExtractor

def recortar_ojos_dataset(ruta_entrada, ruta_salida):
    """
    PRE-PROCESAMIENTO: Recorta el ojo centrado.
    """
    categorias = ['SIN ANEMIA', 'CON ANEMIA']
    extractor = ConjuntivaExtractor()

    print(f"\n===== INICIANDO RECORTE DE OJOS =====")
    for cat in categorias:
        input_cat = os.path.join(ruta_entrada, cat)
        output_cat = os.path.join(ruta_salida, cat)
        os.makedirs(output_cat, exist_ok=True)

        archivos = glob.glob(os.path.join(input_cat, '*.[jJ][pP][gG]')) + \
                   glob.glob(os.path.join(input_cat, '*.[jJ][pP][eE][gG]'))
        print(f"--- Categoría {cat}: {len(archivos)} imágenes ---")

        for r_img in archivos:
            nombre = os.path.basename(r_img)
            img = cv2.imread(r_img)
            if img is None: 
                print(f"  [X] {nombre}: Error al leer")
                continue
            
            # Detectar ancla final
            center, radius = extractor.detect_eye_anchor(img)
            h_img, w_img = img.shape[:2]
            
            if center == (w_img // 2, h_img // 2):
                print(f"  [!] {nombre}: Ojo no detectado. Recorte fijo aplicado.")
            else:
                print(f"  [OK] {nombre}: Ojo detectado en {center}.")
                
            # Recortar (ahora incluye alineación)
            cropped, _, _ = extractor.crop_to_eye(img, center, radius, align=True)
            
            # Guardar
            cv2.imwrite(os.path.join(output_cat, nombre), cropped)

    print(f"===== RECORTE DE OJOS FINALIZADO =====\n")
