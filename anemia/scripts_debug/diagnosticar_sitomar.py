import os
import cv2
import numpy as np
import sys
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imagenes.tasks.preprocesamiento.core.extractor import ConjuntivaExtractor
from imagenes.tasks.preprocesamiento.validations.quality import es_nitida, tiene_tamano_suficiente
from imagenes.tasks.preprocesamiento.validations.anatomy import ojo_abierto, contiene_esclerotica
from imagenes.tasks.preprocesamiento.validations.conjunctiva import validar_conjuntiva

load_dotenv()

def get_image_paths(txt_path, base_dir):
    paths = []
    if not os.path.exists(txt_path): return paths
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.endswith(':'): continue
            p1 = os.path.join(base_dir, 'CON ANEMIA', line)
            p2 = os.path.join(base_dir, 'SIN ANEMIA', line)
            if os.path.exists(p1): paths.append((line, p1))
            elif os.path.exists(p2): paths.append((line, p2))
    return paths

def diagnosticar():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../media/originales'))
    sitomar = get_image_paths(os.path.join(os.path.dirname(__file__), '../media/sitomar.txt'), base_dir)
    
    extractor = ConjuntivaExtractor()
    
    print(f"Diagnostico de {len(sitomar)} imagenes de sitomar.txt\n", flush=True)
    
    for name, path in sitomar:
        img = cv2.imread(path)
        if img is None:
            print(f"{name}: No se pudo leer la imagen")
            continue
            
        rechazo = []
        if not tiene_tamano_suficiente(img): rechazo.append("Tamano insuficiente")
        if not ojo_abierto(img, extractor): rechazo.append("Ojo no abierto / No detectado")
        if not contiene_esclerotica(img): rechazo.append("No esclerotica")
        
        encontrada, area, mask, forma, pos, pest = validar_conjuntiva(img, extractor)
        if not encontrada: rechazo.append("Conjuntiva no encontrada")
        else:
            min_area = float(os.getenv("CONJUNTIVA_MIN_AREA_PCT", 0.001))
            if area < min_area: rechazo.append(f"Area insuficiente ({area:.6f} < {min_area:.6f})")
            if not forma: rechazo.append("Forma no valida")
            if not pest: rechazo.append("Bloqueo pestanas")
            
        if not es_nitida(img, mask): rechazo.append("No nitida")
        
        if not rechazo:
            print(f"{name}: [PASS]", flush=True)
        else:
            print(f"{name}: [FAIL] -> {', '.join(rechazo)}", flush=True)

if __name__ == "__main__":
    diagnosticar()
