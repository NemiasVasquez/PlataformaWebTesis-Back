import os
import cv2
import sys
import random
from dotenv import set_key

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imagenes.tasks.preprocesamiento.core.extractor import ConjuntivaExtractor
from imagenes.tasks.preprocesamiento.validations.quality import es_nitida, tiene_tamano_suficiente
from imagenes.tasks.preprocesamiento.validations.anatomy import ojo_abierto, contiene_esclerotica
from imagenes.tasks.preprocesamiento.validations.conjunctiva import validar_conjuntiva

ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.env'))

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

anchor_cache = {}

def evaluar_imagen(img, extractor, img_name):
    if not tiene_tamano_suficiente(img): return False
    
    if img_name not in anchor_cache:
        anchor_cache[img_name] = extractor.detect_eye_anchor(img)
    
    anchor, radius = anchor_cache[img_name]
    if anchor is None: return False
    
    original_detect = extractor.detect_eye_anchor
    extractor.detect_eye_anchor = lambda x: (anchor, radius)
    
    if not ojo_abierto(img, extractor): 
        extractor.detect_eye_anchor = original_detect
        return False
    if not contiene_esclerotica(img): 
        extractor.detect_eye_anchor = original_detect
        return False
    
    encontrada, area, mask, forma, pos, pest = validar_conjuntiva(img, extractor)
    extractor.detect_eye_anchor = original_detect
    
    if not encontrada: return False
    if area < float(os.getenv("CONJUNTIVA_MIN_AREA_PCT", 0.001)) or not forma: return False
    if not pest: return False
    
    if not es_nitida(img, mask): return False
    
    return True

def actualizar_env(params):
    for k, v in params.items():
        os.environ[k] = str(v)
        set_key(ENV_PATH, k, str(v))

def optimize():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../media/originales'))
    sitomar = get_image_paths(os.path.join(os.path.dirname(__file__), '../media/sitomar.txt'), base_dir)
    notomar = get_image_paths(os.path.join(os.path.dirname(__file__), '../media/notomar.txt'), base_dir)
    
    print(f"Iniciando optimizador... SI: {len(sitomar)}, NO: {len(notomar)}", flush=True)

    imgs_si = [(n, cv2.imread(p)) for n, p in sitomar if cv2.imread(p) is not None]
    imgs_no = [(n, cv2.imread(p)) for n, p in notomar if cv2.imread(p) is not None]

    best_score = -9999
    
    iteration = 0
    while True:
        iteration += 1
        params = {
            "NITIDEZ_UMBRAL_LAP": random.choice([1.0, 1.5, 2.0, 2.5]),
            "NITIDEZ_UMBRAL_TENENGRAD": random.choice([1.0, 1.5, 2.0, 2.5]),
            "NITIDEZ_UMBRAL_VMLAP_MASCARA": random.choice([5, 10, 15]),
            "CONJUNTIVA_MIN_AREA_PCT": random.choice([0.0001, 0.0003, 0.0005]),
            "CONJUNTIVA_MIN_ASPECT_RATIO": random.choice([0.3, 0.5, 0.8]),
            "SEG_MIN_AREA": random.choice([50, 100, 200]),
            "SEG_MIN_ASPECT_PRE": random.choice([1.0, 1.1])
        }
        
        for k, v in params.items():
            os.environ[k] = str(v)
            
        extractor = ConjuntivaExtractor() 
        
        pasados_si = sum(1 for n, img in imgs_si if evaluar_imagen(img, extractor, n))
        pasados_no = sum(1 for n, img in imgs_no if evaluar_imagen(img, extractor, n))
        
        score = pasados_si - (pasados_no * 2)
        
        if score > best_score:
            best_score = score
            print(f"[{iteration}] !NUEVO MEJOR! Score: {score} | Pasan SI: {pasados_si}/{len(imgs_si)} | Pasan NO (malos): {pasados_no}/{len(imgs_no)}", flush=True)
            print(f"Guardando en .env: {params}", flush=True)
            actualizar_env(params)
        
        if iteration % 10 == 0:
            print(f"[{iteration}] Iterando... Mejor score actual: {best_score}", flush=True)

if __name__ == "__main__":
    optimize()
