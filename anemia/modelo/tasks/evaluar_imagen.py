# tasks/evaluar_imagen.py
from django.http import JsonResponse
from . import cargar_datos, entrenar, evaluar
from django.views.decorators.csrf import csrf_exempt
from .dataset import ConjuntivaDataset
from torch.utils.data import DataLoader
import numpy as np
import os, cv2, uuid
from .config import MODELO_NFNET_PATH, IMG_WIDTH, IMG_HEIGHT
from sklearn.metrics import confusion_matrix
from imagenes.tasks.preprocesamiento.filtrarImagenes import filtrar_conjuntiva
from imagenes.tasks.preprocesamiento.extraccionConjuntiva import segmentar_y_recortar_conjuntiva
from imagenes.tasks.preprocesamiento.resizeImagenes import redimensionar_imagenes
from django.conf import settings
from pathlib import Path
from django.conf.urls.static import static
import shutil 
import torch

def evaluar_imagen_individual(imagen):
    base_dir = os.path.join(settings.MEDIA_ROOT, 'pruebas')
    os.makedirs(base_dir, exist_ok=True)

    existentes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    numeracion = sorted([int(d) for d in existentes if d.isdigit()], reverse=True)
    nuevo_id = str(numeracion[0] + 1 if numeracion else 1).zfill(4)

    ruta_base = os.path.join(base_dir, nuevo_id)
    ruta_entrada = os.path.join(ruta_base, 'entrada')
    ruta_filtrada = os.path.join(ruta_base, 'filtrada')
    ruta_segmentada = os.path.join(ruta_base, 'segmentada')
    ruta_recortada = os.path.join(ruta_base, 'recortada')
    ruta_png = os.path.join(ruta_base, 'png')
    ruta_resize = os.path.join(ruta_base, 'resize')
    ruta_no_filtrados = os.path.join(ruta_base, 'no_filtrados')
    ruta_reporte = os.path.join(ruta_base, 'reporte.txt')

    os.makedirs(os.path.join(ruta_entrada, 'SIN ANEMIA'), exist_ok=True)
    path_img = os.path.join(ruta_entrada, 'SIN ANEMIA', 'imagen.jpeg')
    with open(path_img, 'wb') as f:
        for chunk in imagen.chunks():
            f.write(chunk)

    filtrar_conjuntiva(ruta_entrada, ruta_filtrada, ruta_no_filtrados, ruta_reporte)

    ruta_filtrada_sin_anemia = os.path.join(ruta_filtrada, 'SIN ANEMIA')
    num_filtrados = len(os.listdir(ruta_filtrada_sin_anemia)) if os.path.exists(ruta_filtrada_sin_anemia) else 0
    print(f"DEBUG: Imagenes despues de filtrado: {num_filtrados}")

    if num_filtrados == 0:
        razon_rechazo = "Imagen rechazada por calidad (Iris/Nitidez/Color)"
        if os.path.exists(ruta_reporte):
            with open(ruta_reporte, encoding='utf-8') as f:
                lineas = [l.strip() for l in f if l.strip()]
                if len(lineas) > 1:
                    razon_rechazo = lineas[1]
        print(f"DEBUG: Fallo en filtrado. Razon: {razon_rechazo}")
        return {
            'valida': False,
            'razon': razon_rechazo,
            'directorio': f"media/pruebas/{nuevo_id}"
        }

    segmentar_y_recortar_conjuntiva(ruta_filtrada, ruta_segmentada, ruta_recortada, ruta_png)
    
    num_pngs = len(list(Path(os.path.join(ruta_png, 'SIN ANEMIA')).glob('*.png'))) if os.path.exists(os.path.join(ruta_png, 'SIN ANEMIA')) else 0
    print(f"DEBUG: Imagenes segmentadas (PNG): {num_pngs}")

    redimensionar_imagenes(ruta_png, ruta_resize, size=(IMG_WIDTH, IMG_HEIGHT))

    final_imgs = list(Path(os.path.join(ruta_resize, 'SIN ANEMIA')).glob('*.png'))
    print(f"DEBUG: Imagenes finales listas: {len(final_imgs)}")

    if not final_imgs:
        print(f"DEBUG: Fallo. No hay imagenes tras segmentacion y resize.")
        return {
            'valida': False,
            'razon': 'Segmentacion no detectó tejido palpebral. Intenta con mejor iluminación.',
            'directorio': f"media/pruebas/{nuevo_id}"
        }

    img = cv2.imread(str(final_imgs[0]), cv2.IMREAD_UNCHANGED)
    if img is None:
        return {
            'valida': False,
            'razon': 'Error critico leyendo archivo procesado',
            'directorio': f"media/pruebas/{nuevo_id}"
        }

    x = np.array([img])
    y = np.array([0])
    dataset = ConjuntivaDataset(x, y, in_chans=4)

    model, device = entrenar.cargar_modelo_entrenado()
    model.eval()
    with torch.no_grad():
        tensor_img = torch.tensor(x).permute(0, 3, 1, 2).float().to(device)  # NCHW
        # Normalizar a 0-1 (el modelo fue entrenado con valores normalizados)
        tensor_img = tensor_img / 255.0
        
        outputs = model(tensor_img)
        probs = torch.softmax(outputs, dim=1)
        prob_anemia = probs[0][1].item()  # Clase 1 = Anemia
        pred = torch.argmax(probs, dim=1).item()
        
        # DEBUG CAVERNÍCOLA
        print(f"DEBUG CONFIANZA:")
        print(f"  Raw outputs: {outputs[0].cpu().numpy()}")
        print(f"  Softmax: {probs[0].cpu().numpy()}")
        print(f"  prob_anemia: {prob_anemia:.4f}")
        print(f"  pred: {pred} ({'CON ANEMIA' if pred == 1 else 'SIN ANEMIA'})")
        
    resultado_clase = "CON ANEMIA" if pred == 1 else "SIN ANEMIA"
    
    # Renombrar todas las carpetas 'SIN ANEMIA' al resultado real si es CON ANEMIA
    if resultado_clase == "CON ANEMIA":
        pasos = ['entrada', 'filtrada', 'area', 'segmentada', 'recortada', 'png', 'resize']
        for paso in pasos:
            ruta_paso_old = os.path.join(ruta_base, paso, 'SIN ANEMIA')
            ruta_paso_new = os.path.join(ruta_base, paso, 'CON ANEMIA')
            if os.path.exists(ruta_paso_old):
                os.makedirs(os.path.dirname(ruta_paso_new), exist_ok=True)
                if os.path.exists(ruta_paso_new):
                    shutil.rmtree(ruta_paso_new)
                os.rename(ruta_paso_old, ruta_paso_new)

    confianza = prob_anemia if pred == 1 else (1 - prob_anemia)
    
    return {
        'valida': True,
        'directorio': f"media/pruebas/{nuevo_id}",
        'prediccion': pred,
        'probable_clase': "Con Anemia" if pred == 1 else "Sin Anemia",
        'categoria': resultado_clase,
        'confianza': round(confianza * 100, 1),  # Porcentaje real del modelo
    }