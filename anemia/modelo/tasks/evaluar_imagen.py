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
from .explicabilidad import generate_smoothgrad
from ..indicadores.nivel_detalle import calcular_nivel_detalle

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

    # === EXPLICABILIDAD SMOOTHGRAD ===
    try:
        ruta_exp = os.path.join(ruta_base, 'explicabilidad', resultado_clase)
        os.makedirs(ruta_exp, exist_ok=True)
        
        path_original = os.path.join(ruta_base, 'entrada', resultado_clase, 'imagen.jpeg')
        img_original = cv2.imread(path_original)
        
        # Ojo, al usar glob el nombre del area podría cambiar si usa uuid, busquemos el primero
        path_area_dir = os.path.join(ruta_base, 'area', resultado_clase)
        img_area = None
        if os.path.exists(path_area_dir):
            archivos_area = os.listdir(path_area_dir)
            if archivos_area:
                img_area = cv2.imread(os.path.join(path_area_dir, archivos_area[0]))
        
        # SmoothGrad requiere gradientes, por lo que creamos tensor y lo enviamos sin with torch.no_grad()
        tensor_img_grad = torch.tensor(x).permute(0, 3, 1, 2).float().to(device) / 255.0
        
        heatmap, overlay = generate_smoothgrad(model, device, tensor_img_grad, img_original, pred)
        
        if img_area is not None:
            if img_area.shape[:2] != heatmap.shape[:2]:
                img_area = cv2.resize(img_area, (heatmap.shape[1], heatmap.shape[0]))
            overlay_delineado = cv2.addWeighted(img_area, 0.6, heatmap, 0.4, 0)
        else:
            overlay_delineado = overlay
            
        cv2.imwrite(os.path.join(ruta_exp, 'heatmap.jpeg'), heatmap)
        cv2.imwrite(os.path.join(ruta_exp, 'overlay.jpeg'), overlay)
        cv2.imwrite(os.path.join(ruta_exp, 'overlay_delineado.jpeg'), overlay_delineado)
        
        # Calcular indicador RCAP para esta imagen
        try:
            gray_map = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
            # img es cv2 IMREAD_UNCHANGED (tiene alpha si es png)
            img_rgb_o_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA) if img.shape[-1] == 4 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res_ind = calcular_nivel_detalle([img_rgb_o_rgba], [gray_map], model, device=device, class_idx=pred)
            rcap_val = round(res_ind['RCAP_valores'][0], 4)
        except Exception as e_ind:
            import traceback
            print(f"Error en calcular_nivel_detalle:\n{traceback.format_exc()}")
            rcap_val = 0.0

        print("  [OK] SmoothGrad generado exitosamente.")
    except Exception as e:
        print(f"  [X] Error generando SmoothGrad: {e}")
        rcap_val = 0.0
        
    confianza = prob_anemia if pred == 1 else (1 - prob_anemia)
    
    return {
        'valida': True,
        'directorio': f"media/pruebas/{nuevo_id}",
        'prediccion': pred,
        'probable_clase': "Con Anemia" if pred == 1 else "Sin Anemia",
        'categoria': resultado_clase,
        'confianza': round(confianza * 100, 1),
        'rcap': rcap_val
    }