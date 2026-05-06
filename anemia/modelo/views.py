from django.http import JsonResponse
from .tasks import cargar_datos, entrenar, evaluar
from django.views.decorators.csrf import csrf_exempt
from .tasks.dataset import ConjuntivaDataset
from torch.utils.data import DataLoader
import numpy as np
import os, cv2, uuid
from .tasks.config import MODELO_NFNET_PATH, IMG_WIDTH, IMG_HEIGHT
from sklearn.metrics import confusion_matrix
from imagenes.tasks.preprocesamiento.filtrarImagenes import filtrar_conjuntiva
from imagenes.tasks.preprocesamiento.extraccionConjuntiva import segmentar_y_recortar_conjuntiva
from imagenes.tasks.preprocesamiento.resizeImagenes import redimensionar_imagenes
from django.conf import settings
from pathlib import Path
from django.conf.urls.static import static
from .tasks.evaluar_imagen import evaluar_imagen_individual
import time

def entrenar_modelo_nfnet(request):
    entrenamientoCompleto = request.GET.get('entrenamientoCompleto', 'true').lower() == 'true'

    x_train, x_test, y_train, y_test = cargar_datos.cargar_imagenes()

    if entrenamientoCompleto or not os.path.exists(MODELO_NFNET_PATH):
        model, device = entrenar.entrenar_nfnet(x_train, y_train)
    else:
        model, device = entrenar.cargar_modelo_entrenado()
    
    acc, reporte, y_true, y_pred = evaluar.evaluar_modelo(model, x_test, y_test)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    evaluar.graficar_matriz_confusion(cm, labels=["Sin Anemia", "Con Anemia"])  # solo guarda el PNG

    return JsonResponse({
        "mensaje": "Evaluación completada con modelo NFNet",
        "exactitud": acc,
        "reporte": reporte,
        "confusion_matrix": cm.tolist()
    })


@csrf_exempt
def evaluar_imagen_anemia(request):
    if request.method != 'POST':
        return JsonResponse({'alert': 'Método no permitido'}, status=405)

    imagen = request.FILES.get('imagen')
    if not imagen:
        return JsonResponse({'alert': 'No se envió imagen'}, status=400)

    resultado = evaluar_imagen_individual(imagen)

    if not resultado['valida']:
        return JsonResponse({
            'error': 'Imagen no válida para el análisis',
            'alert': resultado['razon'],
            'directorio_procesado': resultado['directorio']
        },status=400)

    return JsonResponse({
        "mensaje": "Evaluación realizada",
        "directorio_procesado": resultado['directorio'],
        "prediccion": resultado['prediccion'],
        "probable_clase": resultado['probable_clase'],
        "categoria": resultado.get('categoria', 'SIN ANEMIA'),
        "confianza": resultado.get('confianza'),
        "rcap": resultado.get('rcap'),
        "exactitud": resultado.get('exactitud'),
        "robustez": resultado.get('robustez'),
        "transparencia": resultado.get('transparencia'),
        "tiempo": resultado.get('tiempo'),
    })

@csrf_exempt
def evaluar_indicadores(request):
    if request.method != 'POST':
        return JsonResponse({'alert': 'Método no permitido'}, status=405)

    from .tasks import cargar_datos, entrenar
    from .tasks.explicabilidad import generate_smoothgrad, calcular_nivel_detalle, calcular_exactitud_areas
    from .indicadores.robustez import calcular_robustez_general, calcular_robustez_imagen
    from .indicadores.transparencia import calcular_transparencia_diagnostico, calcular_transparencia_general
    import torch

    # Cargar 5 imagenes de test para evaluar
    x_train, x_test, y_train, y_test = cargar_datos.cargar_imagenes()
    model, device = entrenar.cargar_modelo_entrenado()
    model.eval()

    num_imgs = min(5, len(x_test))
    x_test_sample = x_test[:num_imgs]
    y_test_sample = y_test[:num_imgs]

    saliency_maps = []
    images_rgb = []
    tiempos = []
    
    for i in range(num_imgs):
        tensor_img = torch.tensor([x_test_sample[i]]).permute(0, 3, 1, 2).float().to(device) / 255.0
        img_np = x_test_sample[i]
        
        inicio_t = time.time()
        heatmap, overlay = generate_smoothgrad(model, device, tensor_img, img_np, int(y_test_sample[i]))
        fin_t = time.time()
        tiempos.append(fin_t - inicio_t)
        
        gray_map = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        
        saliency_maps.append(gray_map)
        images_rgb.append(img_np)
        
        # Guardar tensores para robustez
        if 'images_tensors' not in locals(): images_tensors = []
        images_tensors.append(tensor_img)
        
        # Calcular ti individual para el lote
        if 'ti_list' not in locals(): ti_list = []
        ti_val = calcular_transparencia_diagnostico(model, tensor_img, gray_map, device=device)
        ti_list.append(ti_val)
        
    t_promedio = sum(tiempos) / len(tiempos) if tiempos else 0.0
        
    # Calcular D (Nivel de Detalle)
    res_ind = calcular_nivel_detalle(images_rgb, saliency_maps, model, device=device)
    d_val = round(res_ind['D'], 2)
    rcap_list = [round(v, 4) for v in res_ind['RCAP_valores']]
    
    # Calcular P (Exactitud de Áreas)
    res_p = calcular_exactitud_areas(images_rgb, saliency_maps, threshold=0.1)
    p_val = round(res_p['P'], 2)
    p_list = [round(v * 100, 2) for v in res_p['P_valores']]
    
    # Calcular RG (Robustez General)
    res_rg = calcular_robustez_general(model, images_tensors, device=device)
    rg_val = round(res_rg['RG'], 4)
    rg_list = [round(v, 4) for v in res_rg['rhos_individuales']]
    
    # Calcular NT (Transparencia General)
    nt_val = round(calcular_transparencia_general(ti_list), 2)
    ti_final_list = [round(v, 2) for v in ti_list]
    
    return JsonResponse({
        'status': 'success',
        'd_metric': d_val,
        'rcap_individuales': rcap_list,
        'p_metric': p_val,
        'p_individuales': p_list,
        'rg_metric': rg_val,
        'rg_individuales': rg_list,
        'nt_metric': nt_val,
        'nt_individuales': ti_final_list,
        't_promedio': round(t_promedio, 3),
        'procesadas': len(images_rgb)
    })