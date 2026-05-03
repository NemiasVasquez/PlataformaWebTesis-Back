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
    })

@csrf_exempt
def evaluar_indicadores(request):
    if request.method != 'POST':
        return JsonResponse({'alert': 'Método no permitido'}, status=405)

    from .tasks import cargar_datos, entrenar
    from .tasks.explicabilidad import generate_smoothgrad, calcular_nivel_detalle
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
    
    for i in range(num_imgs):
        img_np = x_test_sample[i]
        tensor_img = torch.tensor(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        
        heatmap, overlay = generate_smoothgrad(model, device, tensor_img, img_np, int(y_test_sample[i]))
        gray_map = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        
        saliency_maps.append(gray_map)
        images_rgb.append(img_np)
        
    resultado = calcular_nivel_detalle(images_rgb, saliency_maps, model, device=device)
    
    return JsonResponse({
        "mensaje": "Indicadores calculados exitosamente",
        "D": round(resultado['D'], 2),
        "RCAP": [round(val, 4) for val in resultado['RCAP_valores']],
        "imagenes_evaluadas": num_imgs
    })
    