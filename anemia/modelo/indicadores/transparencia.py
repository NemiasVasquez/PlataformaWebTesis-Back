import numpy as np
import torch
import shap

def calcular_transparencia_diagnostico(model, image_tensor, smoothgrad_map, device='cuda', threshold_percent=15):
    """
    Calcula el grado de transparencia (ti) comparando SHAP (A) con SmoothGrad (B).
    ti = (|A INTER B| / |B|) * 100
    """
    model.eval()
    # 1. Obtener Mapa SHAP (A)
    # GradientExplainer es eficiente para redes neuronales en PyTorch
    # Necesitamos una imagen de fondo (background) para el explainer, 
    # o podemos usar la misma imagen con ruido/ceros si es individual.
    # Para simplificar y siguiendo el estándar de SHAP con PyTorch:
    background = torch.zeros_like(image_tensor).to(device)
    explainer = shap.GradientExplainer(model, background)
    
    # shap_values es una lista por clase, tomamos la clase predicha
    with torch.no_grad():
        output = model(image_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        
    shap_values = explainer.shap_values(image_tensor)
    print(f"DEBUG SHAP: Raw type={type(shap_values)}, shape={np.array(shap_values).shape if not isinstance(shap_values, list) else [v.shape for v in shap_values]}")
    
    # Manejar si es lista (una por clase) o array único
    if isinstance(shap_values, list):
        target_shap = shap_values[pred_idx] if len(shap_values) > pred_idx else shap_values[0]
    else:
        target_shap = shap_values

    # Manejar dimensiones extras (algunas versiones de SHAP añaden clase al final)
    if len(target_shap.shape) == 5: # [batch, channels, H, W, classes]
        target_shap = target_shap[..., pred_idx]

    # Asegurar que tomamos la primera imagen del batch y sumamos canales
    if len(target_shap.shape) == 4: # [batch, channels, H, W]
        map_a = np.abs(target_shap[0]).sum(axis=0)
    elif len(target_shap.shape) == 3: # [channels, H, W]
        map_a = np.abs(target_shap).sum(axis=0)
    else:
        map_a = np.abs(target_shap)
    
    # 2. Mapa SmoothGrad (B)
    map_b = smoothgrad_map.astype(np.float32)
    
    print(f"DEBUG SHAP: Final Map A shape={map_a.shape}, Map B shape={map_b.shape}")

    if map_b.shape != map_a.shape:
        # Esto no debería pasar si vienen de la misma imagen, pero por si acaso:
        from cv2 import resize
        map_b = resize(map_b, (map_a.shape[1], map_a.shape[0]))

    # 3. Binarización (Top 15%)
    def binarizar_top(m, percent):
        thresh = np.percentile(m, 100 - percent)
        return m >= thresh

    bin_a = binarizar_top(map_a, threshold_percent)
    bin_b = binarizar_top(map_b, threshold_percent)

    # 4. Cálculo de Coincidencia (ti)
    interseccion = np.sum(bin_a & bin_b)
    total_b = np.sum(bin_b)
    
    print(f"DEBUG SHAP: Interseccion={interseccion}, Total B={total_b}")

    if total_b == 0:
        ti = 0.0
    else:
        ti = (interseccion / total_b) * 100

    return ti

def calcular_transparencia_general(ti_list):
    """
    Calcula la Visibilidad de Características Claras (NT) como promedio.
    """
    if not ti_list:
        return 0.0
    return np.mean(ti_list)
