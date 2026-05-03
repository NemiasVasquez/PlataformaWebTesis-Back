import numpy as np
import torch
from PIL import Image

def calcular_nivel_detalle(images, saliency_maps, model, transform=None, device='cuda', j=10, class_idx=1):
    """
    Calcula la métrica de Nivel de Detalle (D) basada en el RCAP para evaluar
    las explicaciones (mapas de prominencia) del modelo NFNet en la tesis.
    
    Parámetros:
    - images: Lista de imágenes segmentadas (NumPy arrays HxWxC). Puede ser RGBA (PNG transparente) o RGB.
              También acepta una sola imagen.
    - saliency_maps: Lista de mapas de prominencia/calor (M) correspondientes (NumPy arrays HxW).
    - model: Modelo de PyTorch (NFNet) cargado y en modo evaluación.
    - transform: Funciones de transformación (torchvision.transforms) para el preprocesamiento de la imagen.
    - device: Dispositivo para ejecutar la inferencia ('cuda' o 'cpu').
    - j: Número de particiones (por defecto 10, correspondiente a deciles).
    - class_idx: Índice de la clase positiva (por defecto 1, asumiendo 0=sano, 1=anemia).
    
    Retorna:
    - dict: Contiene el valor general 'D' (porcentaje) y la lista de valores individuales 'RCAP' por imagen.
    """
    # Convertir a lista si se pasa un solo elemento
    if not isinstance(images, (list, tuple)):
        images = [images]
        saliency_maps = [saliency_maps]
        
    model.eval()
    model.to(device)
    
    rcap_values = []
    
    for img, m_map in zip(images, saliency_maps):
        # 1. Identificar píxeles válidos (ignorar fondo transparente o negro)
        # Asegurarse de que la máscara se aplique correctamente ignorando el fondo
        if img.shape[-1] == 4:
            # Caso PNG transparente: el canal Alpha > 0 define la conjuntiva
            valid_mask = img[..., 3] > 0
        else:
            # Caso RGB: los píxeles mayores a 0 en cualquier canal definen la conjuntiva
            valid_mask = np.any(img > 0, axis=-1)
            
        total_intensity = np.sum(m_map[valid_mask])
        
        # Evitar errores si la imagen es puro fondo o el mapa está vacío
        if total_intensity == 0 or not np.any(valid_mask):
            rcap_values.append(0.0)
            continue
            
        valid_m_values = m_map[valid_mask]
        rcap_sum = 0.0
        
        # 2. Iterar sobre las j particiones (ej: top 10%, 20%, ..., 100%)
        for step in range(1, j + 1):
            k_percent = (step / j) * 100
            
            # Encontrar el umbral (threshold) para el top k% de importancia
            # Usamos 100 - k_percent porque percentile() toma percentiles desde el más bajo.
            threshold = np.percentile(valid_m_values, 100 - k_percent)
            
            # Partición p_k: Píxeles que superan el umbral Y son parte de la conjuntiva válida
            p_k_mask = (m_map >= threshold) & valid_mask
            
            # 3. Ruido Visual: proporción de la intensidad de p_k respecto al total de M
            ruido_visual = np.sum(m_map[p_k_mask]) / total_intensity
            
            # 4. Localización: Crear la versión "enmascarada" de la imagen
            img_masked = np.zeros_like(img)
            img_masked[p_k_mask] = img[p_k_mask]
            
            # Usar la imagen completa (con sus canales originales, ej. RGBA) porque el modelo la espera así
            img_to_model = img_masked
            
            # Inferencia con PyTorch
            if transform:
                img_pil = Image.fromarray(img_to_model.astype('uint8'))
                img_tensor = transform(img_pil).unsqueeze(0).to(device)
            else:
                # Preprocesamiento básico si no se proporcionan transforms
                img_tensor = torch.from_numpy(img_to_model).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                img_tensor = img_tensor.to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                
                # Obtener el score de confianza para la clase positiva (sigma(fc(I_pk)))
                if outputs.shape[-1] == 1:
                    # Caso de salida binaria (1 neurona)
                    score = torch.sigmoid(outputs).item()
                else:
                    # Caso de múltiples clases (softmax)
                    score = torch.softmax(outputs, dim=-1)[0, class_idx].item()
                    
            localizacion = score
            
            # Aplicar sumatoria: (Ruido x Localización)
            rcap_sum += (ruido_visual * localizacion)
            
        # Calcular el RCAP final de la imagen actual
        rcap = rcap_sum / j
        rcap_values.append(rcap)
        
    # Cálculo de D (Nivel de Detalle)
    # Promedio de los RCAP de las n imágenes multiplicado por 100
    D = np.mean(rcap_values) * 100 if rcap_values else 0.0
    
    return {
        'D': D,
        'RCAP_valores': rcap_values
    }
