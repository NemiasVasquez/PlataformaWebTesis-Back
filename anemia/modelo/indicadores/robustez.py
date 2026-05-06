import torch
import numpy as np

def calcular_robustez_imagen(model, image_tensor, device='cuda'):
    """
    Calcula la métrica de robustez rho(x) para una imagen individual.
    rho(x) = min_{j != i} (Psi_i(x) - Psi_j(x)) / ||grad(Psi_i(x)) - grad(Psi_j(x))||
    """
    model.eval()
    # Asegurar que la imagen requiere gradiente
    x = image_tensor.clone().detach().to(device)
    x.requires_grad = True
    
    # Obtener logits (Psi)
    outputs = model(x) # Shape [1, num_classes]
    num_classes = outputs.shape[1]
    
    # Identificar clase predicha (i)
    pred_idx = torch.argmax(outputs, dim=1).item()
    psi_i = outputs[0, pred_idx]
    
    # Calcular gradiente para la clase predicha i
    grad_i = torch.autograd.grad(psi_i, x, retain_graph=True)[0]
    
    min_rho = float('inf')
    
    # Comparar con todas las demas clases j
    for j in range(num_classes):
        if j == pred_idx:
            continue
            
        psi_j = outputs[0, j]
        
        # Calcular gradiente para la clase j
        grad_j = torch.autograd.grad(psi_j, x, retain_graph=True)[0]
        
        # Diferencial de puntuaciones
        diff_psi = (psi_i - psi_j).item()
        
        # Norma de la diferencia de gradientes
        grad_diff_norm = torch.norm(grad_i - grad_j).item()
        
        # Evitar división por cero
        rho_j = diff_psi / grad_diff_norm if grad_diff_norm != 0 else 0.0
            
        if rho_j < min_rho:
            min_rho = rho_j
            
    return min_rho if min_rho != float('inf') else 0.0

def calcular_robustez_general(model, images_tensors, device='cuda'):
    """
    Calcula la Robustez General (RG) como el promedio de rho(x) para un conjunto de imagenes.
    """
    rhos = []
    for img_t in images_tensors:
        # Asegurarse de que sea un batch de 1 si viene como CHW
        if len(img_t.shape) == 3:
            img_t = img_t.unsqueeze(0)
            
        rho = calcular_robustez_imagen(model, img_t, device)
        rhos.append(rho)
        
    rg = np.mean(rhos) if rhos else 0.0
    return {
        'RG': rg,
        'rhos_individuales': rhos
    }
