import torch
import numpy as np
import cv2

def get_raw_smoothgrad_map(model, device, img_tensor, class_idx, num_samples=20, stdev_spread=0.15):
    """
    Versión simplificada de generate_smoothgrad que retorna el mapa de gradientes crudo (0-1).
    """
    model.eval()
    stdev = stdev_spread * (img_tensor.max() - img_tensor.min()).item()
    total_gradients = torch.zeros_like(img_tensor)
    
    # Fijar semilla para que el ruido sea determinista en la comparación
    torch.manual_seed(42)
    
    for _ in range(num_samples):
        noise = torch.randn_like(img_tensor).to(device) * stdev
        noisy_img = (img_tensor + noise).detach().requires_grad_(True)
        
        outputs = model(noisy_img)
        score = outputs[0, class_idx]
        
        model.zero_grad()
        score.backward()
        
        total_gradients += noisy_img.grad.abs()
        
    avg_gradients = total_gradients / num_samples
    grad_np = avg_gradients[0].detach().cpu().numpy()
    grad_np = np.max(grad_np, axis=0)
    
    # Normalizar 0-1
    denom = np.max(grad_np) - np.min(grad_np)
    if denom > 0:
        grad_np = (grad_np - np.min(grad_np)) / denom
    else:
        grad_np = np.zeros_like(grad_np)
        
    return grad_np

def calcular_sensibilidad_explicabilidad(model, device, img_tensor, class_idx, epsilon=1e-5):
    """
    Implementa la métrica de Sensibilidad (S).
    S(x) = || (Phi(x + eps*e) - Phi(x)) / eps ||
    """
    # 1. Mapa original Phi(x)
    phi_x = get_raw_smoothgrad_map(model, device, img_tensor, class_idx)
    
    # 2. Perturbación mínima epsilon * e_j
    # Usamos un ruido unitario escalado por epsilon
    noise = torch.randn_like(img_tensor).to(device)
    # Normalizar ruido para que sea "unitario" en norma L2
    noise = noise / (torch.norm(noise) + 1e-8)
    
    img_perturbed = img_tensor + (epsilon * noise)
    
    # 3. Mapa perturbado Phi(x + eps*e)
    phi_x_eps = get_raw_smoothgrad_map(model, device, img_perturbed, class_idx)
    
    # 4. Diferencial aproximado
    # Gradiente = (Phi(x+eps) - Phi(x)) / eps
    diff = (phi_x_eps - phi_x) / epsilon
    
    # Magnitud (Norma Frobenius / L2 de la diferencia)
    sensibilidad = np.linalg.norm(diff)
    
    # Normalizar por el número de píxeles para que sea comparable
    sensibilidad = sensibilidad / np.sqrt(diff.size)
    
    return sensibilidad

def calcular_sensibilidad_general(sens_list):
    """
    Media aritmética de la sensibilidad.
    """
    if not sens_list:
        return 0.0
    return np.mean(sens_list)
