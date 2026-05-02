import cv2
import numpy as np
import os

def ojo_abierto(img, extractor):
    """
    Usa el extractor para detectar el iris y validar que el área ocupada 
    sea suficiente para considerar que el ojo está abierto.
    """
    _, radius = extractor.detect_eye_anchor(img)
    if radius is None:
        return False
    area_img = img.shape[0] * img.shape[1]
    area_iris = np.pi * radius**2
    umbral = float(os.getenv("OJO_MIN_AREA_FRACCION", 0.005))
    return area_iris > area_img * umbral

def contiene_esclerotica(img):
    """
    Busca tonos blancos en el ojo para confirmar presencia de esclerótica.
    Acepta esclerótica levemente enrojecida (conjuntiva anémica).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Relajado: S<80 y V>130 cubre blanco puro y blanco-rosado de anemia
    mask_white = cv2.inRange(hsv, np.array([0, 0, 130]), np.array([180, 80, 255]))
    umbral = float(os.getenv("ESCLEROTICA_UMBRAL_AREA", 0.00001))
    return (np.count_nonzero(mask_white) / (img.shape[0] * img.shape[1])) > umbral
