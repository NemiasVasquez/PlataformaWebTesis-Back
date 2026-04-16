import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from .config import DATA_DIR, CATEGORIAS, TEST_SIZE, RANDOM_STATE

def cargar_imagenes():
    X, Y = [], []
    for i, categoria in enumerate(CATEGORIAS):
        ruta = glob.glob(f"{DATA_DIR}/{categoria}/*.png")
        for img_path in ruta:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None or img.shape[2] != 4: continue
            X.append(img)
            Y.append(i)
    return train_test_split(np.array(X), np.array(Y), test_size=TEST_SIZE, random_state=RANDOM_STATE)
