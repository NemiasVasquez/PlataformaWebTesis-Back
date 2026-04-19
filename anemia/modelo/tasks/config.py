import os

CATEGORIAS = ['SIN ANEMIA', 'CON ANEMIA']
DATA_DIR = os.path.join('media', 'procesadas', 'resize')
IMG_WIDTH, IMG_HEIGHT = 224, 224
EPOCHS = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODELO_NFNET_PATH = os.path.join('modelos_guardados', 'nfnet_f0_entrenado.pt')
