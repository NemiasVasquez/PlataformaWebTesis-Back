import os
from PIL import Image

def redimensionar_imagenes(ruta_entrada, ruta_salida, size=(150, 150)):
    """
    REDIMENSIONADO: Ajusta todas las imágenes a un tamaño uniforme 
    para que la red neuronal pueda procesarlas correctamente.
    """
    for root, _, files in os.walk(ruta_entrada):
        rel_path = os.path.relpath(root, ruta_entrada)
        target_dir = os.path.join(ruta_salida, rel_path)
        os.makedirs(target_dir, exist_ok=True)

        for f in files:
            if f.lower().endswith(('png', 'jpg', 'jpeg')):
                try:
                    img = Image.open(os.path.join(root, f))
                    img.resize(size, Image.LANCZOS).save(os.path.join(target_dir, f))
                except Exception as e:
                    print(f"Error procesando {f}: {e}")

    print(f"Redimensionamiento a {size} completado.")
