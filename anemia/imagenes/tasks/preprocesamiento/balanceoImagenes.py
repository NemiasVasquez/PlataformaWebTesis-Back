import os
import random
import shutil

def balancear_dataset(ruta_control, ruta_anemia, salida_control, salida_anemia):
    """
    BALANCEO DE CLASES: Asegura que haya la misma cantidad de imágenes 
    de personas con y sin anemia, duplicando imágenes aleatoriamente 
    si es necesario para evitar sesgos en el modelo.
    """
    os.makedirs(salida_control, exist_ok=True)
    os.makedirs(salida_anemia, exist_ok=True)

    archivos_c = [f for f in os.listdir(ruta_control) if os.path.isfile(os.path.join(ruta_control, f))]
    archivos_a = [f for f in os.listdir(ruta_anemia) if os.path.isfile(os.path.join(ruta_anemia, f))]

    if not archivos_c or not archivos_a:
        print("Faltan imágenes para el balanceo.")
        return

    n_max = max(len(archivos_c), len(archivos_a))

    def copiar_balanceado(lista, origen, destino, total):
        extra = random.choices(lista, k=total - len(lista))
        for i, f in enumerate(lista + extra):
            shutil.copy(os.path.join(origen, f), os.path.join(destino, f"{i}_{f}"))

    copiar_balanceado(archivos_c, ruta_control, salida_control, n_max)
    copiar_balanceado(archivos_a, ruta_anemia, salida_anemia, n_max)

    print(f"Balanceo completado: {n_max} imágenes por categoría.")
