import os
import shutil

def limpiar_carpeta(ruta):
    """
    Elimina todos los archivos y subdirectorios dentro de una ruta específica.
    """
    if os.path.exists(ruta):
        for archivo in os.listdir(ruta):
            ruta_completa = os.path.join(ruta, archivo)
            try:
                if os.path.isfile(ruta_completa) or os.path.islink(ruta_completa):
                    os.unlink(ruta_completa)
                elif os.path.isdir(ruta_completa):
                    shutil.rmtree(ruta_completa)
            except Exception as e:
                print(f"Error eliminando {ruta_completa}: {e}")

def asegurar_carpetas(rutas):
    """
    Crea las carpetas de la lista si no existen.
    """
    for ruta in rutas:
        if ruta:
            os.makedirs(ruta, exist_ok=True)
