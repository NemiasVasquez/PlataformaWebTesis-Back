import os
import shutil
from django.conf import settings
from .tasks.preprocesamiento.utils.folders import limpiar_carpeta, asegurar_carpetas
from .tasks.preprocesamiento.filtrarImagenes import filtrar_conjuntiva
from .tasks.preprocesamiento.balanceoImagenes import balancear_dataset
from .tasks.preprocesamiento.extraccionConjuntiva import segmentar_y_recortar_conjuntiva
from .tasks.preprocesamiento.resizeImagenes import redimensionar_imagenes
from .tasks.preprocesamiento.aumentarImagenes import aumentar_dataset

def procesar_logica_carpetas():
    """Crea la estructura de directorios necesaria para el procesamiento."""
    rutas = [
        os.getenv("CARPETA_PROCESADA"),
        os.path.dirname(os.getenv("RUTA_REPORTE_TXT")),
        os.getenv("BALANCEO_CONTROL_ORIGEN"),
        os.getenv("BALANCEO_ANEMIA_ORIGEN"),
        os.getenv("BALANCEO_CONTROL_SALIDA"),
        os.getenv("BALANCEO_ANEMIA_SALIDA"),
        os.getenv("RUTA_BALANCEADAS"),
        os.getenv("RUTA_SEGMENTADAS"),
        os.getenv("RUTA_RECORTADAS"),
        os.getenv("RUTA_PNG"),
        os.getenv("RUTA_AREA"),
        os.getenv("RUTA_PNG_RESIZE"),
        os.getenv("RUTA_AUMENTATION"),
        os.getenv("RUTA_ENTRADA"),
        os.getenv("RUTA_SALIDA"),
        os.getenv("RUTA_NO_FILTRADOS"),
    ]
    abs_paths = [os.path.join(settings.BASE_DIR, r) for r in rutas if r]
    asegurar_carpetas(abs_paths)
    return abs_paths

def ejecutar_paso_filtrado():
    """Ejecuta el filtrado de imágenes por calidad y anatomía."""
    salida = os.path.join(settings.BASE_DIR, os.getenv("RUTA_SALIDA"))
    no_filtrados = os.path.join(settings.BASE_DIR, os.getenv("RUTA_NO_FILTRADOS"))
    reporte = os.path.join(settings.BASE_DIR, os.getenv("RUTA_REPORTE_TXT"))
    
    limpiar_carpeta(salida)
    limpiar_carpeta(no_filtrados)
    if os.path.exists(reporte): os.remove(reporte)
    
    filtrar_conjuntiva(
        os.path.join(settings.BASE_DIR, os.getenv("RUTA_ENTRADA")),
        salida, no_filtrados, reporte
    )

def ejecutar_paso_balanceo():
    """Equilibra el número de imágenes entre clases."""
    c_orig = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_CONTROL_ORIGEN"))
    a_orig = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_ANEMIA_ORIGEN"))
    c_out = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_CONTROL_SALIDA"))
    a_out = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_ANEMIA_SALIDA"))
    for r in [c_out, a_out]: limpiar_carpeta(r)
    balancear_dataset(c_orig, a_orig, c_out, a_out)

def ejecutar_paso_segmentacion():
    """Ejecuta la segmentación y extracción de la conjuntiva."""
    entrada = os.path.join(settings.BASE_DIR, os.getenv("RUTA_BALANCEADAS"))
    segmentadas = os.path.join(settings.BASE_DIR, os.getenv("RUTA_SEGMENTADAS"))
    recortadas = os.path.join(settings.BASE_DIR, os.getenv("RUTA_RECORTADAS"))
    png = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG"))
    area = os.path.join(settings.BASE_DIR, os.getenv("RUTA_AREA"))
    
    for r in [segmentadas, recortadas, png, area]: limpiar_carpeta(r)
    segmentar_y_recortar_conjuntiva(entrada, segmentadas, recortadas, png, area)

def ejecutar_paso_redimensionamiento():
    """Ajusta las imágenes al tamaño final para la red neuronal."""
    input_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG"))
    output_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG_RESIZE"))
    limpiar_carpeta(output_folder)
    redimensionar_imagenes(input_folder, output_folder)

def ejecutar_paso_aumentacion():
    """Aplica data augmentation para balancear y expandir el dataset."""
    input_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG_RESIZE"))
    output_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_AUMENTATION"))
    min_img = int(os.getenv("MIN_IMAGES_AUGMENTATION", 300))
    limpiar_carpeta(output_folder)
    aumentar_dataset(input_folder, output_folder, min_imagenes=min_img)

def mover_basura_imagen(nombre, categoria, razon):
    """Mueve una imagen de las carpetas aceptadas a la carpeta de no filtradas."""
    carpetas = [
        os.getenv("RUTA_SALIDA"), os.getenv("RUTA_BALANCEADAS"),
        os.getenv("RUTA_SEGMENTADAS"), os.getenv("RUTA_RECORTADAS"),
        os.getenv("RUTA_PNG"), os.getenv("RUTA_AREA"), 
        os.getenv("RUTA_PNG_RESIZE"), os.getenv("RUTA_AUMENTATION")
    ]
    
    destino_base = os.path.join(settings.BASE_DIR, os.getenv("RUTA_NO_FILTRADOS"), categoria, razon)
    os.makedirs(destino_base, exist_ok=True)
    
    encontrado = False
    basename = os.path.splitext(nombre)[0]
    
    for folder in carpetas:
        if not folder: continue
        ruta_cat = os.path.join(settings.BASE_DIR, folder, categoria)
        for ext in ['.jpeg', '.jpg', '.png']:
            test_path = os.path.join(ruta_cat, basename + ext)
            if os.path.exists(test_path):
                if not encontrado:
                    shutil.move(test_path, os.path.join(destino_base, basename + ext))
                    encontrado = True
                else:
                    os.remove(test_path)
    return encontrado

def preparar_dataset_modelo():
    """Crea el archivo zip con las imágenes aumentadas para el entrenamiento del modelo."""
    aumentation_path = os.path.join(settings.BASE_DIR, os.getenv("RUTA_AUMENTATION", "media/procesadas/aumentation"))
    data_modelo_path = os.path.join(settings.BASE_DIR, 'media', 'data_modelo')
    conjuntiva_path = os.path.join(data_modelo_path, 'Conjuntiva')
    zip_name = "ConjuntivaPng"
    zip_full_path = os.path.join(data_modelo_path, zip_name)

    # 1. Asegurar carpeta data_modelo y limpiar temporal
    os.makedirs(data_modelo_path, exist_ok=True)
    if os.path.exists(conjuntiva_path):
        shutil.rmtree(conjuntiva_path)
    os.makedirs(conjuntiva_path, exist_ok=True)

    # 2. Copiar carpetas de aumentation
    for cat in ['CON ANEMIA', 'SIN ANEMIA']:
        src = os.path.join(aumentation_path, cat)
        dst = os.path.join(conjuntiva_path, cat)
        if os.path.exists(src):
            shutil.copytree(src, dst)
        else:
            os.makedirs(dst, exist_ok=True)

    # 3. Crear el ZIP (incluyendo la carpeta Conjuntiva dentro)
    shutil.make_archive(zip_full_path, 'zip', root_dir=data_modelo_path, base_dir='Conjuntiva')

    # 4. Limpiar temporal
    shutil.rmtree(conjuntiva_path)
    
    return zip_full_path + ".zip"
