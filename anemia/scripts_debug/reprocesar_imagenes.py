import os
import django
import sys

# Añadir el directorio raíz al path para que encuentre los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar el entorno de Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'anemia.settings')
django.setup()

from django.conf import settings
from dotenv import load_dotenv
from imagenes.tasks.preprocesamiento.filtrarImagenes import filtrar_conjuntiva
from imagenes.tasks.preprocesamiento.balanceoImagenes import balancear_dataset
from imagenes.tasks.preprocesamiento.extraccionConjuntiva import segmentar_y_recortar_conjuntiva
from imagenes.tasks.preprocesamiento.resizeImagenes import redimensionar_imagenes
from imagenes.tasks.preprocesamiento.recortarOjo import recortar_ojos_dataset
from imagenes.tasks.preprocesamiento.utils.folders import limpiar_carpeta

# Cargar variables de entorno
load_dotenv()

def ejecutar_reproceso():
    print("\n--- INICIANDO REPROCESAMIENTO DE IMÁGENES ---")
    
    # 1. RECORTE DE OJOS (Nuevo paso para centrar antes de filtrar)
    print("\n[1/5] Ejecutando Recorte de Ojos...")
    ruta_originales = os.path.join(settings.BASE_DIR, os.getenv("RUTA_ENTRADA"))
    ruta_recortados = os.path.join(settings.BASE_DIR, os.getenv("RUTA_RECORTADO_OJO"))
    
    limpiar_carpeta(ruta_recortados)
    recortar_ojos_dataset(ruta_originales, ruta_recortados)

    # 2. FILTRADO
    print("\n[2/5] Ejecutando Filtrado...")
    # Usamos los recortes como entrada para mayor precisión
    ruta_entrada = ruta_recortados 
    ruta_salida = os.path.join(settings.BASE_DIR, os.getenv("RUTA_SALIDA"))
    ruta_no_filtrados = os.path.join(settings.BASE_DIR, os.getenv("RUTA_NO_FILTRADOS"))
    ruta_reporte = os.path.join(settings.BASE_DIR, os.getenv("RUTA_REPORTE_TXT"))
    
    limpiar_carpeta(ruta_salida)
    limpiar_carpeta(ruta_no_filtrados)
    if os.path.exists(ruta_reporte): os.remove(ruta_reporte)
    
    filtrar_conjuntiva(ruta_entrada, ruta_salida, ruta_no_filtrados, ruta_reporte)

    # 3. BALANCEO
    print("\n[3/5] Ejecutando Balanceo...")
    control_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_CONTROL_ORIGEN"))
    anemia_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_ANEMIA_ORIGEN"))
    output_control_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_CONTROL_SALIDA"))
    output_anemia_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_ANEMIA_SALIDA"))
    
    limpiar_carpeta(output_control_folder)
    limpiar_carpeta(output_anemia_folder)
    balancear_dataset(control_folder, anemia_folder, output_control_folder, output_anemia_folder)

    # 4. SEGMENTACIÓN
    print("\n[4/5] Ejecutando Segmentación (Extracción de Conjuntiva)...")
    entrada = os.path.join(settings.BASE_DIR, os.getenv("RUTA_BALANCEADAS"))
    salida_segmentadas = os.path.join(settings.BASE_DIR, os.getenv("RUTA_SEGMENTADAS"))
    salida_recortadas = os.path.join(settings.BASE_DIR, os.getenv("RUTA_RECORTADAS"))
    salida_png = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG"))
    salida_area = os.path.join(settings.BASE_DIR, os.getenv("RUTA_AREA"))
    
    limpiar_carpeta(salida_segmentadas)
    limpiar_carpeta(salida_recortadas)
    limpiar_carpeta(salida_png)
    limpiar_carpeta(salida_area)
    
    segmentar_y_recortar_conjuntiva(entrada, salida_segmentadas, salida_recortadas, salida_png, salida_area)

    # 5. REDIMENSIONAMIENTO
    print("\n[5/5] Ejecutando Redimensionamiento...")
    input_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG"))
    output_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG_RESIZE"))
    
    limpiar_carpeta(output_folder)
    redimensionar_imagenes(input_folder, output_folder)

    print("\n PROCESO COMPLETADO EXITOSAMENTE.")
    print(f"Revisa las carpetas en: {os.path.join(settings.BASE_DIR, 'media/procesadas')}")

if __name__ == "__main__":
    ejecutar_reproceso()
