import os
import django
import sys

# Añadir el directorio actual al path para que encuentre los módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configurar el entorno de Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'anemia.settings')
django.setup()

from django.conf import settings
from dotenv import load_dotenv
from imagenes.tasks.preprocesamiento.filtrarImagenes import filtrar_conjuntiva
from imagenes.tasks.preprocesamiento.balanceoImagenes import balancear_dataset
from imagenes.tasks.preprocesamiento.extraccionConjuntiva import segmentar_y_recortar_conjuntiva
from imagenes.tasks.preprocesamiento.resizeImagenes import redimensionar_imagenes
from imagenes.views import limpiar_carpeta

# Cargar variables de entorno
load_dotenv()

def ejecutar_reproceso():
    print("\n--- INICIANDO REPROCESAMIENTO DE IMÁGENES ---")
    
    # 1. FILTRADO
    print("\n[1/4] Ejecutando Filtrado...")
    ruta_entrada = os.path.join(settings.BASE_DIR, os.getenv("RUTA_ENTRADA"))
    ruta_salida = os.path.join(settings.BASE_DIR, os.getenv("RUTA_SALIDA"))
    ruta_no_filtrados = os.path.join(settings.BASE_DIR, os.getenv("RUTA_NO_FILTRADOS"))
    ruta_reporte = os.path.join(settings.BASE_DIR, os.getenv("RUTA_REPORTE_TXT"))
    
    limpiar_carpeta(ruta_salida)
    limpiar_carpeta(ruta_no_filtrados)
    if os.path.exists(ruta_reporte): os.remove(ruta_reporte)
    
    filtrar_conjuntiva(ruta_entrada, ruta_salida, ruta_no_filtrados, ruta_reporte)

    # 2. BALANCEO
    print("\n[2/4] Ejecutando Balanceo...")
    control_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_CONTROL_ORIGEN"))
    anemia_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_ANEMIA_ORIGEN"))
    output_control_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_CONTROL_SALIDA"))
    output_anemia_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_ANEMIA_SALIDA"))
    
    limpiar_carpeta(output_control_folder)
    limpiar_carpeta(output_anemia_folder)
    balancear_dataset(control_folder, anemia_folder, output_control_folder, output_anemia_folder)

    # 3. SEGMENTACIÓN (Aquí es donde está el nuevo código de cierre de medialuna)
    print("\n[3/4] Ejecutando Segmentación (Extracción de Conjuntiva)...")
    entrada = os.path.join(settings.BASE_DIR, os.getenv("RUTA_BALANCEADAS"))
    salida_segmentadas = os.path.join(settings.BASE_DIR, os.getenv("RUTA_SEGMENTADAS"))
    salida_recortadas = os.path.join(settings.BASE_DIR, os.getenv("RUTA_RECORTADAS"))
    salida_png = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG"))
    
    limpiar_carpeta(salida_segmentadas)
    limpiar_carpeta(salida_recortadas)
    limpiar_carpeta(salida_png)
    
    segmentar_y_recortar_conjuntiva(entrada, salida_segmentadas, salida_recortadas, salida_png)

    # 4. REDIMENSIONAMIENTO
    print("\n[4/4] Ejecutando Redimensionamiento...")
    input_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG"))
    output_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG_RESIZE"))
    
    limpiar_carpeta(output_folder)
    redimensionar_imagenes(input_folder, output_folder)

    print("\n✅ PROCESO COMPLETADO EXITOSAMENTE.")
    print(f"Revisa las carpetas en: {os.path.join(settings.BASE_DIR, 'media/procesadas')}")

if __name__ == "__main__":
    ejecutar_reproceso()
