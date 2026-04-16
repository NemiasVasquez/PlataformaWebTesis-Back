from django.conf import settings
from django.http import JsonResponse
from imagenes.tasks.preprocesamiento.filtrarImagenes import filtrar_conjuntiva
from imagenes.tasks.preprocesamiento.balanceoImagenes import balancear_dataset
from imagenes.tasks.preprocesamiento.extraccionConjuntiva import segmentar_y_recortar_conjuntiva
from imagenes.tasks.preprocesamiento.resizeImagenes import redimensionar_imagenes
import shutil
import os
from dotenv import load_dotenv

load_dotenv()  # Carga el .env solo una vez al inicio

def limpiar_carpeta(ruta):
    if os.path.exists(ruta):
        for archivo in os.listdir(ruta):
            ruta_completa = os.path.join(ruta, archivo)
            try:
                if os.path.isfile(ruta_completa) or os.path.islink(ruta_completa):
                    os.unlink(ruta_completa)
                elif os.path.isdir(ruta_completa):
                    shutil.rmtree(ruta_completa)
            except Exception as e:
                print(f"❌ Error eliminando {ruta_completa}: {e}")

#1º Vista: crear carpetas iniciales
def crear_carpetas_iniciales(request):
    rutas = [
        # Carpetas base
        os.getenv("CARPETA_PROCESADA"),

        # Filtrado de imágenes (solo carpeta del reporte)
        os.path.dirname(os.getenv("RUTA_REPORTE_TXT")),

        # Balanceo
        os.getenv("BALANCEO_CONTROL_ORIGEN"),
        os.getenv("BALANCEO_ANEMIA_ORIGEN"),
        os.getenv("BALANCEO_CONTROL_SALIDA"),
        os.getenv("BALANCEO_ANEMIA_SALIDA"),

        # Segmentación
        os.getenv("RUTA_BALANCEADAS"),
        os.getenv("RUTA_SEGMENTADAS"),
        os.getenv("RUTA_RECORTADAS"),
        os.getenv("RUTA_PNG"),

        # Resize
        os.getenv("RUTA_PNG_RESIZE"),

        # Entradas y salidas
        os.getenv("RUTA_ENTRADA"),
        os.getenv("RUTA_SALIDA"),
        os.getenv("RUTA_NO_FILTRADOS"),

        # Carpeta del modelo
        os.path.dirname(os.getenv("CARPETA_MODELO")),
    ]

    creadas = []
    for ruta in rutas:
        if ruta:
            abs_path = os.path.join(settings.BASE_DIR, ruta)
            os.makedirs(abs_path, exist_ok=True)
            creadas.append(abs_path)

    return JsonResponse({"mensaje": "Carpetas creadas", "rutas": creadas})


#2º Vista: ejecutar filtrado de conjuntiva
def ejecutar_filtrado(request):
    ruta_entrada = os.path.join(settings.BASE_DIR, os.getenv("RUTA_ENTRADA"))
    ruta_salida = os.path.join(settings.BASE_DIR, os.getenv("RUTA_SALIDA"))
    ruta_no_filtrados = os.path.join(settings.BASE_DIR, os.getenv("RUTA_NO_FILTRADOS"))
    ruta_reporte = os.path.join(settings.BASE_DIR, os.getenv("RUTA_REPORTE_TXT"))

    if not all([ruta_entrada, ruta_salida, ruta_no_filtrados, ruta_reporte]):
        return JsonResponse({"error": "Faltan rutas en el .env"}, status=400)

    # Limpiar salidas antes de procesar
    limpiar_carpeta(ruta_salida)
    limpiar_carpeta(ruta_no_filtrados)
    if os.path.exists(ruta_reporte):
        os.remove(ruta_reporte)

    filtrar_conjuntiva(ruta_entrada, ruta_salida, ruta_no_filtrados, ruta_reporte)
    return JsonResponse({"mensaje": "Filtrado completado correctamente"})


#3º Vista: ejecutar balanceo de imágenes
def ejecutar_balanceo(request):
    control_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_CONTROL_ORIGEN"))
    anemia_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_ANEMIA_ORIGEN"))
    output_control_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_CONTROL_SALIDA"))
    output_anemia_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_ANEMIA_SALIDA"))

    if not all([control_folder, anemia_folder, output_control_folder, output_anemia_folder]):
        return JsonResponse({"error": "Faltan variables en el .env"}, status=400)
    # Limpiar salidas de balanceo antes de procesar
    limpiar_carpeta(output_control_folder)
    limpiar_carpeta(output_anemia_folder)
    
    balancear_dataset(control_folder, anemia_folder, output_control_folder, output_anemia_folder)
    return JsonResponse({"mensaje": "Balanceo completado correctamente"})


#4º Vista: ejecutar segmentación de conjuntiva
def ejecutar_segmentacion(request):
    entrada = os.path.join(settings.BASE_DIR, os.getenv("RUTA_BALANCEADAS"))
    salida_segmentadas = os.path.join(settings.BASE_DIR, os.getenv("RUTA_SEGMENTADAS"))
    salida_recortadas = os.path.join(settings.BASE_DIR, os.getenv("RUTA_RECORTADAS"))
    salida_png = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG"))

    if not all([entrada, salida_segmentadas, salida_recortadas, salida_png]):
        return JsonResponse({"error": "Faltan rutas en el .env"}, status=400)

    # Limpiar salidas de segmentación
    limpiar_carpeta(salida_segmentadas)
    limpiar_carpeta(salida_recortadas)
    limpiar_carpeta(salida_png)
    
    segmentar_y_recortar_conjuntiva(entrada, salida_segmentadas, salida_recortadas, salida_png)
    return JsonResponse({"mensaje": "Segmentación completada"})


# Vista: ejecutar redimensionamiento
def ejecutar_redimensionamiento(request):
    input_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG"))
    output_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG_RESIZE"))

    if not all([input_folder, output_folder]):
        return JsonResponse({"error": "Faltan rutas en el .env"}, status=400)
    # Limpiar carpeta de salida antes de guardar imágenes redimensionadas
    limpiar_carpeta(output_folder)
    
    redimensionar_imagenes(input_folder, output_folder)
    return JsonResponse({"mensaje": "Redimensionamiento completado correctamente"})
