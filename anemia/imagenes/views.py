from django.conf import settings
from django.http import JsonResponse
from imagenes.tasks.preprocesamiento.filtrarImagenes import filtrar_conjuntiva
from imagenes.tasks.preprocesamiento.balanceoImagenes import balancear_dataset
from imagenes.tasks.preprocesamiento.extraccionConjuntiva import segmentar_y_recortar_conjuntiva
from imagenes.tasks.preprocesamiento.resizeImagenes import redimensionar_imagenes
import shutil
import os
import cv2
from dotenv import load_dotenv

load_dotenv()

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

# --- Funciones Internas para Lógica Reutilizable ---

def _crear_carpetas():
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
        os.getenv("RUTA_ENTRADA"),
        os.getenv("RUTA_SALIDA"),
        os.getenv("RUTA_NO_FILTRADOS"),
        os.path.dirname(os.getenv("CARPETA_MODELO")),
    ]
    creadas = []
    for ruta in rutas:
        if ruta:
            abs_path = os.path.join(settings.BASE_DIR, ruta)
            os.makedirs(abs_path, exist_ok=True)
            creadas.append(abs_path)
    return creadas

def _filtrar():
    ruta_entrada = os.path.join(settings.BASE_DIR, os.getenv("RUTA_ENTRADA"))
    ruta_salida = os.path.join(settings.BASE_DIR, os.getenv("RUTA_SALIDA"))
    ruta_no_filtrados = os.path.join(settings.BASE_DIR, os.getenv("RUTA_NO_FILTRADOS"))
    ruta_reporte = os.path.join(settings.BASE_DIR, os.getenv("RUTA_REPORTE_TXT"))
    limpiar_carpeta(ruta_salida)
    limpiar_carpeta(ruta_no_filtrados)
    if os.path.exists(ruta_reporte): os.remove(ruta_reporte)
    filtrar_conjuntiva(ruta_entrada, ruta_salida, ruta_no_filtrados, ruta_reporte)

def _balancear():
    control_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_CONTROL_ORIGEN"))
    anemia_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_ANEMIA_ORIGEN"))
    output_control_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_CONTROL_SALIDA"))
    output_anemia_folder = os.path.join(settings.BASE_DIR, os.getenv("BALANCEO_ANEMIA_SALIDA"))
    limpiar_carpeta(output_control_folder)
    limpiar_carpeta(output_anemia_folder)
    balancear_dataset(control_folder, anemia_folder, output_control_folder, output_anemia_folder)

def _segmentar():
    entrada = os.path.join(settings.BASE_DIR, os.getenv("RUTA_BALANCEADAS"))
    salida_segmentadas = os.path.join(settings.BASE_DIR, os.getenv("RUTA_SEGMENTADAS"))
    salida_recortadas  = os.path.join(settings.BASE_DIR, os.getenv("RUTA_RECORTADAS"))
    salida_png         = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG"))
    salida_area        = os.path.join(settings.BASE_DIR, os.getenv("RUTA_AREA"))
    limpiar_carpeta(salida_segmentadas)
    limpiar_carpeta(salida_recortadas)
    limpiar_carpeta(salida_png)
    limpiar_carpeta(salida_area)
    segmentar_y_recortar_conjuntiva(entrada, salida_segmentadas, salida_recortadas, salida_png, salida_area)

def _redimensionar():
    input_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG"))
    output_folder = os.path.join(settings.BASE_DIR, os.getenv("RUTA_PNG_RESIZE"))
    limpiar_carpeta(output_folder)
    redimensionar_imagenes(input_folder, output_folder)

# --- Vistas de Django ---

def crear_carpetas_iniciales(request):
    creadas = _crear_carpetas()
    return JsonResponse({"mensaje": "Carpetas creadas", "rutas": creadas})

def ejecutar_filtrado(request):
    _filtrar()
    return JsonResponse({"mensaje": "Filtrado completado correctamente"})

def ejecutar_balanceo(request):
    _balancear()
    return JsonResponse({"mensaje": "Balanceo completado correctamente"})

def ejecutar_segmentacion(request):
    _segmentar()
    return JsonResponse({"mensaje": "Segmentación completada"})

def ejecutar_redimensionamiento(request):
    _redimensionar()
    return JsonResponse({"mensaje": "Redimensionamiento completado correctamente"})

def ejecutar_todo(request):
    try:
        _crear_carpetas()
        _filtrar()
        _balancear()
        _segmentar()
        _redimensionar()
        return JsonResponse({"mensaje": "Todo el proceso ejecutado correctamente"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def listar_imagenes(request):
    carpetas = {
        "originales": os.getenv("RUTA_ENTRADA"),
        "filtradas": os.getenv("RUTA_SALIDA"),
        "segmentadas": os.getenv("RUTA_SEGMENTADAS"),
        "recortadas": os.getenv("RUTA_RECORTADAS"),
        "png": os.getenv("RUTA_PNG"),
        "resize": os.getenv("RUTA_PNG_RESIZE")
    }
    
    resultado = {}
    for clave, ruta in carpetas.items():
        lista = []
        if not ruta: continue
        abs_path = os.path.join(settings.BASE_DIR, ruta)
        if os.path.exists(abs_path):
            for root, dirs, files in os.walk(abs_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        if abs_path.startswith(str(settings.MEDIA_ROOT)):
                            rel_path = os.path.relpath(os.path.join(root, file), settings.MEDIA_ROOT)
                            url = f"{settings.MEDIA_URL}{rel_path.replace(os.sep, '/')}"
                        else:
                            url = f"/media/{root.replace(str(settings.BASE_DIR), '').strip(os.sep)}/{file}".replace('\\', '/')
                        lista.append(url)
        resultado[clave] = lista
        
    return JsonResponse(resultado)

def explorar_carpetas(request):
    base_dir = os.path.join(settings.BASE_DIR, 'media')
    rel_path = request.GET.get('path', '').strip('/')
    try:
        page = int(request.GET.get('page', 1))
        page_size = int(request.GET.get('page_size', 16))
    except (ValueError, TypeError):
        page = 1
        page_size = 16
    
    target_dir = os.path.abspath(os.path.join(base_dir, rel_path))
    
    if not target_dir.startswith(os.path.abspath(base_dir)):
        return JsonResponse({"error": "Acceso denegado"}, status=403)
        
    if not os.path.exists(target_dir):
        return JsonResponse({"error": "No existe el directorio", "path": rel_path}, status=404)
        
    items = os.listdir(target_dir)
    folders = []
    files = []
    
    for item in items:
        item_path = os.path.join(target_dir, item)
        if os.path.isdir(item_path):
            folders.append(item)
        elif item.lower().endswith(('.png', '.jpg', '.jpeg')):
            media_rel = os.path.relpath(item_path, settings.MEDIA_ROOT)
            url = f"{settings.MEDIA_URL}{media_rel.replace(os.sep, '/')}"
            
            # Obtener metadatos
            try:
                size_bytes = os.path.getsize(item_path)
                size_str = f"{round(size_bytes / 1024, 1)} KB"
                if size_bytes > 1024 * 1024:
                    size_str = f"{round(size_bytes / (1024 * 1024), 2)} MB"
                
                img = cv2.imread(item_path)
                if img is not None:
                    h, w, _ = img.shape
                    dims = f"{w}x{h} px"
                else:
                    dims = "N/A"
            except:
                size_str = "Error"
                dims = "Error"

            files.append({
                "name": item, 
                "url": url,
                "size": size_str,
                "dimensions": dims
            })
            
    files.sort(key=lambda x: x['name'])
    total_files = len(files)
    start = (page - 1) * page_size
    end = start + page_size
    paged_files = files[start:end]
    
    return JsonResponse({
        "current_path": rel_path,
        "folders": sorted(folders),
        "files": paged_files,
        "pagination": {
            "total": total_files,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_files + page_size - 1) // page_size if total_files > 0 else 0
        }
    })
