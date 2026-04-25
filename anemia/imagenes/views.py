from django.conf import settings
from django.http import JsonResponse
import os
import cv2
import json
from . import services

def crear_carpetas_iniciales(request):
    """Crea la estructura de carpetas necesaria para el sistema."""
    rutas = services.procesar_logica_carpetas()
    return JsonResponse({"mensaje": "Carpetas creadas", "rutas": [str(r) for r in rutas]})

def ejecutar_filtrado(request):
    """Luce filtrado de calidad y anatomía."""
    services.ejecutar_paso_filtrado()
    return JsonResponse({"mensaje": "Filtrado completado correctamente"})

def ejecutar_balanceo(request):
    """Equilibra la cantidad de imágenes entre clases."""
    services.ejecutar_paso_balanceo() # Si existe en services, sino llamar directo
    return JsonResponse({"mensaje": "Balanceo completado correctamente"})

def ejecutar_segmentacion(request):
    """Extrae la conjuntiva de las imágenes aceptadas."""
    services.ejecutar_paso_segmentacion()
    return JsonResponse({"mensaje": "Segmentación completada"})

def ejecutar_redimensionamiento(request):
    """Cambia el tamaño de las imágenes para el modelo."""
    services.ejecutar_paso_redimensionamiento()
    return JsonResponse({"mensaje": "Redimensionamiento completado"})

def ejecutar_todo(request):
    """Ejecuta el pipeline completo de procesamiento."""
    try:
        services.procesar_logica_carpetas()
        services.ejecutar_paso_filtrado()
        # services.ejecutar_paso_balanceo()
        services.ejecutar_paso_segmentacion()
        # services.ejecutar_paso_redimensionamiento()
        return JsonResponse({"mensaje": "Proceso completo ejecutado"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def mover_archivo(request):
    """Descartar imagen y moverla a la sección de fallidas."""
    if request.method != 'POST': return JsonResponse({"error": "Solo POST"}, status=405)
    data = json.loads(request.body)
    ok = services.mover_basura_imagen(data.get('nombre_archivo'), data.get('categoria'), data.get('razon_rechazo'))
    if ok: return JsonResponse({"mensaje": "Imagen Descartada"})
    return JsonResponse({"error": "No se encontró la imagen en Carpetas Filtradas"}, status=404)

def explorar_carpetas(request):
    """Explorador de archivos para el frontend."""
    base_dir = os.path.join(settings.BASE_DIR, 'media')
    rel_path = request.GET.get('path', '').strip('/')
    target_dir = os.path.abspath(os.path.join(base_dir, rel_path))
    
    if not target_dir.startswith(os.path.abspath(base_dir)) or not os.path.exists(target_dir):
        return JsonResponse({"error": "Ruta inválida"}, status=404)
        
    page = int(request.GET.get('page', 1))
    page_size = int(request.GET.get('page_size', 16))
    
    items = os.listdir(target_dir)
    folders = [i for i in items if os.path.isdir(os.path.join(target_dir, i))]
    files_info = []
    
    for f in items:
        f_path = os.path.join(target_dir, f)
        if os.path.isfile(f_path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            media_rel = os.path.relpath(f_path, settings.MEDIA_ROOT)
            url = f"{settings.MEDIA_URL}{media_rel.replace(os.sep, '/')}"
            files_info.append({"name": f, "url": url, "size": f"{round(os.path.getsize(f_path)/1024, 1)} KB"})

    files_info.sort(key=lambda x: x['name'])
    start, end = (page-1)*page_size, page*page_size
    
    return JsonResponse({
        "current_path": rel_path,
        "folders": sorted(folders),
        "files": files_info[start:end],
        "pagination": {"total": len(files_info), "page": page, "total_pages": (len(files_info)+page_size-1)//page_size}
    })

def listar_imagenes(request):
    """Alias para mantener compatibilidad con el frontend."""
    return explorar_carpetas(request)
