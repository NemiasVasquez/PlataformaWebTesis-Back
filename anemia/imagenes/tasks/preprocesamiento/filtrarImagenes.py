import os
import cv2
import glob
import shutil
from .core.extractor import ConjuntivaExtractor
from .validations.quality import es_nitida, tiene_tamano_suficiente
from .validations.anatomy import ojo_abierto, contiene_esclerotica
from .validations.conjunctiva import validar_conjuntiva

def filtrar_conjuntiva(ruta_entrada, ruta_salida, ruta_no_filtrados, ruta_reporte_txt):
    """
    PROCESO DE FILTRADO CON LOGS: Indica por qué se rechaza cada imagen.
    """
    categorias = ['SIN ANEMIA', 'CON ANEMIA']
    razones_nombres = {
        "iris": "Ojo cerrado", "conj": "Sin conjuntiva", "blur": "Efecto Blur",
        "size": "Tamaño insuficiente", "escl": "Sin esclerotica", 
        "area": "Area insuficiente", "pos": "Posicion incorrecta", "pest": "Bloqueo pestanas"
    }

    # Preparar carpetas
    for cat in categorias:
        os.makedirs(os.path.join(ruta_salida, cat), exist_ok=True)
        for r in razones_nombres.values():
            os.makedirs(os.path.join(ruta_no_filtrados, cat, r), exist_ok=True)

    extractor = ConjuntivaExtractor()
    
    # Asegurar que el directorio del reporte existe
    os.makedirs(os.path.dirname(ruta_reporte_txt), exist_ok=True)

    print(f"===== INICIANDO FILTRADO DE IMÁGENES =====")
    with open(ruta_reporte_txt, 'w', encoding='utf-8') as f:
        f.write("Reporte de imágenes rechazadas:\n\n")

    for cat in categorias:
        archivos = glob.glob(os.path.join(ruta_entrada, cat, '*.[jJ][pP][gG]')) + \
                   glob.glob(os.path.join(ruta_entrada, cat, '*.[jJ][pP][eE][gG]'))
        print(f"--- Categoría {cat}: {len(archivos)} imágenes ---")

        for r_img in archivos:
            img = cv2.imread(r_img)
            if img is None: continue
            
            nombre = os.path.basename(r_img)
            rechazos = []

            # Validaciones
            if not tiene_tamano_suficiente(img): rechazos.append("size")
            if not ojo_abierto(img, extractor): rechazos.append("iris")
            if not contiene_esclerotica(img): rechazos.append("escl")

            encontrada, area, mask, forma, pos, pest = validar_conjuntiva(img, extractor)
            
            if not encontrada: rechazos.append("conj")
            else:
                if area < float(os.getenv("CONJUNTIVA_MIN_AREA_PCT", 0.018)) or not forma: rechazos.append("area")
                # if not pos: rechazos.append("pos") # Desactivado por ahora
                if not pest: rechazos.append("pest")
            
            if not es_nitida(img, mask): rechazos.append("blur")

            if not rechazos:
                print(f"  [PASS] {nombre}")
                shutil.copy(r_img, os.path.join(ruta_salida, cat, nombre))
            else:
                razones_str = ", ".join([razones_nombres[r] for r in rechazos])
                print(f"  [REJECT] {nombre}: {razones_str}")
                for r in rechazos:
                    destino = os.path.join(ruta_no_filtrados, cat, razones_nombres[r], nombre)
                    cv2.imwrite(destino, img) # Guardar copia para revisar
                
                with open(ruta_reporte_txt, 'a', encoding='utf-8') as f_rep:
                    f_rep.write(f"{nombre} ({cat}): {razones_str}\n")

    print(f"===== FILTRADO FINALIZADO. Reporte: {ruta_reporte_txt} =====\n")
