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
    PROCESO DE FILTRADO: Analiza cada imagen para asegurar que tenga nitidez, 
    el ojo esté abierto, se vea la esclerótica y la conjuntiva sea válida.
    Si una imagen falla, se mueve a 'no_filtradas' con la razón del rechazo.
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
    with open(ruta_reporte_txt, 'w', encoding='utf-8') as f:
        f.write("Reporte de imágenes rechazadas:\n\n")

    for cat in categorias:
        for r_img in glob.glob(os.path.join(ruta_entrada, cat, '*.jpeg')):
            img = cv2.imread(r_img)
            if img is None: continue
            
            nombre = os.path.basename(r_img)
            rechazos = []

            # Ejecutar validaciones
            if not tiene_tamano_suficiente(img): rechazos.append("size")
            if not ojo_abierto(img, extractor): rechazos.append("iris")
            if not contiene_esclerotica(img): rechazos.append("escl")

            encontrada, area, mask, forma, pos, pest = validar_conjuntiva(img, extractor)
            
            if not encontrada: rechazos.append("conj")
            else:
                if area < float(os.getenv("CONJUNTIVA_MIN_AREA_PCT", 0.01)) or not forma: rechazos.append("area")
                if not pos: rechazos.append("pos")
                if not pest: rechazos.append("pest")
            
            if not es_nitida(img, mask): rechazos.append("blur")

            # Decisión final
            if not rechazos:
                shutil.copy(r_img, os.path.join(ruta_salida, cat, nombre))
            else:
                for r in rechazos:
                    destino = os.path.join(ruta_no_filtrados, cat, razones_nombres[r], nombre)
                    img_debug = img.copy()
                    if encontrada:
                        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(img_debug, cnts, -1, (0, 255, 0), 2)
                    cv2.imwrite(destino, img_debug)
                
                with open(ruta_reporte_txt, 'a', encoding='utf-8') as f_rep:
                    f_rep.write(f"{nombre} ({cat}): {', '.join([razones_nombres[r] for r in rechazos])}\n")

    print(f"Filtrado finalizado. Reporte en {ruta_reporte_txt}")
