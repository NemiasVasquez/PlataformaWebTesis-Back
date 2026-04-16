# imagenes/tasks/preprocesamiento/filtrarImagenes.py

import os
import cv2
import numpy as np
import glob
import shutil

def filtrar_conjuntiva(ruta_entrada, ruta_salida, ruta_no_filtrados, ruta_reporte_txt):
    categorias = ['SIN ANEMIA', 'CON ANEMIA']
    razones_rechazo = {
        "Ojo no está abierto (no se detecta iris adecuado)": "Ojo cerrado",
        "No se detecta conjuntiva": "Sin conjuntiva",
        "Efecto Blur detectado": "Efecto Blur",
        "Tamaño insuficiente": "Tamaño insuficiente",
        "No se detecta esclerótica": "Sin esclerótica"
    }

    for categoria in categorias:
        os.makedirs(os.path.join(ruta_salida, categoria), exist_ok=True)
        for razon in razones_rechazo.values():
            os.makedirs(os.path.join(ruta_no_filtrados, categoria, razon), exist_ok=True)

    with open(ruta_reporte_txt, 'w', encoding='utf-8') as f:
        f.write("📋 Reporte de imágenes rechazadas y razones:\n\n")

    # Métricas
    contador_total = contador_filtradas = contador_sin_iris = 0
    contador_sin_conjuntiva = contador_desenfoque = 0
    contador_tamano_insuficiente = contador_sin_esclerotica = 0

    # Funciones auxiliares
    def es_nitida(img, umbral=7):
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gris, cv2.CV_64F)
        return lap.var() > umbral

    def ojo_abierto(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_iris = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 60]))
        contornos, _ = cv2.findContours(mask_iris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            if cv2.contourArea(c) > 100 and len(c) >= 4:
                _, (MA, ma), _ = cv2.fitEllipse(c)
                if 0.01 < MA / ma < 2:
                    return True
        return False

    def forma_eliptica_detectada(mask):
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            if len(c) >= 5:
                _, (MA, ma), _ = cv2.fitEllipse(c)
                if 0.01 < MA / ma < 0.9 and cv2.contourArea(c) > 100:
                    return True
        return False

    def contiene_conjuntiva(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 30, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 30, 50]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask1, mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        return np.count_nonzero(mask_clean) / (img.shape[0] * img.shape[1]) > 0.01 or forma_eliptica_detectada(mask_clean)

    def tiene_desenfoque(img, umbral=8):
        return not es_nitida(img, umbral)

    def tiene_tamano_suficiente(img, min_tamano=200):
        alto, ancho = img.shape[:2]
        return alto >= min_tamano and ancho >= min_tamano

    def contiene_esclerotica(img, umbral_area=0.001):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask_white = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_clean = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        return np.count_nonzero(mask_clean) / (img.shape[0] * img.shape[1]) > umbral_area

    # Procesamiento
    for categoria in categorias:
        imagenes = glob.glob(os.path.join(ruta_entrada, categoria, '*.jpeg'))

        for ruta_img in imagenes:
            img = cv2.imread(ruta_img)
            if img is None:
                print(f"⚠️ No se pudo leer: {ruta_img}")
                continue

            contador_total += 1
            nombre_img = os.path.basename(ruta_img)

            razones = []
            pasa_iris = ojo_abierto(img)
            pasa_conjuntiva = contiene_conjuntiva(img)
            pasa_desenfoque = not tiene_desenfoque(img)
            pasa_tamano = tiene_tamano_suficiente(img)
            pasa_esclerotica = contiene_esclerotica(img)

            if not pasa_iris:
                razones.append("Ojo no está abierto (no se detecta iris adecuado)")
                contador_sin_iris += 1
            if not pasa_conjuntiva:
                razones.append("No se detecta conjuntiva")
                contador_sin_conjuntiva += 1
            if not pasa_desenfoque:
                razones.append("Efecto Blur detectado")
                contador_desenfoque += 1
            if not pasa_tamano:
                razones.append("Tamaño insuficiente")
                contador_tamano_insuficiente += 1
            if not pasa_esclerotica:
                razones.append("No se detecta esclerótica")
                contador_sin_esclerotica += 1

            if all([pasa_iris, pasa_conjuntiva, pasa_desenfoque, pasa_tamano, pasa_esclerotica]):
                shutil.copy(ruta_img, os.path.join(ruta_salida, categoria, nombre_img))
                contador_filtradas += 1
            else:
                for razon in razones:
                    destino = os.path.join(ruta_no_filtrados, categoria, razones_rechazo[razon], nombre_img)
                    shutil.copy(ruta_img, destino)
                with open(ruta_reporte_txt, 'a', encoding='utf-8') as f:
                    f.write(f"{nombre_img} ({categoria}): {', '.join(razones)}\n")

    print("\n📊 MÉTRICAS DE FILTRADO FINAL:")
    print(f"🔢 Total imágenes procesadas: {contador_total}")
    print(f"✅ Aceptadas: {contador_filtradas}")
    print(f"❌ Rechazadas por ojo cerrado: {contador_sin_iris}")
    print(f"❌ Rechazadas sin conjuntiva: {contador_sin_conjuntiva}")
    print(f"❌ Rechazadas por desenfoque: {contador_desenfoque}")
    print(f"❌ Rechazadas por tamaño: {contador_tamano_insuficiente}")
    print(f"❌ Rechazadas por esclerótica: {contador_sin_esclerotica}")
    print(f"📋 Reporte guardado en: {ruta_reporte_txt}")
