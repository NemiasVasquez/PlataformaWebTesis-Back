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
    def es_nitida(img, umbral_lap=10, umbral_tenengrad=8, umbral_hf_ratio=0.005):
        """
        Triple verificación de nitidez:
        1. Laplacian variance: sensible a bordes generales.
        2. Tenengrad (Sobel): detecta gradientes locales, muy sensible al foco óptico.
        3. Ratio de alta frecuencia (FFT): imágenes con bokeh/desenfoque óptico tienen
           muy poca energía en frecuencias altas respecto al total.
        """
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- Métrica 1: Varianza del Laplaciano ---
        lap_var = cv2.Laplacian(gris, cv2.CV_64F).var()
        
        # --- Métrica 2: Tenengrad (energía de gradientes Sobel) ---
        sobelx = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.sqrt(sobelx**2 + sobely**2).mean()
        
        # --- Métrica 3: Ratio de energía en altas frecuencias (FFT) ---
        f = np.fft.fft2(gris)
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        h, w = gris.shape
        # Máscara para zona central (bajas frecuencias) - 20% del espectro
        cy, cx = h // 2, w // 2
        r = int(min(h, w) * 0.10)
        y_grid, x_grid = np.ogrid[:h, :w]
        mask_low = (y_grid - cy)**2 + (x_grid - cx)**2 <= r**2
        energia_total = np.sum(mag) + 1e-8
        energia_alta = np.sum(mag[~mask_low])
        hf_ratio = energia_alta / energia_total
        
        pasa_lap = lap_var > umbral_lap
        pasa_tenengrad = tenengrad > umbral_tenengrad
        pasa_hf = hf_ratio > umbral_hf_ratio
        
        return pasa_lap and pasa_tenengrad and pasa_hf




    def ojo_abierto(img):
        """
        Detecta si el ojo está abierto buscando el iris o la pupila.
        Mejorado para ser más robusto ante diferentes tonos de iris y sombras.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Buscar zonas oscuras (Iris/Pupila) en HSV y escala de grises
        # Ampliamos un poco el rango de valor (V) para captar irises en sombra
        mask_dark_hsv = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 75]))
        _, mask_dark_gray = cv2.threshold(gris, 65, 255, cv2.THRESH_BINARY_INV)
        
        mask_iris = cv2.bitwise_or(mask_dark_hsv, mask_dark_gray)
        
        # Limpieza morfológica suave
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_iris = cv2.morphologyEx(mask_iris, cv2.MORPH_OPEN, kernel)
        
        contornos, _ = cv2.findContours(mask_iris, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in contornos:
            area = cv2.contourArea(c)
            # Ser más permisivo con el tamaño y la forma (iris parcial al mirar arriba)
            if 150 < area < (img.shape[0] * img.shape[1] * 0.25): 
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / h
                if 0.3 < aspect_ratio < 3.5:
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
        """
        Detecta si hay áreas con colores típicos de conjuntiva (rojos, rosas, naranjas suaves).
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Rango para rojos/rosas en HSV (saturacion mas baja para ojos palidos/anemicos)
        mask1 = cv2.inRange(hsv, np.array([0, 20, 40]), np.array([20, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([160, 20, 40]), np.array([180, 255, 255]))
        mask_red = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_clean = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
        
        porcentaje_rojo = np.count_nonzero(mask_clean) / (img.shape[0] * img.shape[1])
        
        # Si hay suficiente área roja o una forma compatible con la conjuntiva
        return porcentaje_rojo > 0.005 or forma_eliptica_detectada(mask_clean)


    def tiene_desenfoque(img):
        return not es_nitida(img)



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

    print("\nMETRICAS DE FILTRADO FINAL:")
    print(f"Total imagenes procesadas: {contador_total}")
    print(f"Aceptadas: {contador_filtradas}")
    print(f"Rechazadas por ojo cerrado: {contador_sin_iris}")
    print(f"Rechazadas sin conjuntiva: {contador_sin_conjuntiva}")
    print(f"Rechazadas por desenfoque: {contador_desenfoque}")
    print(f"Rechazadas por tamano: {contador_tamano_insuficiente}")
    print(f"Rechazadas por esclerotica: {contador_sin_esclerotica}")
    print(f"Reporte guardado en: {ruta_reporte_txt}")
