import os
import re

def ajustar_env_permisivo(filepath=".env"):
    """
    SCRIPT CAVERNÍCOLA: Cambia valores de .env para que pasen más imágenes.
    Relaja nitidez, área y forma de conjuntiva.
    """
    if not os.path.exists(filepath):
        print(f"No hay piedra .env en {filepath}")
        return

    cambios = {
        # Nitidez (Menos exigente)
        "NITIDEZ_UMBRAL_LAP": "2",
        "NITIDEZ_UMBRAL_TENENGRAD": "2",
        "NITIDEZ_UMBRAL_VMLAP_MASCARA": "35",
        
        # Área de conjuntiva (Mucho más pequeña)
        "CONJUNTIVA_MIN_AREA_PCT": "0.007",
        "SEG_MIN_AREA": "200",
        
        # Forma y posición (Más flexible)
        "CONJUNTIVA_MIN_ASPECT_RATIO": "0.7",
        "CONJUNTIVA_MIN_ANCHO_FRACCION": "0.08",
        "SEG_MIN_ASPECT_PRE": "0.8",
        "SEG_SOLIDITY_TARGET": "0.45",
        "SEG_Y_TARGET_FACTOR": "1.4", # Buscar un poco más abajo
        
        # Iris y Anatomía
        "IRIS_HOUGH_P2": "20",
        "IRIS_MIN_AREA": "200",
        "ESCLEROTICA_UMBRAL_AREA": "0.001", # Casi ignorar si no hay blanco
    }

    with open(filepath, 'r', encoding='utf-8') as f:
        contenido = f.read()

    for clave, nuevo_valor in cambios.items():
        # Buscar la clave y reemplazar su valor
        patron = rf"^({clave})=.*"
        if re.search(patron, contenido, re.MULTILINE):
            contenido = re.sub(patron, rf"\1={nuevo_valor}", contenido, flags=re.MULTILINE)
        else:
            # Si no existe, agregar al final
            contenido += f"\n{clave}={nuevo_valor}"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(contenido)

    print("Piedra .env actualizada. Ahora proceso más permisivo.")

if __name__ == "__main__":
    ajustar_env_permisivo()
