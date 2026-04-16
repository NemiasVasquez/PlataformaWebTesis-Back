# imagenes/tasks/preprocesamiento/resizeImagenes.py

import os
from PIL import Image

def redimensionar_imagenes(input_folder, output_folder, size=(150, 150)):
    for root, dirs, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        output_path = os.path.join(output_folder, relative_path)
        os.makedirs(output_path, exist_ok=True)

        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp')):
                try:
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    img = img.resize(size, Image.LANCZOS)
                    new_img_path = os.path.join(output_path, file)
                    img.save(new_img_path)
                except Exception as e:
                    print(f"⚠️ Error con imagen {file}: {e}")

    print("✅ Redimensionamiento completado.")
