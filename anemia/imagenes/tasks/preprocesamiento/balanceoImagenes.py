# imagenes/tasks/balanceoImagenes.py
from django.conf import settings
import os
import random
import shutil

def balancear_dataset(control_folder, parkinson_folder, output_control_folder, output_parkinson_folder):
    os.makedirs(output_control_folder, exist_ok=True)
    os.makedirs(output_parkinson_folder, exist_ok=True)

    control_files = [f for f in os.listdir(control_folder) if os.path.isfile(os.path.join(control_folder, f))]
    parkinson_files = [f for f in os.listdir(parkinson_folder) if os.path.isfile(os.path.join(parkinson_folder, f))]

    n_parkinson = len(parkinson_files)
    if len(control_files) == 0 or len(parkinson_files) == 0:
        print("❌ No hay imágenes suficientes para balancear.")
        return

    additional_control_files = random.choices(control_files, k=n_parkinson - len(control_files))
    balanced_control_files = control_files + additional_control_files

    for i, file in enumerate(balanced_control_files):
        src_path = os.path.join(control_folder, file)
        dest_path = os.path.join(output_control_folder, f"{i}_{file}")
        shutil.copy(src_path, dest_path)

    for file in parkinson_files:
        src_path = os.path.join(parkinson_folder, file)
        dest_path = os.path.join(output_parkinson_folder, file)
        shutil.copy(src_path, dest_path)

    print(f"✅ Balanceo completo. {len(balanced_control_files)} imágenes en control, {len(parkinson_files)} en parkinson.")
