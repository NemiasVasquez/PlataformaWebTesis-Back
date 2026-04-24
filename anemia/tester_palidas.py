import os
import cv2
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imagenes.tasks.preprocesamiento.extraccionConjuntiva import ConjuntivaExtractor

extractor = ConjuntivaExtractor()

def test_image(img_name):
    img_path = os.path.join(r"c:\laragon\www\PlataformaWebTesis\BackDjango\anemia\media\originales\CON ANEMIA", img_name)
    if not os.path.exists(img_path): return

    print(f"\n===========================================")
    print(f"TESTING: {img_name}")
    print(f"===========================================")
    img = cv2.imread(img_path)
    if img is None: return

    anchor, radius = extractor.detect_eye_anchor(img)
    win_mask = extractor.get_search_window(img, anchor, radius)[0]
    
    # 1. Normal call (it will internally retry with rojo_offset=15 if area is below 4500)
    final_mask = extractor.find_medialuna_by_contrast(img, win_mask, anchor, radius)
    area_final = np.count_nonzero(final_mask)
    print(f"Area Final: {area_final} pixels")

def main():
    test_files = [
        "4XY2-4ICA5F.jpeg",  # la rectangular 
        "4XY-4DCA4M.jpeg",
        "2XY-9DCA2M.jpeg",
        "8X2DCA6M.jpeg",   # BLANQUIÑOSA
        "3XY-1DCA26M.jpeg", # palida
        "8X7DCA6M.jpeg" # palida
    ]
    for f in test_files:
        test_image(f)

if __name__ == '__main__':
    main()
