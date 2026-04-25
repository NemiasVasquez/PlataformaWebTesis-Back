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

    anchor, radius = extractor.detect_eye_anchor(img)
    win_mask = extractor.get_search_window(img, anchor, radius)[0]
    
    final_mask = extractor.find_medialuna_by_contrast(img, win_mask, anchor, radius)
    area_final = np.count_nonzero(final_mask)
    
    if area_final > 0:
        cnts, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cnts, key=cv2.contourArea)
        xc, yc, wc, hc = cv2.boundingRect(c)
        aspect = wc / max(hc, 1)
        print(f"Area: {area_final}, AspectRatio: {aspect:.2f}, w={wc}, h={hc}, x={xc}, y={yc}")

def main():
    test_files = ["8X3ICA6M.jpeg"]
    for f in test_files:
        test_image(f)

if __name__ == '__main__':
    main()
