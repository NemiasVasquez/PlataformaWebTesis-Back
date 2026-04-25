import os
import cv2
import numpy as np
import sys
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from imagenes.tasks.preprocesamiento.extraccionConjuntiva import ConjuntivaExtractor

def debug_image(img_name):
    img_path = os.path.join(r"BackDjango\anemia\media\originales\CON ANEMIA", img_name)
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        return

    img = cv2.imread(img_path)
    extractor = ConjuntivaExtractor()
    
    anchor, radius = extractor.detect_eye_anchor(img)
    win_mask, _, _ = extractor.get_search_window(img, anchor, radius)
    
    # Replicate find_medialuna_by_contrast steps
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a_ch, b_ch = cv2.split(lab)
    (ax, ay) = anchor 

    pelo_l = 75
    _, mask_no_pelo = cv2.threshold(L, pelo_l, 255, cv2.THRESH_BINARY)
    mask_no_piel = cv2.compare(a_ch, b_ch, cv2.CMP_GT)

    blanco_l = 185
    _, mask_blanco = cv2.threshold(L, blanco_l, 255, cv2.THRESH_BINARY)
    mask_blanco = cv2.morphologyEx(mask_blanco, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))
    
    clip = 3.0
    grid = 8
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    a_fuerte = clahe.apply(a_ch)
    
    rojo_a = 138
    _, mask_rojo = cv2.threshold(a_fuerte, rojo_a, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_and(mask_rojo, mask_no_pelo)
    combined = cv2.bitwise_and(combined, mask_no_piel)
    combined = cv2.bitwise_and(combined, win_mask)
    
    combined_pre_blanco = combined.copy()
    combined = cv2.subtract(combined, mask_blanco) 

    y_wall_factor = 0.8
    y_pared = ay + int(radius * y_wall_factor)
    combined_pre_wall = combined.copy()
    combined[0:y_pared, :] = 0

    # Save masks for inspection
    debug_dir = r"BackDjango\anemia\media\debug_8X1DCA6M"
    if not os.path.exists(debug_dir): os.makedirs(debug_dir)
    
    cv2.imwrite(os.path.join(debug_dir, "0_original.jpg"), img)
    cv2.imwrite(os.path.join(debug_dir, "1_mask_rojo.jpg"), mask_rojo)
    cv2.imwrite(os.path.join(debug_dir, "2_mask_blanco.jpg"), mask_blanco)
    cv2.imwrite(os.path.join(debug_dir, "3_mask_no_piel.jpg"), mask_no_piel)
    cv2.imwrite(os.path.join(debug_dir, "4_win_mask.jpg"), win_mask)
    cv2.imwrite(os.path.join(debug_dir, "5_combined_pre_blanco.jpg"), combined_pre_blanco)
    cv2.imwrite(os.path.join(debug_dir, "6_combined_post_blanco.jpg"), combined)
    cv2.imwrite(os.path.join(debug_dir, "7_combined_post_wall.jpg"), combined)
    
    print(f"Debug masks saved in {debug_dir}")

if __name__ == "__main__":
    debug_image("8X1DCA6M.jpeg")
