import cv2
import numpy as np
import os

img_path = r"BackDjango\anemia\media\originales\CON ANEMIA\8X1DCA6M.jpeg"
img = cv2.imread(img_path)
if img is None:
    print("Image not found")
    exit()

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, a, b = cv2.split(lab)

# Define a ROI where the conjunctiva should be (manually for testing)
# Based on the image, let's say it's around cy + 1.5*radius
# Let's just print stats of the whole bottom area
h, w = img.shape[:2]
bottom_half = a[h//2:, :]
print(f"A channel stats (bottom half): min={np.min(bottom_half)}, max={np.max(bottom_half)}, mean={np.mean(bottom_half)}")

# Check some points
# The skin at bottom left (roughly y=0.8h, x=0.2w) 
# The conjunctiva at center (roughly y=0.7h, x=0.5w)

print(f"LAB at [0.8h, 0.2w] (Skin?): {lab[int(h*0.8), int(w*0.2)]}")
print(f"LAB at [0.65h, 0.5w] (Conjunctiva?): {lab[int(h*0.65), int(w*0.5)]}")
