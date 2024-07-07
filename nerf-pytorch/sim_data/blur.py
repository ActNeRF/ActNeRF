import os
import sys
import shutil
import cv2
import numpy as np

input_path = sys.argv[1]
output_path = sys.argv[2]

os.makedirs(output_path, exist_ok=True)

for root, dirs, files in os.walk(input_path):
  rel_path = os.path.relpath(root, input_path)
  dest_folder = os.path.join(output_path, rel_path)

  if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

  for file in files:
    source_file = os.path.join(root, file)
    dest_file = os.path.join(dest_folder, file)

    if file.endswith('.png'):
      image = cv2.imread(source_file, cv2.IMREAD_UNCHANGED)
      blurred_image = cv2.GaussianBlur(image, (51, 51), 0)
      cv2.imwrite(dest_file, blurred_image)
    else:
      shutil.copy2(source_file, dest_file)