import os
import sys
import shutil
import cv2
import numpy as np
dir_path = sys.argv[1]
new_dir = sys.argv[2]

os.makedirs(new_dir, exist_ok=True)

for fname in os.listdir(dir_path):
  old_path = os.path.join(dir_path, fname)
  new_path = os.path.join(new_dir, fname)
  print(old_path)

  if fname.endswith('png'):
    img = cv2.imread(old_path)
    # downsample the image by 2
    shape = (int(img.shape[1] // 4), int(img.shape[0] // 4))
    img = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)
    cv2.imsave(new_path, img)
  else:
    shutil.copyfile(old_path, new_path)