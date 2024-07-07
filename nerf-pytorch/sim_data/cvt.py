import os
import sys
import shutil
from PIL import Image
import numpy as np
dir_path = sys.argv[1]
new_dir = sys.argv[2]

os.makedirs(new_dir, exist_ok=True)

for fname in os.listdir(dir_path):
  old_path = os.path.join(dir_path, fname)
  new_path = os.path.join(new_dir, fname)
  print(old_path)

  if fname.endswith('png'):
    img = Image.open(old_path)
    img = img.convert('RGBA')
    data = img.getdata()
    # data[data[:3] == np.array([0,0,0])] = np.array([255,255,255,0])
    new_data = []
    for item in data:
      if item[0] == 0 and item[1] == 0 and item[2] == 0:
        new_data.append((255, 255, 255, 0))
      else:
        new_data.append(item)
    img.putdata(new_data)
    img.save(new_path, 'PNG')
  else:
    shutil.copyfile(old_path, new_path)


