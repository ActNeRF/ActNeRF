import os
import sys
import shutil
import os.path as osp
import numpy as np
import cv2

obj_dir = sys.argv[1]
env_dir = sys.argv[2]
out_dir = sys.argv[3]
mask_dir = out_dir + '/mask'

os.makedirs(out_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

obj_img_paths = []
env_img_paths = []

for fname in sorted(os.listdir(obj_dir)):
  obj_path = osp.join(obj_dir, fname)
  print(obj_path)
  if fname.endswith('.png'):
    obj_img_paths.append(obj_path)

for fname in sorted(os.listdir(env_dir), key=lambda x: int(x.split('-')[0])):
  env_path = osp.join(env_dir, fname)
  print(env_path)
  if fname.endswith('.png'):
    env_img_paths.append(env_path)

assert len(env_img_paths) == len(obj_img_paths)
for i in range(len(env_img_paths)):
  env_img = cv2.imread(env_img_paths[i])
  obj_img = cv2.imread(obj_img_paths[i])
  # cv2.imshow('env', env_img)
  # cv2.imshow('obj', obj_img)
  # cv2.waitKey(0)
  sub_img = cv2.absdiff(obj_img, env_img)
  print(sub_img.mean(), sub_img.max())
  mask = (sub_img.mean(axis=-1) > 25)
  mask = mask.astype(np.uint8) * 255
  cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), mask)
  cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((10, 10), np.uint8), mask)
  # cv2.imshow('mask', mask)
  print(mask.shape)
  new_img = np.zeros_like(obj_img)
  new_img[mask > 0] = obj_img[mask > 0]

  part_image = new_img.copy()
  part_image[mask == 0] = 255

  new_img = cv2.merge([*cv2.split(part_image), mask], 4)

  out_path = osp.join(out_dir, obj_img_paths[i].split('/')[-1])
  mask_out_path = osp.join(mask_dir, obj_img_paths[i].split('/')[-1])
  print(out_path)
  print(mask_out_path)
  cv2.imwrite(out_path, new_img)
  cv2.imwrite(mask_out_path, mask)