import os
import sys
import json
import shutil
import numpy as np

train_frac = .8
test_frac = 0

input_dir, output_dir = sys.argv[1:]
os.makedirs(f'{output_dir}', exist_ok=True)

num_frames = 0

param_file_names = []

for file in os.listdir(input_dir):
  if file.endswith(".json"):
    num_frames += 1
    param_file_names.append(file)

num_train = int(train_frac * num_frames)
num_test = int(test_frac * num_frames)

np.random.shuffle(param_file_names)

train_file_names  = param_file_names[:num_train]
test_file_names   = param_file_names[num_train:num_train+num_test]
other_names       = param_file_names[num_train+num_test:]

for type, file_list in [
    ('train', train_file_names), 
    ('test', test_file_names),
    ('other', other_names)
  ]:

  os.makedirs(f'{output_dir}/{type}', exist_ok=True)
  for file in file_list:
    fname = file.split('.')[0].split('_')[-1]
    param_name = file
    image_name = f'rgb_{fname}.png'
    shutil.copy(f'{input_dir}/{param_name}', f'{output_dir}/{type}/{param_name}')
    shutil.copy(f'{input_dir}/{image_name}', f'{output_dir}/{type}/{image_name}')
