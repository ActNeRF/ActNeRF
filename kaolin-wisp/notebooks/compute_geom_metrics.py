
import os
import shutil
import subprocess

objects=["small_mug"]
# objects=["cheezit", "basket", "small_mug", "rubik", "spam"]
get_model_idc = lambda obj: [1,2,3] 

metrics_dir = f'../geom_metrics_smug/'
os.makedirs(metrics_dir, exist_ok=True)

for obj in objects:
  for idx in get_model_idc(obj):
    run_names = [
      f'{obj}_5_2_val4_{idx}',
      f'{obj}_5_2_val4_noflip_{idx}',
      f'{obj}_5_2_val4_random_noflip_{idx}',
      f'{obj}_5_2_val4_random_singleflip_{idx}',
      f'{obj}_5_2_val4_furthest_noflip_{idx}',
      f'{obj}_5_2_val4_furthest_singleflip_{idx}',
    ]

    for i, run_name in enumerate(run_names):
      if i == 2 and idx > 1:
        continue
      print(run_name)
      subprocess.run(f'python geom_metrics.py -o {obj} -e {run_name} -out {metrics_dir}/{run_name}.json -ft 0.0025', shell=True)

