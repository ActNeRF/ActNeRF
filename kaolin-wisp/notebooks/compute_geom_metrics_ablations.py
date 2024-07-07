
import os
import shutil
import subprocess

# objects=["small_mug"]
# objects=["cheezit", "basket", "small_mug", "rubik", "spam"]
# get_model_idc = lambda obj: [1,2,3] if obj != "spam" else [1,2]

objects=["spam"]
get_model_idc = lambda obj: [3]

metrics_dir = f'../geom_metrics_ablations/'
os.makedirs(metrics_dir, exist_ok=True)

for obj in objects:
  for idx in get_model_idc(obj):
    run_names = [
      f'{obj}_5_2_val4_lee_chen2_noflip_{idx}',
    ]
    for i, run_name in enumerate(run_names):
      print(run_name)
      subprocess.run(f'python geom_metrics.py -o {obj} -e {run_name} -out {metrics_dir}/{run_name}.json -ft 0.0025', shell=True)

