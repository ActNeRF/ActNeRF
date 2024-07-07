import os
import shutil
import subprocess

metrics_dir = '../geom_metrics'
lines = open('leftovers2.txt').readlines()
for exp_name in lines:
  exp_name = exp_name.strip()
  if exp_name != '':
    print(exp_name)
    obj = exp_name.split('_5')[0]
    subprocess.run(f'python geom_metrics.py -o {obj} -e {exp_name} -out {metrics_dir}/{exp_name}.json -ft 0.0025', shell=True)