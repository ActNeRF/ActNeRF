import os
import json
import subprocess

workdir_path = '/home/saptarshi/dev/CustomComposer/workdir'
old_metrics_path = '../geom_metrics_ablations'
new_metrics_path = '../geom_metrics_ablations_final'

os.makedirs(new_metrics_path, exist_ok=True)

filelist = os.listdir(old_metrics_path)

for fname in filelist:
  # print(fname)
  exp_name = fname.split('.json')[0]
  # an = '_an_' in fname
  obj = exp_name.split('_5_2')[0]
  run_id = int(exp_name.split('_')[-1])

  images_path = f'{workdir_path}/{exp_name}'

  with open(f'{old_metrics_path}/{fname}', 'r') as f:
    json_data = json.load(f)
  
  fscores = json_data['fscores']
  chamfers = json_data['chamfer_dists']
  new_fscores = []
  new_chamfers = []

  for i in range(len(fscores)):
    gen_images_path = f'{images_path}/generated_images_{i}'
    if len(os.listdir(gen_images_path)) > 1:
      new_fscores.append(fscores[i])
      new_chamfers.append(chamfers[i])

  print(f'old - {len(fscores)}, new - {len(new_fscores)} :: {exp_name}')

  json_data['fscores'] = new_fscores
  if fname == 'basket_5_2_val4_lee_chen_noflip_1.json':
    breakpoint()
  json_data['chamfer_dists'] = new_chamfers

  with open(f'{new_metrics_path}/{fname}', 'w') as f:
    json.dump(json_data, f)
  

