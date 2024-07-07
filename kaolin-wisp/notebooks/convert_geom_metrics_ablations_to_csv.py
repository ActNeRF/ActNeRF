import os
import sys
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cost-limit', action='store_true', help='Use cost limit')

args = parser.parse_args()

objects=["basket", "cheezit", "small_mug", "rubik", "spam"]

run_names = [
  '5_2_val4_lee_chen2_noflip',
  # '5_2_val4_lee_chen_noflip',
  '5_2_val4_density_noflip',
  '5_2_val4_dan_noflip'
]

cost_iters = {
  'basket': {
    '5_2_val4_lee_chen2_noflip' : [21, 21, 21],
    '5_2_val4_lee_chen_noflip' : [17, 21, 21],
    '5_2_val4_density_noflip' : [15, 14, 14],
    '5_2_val4_dan_noflip': [21, 21, 20]
  },
  'cheezit': {
    '5_2_val4_lee_chen2_noflip' : [14, 21, 18],
    '5_2_val4_lee_chen_noflip' : [18, 21, 17],
    '5_2_val4_density_noflip' : [14, 15, 13],
    '5_2_val4_dan_noflip': [21, 21, 21]
  },
  'small_mug': {
    '5_2_val4_lee_chen2_noflip' : [21, 20, 21],
    '5_2_val4_lee_chen_noflip' : [21, 21, 21],
    '5_2_val4_density_noflip' : [14, 16, 14],
    '5_2_val4_dan_noflip': [21, 21, 21]
  },
  'rubik': {
    '5_2_val4_lee_chen2_noflip' : [21, 21, 21],
    '5_2_val4_lee_chen_noflip' : [21, 21, 21],
    '5_2_val4_density_noflip' : [21, 18, 17],
    '5_2_val4_dan_noflip': [21, 21, 21]
  },
  'spam': {
    '5_2_val4_lee_chen2_noflip' : [21, 21, 21],
    '5_2_val4_lee_chen_noflip' : [21, 21, 21],
    '5_2_val4_density_noflip' : [15, 21, 15],
    '5_2_val4_dan_noflip': [21, 21, 21]
  }
}


def get_model_idc(obj, model_type_idx): 
  # if (obj == "cheezit" and model_type_idx == 0) :
  #   return [2,7,9] 
  # elif (obj == 'rubik' and model_type_idx == 4):
  #   return [1,3]
  # elif model_type_idx == 6:
  #   return [1]
  # else:
  return [1,2,3]

df = pd.DataFrame()

all_run_means = [[] for i in range(len(run_names))]
for obj in objects:
  all_mean_stds = []
  for model_type_idx, run_name in enumerate(run_names):
    all_vals = []
    run_name_old = run_name
    for run_idx, run_id in enumerate(get_model_idc(obj, model_type_idx)):
      exp_name = f'{obj}_{run_name}_{run_id}'
      print(exp_name)
      if not os.path.exists(f'../geom_metrics_ablations_final/{exp_name}.json'):
        print(f'Warning: {exp_name} does not exist')
        val = -1
      else:
        metrics = pd.read_json(f'../geom_metrics_ablations_final/{exp_name}.json')
        if args.cost_limit:
          cost_iter_id = cost_iters[obj][run_name_old][run_idx] - 1
        else:
          cost_iter_id = -1
        
        if len(metrics['fscores']) <= cost_iter_id:
          print(f'Warning: {exp_name} has less than {cost_iter_id} cost iterations')
          val = list(metrics['fscores'])[-1]
        else:
          val = list(metrics['fscores'])[cost_iter_id]
      all_vals.append(val)
    mean = np.round(np.mean(all_vals), 2) * 10
    std = np.round(np.std(all_vals), 2) * 10

    # if obj == 'rubik':
    #   breakpoint()

    mean_std = f'\\val{{{mean:0.1f}}}{{{std:0.1f}}}'
    all_mean_stds.append(mean_std)
    all_run_means[model_type_idx].append(mean)
  df[obj] = all_mean_stds

print(np.array(all_run_means).mean(axis=1))

# add rowsnames to df
df.index = run_names
print(df)
# save df to csv
# csv_file_name = 'fscores_out_iter'
csv_file_name = 'fscores_out_cost' if args.cost_limit else 'fscores_out_iter'
df.to_csv(f'../geom_workdir_ablations/noflip_{csv_file_name}.csv')