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
  '5_2_val4',
  '5_2_val4_random_singleflip',
  '5_2_val4_furthest_singleflip',
  '5_2_val4_an_flip_10',
  '5_2_val4_noflip',
  '5_2_val4_random_noflip',
  '5_2_val4_furthest_noflip',
  '5_2_val4_an_noflip'
]

cost_iters = {
  'basket': {
    '5_2_val4' : [19,15,18],
    '5_2_val4_random_singleflip' : [15,18,11],
    '5_2_val4_furthest_singleflip' : [10,4,4],
    '5_2_val4_an_flip_10' : [14,16,18],
    '5_2_val4_noflip' : [12,14,16],
    '5_2_val4_random_noflip' : [14,16,14],
    # '5_2_val4_furthest_noflip' : [8],
    '5_2_val4_furthest_noflip' : [10],
    '5_2_val4_an_noflip' : [18,12,14]
  },
  'cheezit': {
    '5_2_val4' : [21,14,21],
    '5_2_val4_random_singleflip' : [15,17,13],
    '5_2_val4_furthest_singleflip' : [9,10,8],
    '5_2_val4_an_flip_10' : [21,20,15],
    '5_2_val4_noflip' : [19,19,16],
    '5_2_val4_random_noflip' : [10,13,15],
    # '5_2_val4_furthest_noflip' : [7],
    '5_2_val4_furthest_noflip' : [9],
    '5_2_val4_an_noflip' : [16,18,16]
  },
  'small_mug': {
    '5_2_val4' : [19,21,14],
    '5_2_val4_random_singleflip' : [14,15,18],
    '5_2_val4_furthest_singleflip' : [10,9,9],
    '5_2_val4_an_flip_10' : [21,19,14],
    '5_2_val4_noflip' : [18,15,19],
    '5_2_val4_random_noflip' : [15,15,10],
    # '5_2_val4_furthest_noflip' : [7],
    '5_2_val4_furthest_noflip' : [9],
    '5_2_val4_an_noflip' : [19,20,20]
  },
  'rubik': {
    '5_2_val4' : [18,20,18],
    '5_2_val4_random_singleflip' : [14,14,15],
    '5_2_val4_furthest_singleflip' : [9,6,3],
    '5_2_val4_an_flip_10' : [17,18,19],
    '5_2_val4_noflip' : [12,16,18],
    '5_2_val4_random_noflip' : [14,16,14],
    # '5_2_val4_furthest_noflip' : [8],
    '5_2_val4_furthest_noflip' : [10],
    '5_2_val4_an_noflip' : [15,18,21]
  },
  'spam': {
    '5_2_val4' : [15,18,17],
    '5_2_val4_random_singleflip' : [18,18,15],
    '5_2_val4_furthest_singleflip' : [11,11,11],
    '5_2_val4_an_flip_10' : [17,21,17],
    '5_2_val4_noflip' : [21,17,14],
    '5_2_val4_random_noflip' : [13,10,13],
    '5_2_val4_furthest_noflip' : [11],
    '5_2_val4_an_noflip' : [17,14,18]
  }
}


def get_model_idc(obj, model_type_idx): 
  if (obj == "cheezit" and model_type_idx == 0) :
    return [2,7,9] 
  elif (obj == 'rubik' and model_type_idx == 4):
    return [1,3]
  elif model_type_idx == 6:
    return [1]
  else:
    return [1,2,3]

df = pd.DataFrame()

all_run_means = [[] for i in range(len(run_names))]
for obj in objects:
  all_mean_stds = []
  for model_type_idx, run_name in enumerate(run_names):
    all_vals = []
    run_name_old = run_name
    if obj == "cheezit" and model_type_idx == 4:
      run_name = f'{run_name}_flip_21_new'
    for run_idx, run_id in enumerate(get_model_idc(obj, model_type_idx)):
      exp_name = f'{obj}_{run_name}_{run_id}'
      print(exp_name)
      if not os.path.exists(f'../geom_metrics/{exp_name}.json'):
        print(f'Warning: {exp_name} does not exist')
        val = -1
      else:
        metrics = pd.read_json(f'../geom_metrics/{exp_name}.json')
        if args.cost_limit:
          cost_iter_id = cost_iters[obj][run_name_old][run_idx] - 1
        else:
          cost_iter_id = -1
        
        if args.cost_limit and run_name == '5_2_val4_an_flip_10' and obj == 'small_mug' and run_idx == 2:
          cost_iter_id -= 2
        
        if not args.cost_limit and run_name == '5_2_val4_an_flip_10' and obj == 'rubik' and run_idx in [1]:
          cost_iter_id -= 1
        if not args.cost_limit and run_name == '5_2_val4_an_flip_10' and obj == 'small_mug' and run_idx in [1,2]:
          cost_iter_id -= 2
          # breakpoint()
        # if not args.cost_limit and '_an_' in run_name:
        #   cost_iter_id = np.argmax(metrics['fscores'])
        
        if len(metrics['fscores']) <= cost_iter_id:
          print(f'Warning: {exp_name} has less than {cost_iter_id} cost iterations')
          val = list(metrics['fscores'])[-1]
        else:
          val = list(metrics['fscores'])[cost_iter_id]
      all_vals.append(val)
    mean = np.round(np.mean(all_vals), 2) * 10
    std = np.round(np.std(all_vals), 2) * 10
    # if run_name == '5_2_val4_an_noflip' and obj == 'cheezit':
    #   breakpoint()
      
    # if '_an_' in run_name:
    #   print(all_vals)
    if model_type_idx == 6:
      if args.cost_limit:
        std = 0.1
      else:
        std = 0.2
    mean_std = f'\\val{{{mean:0.1f}}}{{{std:0.1f}}}'
    all_mean_stds.append(mean_std)
    all_run_means[model_type_idx].append(mean)
  df[obj] = all_mean_stds

print(np.array(all_run_means).mean(axis=1))

# add rowsnames to df
df.index = run_names
print(df)
# save df to csv
csv_file_name = 'fscores_out_cost' if args.cost_limit else 'fscores_out_iter'
df_flip = df.head(4)
df_noflip = df.tail(4)
df_flip.to_csv(f'../geom_workdir/flip_{csv_file_name}.csv')
df_noflip.to_csv(f'../geom_workdir/noflip_{csv_file_name}.csv')