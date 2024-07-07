import os
import subprocess

obj = 'cheezit'

for model_id in [2,3]:
  for iter_id in [10,24]:
    run_name = f'{obj}_5_2_val4_furthest_noflip_{model_id}'
    images_dir = f'/home/saptarshi/dev/CustomComposer/workdir/{run_name}/merged_images_{iter_id}'
    exp_name = f"workdir/{run_name}/rob0_{obj}_single_side_env_nobg_sam_iteration_{iter_id}/model_{model_id}"
    nerf_cmd = f'WISP_HEADLESS=1 python app/nerf/main_nerf.py --dataset-path {images_dir} --config app/nerf/configs/nerf_hash.yaml --log-dir=_results3/ensembles --exp-name={exp_name} --mip 0'

    print(nerf_cmd)
    subprocess.run(nerf_cmd, shell=True)
    
