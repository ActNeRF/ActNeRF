WISP_HEADLESS=1 python app/nerf/main_nerf.py --dataset-path ~/dev/nerf-pytorch/sim_data/processed/cheezit_with_env --config app/nerf/configs/nerf_hash.yaml  >outs/cheezit_with_env_new.txt
WISP_HEADLESS=1 python app/nerf/main_nerf.py --dataset-path ~/dev/nerf-pytorch/sim_data/processed/crate_with_env --config app/nerf/configs/nerf_hash.yaml  >outs/crate_with_env_new.txt
WISP_HEADLESS=1 python app/nerf/main_nerf.py --dataset-path ~/dev/nerf-pytorch/sim_data/processed/rubik_with_env --config app/nerf/configs/nerf_hash.yaml  >outs/rubik_with_env_new.txt
