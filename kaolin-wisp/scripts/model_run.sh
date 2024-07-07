for i in {1..10}
do
    WISP_HEADLESS=1 python app/nerf/main_nerf.py --dataset-path ~/dev/nerf-pytorch/sim_data/processed/cheezit_triple_side/ --config app/nerf/configs/nerf_hash.yaml
done