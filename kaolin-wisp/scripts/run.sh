for i in {1..10}
do
    WISP_HEADLESS=1 python app/nerf/main_nerf_density.py --dataset-path ~/dev/nerf-pytorch/sim_data/processed/cheezit_triple_side --config app/nerf/configs/nerf_hash.yaml --pretrained /home/optimus/dev/kaolin-wisp/_results/logs/runs/test-nerf/cheezit_triple_ep25_${i}/model.pth --valid-only
done