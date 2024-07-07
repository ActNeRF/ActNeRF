for i in {1..10}
do
    WISP_HEADLESS=1 python app/nerf/main_nerf.py --dataset-path ~/dev/nerf-pytorch/sim_data/processed/$1 --config app/nerf/configs/nerf_hash.yaml --log-dir=_results3/ensembles --exp-name=$1/model_$i --mip=$2
done