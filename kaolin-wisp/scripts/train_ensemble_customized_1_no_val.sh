
WISP_HEADLESS=1 python app/nerf/main_nerf.py \
    --dataset-path $1 \
    --config app/nerf/configs/nerf_hash2.yaml \
    --log-dir=_results3/ensembles \
    --exp-name="$2/model_1" \
    --mip=$3 
