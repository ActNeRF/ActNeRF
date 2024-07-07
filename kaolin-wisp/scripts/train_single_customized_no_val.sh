
WISP_HEADLESS=1 python app/nerf/main_nerf.py \
    --dataset-path $1 \
    --config app/nerf/configs/nerf_hash2.yaml \
    --log-dir=_results3/single \
    --exp-name="$2" \
    --mip=$3 
