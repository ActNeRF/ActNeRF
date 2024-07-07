for i in {1..5}
do  
    echo $i
    WISP_HEADLESS=1 python app/nerf/main_nerf.py \
        --dataset-path $1 \
        --config app/nerf/configs/nerf_hash.yaml \
        --log-dir=_results3/ensembles \
        --exp-name="$2/model_$i" \
        --mip=$3 
done