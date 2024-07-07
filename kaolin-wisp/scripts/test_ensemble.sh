for iter in {2..19}
do
    for i in {1..5}
    do
        model_path_candidates="_results3/ensembles/$1/$2/${3}_${iter}/model_$i/**/model.pth"
        model_path=`ls -td -- $model_path_candidates | head -n 1`
        echo $model_path
        a1=$1
        a2=$2
        a3=$3
        a4=$4
        WISP_HEADLESS=1 python app/nerf/main_nerf.py \
        --dataset-path /home/saptarshi/dev/ros_data_formatted/case_final_last_nobg_sam \
        --config app/nerf/configs/nerf_hash.yaml \
        --log-dir=_results3/ensembles \
        --exp-name=${a3}_$iter/model_$i \
        --pretrained=$model_path \
        --valid-only \
        --mip=$a4
    done
done

# for i in {1..10}
# do
#     model_path="_results_new/cheezit_single_side_env2_nobg_sam_scale10/model_$i/**/model.pth"
#     WISP_HEADLESS=1 python app/nerf/main_nerf.py \
#     --dataset-path ~/dev/nerf-pytorch/sim_data/processed/cheezit_single_side_new_nobg_sam \
#     --config app/nerf/configs/nerf_hash.yaml \
#     --log-dir=_results_new \
#     --exp-name=cheezit_single_side_new_nobg_sam/model_$i \
#     --pretrained=$model_path \
#     --valid-only \
#     --mip=2
# done