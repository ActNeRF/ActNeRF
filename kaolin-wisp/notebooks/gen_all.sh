export CUDA_VISIBLE_DEVICES=1

dirtype=model_videos_5_2
# for obj in cheezit rubik banana can crate mug 
# do 
#   dirname="results2/${dirtype}/$obj"
#   mkdir -p $dirname
#   for ((i = 0; i <= 10; i++)); do
#     python gen_revolve_video.py \
#       workdir/${obj}_5_2_nnval/rob0_${obj}_single_side_env_nobg_sam_iteration_$i \
#       results/${dirtype}/$obj/var_iter_$i.avi \
#       ${obj} 
#   done
# done

for obj in crate
do 
  dirname="results/${dirtype}/$obj"
  mkdir -p $dirname
  for ((i = 8; i <= 8; i++)); do
    python gen_revolve_video.py \
      workdir/${obj}_10_5_nnval/rob0_${obj}_single_side_env_nobg_sam_iteration_$i \
      results/${dirtype}/$obj/iter_$i.avi \
      ${obj} 
  done
done