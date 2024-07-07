for i in {21..30}
do
    python src/metrics.py -m workdir/$1/rob0_cheezit_single_side_env_nobg_sam_iteration_$i -o /home/saptarshi/dev/CustomComposer/workdir/$1/pose_actual/iter_0.pkl -i /home/saptarshi/dev/CustomComposer/workdir/$1/merged_images_$i
done