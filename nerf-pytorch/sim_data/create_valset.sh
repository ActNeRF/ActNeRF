echo $2
echo $3
python modify_data.py -i /home/saptarshi/dev/datasets/"$1"_val/ -o /home/saptarshi/dev/nerf-pytorch/sim_data/processed/$1/ -d val
if [[ "$3" == "--no-morph" ]]
then
    ./sam_backsub_test_no_morph.sh $1 "$2"
else
    ./sam_backsub_test.sh $1 "$2"
fi
cp /home/saptarshi/dev/nerf-pytorch/sim_data/processed/"$1"_nobg_sam/transforms_val.json /home/saptarshi/dev/nerf-pytorch/sim_data/processed/"$1"_nobg_sam/transforms_test.json 