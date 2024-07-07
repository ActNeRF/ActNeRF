# $1 - dataset path
# $2 - env model path
# $3 - optional output path

# compute train transforms

ds_path="processed/$1"
wisp="/home/optimus/dev/kaolin-wisp/"
nerf_path="$wisp/app/nerf/main_nerf.py"
config_path="$wisp/app/nerf/configs/nerf_hash.yaml"

if [ -z "$3" ]
  then
    out_path="$ds_path"_nobg
  else
    out_path=$3
fi

rm -rf $out_path
rm -rf _env_data/"$1"_train
rm -rf _env_data/"$1"_val

mv $ds_path/transforms_val.json $ds_path/transforms_val.json.1
mv $ds_path/transforms_test.json $ds_path/transforms_test.json.1
WISP_HEADLESS=1 python $nerf_path --dataset-path $ds_path --config $config_path --pretrained=$2 --valid-only --log-dir=_env_data --exp-name="$1"_train --mip=0

mv $ds_path/transforms_val.json.1 $ds_path/transforms_val.json
mv $ds_path/transforms_test.json.1 $ds_path/transforms_test.json
WISP_HEADLESS=1 python $nerf_path --dataset-path $ds_path --config $config_path --pretrained=$2 --valid-only --log-dir=_env_data --exp-name="$1"_val --mip=0


python back_sub.py  $ds_path/train  _env_data/"$1"_train/*/val/  $out_path/train
python back_sub.py  $ds_path/val  _env_data/"$1"_val/*/val/  $out_path/val 

cp $ds_path/*.json $out_path
cp $ds_path/train/*.npy $out_path/train 2>/dev/null || :
cp $ds_path/val/*.npy $out_path/val 2>/dev/null || :




