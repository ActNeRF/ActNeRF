
ds_path=$1
prompt=$2
if [ -z "$3" ]
  then
  out_path="$ds_path"_nobg_sam
  else
  out_path=$3
fi
echo $out_path
cd ~/dev/groundingSAM/Grounded-Segment-Anything

rm -rf $out_path

python grounded_sam_demo_backsub.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint groundingdino_swint_ogc.pth --sam_checkpoint sam_vit_h_4b8939.pth --input_image_dir $ds_path/train  --output_dir $out_path/train --box_threshold 0.3 --text_threshold 0.25 --text_prompt "$prompt" --device "cuda"

python grounded_sam_demo_backsub.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint groundingdino_swint_ogc.pth --sam_checkpoint sam_vit_h_4b8939.pth --input_image_dir $ds_path/val  --output_dir $out_path/val --box_threshold 0.3 --text_threshold 0.25 --text_prompt "$prompt" --device "cuda"

cp $ds_path/*.json $out_path
cp $ds_path/train/*.npy $out_path/train 2>/dev/null || :
