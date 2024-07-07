
img_path=$1
prompt=$2
out_path=$3

echo $out_path
cd ~/dev/groundingSAM/Grounded-Segment-Anything

# rm -rf $out_path

python grounded_sam_demo_backsub.py --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py --grounded_checkpoint groundingdino_swint_ogc.pth --sam_checkpoint sam_vit_h_4b8939.pth --input_image $img_path  --output_dir $out_path --box_threshold 0.3 --text_threshold 0.25 --text_prompt "$prompt" --device "cuda"
