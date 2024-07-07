# objects=("spam" "mustard" "gelatin" "rubik" )
# prompts=("can" "yellow bottle" "box" "rubik's cube" )

# objects=("gelatin" "rubik" )
# prompts=("box" "rubik's cube" )

objects=("gelatin" "spam")
prompts=("red box" "can")

for ((i = 0; i < 2; i++)); do
  obj=${objects[$i]}
  prompt=${prompts[$i]}

  # python src/wrapper_script_with_robot_sapt_inerf_multi_img_inerf_testing.py -r ${obj}_5_2_val4_flip_actpose_test_inerf4 -o $obj -i 0 0 -n 5 2 -g 0 -m 0.5

  cd /home/saptarshi/.local/share/ov/pkg/isaac_sim-2022.2.1
  PYTHONUNBUFFERED=1 ./python.sh /home/saptarshi/dev/CustomComposer/src/two_robots_with_gravity_init_pos.py --obj-type $obj --out-path /home/saptarshi/dev/CustomComposer/scripts/${obj}_pose.pkl

  echo "Getting val images Sim for $obj"

  cd /home/saptarshi/.local/share/ov/pkg/isaac_sim-2022.2.1
  ./python.sh ~/dev/CustomComposer/src/wide_angle_14pics_env.py -o $obj
  
  cd /home/saptarshi/dev/nerf-pytorch/sim_data
  python modify_data.py \
    -i ~/dev/datasets/rob0_${obj}_sides_corners_env_for_robot_with_gravity_val4_val/ \
    -o processed/rob0_${obj}_sides_corners_env_for_robot_with_gravity_val4 \
    -d val
  
  mkdir -p processed/rob0_${obj}_sides_corners_env_for_robot_with_gravity_val4_nobg_sam/val
  if [[ $obj == "cheezit" || $obj == "rubik" ]]; then
    ./sam_backsub_test.sh rob0_${obj}_sides_corners_env_for_robot_with_gravity_val4 "$prompt"
  else
    ./sam_backsub_test_no_morph.sh rob0_${obj}_sides_corners_env_for_robot_with_gravity_val4 "$prompt"
  fi

  cp processed/rob0_${obj}_sides_corners_env_for_robot_with_gravity_val4_nobg_sam/transforms_val.json \
  processed/rob0_${obj}_sides_corners_env_for_robot_with_gravity_val4_nobg_sam/transforms_test.json 
  
done

