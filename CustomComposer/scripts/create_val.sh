objects=("cheezit" "rubik" "crate" "can" "mug" "banana")
prompts=("red box" "rubik's cube" "blue crate" "can" "mug" "banana")

for ((i = 0; i < 6; i++)); do
  obj=${objects[$i]}
  prompt=${prompts[$i]}
  cd /home/saptarshi/.local/share/ov/pkg/isaac_sim-2022.2.1
  ./python.sh ~/dev/CustomComposer/src/wide_angle_14pics_env.py -o $obj
  
  cd /home/saptarshi/dev/nerf-pytorch/sim_data
  python modify_data.py \
    -i ~/dev/datasets/rob0_${obj}_sides_corners_env_for_robot_test/ \
    -o processed/${obj}_with_two_robots_test \
    -d val
  
  if [[ $obj == "cheezit" || $obj == "rubik" ]]; then
    ./sam_backsub_test.sh ${obj}_with_two_robots_test "$prompt"
  else
    ./sam_backsub_test_no_morph.sh ${obj}_with_two_robots_test "$prompt"
  fi

  cp processed/${obj}_with_two_robots_test_nobg_sam/transforms_val.json \
  processed/${obj}_with_two_robots_test_nobg_sam/transforms_test.json 
  
done

