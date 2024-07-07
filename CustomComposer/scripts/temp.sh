rm -r $1/val
rm  $1/transforms_test.json
rm  $1/transforms_val.json
cp -R /home/saptarshi/dev/nerf-pytorch/sim_data/processed/rob0_cheezit_sides_corners_env_for_robot_with_gravity_val4_with_depth_nobg_sam/* $1/