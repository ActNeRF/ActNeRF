echo "starting"
PYTHONUNBUFFERED=1 python src/wrapper_script_with_robot_sapt_inerf_multi_img_inerf_testing.py -r cheezit_5_2_val4_far -o cheezit -i 10 15 -n 5 2 -g 1 -m 0.5 >> cheezit_5_2_val4_far_10_20_singleval 2>&1

# PYTHONUNBUFFERED=1 python src/wrapper_script_with_robot_singleflip_inerf_random.py -r cheezit_5_2_val4_random_singleflip_inerf -o cheezit -i 14 15 -n 5 2 -g 1 -m 0.5 >> cheezit_5_2_val4_random_singleflip_inerf_9_20_singleval 2>&1
# PYTHONUNBUFFERED=1 python src/wrapper_script_with_robot_init_random_inerf_multi_img.py -r cheezit_2_2_val4_random_init4_nba -o cheezit -i 7 15 -n 2 2 -g 1 -m 0.5 -ri 4 > cheezit_2_2_val4_random_init4_nba_7_20_singleval 2>&1
# PYTHONUNBUFFERED=1 python src/wrapper_script_with_robot_singleflip_inerf_random.py -r cheezit_5_2_val4_random_singleflip_inerf -o cheezit -i 16 20 -n 5 2 -g 1 -m 0.5 >> cheezit_5_2_val4_random_singleflip_inerf_9_20_singleval 2>&1
# PYTHONUNBUFFERED=1 python src/wrapper_script_with_robot_init_random_inerf_multi_img.py -r cheezit_2_2_val4_random_init4_nba -o cheezit -i 16 20 -n 2 2 -g 1 -m 0.5 -ri 4 >> cheezit_2_2_val4_random_init4_nba_7_20_singleval 2>&1
