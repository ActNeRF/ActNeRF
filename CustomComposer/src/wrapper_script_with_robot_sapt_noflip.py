import os
import pdb
import sys
import time
import torch
import pickle
import shutil
import argparse
import subprocess
import numpy as np
from utils import *

from scipy.spatial.transform import Rotation as R

workdir = "workdir"
run_name = "with_two_robots"

def bp():
    pdb.set_trace()

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--run-name', type=str, default="with_two_robots")
parser.add_argument('-i', '--iters', nargs=2, type=int, default=[0, 0])
parser.add_argument('-ot', '--object-type', type=str, default="cheezit")
parser.add_argument('-n', '--num-images', nargs=2, type=int, default=[10, 5])
parser.add_argument('-dd','--r-std', type=float, default=0.007)
parser.add_argument('-a','--angle-std', type=float, default=3)
parser.add_argument('-g', '--gpu-id', type=int, default=0)
parser.add_argument('-m', '--move-lambda', type=float, default=0.5)
parser.add_argument('-u', '--unc-lambda', type=float, default=1)
parser.add_argument('-f', '--flip-lambda', type=float, default=1.)
parser.add_argument('-l', '--cost-lim', type=float, default=np.inf)
parser.add_argument('-mi', '--max-iters', type=int, default=np.iinfo(np.int64).max)
parser.add_argument('-st', '--score-th', type=float, default=50.)
args = parser.parse_args()

print(args)
run_name, iter_start, total_iter, obj_type = args.run_name, args.iters[0], args.iters[1], args.object_type
print(iter_start, total_iter)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
my_env = os.environ.copy()

num_caliberation_images = 4

obj_prompt_dict = {
    "cheezit": "red box",
    "rubik" : "multicolored cube",
    "crate" : "blue crate",
    "block" : "multicolored block",
    "can" : "can",
    "banana" : "banana",
    "mug" : "mug",
    "mustard" : "mustard",
    "drill" : "drill",
    "marker" : "marker",
    "spam" : "can",
    "small_mug" : "red mug",
    "basket" : "red bag"
}

sam_file_to_use_dict = {
    "cheezit": "sam_backsub_custom.sh",
    "rubik" : "sam_backsub_custom.sh",
    "crate" : "sam_backsub_custom_no_morph.sh",
    "block" : "sam_backsub_custom.sh",
    "can" : "sam_backsub_custom_no_morph.sh",
    "banana" : "sam_backsub_custom_no_morph.sh",
    "mug" : "sam_backsub_custom_no_morph.sh",
    "mustard" : "sam_backsub_custom_no_morph.sh",
    "drill" : "sam_backsub_custom_no_morph.sh",
    "marker" : "sam_backsub_custom_no_morph.sh",
    "spam" : "sam_backsub_custom_no_morph.sh",
    "small_mug" : "sam_backsub_custom_no_morph.sh",
    "basket" : "sam_backsub_custom.sh"
}

obj_prompt = obj_prompt_dict[obj_type]
sam_file_to_use = sam_file_to_use_dict[obj_type]

os.makedirs(f"/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}", exist_ok=True)
os.makedirs(f"/home/saptarshi/dev/kaolin-wisp/_results3/ensembles/{workdir}/{run_name}", exist_ok=True)

if iter_start == -1:
    done_iters = [int(i.split('_')[-1]) for i in os.listdir(f"/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}") if i.startswith('merged_images_')]
    iter_start = max(done_iters) + 1
    print(f"start iter given -1: Setting iter_start to {iter_start}")

output_args_pkl = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/init.pkl'
range_file = "/home/saptarshi/dev/CustomComposer/robot_range2_lc_0.04.pkl"

r = R.from_rotvec([np.pi/2, 0, 0])
v = np.eye(4)[None,:]
v[0,:3,:3] = r.as_matrix()
v[0,:3,-1] = np.array([0,0.3,0.18])

pickle.dump(np.linalg.inv(v), open(output_args_pkl,'wb'))

# IMP: Should be orientation of object wrt original
pose_dir_actual = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/pose_actual/'
os.makedirs(pose_dir_actual, exist_ok=True)

expected_nerf2ws_dir = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/nerf2ws_expected/'
os.makedirs(expected_nerf2ws_dir, exist_ok=True)
pickle.dump(np.eye(4), open(f"{expected_nerf2ws_dir}/iter_0.pkl", 'wb'))

def get_flipped_object_pose(anygrasp_data_dir, pose_file_actual, nerf2ws_file_expected):
    # c2w_cam = pickle.load(open(f"{anygrasp_data_dir}/c2w.pkl", 'rb'))
    # c2w_final = c2w_cam.copy()
    # c2w_final[:3,:3] = random_rotation_matrix(5 * np.pi/180) @ c2w_final[:3,:3]
    # c2w_final[:3,-1] += np.random.uniform(0,0.05,3) 
    # # pickle.dump(c2w_final, open(f"{anygrasp_data_dir}/c2w_final.pkl", 'wb'))


    sim_grasp_cmd = f"cd /home/saptarshi/.local/share/ov/pkg/isaac_sim-2022.2.1 && PYTHONUNBUFFERED=1 ./python.sh /home/saptarshi/dev/CustomComposer/src/simulation_video_generator_two_robots_with_gravity_testing_multi_img_inerf.py -dd {anygrasp_data_dir} -ot {obj_type} -ooe {nerf2ws_file_expected} -ooa {pose_file_actual}"
    # print(sim_grasp_cmd)
    subprocess.run(sim_grasp_cmd, shell=True, env=my_env)

    actual_pose = pickle.load(open(f"{anygrasp_data_dir}/object_actual_pose.pkl", 'rb'))
    return actual_pose

    # return flip_rot @ cur_orientation

merged_dataset_path = f"/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/merged_images_{iter_start-1}"


if iter_start == 0:
    # prev_data_dir = f"/home/saptarshi/dev/nerf-pytorch/sim_data/processed/{obj_type}_with_two_robots_test_nobg_sam"
    prev_data_dir = f"/home/saptarshi/dev/nerf-pytorch/sim_data/processed/rob0_{obj_type}_sides_corners_env_for_robot_with_gravity_val4_with_depth_nobg_sam"
else:
    model_name = f"{workdir}/{run_name}/rob0_{obj_type}_single_side_env_nobg_sam_iteration_{iter_start-1}"
    prev_data_dir = f"/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/merged_images_{iter_start-1}"



tot_valid_iters = 0
total_action_cost = 0
for j in range(1, iter_start):
    generated_images_dir_j = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/generated_images_{j}'
    captured_imgs = [i for i in os.listdir(generated_images_dir_j) if i.endswith('.png')]
    tot_valid_iters += int(len(captured_imgs) > 0)
    if len(captured_imgs) > 0:
        action_cost_fname = f"/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/iter_{j}_out/action_cost.pkl"
        try:
            cost = pickle.load(open(action_cost_fname, 'rb'))
            total_action_cost += cost
        except:
            pass

print("Valid Iters Done:", tot_valid_iters)
if tot_valid_iters >= args.max_iters:
    print('Max iters already reached')
    exit(0)

init_pose_file_actual = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/pose_actual/iter_0.pkl'


while iter_start <= total_iter:
    print(f"Running Iteration {iter_start}")
    object_center = torch.tensor((0, 0.6, 0.15))
    robot_poses = None 

    rob_pose_file = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/generated_images_{iter_start-1}/robot_poses.pkl'
    
    try:
        robot_poses = pickle.load(open(f"{rob_pose_file}", 'rb'))
    except:
        print(f"{iter_start}: Robot poses not found, setting None")
    
    #TODO: also save actual object_pose and update if after flipping

    nerf2ws_file_expected = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/nerf2ws_expected/iter_{iter_start}.pkl'
    pose_file_actual = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/pose_actual/iter_{iter_start}.pkl'

    cur_pose_actual = np.eye(4)
    cur_nerf2ws_expected = np.eye(4)
    
    # if iter_start > 0:
    if iter_start == 0:
        subprocess.run(f"cd /home/saptarshi/.local/share/ov/pkg/isaac_sim-2022.2.1 && PYTHONUNBUFFERED=1 ./python.sh /home/saptarshi/dev/CustomComposer/src/two_robots_with_gravity_init_pos.py --obj-type {obj_type} --out-path {pose_file_actual}", shell=True, env=my_env)
        # bp()
    
    cur_pose_actual = pickle.load(open(pose_file_actual, 'rb'))
    if iter_start > 1:
        cur_nerf2ws_expected = pickle.load(open(nerf2ws_file_expected, 'rb'))


    # TODO: Take orientation as input in both active learning and flipping file and convert all c2w as c2w_mod = orientation @ c2w and then use c2w_mod in everything. This is to be done for next_best_pose, grasp_pose, c2w.pkl 

    anygrasp_data_dir = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/grasp_data_iter_{iter_start}'
    os.makedirs(anygrasp_data_dir, exist_ok=True)
    
    cur_action_cost = 0
    if iter_start > 0:
        input_args_dir = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/input.pkl'
        output_args_pkl = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/output.pkl'
        output_dump_dir = f"/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/iter_{iter_start}_out"
        os.makedirs(output_dump_dir, exist_ok=True)
        
        pickle.dump([model_name, object_center, robot_poses, cur_nerf2ws_expected], open(input_args_dir, 'wb'))

        import gc; gc.collect()
        sys.stdout.flush()
        import time
        st = time.time()
        print(my_env)
        subprocess.run(f"cd /home/saptarshi/dev/kaolin-wisp/notebooks/ && PYTHONUNBUFFERED=1 python next_best_pose_range_new_flip_aksh.py -i {input_args_dir} -r {range_file} -p {output_args_pkl} -d {output_dump_dir} -a {anygrasp_data_dir} -m {args.move_lambda} -u {args.unc_lambda} -f {args.flip_lambda} --iter {iter_start} -st {args.score_th} --no-flip", shell=True, env=my_env)
        import gc; gc.collect()
        ed = time.time()
        print(f"Took {ed - st} seconds in optimization")
        # bp()
        cur_action_cost =  pickle.load(open(f"{output_dump_dir}/action_cost.pkl", 'rb'))
        total_action_cost += cur_action_cost
        if (total_action_cost > args.cost_lim):
            print("Cost Limit Reached! Can't perform actions. Exiting")
            sys.exit(0)

        flip_transform = None
        try:
            flip_transform = pickle.load(open(f"{output_dump_dir}/flip_transform.pkl", 'rb'))  #TODO: uncomment this
        except:
            # TODO remove
            pass
        # bp()
        if flip_transform is not None:
            print(f"Flipping object in iteration {iter_start}")
            cur_pose_actual = get_flipped_object_pose(anygrasp_data_dir, pose_file_actual, nerf2ws_file_expected)

            cur_nerf2ws_expected = flip_transform @ cur_nerf2ws_expected
            
            # TODO capture pre flip image and save c2w
            inerf_dir = f"/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/inerf_data_iter_{iter_start}"
            os.makedirs(inerf_dir, exist_ok=True)
            target_output_path = inerf_dir

            for idx in range(1, num_caliberation_images+1):
                # break
                post_flip_img_path = f"{anygrasp_data_dir}/post_flip_image{idx}.png"
                subprocess.run(f"/home/saptarshi/dev/nerf-pytorch/sim_data/sam_backsub_custom_single_img.sh {post_flip_img_path} \"{obj_prompt}\" {inerf_dir}",shell=True, env=my_env)
                post_flip_img_path = f"{inerf_dir}/post_flip_image{idx}.png"

                post_flip_c2w_ws = pickle.load(open(f"{anygrasp_data_dir}/post_flip_c2w_ws{idx}.pkl", 'rb'))
                post_flip_c2w_nerf = np.linalg.inv(cur_nerf2ws_expected) @ post_flip_c2w_ws
                post_flip_pre_inerf_c2w_nerf_path = f"{inerf_dir}/post_flip_pre_inerf_c2w{idx}.pkl"
                pickle.dump(post_flip_c2w_nerf, open(post_flip_pre_inerf_c2w_nerf_path, 'wb'))

            ans_file = f'{inerf_dir}/ans.pkl'
            # pickle.dump(pickle.load(open(pose_file_actual, 'rb')) @ np.linalg.inv(cur_pose_actual) @ post_flip_c2w_ws, open(ans_file, 'wb'))

            inerf_cmd = f'cd /home/saptarshi/dev/kaolin-wisp/notebooks/ && PYTHONUNBUFFERED=1 python inerf_multi_img_new.py \
                    -mn {model_name} \
                    -dp {merged_dataset_path} \
                    -o {target_output_path} \
                    --c2w_ans {ans_file}'
            
            # inerf_cmd = f'cd /home/saptarshi/dev/kaolin-wisp/notebooks/ && PYTHONUNBUFFERED=1 python inerf2.py \
            #         -mn {model_name} \
            #         -dp {merged_dataset_path} \
            #         -tp {post_flip_img_path1} \
            #         -o {target_output_path} \
            #         -c2w {post_flip_pre_inerf_c2w_nerf_path1} \
            #         --c2w_ans {ans_file}'
            print(inerf_cmd)
            # sys.exit(0)
            subprocess.run(inerf_cmd, shell=True, env=my_env)
            # bp()

            post_flip_post_inerf_c2w_nerf_path = f"{inerf_dir}/c2w_estimated.pkl"
            post_flip_post_inerf_c2w_nerf = pickle.load(open(post_flip_post_inerf_c2w_nerf_path, 'rb'))
            cur_nerf2ws_expected = pickle.load(open(f"{anygrasp_data_dir}/post_flip_c2w_ws1.pkl", 'rb')) @ np.linalg.inv(post_flip_post_inerf_c2w_nerf)
            
            cur_nerf2ws_expected_gold = np.linalg.inv(pickle.load(open(init_pose_file_actual, 'rb')) @ np.linalg.inv(cur_pose_actual))

            dist_diff = np.linalg.norm(cur_nerf2ws_expected[:3,-1] - cur_nerf2ws_expected_gold[:3,-1])
            angle_diff = np.rad2deg(np.arccos((np.trace(cur_nerf2ws_expected[:3,:3].T @ cur_nerf2ws_expected_gold[:3,:3]) - 1)/2))

            print('diffs', dist_diff, angle_diff)

    nerf2ws_file_expected_next_iter = f"{expected_nerf2ws_dir}/iter_{iter_start+1}.pkl"
    pose_file_actual_next_iter = f"{pose_dir_actual}/iter_{iter_start+1}.pkl"

    # cur_nerf2ws_expected = pickle.load(open(pose_file_actual, 'rb')) @ np.linalg.inv(cur_pose_actual)
    # cur_nerf2ws_expected = np.linalg.inv(pickle.load(open(pose_file_actual, 'rb'))) @ cur_pose_actual

    # bp()
    pickle.dump(cur_nerf2ws_expected, open(nerf2ws_file_expected_next_iter, 'wb'))
    pickle.dump(cur_pose_actual, open(pose_file_actual_next_iter, 'wb'))

    generated_images_dir = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/generated_images_{iter_start}'
    generated_images_formatted_dir = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/generated_images_formatted_{iter_start}'
    generated_images_nobg_dir = f'/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/generated_images_formatted_{iter_start}_nobg_sam'
    merged_dataset_path = f"/home/saptarshi/dev/CustomComposer/{workdir}/{run_name}/merged_images_{iter_start}"
    model_name = f"{workdir}/{run_name}/rob0_{obj_type}_single_side_env_nobg_sam_iteration_{iter_start}"
    num_images = args.num_images[0] if iter_start == 0 else args.num_images[1]

    print(generated_images_nobg_dir)
    sys.stdout.flush()
    subprocess.run(f"cd /home/saptarshi/.local/share/ov/pkg/isaac_sim-2022.2.1 && PYTHONUNBUFFERED=1 ./python.sh /home/saptarshi/dev/CustomComposer/src/active_learning_with_two_robots_sapt.py -p {output_args_pkl} -o {generated_images_dir} -t {obj_type} -n {num_images} -r {args.r_std} -a {args.angle_std} -rbp {rob_pose_file} -ooa {pose_file_actual_next_iter} -ooe {nerf2ws_file_expected_next_iter}", shell=True, env=my_env)
    prev_dir = os.getcwd()
    os.chdir('/home/saptarshi/dev/nerf-pytorch/sim_data/')
    # bp()
   
    subprocess.run(f"python modify_data.py --input-dir {generated_images_dir} --output-dir {generated_images_formatted_dir} --dataset-type train", shell=True, env=my_env)
    import gc; gc.collect()

    print(obj_prompt)
    sam_str = f"./{sam_file_to_use} {generated_images_formatted_dir} \"{obj_prompt}\""
    print(sam_str)
    subprocess.run(sam_str, shell=True, env=my_env)
    import gc; gc.collect()
    # bp()

    if iter_start == 0:
        shutil.rmtree(merged_dataset_path, ignore_errors=True)
        shutil.copytree(generated_images_nobg_dir, merged_dataset_path)
        shutil.copytree(prev_data_dir, merged_dataset_path, dirs_exist_ok=True)
        new_dataset_path = merged_dataset_path
    else:
        subprocess.run(f"python combine_datasets.py {prev_data_dir} {generated_images_nobg_dir} {merged_dataset_path} iter{iter_start}", shell=True)
        import gc; gc.collect()
        new_dataset_path = merged_dataset_path
    
    st = time.time()
    subprocess.run(f"/bin/bash -c 'cd /home/saptarshi/dev/kaolin-wisp; ./scripts/train_ensemble_customized.sh {new_dataset_path} {model_name} 0'", shell=True, env=my_env)
    ed = time.time()

    print(f"Took {ed - st} secs in 5 Nerf training")
    sys.stdout.flush()
    captured_imgs = [i for i in os.listdir(generated_images_dir) if i.endswith('.png')]
    if len(captured_imgs) == 0:
        print("0 Images Captured- skipping this iteration metrics")
    else:
        if iter_start != 0:
            tot_valid_iters += 1
        print(f"Action Cost in iter {iter_start}: {cur_action_cost}")
        print(f"Total Action Cost after iter {iter_start}: {total_action_cost}")
        subprocess.run(f"python /home/saptarshi/dev/CustomComposer/src/metrics.py -m {model_name} -o {init_pose_file_actual} -i {merged_dataset_path}", shell=True, env=my_env)
        subprocess.run(f"/bin/bash -c 'cd /home/saptarshi/dev/kaolin-wisp/notebooks/; python geom_metrics.py -o {obj_type} -e {run_name} -i {iter_start} --print-stdout'", shell=True, env=my_env)
    sys.stdout.flush()
    
    prev_data_dir = new_dataset_path
    
    iter_start += 1
    os.chdir(prev_dir)

    if tot_valid_iters >= args.max_iters:
        break

