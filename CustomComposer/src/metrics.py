
import numpy as np
import pickle as pkl
import json
import os
import cv2
import matplotlib.pyplot as plt
import argparse
import skimage
import skimage.metrics
from tqdm import tqdm

def psnr(rgb, gts):
    """Calculate the PSNR metric.

    Assumes the RGB image is in [0,1]
    """
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    assert (rgb.shape[-1] == 3)
    assert (gts.shape[-1] == 3)

    mse = np.mean((rgb[..., :3] - gts[..., :3]) ** 2).item()
    return 10 * np.log10(1.0 / mse)

def ssim(rgb, gts):
    """Calculate the SSIM metric.

    Assumes the RGB image is in [0,1]
    """
    assert (rgb.max() <= 1.05 and rgb.min() >= -0.05)
    assert (gts.max() <= 1.05 and gts.min() >= -0.05)
    return skimage.metrics.structural_similarity(
        rgb[..., :3],
        gts[..., :3],
        multichannel=True,
        data_range=1,
        gaussian_weights=True,
        sigma=1.5,
        channel_axis=-1)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model-name', type=str)
parser.add_argument('-o', '--obj-pos', type=str)
parser.add_argument('-i', '--merged-images-dir', type=str)
parser.add_argument('-n', '--num-pipelines', type=int, default=5)
parser.add_argument('-an', '--active-nerf', action='store_true', default=False)

args = parser.parse_args()
# model_name = "workdir/cheezit_5_2_val4_perfectinerf2/rob0_cheezit_single_side_env_nobg_sam_iteration_7"
# obj_pos = f'/home/saptarshi/dev/CustomComposer/workdir/cheezit_5_2_val4_perfectinerf2/pose_actual/iter_0.pkl'
# merged_images_dir = f'/home/saptarshi/dev/CustomComposer/workdir/cheezit_5_2_val4_perfectinerf2/merged_images_7'
model_name, obj_pos, merged_images_dir = args.model_name, args.obj_pos, args.merged_images_dir

num_pipelines = args.num_pipelines
model_dirs = []
root_dir = '/home/saptarshi/dev/kaolin-wisp/_results3/ensembles/' + model_name + '/'
if args.active_nerf:
    root_dir = '/home/saptarshi/dev/an-wisp/_results3/ensembles/' + model_name + '/'

for i in range(1,num_pipelines+1):
    model_dir = os.path.join(root_dir, f"model_{i}")
    name_list = list(sorted(os.listdir(model_dir)))
    if name_list[-1] == "logs.parquet":
        name_list = name_list[:(len(name_list)-1)]
    model_path = os.path.join(model_dir, name_list[-1])
    model_dirs.append(model_path)
print(model_dirs)

files = os.listdir(model_dirs[0]+'/val')
files_1 = [i for i in files if not (i.startswith('depth') or i.startswith('unc') or i.startswith('alpha'))]
files_2 = [i for i in files if i.endswith('.pkl')]

rgb_filenames = sorted(files_1, key= lambda x : int(x.split('-')[0]))
depth_filenames = sorted(files_2, key= lambda x : int(x.split('-')[0].split('_')[-1]))

pos = pkl.load(open(obj_pos, 'rb'))
pos = pos[:3, 3]

val_data = json.load(open(f'{merged_images_dir}/transforms_val.json', 'r'))
num_frames = len(val_data['frames'])
depth_available = False
if (len(depth_filenames) > 0) and ('depth_path' in val_data['frames'][0].keys()):
    depth_available = True

def angle_z (cam_pos, pos):
    return np.arcsin((cam_pos[2] - pos[2])/np.linalg.norm(cam_pos - pos))

exposed_psnr = 0
# exposed_ssim = 0
unexposed_psnr = 0
# unexposed_ssim = 0
exposed_count = 0

# exposed_depth_mse = 0
# unexposed_depth_mse = 0

for (i, frame) in tqdm(enumerate(val_data['frames']), 'Calculating Metrics'):
    gold = cv2.imread(f"{merged_images_dir}/{frame['file_path']}.png")
    gold = gold / 255

    pred = np.zeros_like(gold)
    for j in range(num_pipelines):
        img = cv2.imread(f'{model_dirs[j]}/val/{rgb_filenames[i]}')
        img = img/255
        pred += img
    pred /= num_pipelines

    cur_psnr = psnr(pred, gold)
    # cur_ssim = ssim(pred, gold)

    depth_mse = 0
    if depth_available:
        nerf_depth = pkl.load(open(f'{model_dirs[j]}/val/{depth_filenames[i]}', 'rb'))
        nerf_depth = (nerf_depth * 1.25) / 6
        
        actual_depth = np.load(f"{merged_images_dir}/{frame['depth_path']}.npy")
        mask = cv2.imread(f"{merged_images_dir}/val/mask/{frame['file_path'].split('/')[-1]}.png")
        mask = mask[..., 0]
        actual_depth[mask != 255] = (10 * 1.25) / 6
        actual_depth[actual_depth > ((10 * 1.25) / 6)] = (10 * 1.25) / 6

        actual_depth /= ((10 * 1.25) / 6)
        nerf_depth /= ((10 * 1.25) / 6)

        depth_mse = np.mean(np.power(actual_depth - nerf_depth, 2))
        # print(depth_mse)
    cam_pos = np.array(frame['transform_matrix'])[:3, 3]
    # if cam_pos[2] >= pos[2]:
    if angle_z(cam_pos, pos) >= 0:
        exposed_psnr += cur_psnr
        # exposed_ssim += cur_ssim
        # exposed_depth_mse += depth_mse
        exposed_count += 1

    else:
        # plt.imshow(gold)
        # plt.show()
        # plt.imshow(pred)
        # plt.show()
        unexposed_psnr += cur_psnr
        # unexposed_ssim += cur_ssim
        # unexposed_depth_mse += depth_mse
    


exposed_psnr /= exposed_count
# exposed_ssim /= exposed_count
# exposed_depth_mse /= exposed_count

unexposed_psnr /= (num_frames - exposed_count)
# unexposed_ssim /= (num_frames - exposed_count)
# unexposed_depth_mse /= (num_frames - exposed_count)

total_psnr = (exposed_psnr * exposed_count + unexposed_psnr * (num_frames - exposed_count))/num_frames
# total_ssim = (exposed_ssim * exposed_count + unexposed_ssim * (num_frames - exposed_count))/num_frames
# total_depth_mse = (exposed_depth_mse * exposed_count + unexposed_depth_mse * (num_frames - exposed_count))/num_frames

print("PSNR Exposed:", exposed_psnr)
# print("SSIM Exposed:", exposed_ssim)
print("PSNR Unexposed:", unexposed_psnr)
# print("SSIM Unexposed:", unexposed_ssim)
print("PSNR Total:", total_psnr)
# print("SSIM Total:", total_ssim)
# if depth_available:
#     print("Avg Depth MSE Exposed:", exposed_depth_mse)
#     print("Avg Depth MSE Unexposed:", unexposed_depth_mse)
#     print("Avg Depth MSE Total:", total_depth_mse)
    
# print(exposed_psnr, exposed_ssim, unexposed_psnr, unexposed_ssim)
