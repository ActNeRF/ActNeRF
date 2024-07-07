import os
import sys
import cv2
import pdb
import json
import copy
import torch
import pickle
import imageio
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from deap import base, creator, tools, algorithms
from nba_utils import *

from tqdm import tqdm
from typing import Tuple
from math import sin, cos, tan
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R


ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "inerf_utils"))

import wisp
from wisp.core import RenderBuffer, Rays
# %%
from inerf_utils import *
from wisp.datasets import SampleRays
from wisp.framework import WispState
from wisp.trainers import MultiviewTrainer
from wisp.models.pipeline import Pipeline
from kaolin.render.camera import Camera, blender_coords, opengl_coords
from wisp.ops.raygen import generate_pinhole_rays, generate_centered_pixel_coords

def bp():
    pdb.set_trace()


def make_model(args, args_dict, extra_args, train_dataset, validation_dataset):
    pipeline = torch.load(args.pretrained)
    pipeline.to(extra_args['device'])
    scene_state = WispState()
    return pipeline
    # return pipeline

def gen_camera(pose, focal, w, h, far, extra_args):
    view_matrix = torch.zeros_like(pose)
    view_matrix[:3, :3] = pose[:3, :3].T
    view_matrix[:3, -1] = torch.matmul(-view_matrix[:3, :3], pose[:3, -1])
    view_matrix[3, 3] = 1.0

    x0 = 0.0
    y0 = 0.0
    # print(view_matrix)
    camera = Camera.from_args(
        view_matrix=view_matrix,
        focal_x=focal,
        focal_y=focal,
        width=w,
        height=h,
        far=far,
        near=0.0,
        x0=x0,
        y0=y0,
        dtype=torch.float32,
        device=extra_args['device']
    )
    camera.change_coordinate_system(blender_coords())
    # print(camera.extrinsics.parameters().reshape((4,4)))
    # mask = camera.extrinsics.gradient_mask('t')
    # camera.extrinsics.parameters.register_hook(lambda grad: grad * mask.float())
    
    return camera

def gen_rays_only(camera):
    ray_grid = generate_centered_pixel_coords(camera.width, camera.height, camera.width, camera.height, device=extra_args['device'])
    rays = generate_pinhole_rays(camera, ray_grid).reshape(camera.height, camera.width, 3)
    rays = Rays.stack(rays)
    return rays

# %%
render_batch = 40000
def render_pipeline(pipeline, rays, lod_idx):
    rb = RenderBuffer(xyz=None, hit=None, normal=None, shadow=None, ao=None, dirs=None)
    for ray_pack in rays.split(render_batch):
        # print("haha")
        rb  += pipeline.tracer(pipeline.nef, rays=ray_pack, lod_idx=lod_idx)
    return rb

# %%
W, H = 200, 200
fx = (0.5 * H) / np.tan(0.5 * float(1.3213687585295282))
device = 'cuda'
extra_args = {
  'resume' : True,
  'output' : './pose_estimation',
  'device' : 'cuda',
  'z_near' : 0.0,
  'z_far' : 10.0,
  'focal' : fx,
  'lrate' : 3e-3,
  'epochs': 60,
  'wait_epochs': 12
}

# %%
def get_image(cam, pipeline) -> Tuple[np.ndarray, np.ndarray]:
    cam = copy.deepcopy(cam)

    rays = gen_rays_only(cam)
    rays = rays.reshape((rays.shape[0]**2, -1))

    rb = render_pipeline(pipeline, rays, lod_idx=None)
    rgb = rb.rgb.detach().cpu().numpy().reshape((H, W, 3))
    alpha = rb.alpha.detach().cpu().numpy().reshape((H, W, 1))
    # v = np.concatenate([rgb, alpha], axis=2)

    # return v
    return rgb, alpha

def get_image_with_depth(cam, pipeline) -> Tuple[np.ndarray, np.ndarray]:
    cam = copy.deepcopy(cam)

    rays = gen_rays_only(cam)
    rays = rays.reshape((rays.shape[0]**2, -1))

    rb = render_pipeline(pipeline, rays, lod_idx=None)
    rgb = rb.rgb.detach().cpu().numpy().reshape((H, W, 3))
    alpha = rb.alpha.detach().cpu().numpy().reshape((H, W, 1))
    depth = rb.depth.detach().cpu().numpy().reshape((H, W))
    return rgb, alpha, depth


argv_base = [
    '--config=../app/nerf/configs/nerf_hash.yaml',
    '--pretrained=path_to_model',
    '--valid-only'
]

num_pipelines = 5
def get_pipelines(model_name):
    pipelines = []
    root_dir = '/home/saptarshi/dev/kaolin-wisp/_results3/ensembles/' + model_name + '/'
    for i in range(1,num_pipelines+1):
        model_dir = os.path.join(root_dir, f"model_{i}")
        name_list = list(sorted(os.listdir(model_dir)))
        if name_list[-1] == "logs.parquet":
            name_list = name_list[:(len(name_list)-1)]
        model_path = os.path.join(model_dir, name_list[-1], "model.pth")
        sys.argv[1:] = argv_base
        sys.argv[2] = sys.argv[2].replace("path_to_model", model_path)
        print('Model Path : ', sys.argv[2])
        args, args_dict = parse_args()
        pipeline = make_model(
            args, args_dict, extra_args, None, None)
        pipelines.append(pipeline)
    return pipelines

base_scale = 6
base_offset = torch.tensor([0, -0.6, -0.15])
translation_scale = torch.tensor(base_scale/1.25)
translation_offset = translation_scale * base_offset

def is_flip_possible(grasp_data_dir, model_name, iter_id=-1, score_th=50):
    print(os.environ.copy())
    # subprocess.run(f"cd /home/saptarshi/dev/kaolin-wisp/notebooks && python generate_depth_image.py {model_name} {grasp_data_dir}/anygrasp_data/", shell=True, env = os.environ.copy())
    # subprocess.run(f"cd /home/saptarshi/dev/anygrasp/anygrasp_sdk/grasp_detection && python demo_new.py --checkpoint_path log/checkpoint_detection.tar --data-dir {grasp_data_dir}/anygrasp_data/", shell=True, env = os.environ.copy())
    pipelines = get_pipelines(model_name)
    grasp_data = pickle.load(open(f"{grasp_data_dir}/grasp_data.pkl", 'rb'))

    
    grasp_poses, gg_pick_scores, grasp_cosine_scores = grasp_data[0], grasp_data[1], grasp_data[2]
    cam_pose = pickle.load(open(f"{grasp_data_dir}/c2w.pkl", 'rb'))
    # pickle.dump(cam_pose, open(f"{grasp_data_dir}/c2w.pkl", 'wb'))
    cam_pose_scaled = cam_pose.copy()
    cam_pose_scaled[..., :3, 3] *= translation_scale.numpy()
    cam_pose_scaled[..., :3, 3] += translation_offset.numpy()

    images = np.zeros([num_pipelines, H, W, 3])
    projected_depths = np.zeros([num_pipelines, H, W])

    alphas = np.zeros([5, H, W])

    fx, fy = 514.66662, 514.66662
    cx, cy = 400, 400
    for i, pipeline in enumerate(pipelines):
        camera = gen_camera(torch.tensor(cam_pose_scaled).cuda(), extra_args['focal'], W, H, extra_args['z_far'], extra_args)
        images[i], alpha, depth = get_image_with_depth(camera, pipeline)

        alphas[i] = alpha.squeeze()

        xmap, ymap = np.arange(depth.shape[1]), np.arange(depth.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)
        angles = np.arctan(np.power(np.power(xmap - cx, 2) + np.power(ymap - cy, 2), 0.5)/fx)
        # bp()
        projected_depths[i] = depth * np.cos(angles)

    alpha = alphas.mean(axis=0)
    img_base = images[0].copy()
    rgb_uncs = []
    depth_uncs = []
    alpha_means = []
    wf = open(f'{grasp_data_dir}/grasp_scores.csv', 'w')
    wf.write('gg_pick_score,grasp_cosine_score,unc_rgb,unc_depth,gg/u_r,gg/u_d,gc/u_r,gc/u_d\n')

    alpha_th = 0.4
    p1s = []
    p2s = []
    for i, grasp in enumerate(grasp_poses):
        grasp_orig = np.linalg.inv(cam_pose) @ grasp

        grasp_translation = grasp_orig[:3,-1] * [1,-1,-1]
        p1 = point_to_pixel(grasp_translation + np.array((-0.02,-0.02,0)))
        p2 = point_to_pixel(grasp_translation + np.array((0.02,0.02,0)))

        p1s.append(p1)
        p2s.append(p2)

        img_base = cv2.rectangle(img_base, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255), 2)            

        rgb_unc = np.var(images, axis=0).mean(axis=2)[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0])]
        rgb_unc_std = rgb_unc.mean() ** 0.5

        depth_unc = np.var(projected_depths, axis=0)[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0])]
        depth_unc_std = depth_unc.mean() ** 0.5
        alpha_mean = alpha[int(p1[1]):int(p2[1]), int(p1[0]):int(p2[0])].mean()

        alpha_means.append(alpha_mean)
        rgb_uncs.append(rgb_unc_std)
        depth_uncs.append(depth_unc_std)
        if alpha_mean > alpha_th:
            wf.write(f'{gg_pick_scores[i]},{grasp_cosine_scores[i]},{rgb_unc_std},{depth_unc_std},{gg_pick_scores[i]/rgb_unc_std},{gg_pick_scores[i]/depth_unc_std},{grasp_cosine_scores[i]/rgb_unc_std},{grasp_cosine_scores[i]/depth_unc_std}\n')

    wf.close()

    cv2.imwrite(f'{grasp_data_dir}/grasp_patches.png', cv2.cvtColor((img_base*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    
    alpha_means = np.array(alpha_means)
    valid_grasps = alpha_means > alpha_th

    open(f'{grasp_data_dir}/alpha_means.txt', 'w').write(str(alpha_means))
    grasp_poses = np.array(grasp_poses)[valid_grasps]
    gg_pick_scores = np.array(gg_pick_scores)[valid_grasps]
    grasp_cosine_scores = np.array(grasp_cosine_scores)[valid_grasps]
    grasp_angle_scores = 1 - np.arccos(grasp_cosine_scores)

    rgb_uncs = np.array(rgb_uncs)[valid_grasps]
    depth_uncs = np.array(depth_uncs)[valid_grasps] 
    p1s = np.array(p1s)[valid_grasps]
    p2s = np.array(p2s)[valid_grasps]

    g1_u_r = gg_pick_scores / rgb_uncs
    g1_u_d = gg_pick_scores / depth_uncs
    g2_u_r = grasp_cosine_scores / rgb_uncs
    g2_u_d = grasp_cosine_scores / depth_uncs
    g3_u_r = grasp_angle_scores / rgb_uncs
    g3_u_d = grasp_angle_scores / depth_uncs

    gg_best = np.max(gg_pick_scores, initial=-1)
    gc_best = np.max(grasp_cosine_scores, initial=-1)
    ga_best = np.max(grasp_angle_scores, initial=-1)

    rgb_unc_best = np.min(rgb_uncs, initial=np.inf)
    depth_unc_best = np.min(depth_uncs, initial=np.inf)

    g1_u_r_best = np.max(g1_u_r, initial=-1)
    g1_u_d_best = np.max(g1_u_d, initial=-1)
    g2_u_r_best = np.max(g2_u_r, initial=-1)
    g2_u_d_best = np.max(g2_u_d, initial=-1)
    g3_u_r_best = np.max(g3_u_r, initial=-1)
    g3_u_d_best = np.max(g3_u_d, initial=-1)

    # breakpoint()

    # write to a csv file instead of txt
    if iter_id == 1:
        write_file = open(f'{grasp_data_dir}/../best_grasp_scores.csv', 'w')
        write_file.write('iter_id,best gg_pick_score,best grasp_cosine_score,best grasp_angle_scores,best unc_rgb,best unc_depth,best gg/u_r,best gg/u_d,best gc/u_r,best gc/u_d,best ga/u_r,best ga/u_d\n')
    else:
        write_file = open(f'{grasp_data_dir}/../best_grasp_scores.csv', 'a')
    write_file.write(f'{iter_id},{gg_best},{gc_best},{ga_best},{rgb_unc_best},{depth_unc_best},{g1_u_r_best},{g1_u_d_best},{g2_u_r_best},{g2_u_d_best},{g3_u_r_best},{g3_u_d_best}\n')

    write_file.close()
    if (len(g3_u_d) == 0):
        return False
    ind = np.argmax(g3_u_d)
    best_grasp = grasp_poses[ind]
    p1 = p1s[ind]
    p2 = p2s[ind]
    patched_img = cv2.rectangle(images[0].copy(), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255), 2)

    cv2.imwrite(f'{grasp_data_dir}/best_grasp_patch.png', cv2.cvtColor((patched_img*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    pickle.dump(best_grasp, open(f'{grasp_data_dir}/grasp.pkl', 'wb'))

    if g3_u_d_best < score_th:
        return False
    return True

flip_rot = np.array([
    [1., 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])


def point_to_pixel(p):
    fx, fy = 514.66662, 514.66662
    cx, cy = 400, 400
    x, y, z = p
    px = (cx + fx * x / z) / 4  # rendered images are already downsampled
    py = (cy + fy * y / z) / 4

    px = max(px, 0)
    py = max(py, 0)

    px = min(px, 799)
    py = min(py, 799)

    return np.array([px, py])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn', '--model-name', type=str, required=True)
    parser.add_argument('-gd', '--grasp-data-dir', type=str, required=True)
    parser.add_argument('-i', '--iter', type=int, required=False)
    parser.add_argument('-st', '--score-th', type=float, default=50)
    
    args = parser.parse_args()

    flip_possible = is_flip_possible(args.grasp_data_dir, args.model_name, args.iter, args.score_th)
    print("Flip Possible : ", flip_possible)
