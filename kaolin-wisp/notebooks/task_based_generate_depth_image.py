# %%
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from PIL import Image
import os

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
# %%
import os
import sys
import cv2
import pdb
import json
import copy
import torch
import pickle
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from matplotlib.animation import FuncAnimation
from deap import base, creator, tools, algorithms


from tqdm import tqdm
from typing import Tuple
from math import sin, cos, tan
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

def bp():
    pdb.set_trace()

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

def make_model(args, args_dict, extra_args, train_dataset, validation_dataset):
    pipeline = torch.load(args.pretrained)
    pipeline.to(extra_args['device'])
    scene_state = WispState()
    return pipeline

def gen_camera(pose, focal, w, h, far, extra_args):
    view_matrix = torch.zeros_like(pose)
    view_matrix[:3, :3] = pose[:3, :3].T
    view_matrix[:3, -1] = torch.matmul(-view_matrix[:3, :3], pose[:3, -1])
    view_matrix[3, 3] = 1.0

    x0 = 0.0
    y0 = 0.0
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
    return camera

def gen_rays_only(camera):
    ray_grid = generate_centered_pixel_coords(camera.width, camera.height, camera.width, camera.height, device=extra_args['device'])
    rays = generate_pinhole_rays(camera, ray_grid).reshape(camera.height, camera.width, 3)
    rays = Rays.stack(rays)
    return rays

def create_vid(imgs, path, fps=10):
    W,H = imgs[0].shape[1], imgs[0].shape[0]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (W,H))
    for i in range(len(imgs)):
        img = imgs[i]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)
    out.release()

# W, H = 200, 200
W, H = 800, 800

fx = (0.5 * H) / np.tan(0.5 * float(1.3213687585295282))
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

def get_c2w_from_pos(camera_position: np.ndarray, lookat: np.ndarray) -> torch.tensor:
    new_z = (camera_position - lookat)
    new_z = new_z / np.linalg.norm(new_z)

    z = np.array((0,0,1))
    new_y = z - np.dot(z, new_z) * new_z

    if (np.linalg.norm(new_y) == 0):
        new_y = np.array((0,1,0))
    new_y = new_y / np.linalg.norm(new_y)

    r, _ = R.align_vectors([(0,1,0), (0,0,1)], [new_y, new_z])
    c2w = torch.eye(4)
    c2w[:3,:3] = torch.tensor(np.linalg.inv(r.as_matrix()))
    c2w[:3,-1] = (torch.tensor(camera_position) * translation_scale) + translation_offset
    return c2w

def get_c2w_from_pos2(camera_position: np.ndarray, lookat: np.ndarray) -> torch.tensor:
    new_z = (camera_position - lookat)
    new_z = new_z / np.linalg.norm(new_z)

    new_y = np.cross(new_z, np.array((1,0,0)))

    r, _ = R.align_vectors([(0,1,0), (0,0,1)], [new_y, new_z])
    c2w = torch.eye(4)
    c2w[:3,:3] = torch.tensor(np.linalg.inv(r.as_matrix()))
    c2w[:3,-1] = (torch.tensor(camera_position) * translation_scale) + translation_offset
    return c2w


def camera_obj_from_c2w(c2w: np.ndarray):
    camera = gen_camera(c2w, extra_args['focal'], W, H, extra_args['z_far'], extra_args)
    return camera

# %%
render_batch = 40000
def render_pipeline(pipeline, rays, lod_idx):
    rb = RenderBuffer(xyz=None, hit=None, normal=None, shadow=None, ao=None, dirs=None)
    for ray_pack in rays.split(render_batch):
        rb  += pipeline.tracer(pipeline.nef, rays=ray_pack, lod_idx=lod_idx)
    return rb

# %%
def get_images(cam, pipelines) -> np.ndarray:
    all_rgbs = np.zeros((len(pipelines), H, W, 3), dtype=float)
    for i, pipeline in enumerate(pipelines):
        all_rgbs[i,...] = get_image(cam, pipeline)[0]
    mean = np.mean(all_rgbs, axis = 0)
    var = np.mean(np.var(all_rgbs, axis = 0), axis=2)
    return mean, var

def get_image(cam, pipeline) -> Tuple[np.ndarray, np.ndarray]:
    cam = copy.deepcopy(cam)

    rays = gen_rays_only(cam)
    rays = rays.reshape((rays.shape[0]**2, -1))

    rb = render_pipeline(pipeline, rays, lod_idx=None)
    rgb = rb.rgb.detach().cpu().numpy().reshape((H, W, 3))
    alpha = rb.alpha.detach().cpu().numpy().reshape((H, W, 1))
    return rgb, alpha


# %%
argv_base = [
    '--config=../app/nerf/configs/nerf_hash.yaml',
    '--pretrained=path_to_model',
    '--valid-only'
]

base_scale = 6
base_offset = torch.tensor([0, -0.6, -0.15])
translation_scale = torch.tensor(base_scale/1.25)
translation_offset = translation_scale*base_offset


# %%
model_name = sys.argv[1]
output_dir = sys.argv[2]
nerf2ws = pickle.load(open(sys.argv[3], 'rb'))

print(model_name)
print(output_dir)

# %%
root_dir = '/home/saptarshi/dev/kaolin-wisp/_results3/ensembles/' + model_name + '/'
pipelines = []
for i in range(1,6):
    model_dir = os.path.join(root_dir, f"model_{i}")
    name_list = list(sorted(os.listdir(model_dir)))
    if name_list[-1] == "logs.parquet":
        name_list = name_list[:(len(name_list)-1)]
    model_path = os.path.join(model_dir, name_list[-1], "model.pth")
    print(model_path)
    sys.argv[1:] = argv_base
    sys.argv[2] = sys.argv[2].replace("path_to_model", model_path)
    print(sys.argv[2])
    args, args_dict = parse_args()
    pipeline = make_model(args, args_dict, extra_args, None, None)
    pipelines.append(pipeline)
    break

centre = np.array((0,0.6,0.15))
img_means = []

# %%
centre = np.array((0,0.6,0.15))
img_means = []
r = 3
theta = 0

x = centre[0] + r*np.cos(theta)
y = centre[1] + r*np.sin(theta)
z = centre[2]

# %%
# x, y, z = 0.03, 0.55, 0.4
# x, y, z = 0.5, 0.2, 0.3
x, y, z = 0.2, 0.7, 0.45

dirs = [
    [0, 0, 0.2],
    [0.2, 0, 0.2],
    [-0.2, 0, 0.2],
    [0, 0.2, 0.2],
    [0, -0.2, 0.2]
]
breakpoint()
for i in range(5):
    nerf_center = np.array((0,0.6,0.15))
    ws_center = (nerf2ws @ np.concatenate([nerf_center, [1]]))[:3]
    c2w_ws = get_c2w_from_pos(ws_center + np.array(dirs[i]), ws_center) 
    c2w_ws[:3,-1] = (c2w_ws[:3,-1] - translation_offset) / translation_scale
    c2w_nerf = torch.inverse(torch.tensor(nerf2ws, dtype=torch.float32)) @ c2w_ws
    c2w_nerf[:3,-1] = (c2w_nerf[:3,-1] * translation_scale) + translation_offset
    cam = camera_obj_from_c2w(c2w_nerf)

    # %%
    c2w_unscaled = c2w_ws.clone()
    # c2w_unscaled[..., :3, 3] -= translation_offset
    # c2w_unscaled[..., :3, 3] /= translation_scale

    # %%
    # rays = gen_rays_only(cam)
    # rays = rays.reshape((rays.shape[0]**2, -1))
    # rb = render_pipeline(pipeline, rays, lod_idx=None)
    # rgb = rb.rgb.detach().cpu().numpy().reshape((H, W, 3))
    # depth = rb.depth.detach().cpu().numpy().reshape((H, W))
    # alpha = rb.alpha.detach().cpu().numpy().reshape((H, W)) 
    # depth[alpha <= 0.1] = 10
    # depth[alpha > 0.1] = depth[alpha > 0.1] / alpha[alpha > 0.1]

    os.makedirs(output_dir, exist_ok=True)
    # cv2.imwrite(f'{output_dir}/color{i}.png', cv2.cvtColor((rgb*255.).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # Image.fromarray((depth * 1000 / 6 * 1.25).astype(np.int32), 'I').save(f'{output_dir}/depth{i}.png')
    pickle.dump(c2w_unscaled.detach().cpu().numpy(), open(f'{output_dir}/c2w{i}.pkl', 'wb'))
    # breakpoint()