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
        # img = img / img.max()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = cv2.applyColorMap(img, cv2.COLORMAP_HOT)
        out.write(img)
    out.release()

W, H = 200, 200
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
    new_y = new_y / np.linalg.norm(new_y)

    r, _ = R.align_vectors([(0,1,0), (0,0,1)], [new_y, new_z])
    c2w = torch.eye(4)
    c2w[:3,:3] = torch.tensor(np.linalg.inv(r.as_matrix()))
    c2w[:3,-1] = (torch.tensor(camera_position) * translation_scale) + translation_offset
    return c2w

def camera_obj_from_c2w(c2w: np.ndarray):
    camera = gen_camera(c2w, extra_args['focal'], W, H, extra_args['z_far'], extra_args)
    return camera

def get_c2w_from_pos2(camera_position: np.ndarray, lookat: np.ndarray) -> torch.tensor:
    new_z = (camera_position - lookat)
    new_z = new_z / np.linalg.norm(new_z)

    new_y = np.cross(new_z, np.array((1,0,0)))

    r, _ = R.align_vectors([(0,1,0), (0,0,1)], [new_y, new_z])
    c2w = torch.eye(4)
    c2w[:3,:3] = torch.tensor(np.linalg.inv(r.as_matrix()))
    c2w[:3,-1] = (torch.tensor(camera_position) * translation_scale) + translation_offset
    return c2w

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
    var = (var*30).clip(0, 1)
    var_map = cv2.applyColorMap((var*255).astype(np.uint8), cv2.COLORMAP_HOT)
    var_map = cv2.cvtColor(var_map, cv2.COLOR_BGR2RGB).astype(float)/255
    return mean, var_map


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
def main(model_name, output_path, obj_center):
# model_name = "cheezit_single_side_env2_nobg_sam_scale10"
    root_dir = '/home/saptarshi/dev/kaolin-wisp/_results3/ensembles/' + model_name + '/'
    pipelines = []
    for i in range(1,6):
        model_dir = os.path.join(root_dir, f"model_{i}")
        model_path = os.path.join(model_dir, list(sorted(os.listdir(model_dir)))[0], "model.pth")
        print(model_path)
        sys.argv[1:] = argv_base
        sys.argv[2] = sys.argv[2].replace("path_to_model", model_path)
        print(sys.argv[2])
        args, args_dict = parse_args()
        pipeline = make_model(args, args_dict, extra_args, None, None)
        pipelines.append(pipeline)

    r = 0.3
    posses = []
    img_means = []
    img_vars = []
    for theta in np.linspace(-np.pi/2, 3*np.pi/2, 50):
        x = obj_center[0] + r*np.cos(theta)
        y = obj_center[1] + r*np.sin(theta)
        z = obj_center[2]
        pos = np.array((x,y,z))
        c2w = get_c2w_from_pos(pos, obj_center)
        cam = camera_obj_from_c2w(c2w)
        # img_mean, _ = get_image(cam, pipelines[0])
        img_mean, img_var = get_images(cam, pipelines)
        img_means.append(img_mean)
        img_vars.append(img_var)

    for phi in np.linspace(-np.pi, np.pi, 50):
        x = obj_center[0]
        y = obj_center[1] + r*np.cos(phi)
        z = obj_center[2] + r*np.sin(phi)
        pos = np.array((x,y,z))
        c2w = get_c2w_from_pos2(pos, obj_center)
        cam = camera_obj_from_c2w(c2w)
        # img_mean, _ = get_image(cam, pipelines[0])
        img_mean, img_var = get_images(cam, pipelines)
        img_means.append(img_mean)
        img_vars.append(img_var)

    create_vid(img_means, output_path+'_model.avi', fps=30)
    create_vid(img_vars, output_path+'_uncert.avi', fps=30)
        
if __name__ == '__main__':
    model_name = sys.argv[1]
    output_path = sys.argv[2]
    obj_pose_file = sys.argv[3]
    
    obj_center = pickle.load(open(obj_pose_file, 'rb'))[:3, -1]

    main(model_name, output_path, obj_center)
