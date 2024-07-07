
import os
import sys
import copy
import pickle

ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "inerf_utils"))

import json
import torch
import numpy as np
import torchvision.transforms as T
import tqdm
import imageio
import cv2
import matplotlib.pyplot as plt
import wisp
import argparse
from wisp.core import RenderBuffer, Rays

from scipy.spatial.transform import Rotation as R

base_scale = 6
base_offset = torch.tensor([0, -0.6, -0.15])
translation_scale = torch.tensor(base_scale/1.25)
translation_offset = translation_scale * base_offset


parser = argparse.ArgumentParser(description='iNeRF')
parser.add_argument('--config', type=str, default='configs/nerf_hash.yaml', help='Path to config file.')
parser.add_argument('--valid-only', action='store_true', help='Only validate.', default=True)

parser.add_argument('-mn', '--model-name', type=str, default='', help='name of pretrained model.')
parser.add_argument('-dp', '--dataset-path', type=str, default='', help='Path to dataset.')
parser.add_argument('-inp', '--input', type=str, help='Path to input directory.')
parser.add_argument('-out', '--output', type=str, help='Path to output directory.')
parser.add_argument('--num-starts', type=int, default=6, help='Number of initial poses to start from in optimization.')
parser.add_argument('--num-images', type=int, default=4, help='Number of images to optimize over.')

args = parser.parse_args()

root_dir = '/home/saptarshi/dev/kaolin-wisp/_results3/ensembles/' + args.model_name + '/'
model_dir = os.path.join(root_dir, f"model_1")
model_path = os.path.join(model_dir, list(sorted(os.listdir(model_dir)))[0], "model.pth")

sys.argv[1:] = [
    '--config=../app/nerf/configs/nerf_hash.yaml',
    f'--pretrained={model_path}',
    f'--dataset-path={args.dataset_path}',
    '--valid-only'
]


W, H = 200, 200


fx = (0.5 * H) / np.tan(0.5 * float(1.3213687585295282))
extra_args = {
  'resume' : True,
  'output' : './pose_estimation',
  'device' : 'cuda',
  'z_near' : 0.0,
  'z_far' : 10.0,
  'focal' : fx,
  'lrate' : 1e-3,
}

from wisp.framework import WispState
from inerf_utils import *
from kaolin.render.camera import Camera, PinholeIntrinsics, CameraExtrinsics, CameraIntrinsics, blender_coords, opengl_coords
from kaolin.render.camera.extrinsics_backends import _Matrix6DofRotationRep

from wisp.ops.raygen import generate_pinhole_rays, generate_centered_pixel_coords
from wisp.datasets import SampleRays
from wisp.trainers import MultiviewTrainer
from wisp.models.pipeline import Pipeline

from scipy.optimize import minimize
from scipy.optimize import differential_evolution as de
from scipy.linalg import expm
from scipy.linalg import logm

import plotly.express as px

def make_model(args, args_dict, extra_args, train_dataset, validation_dataset):
  pipeline = torch.load(args.pretrained)
  pipeline.to(extra_args['device'])
  scene_state = WispState()
  trainer = load_trainer(
      pipeline=pipeline,
      train_dataset=train_dataset, 
      validation_dataset=validation_dataset,
      device=extra_args['device'], 
      scene_state=scene_state,
      args=args, 
      args_dict=args_dict
  )
  return trainer, pipeline

def gen_camera(pose, focal, w, h, far):
  view_matrix = torch.linalg.inv(pose)

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

inerf_args, inerf_args_dict = parse_args()
train_dataset, validation_dataset = load_dataset(args=inerf_args)

os.makedirs(extra_args['output'], exist_ok=True)

z_near, z_far = extra_args['z_near'], extra_args['z_far']
model, pipeline = make_model(
    inerf_args, inerf_args_dict, extra_args, train_dataset, validation_dataset)

num_images = args.num_images
target_images_np = []
target_images_flatten = []
target_poses = []
for idx in range(num_images):
    target_images_np.append(cv2.resize(cv2.imread(f'{args.input}/post_flip_image{idx+1}.png', cv2.IMREAD_UNCHANGED), (W, H), interpolation=cv2.INTER_AREA))
    target_images_np[idx] = cv2.cvtColor(target_images_np[idx], cv2.COLOR_BGRA2RGBA)

    target_images_flatten.append(np.reshape(target_images_np[idx], [-1, 4]) / 255.0)
    target_images_flatten[idx] = torch.from_numpy(target_images_flatten[idx]).float().to(device=extra_args['device'])

    target_poses.append(torch.tensor(pickle.load(open(f'{args.input}/post_flip_pre_inerf_c2w{idx+1}.pkl', 'rb'))))
    target_poses[idx][:3, 3] *= translation_scale.numpy()
    target_poses[idx][:3, 3] += translation_offset.numpy()


def get_image(cam):
    cam = copy.deepcopy(cam)

    rays = gen_rays_only(cam)
    rays = rays.reshape((rays.shape[0]**2, -1))

    rb = model.renderer.render(pipeline, rays, lod_idx=None)
    rgb = rb.rgb.detach().cpu().numpy().reshape((H, W, 3))
    return rgb

def render_pipeline(renderer, pipeline, rays, lod_idx):
    rb = RenderBuffer(xyz=None, hit=None, normal=None, shadow=None, ao=None, dirs=None)
    for ray_pack in rays.split(renderer.render_batch):
        rb  += pipeline.tracer(pipeline.nef, rays=ray_pack, lod_idx=lod_idx, **renderer.kwargs)
    return rb

curr_pose = torch.tensor(target_poses[0])
camera = gen_camera(curr_pose, extra_args['focal'], W, H, z_far)

def get_euler_pose(rot_pose):
  best_params_se3 = rot_pose.reshape((4,4)).detach().cpu().numpy()
  best_params_euler = np.concatenate([R.from_matrix(best_params_se3[:3, :3]).as_euler('zxz', degrees=True), best_params_se3[:3, -1]])
  return best_params_euler

def get_rot_pose(euler_pose):
  r = R.from_euler('zxz', euler_pose[:3], degrees=True)
  t = euler_pose[3:]
  pose = np.eye(4)
  pose[:3,:3] = r.as_matrix()
  pose[:3, -1] = t
  return pose

def get_se3_pose(pose_6d):
    a,b,c = pose_6d[3:]
    skm = np.array([[0,a,b], [-a,0,c], [-b,-c,0]])
    rot = expm(skm)
    se3_pose = np.eye(4)
    se3_pose[:3,:3] = rot
    se3_pose[:3,-1] = pose_6d[:3]
    return se3_pose

def get_se3_pose_grad(pose_6d):
    a,b,c = pose_6d[3:]
    zero = torch.tensor(0, device='cuda', dtype=torch.float32)
    # skm = torch.tensor([[0,a,b], [-a,0,c], [-b,-c,0]])
    skm = torch.stack([
        torch.stack([zero, a, b]),
        torch.stack([-a, zero, c]),
        torch.stack([-b, -c, zero])
    ])
    rot = torch.matrix_exp(skm)
    se3_pose = torch.eye(4, device='cuda', dtype=torch.float32)
    se3_pose[:3,:3] = rot
    se3_pose[:3,-1] = pose_6d[:3]
    return se3_pose

def get_pose_6d(se3_pose):
    rot = se3_pose[:3,:3]
    skm = logm(rot)
    pose_6d = np.zeros(6)
    pose_6d[:3] = se3_pose[:3,-1]
    pose_6d[3:] = np.array((skm[0,1], skm[0,2], skm[1,2]))
    return pose_6d

def remove_spurious(image, alpha):
    orig_image = image.copy()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image[alpha < 0.1] = 0

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    largest_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour

    new_image = orig_image.copy()
    if largest_contour is not None:
        new_image = np.ones_like(orig_image)
        x, y, w, h = cv2.boundingRect(largest_contour)
        new_image[y:y+h, x:x+w] = orig_image[y:y+h, x:x+w]
    
    return new_image


def lossfn2(p, target_image_flatten, get_grad=False, rewrite=False, opt_all=False, resample=False):
    curr_pose = torch.tensor(target_poses[0]).detach()
    camera = gen_camera(curr_pose, extra_args['focal'], W, H, z_far)

    if not get_grad:
        camera.extrinsics._backend.params = torch.tensor(p, device='cuda', dtype=torch.float32)
        camera.switch_backend('matrix_6dof_rotation')
        rays = gen_rays_only(camera)
        rays = rays.reshape((rays.shape[0]**2, -1))
        rb = pipeline(rays)

        rgb = rb.rgb.detach().cpu().numpy().reshape((H, W, 3))
        alpha = rb.alpha.detach().cpu().numpy().reshape((H, W))
        image = remove_spurious(rgb, alpha)
        image = image.reshape((-1,3))
        loss = torch.abs(torch.tensor(image).cuda() - target_image_flatten[:,:3]).mean()
        return loss.item()
    
    else:
        camera.switch_backend('matrix_6dof_rotation')
        camera.extrinsics.requires_grad = True
        camera.extrinsics._backend.params = _Matrix6DofRotationRep.convert_from_mat(p[None, :])
        rays = gen_rays_only(camera)
        rays = rays.reshape((rays.shape[0]**2, -1))
        rb = pipeline(rays)

        rgb = rb.rgb.reshape((H, W, 3))
        alpha = rb.alpha.reshape((H, W))
        # image = remove_spurious(rgb, alpha)
        image = rgb.reshape((-1,3))
        loss = torch.linalg.norm(image - target_image_flatten[:,:3])
        return loss


def cvt_pose(pose):
    curr_pose = torch.clone(pose).detach()
    camera = gen_camera(curr_pose, extra_args['focal'], W, H, z_far)
    return camera.extrinsics._backend.params

def cvt_pose_se3(pose):
    pose = torch.clone(pose).detach()
    camera = gen_camera(pose, extra_args['focal'], W, H, z_far)
    pose = camera.extrinsics._backend.params.cpu().numpy().reshape((4,4))
    return pose

def uncvt_pose_se3_temp(pose):
    pose = torch.clone(pose).detach()
    camera = gen_camera(pose, extra_args['focal'], W, H, z_far)
    camera.change_coordinate_system(blender_coords())
    camera.change_coordinate_system(blender_coords())
    return pose

def cvt_pose_se3_pred(pose):
    pose = torch.clone(pose).detach()
    vm = torch.linalg.inv(pose)
    new_vm = vm @ blender_coords().float().cuda()
    return torch.linalg.inv(new_vm)

def uncvt_pose_se3(vm):
    pose = np.linalg.inv(vm)
    blender_to_opengl = np.array([
        [ 1.,  0.,  0.],
        [-0., -0., -1.],
        [ 0.,  1.,  0.]
        ])
    fin_pose = np.eye(4)
    fin_pose[:3, :3] = blender_to_opengl @ pose[:3, :3]
    fin_pose[:3, -1] = (blender_to_opengl @ pose[:3, 3].reshape((3,1))).squeeze()
    return fin_pose

def get_scipy_image(t):
    camera = gen_camera(curr_pose, extra_args['focal'], W, H, z_far)
    camera.extrinsics._backend.params = torch.tensor(t, device='cuda', dtype=torch.float32)
    print(camera.extrinsics.parameters().dtype, camera.intrinsics.parameters().dtype)
    im = get_image(camera)
    return (255*im).astype(np.uint8) 

def lossfn_combined_scipy(p):
    se3_pose0 = get_se3_pose(p)
    total_loss = 0
    for idx in range(num_images):
        se3_pose = torch.linalg.inv(target_poses[idx]) @ target_poses[0] @ se3_pose0 if idx > 0 else se3_pose0
        loss = lossfn2(se3_pose, target_images_flatten[idx], rewrite=True, opt_all=True)
        total_loss += loss
    return total_loss

def lossfn_combined_grad(p):
    se3_pose0 = get_se3_pose_grad(p)
    total_loss = 0
    for idx in range(num_images):
        rot_factor = (torch.linalg.inv(target_poses[idx]) @ target_poses[0]).to(torch.float32).cuda()
        se3_pose = rot_factor @ se3_pose0 if idx > 0 else se3_pose0
        loss = lossfn2(se3_pose, target_images_flatten[idx], rewrite=True, opt_all=True, get_grad=True)
        total_loss += loss
    return total_loss


base_position = target_poses[0][:3, -1].numpy()
base_rotation = target_poses[0][:3, :3].numpy()

r_std = 0.15
angle_std = 4
num_poses = args.num_starts - 1
posses = [base_position]
rots = [base_rotation]

posses += [base_position + np.random.normal(scale=r_std, size=(3,)) for i in range(num_poses)]
rots += [base_rotation @ R.from_euler('xyz', np.random.normal(scale=angle_std, size=(3,)), degrees=True).as_matrix() for i in range(num_poses)]

def optimize_pose_gd():
    idx = 0
    n_steps = 100

    losses = []
    poses = []
    cam_poses = []

    for i in range(len(posses)):
        start_pose = np.eye(4)
        start_pose[:3,-1] = posses[i]
        start_pose[:3,:3] = rots[i]
        curr_pose = torch.tensor(start_pose)

        pose_6d_vars = torch.tensor(get_pose_6d(cvt_pose_se3(curr_pose)), device='cuda', dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam(params=[pose_6d_vars], lr=0.03)

        for ep in range(n_steps):
            optimizer.zero_grad()

            loss = lossfn_combined_grad(pose_6d_vars)
            loss.backward()
            optimizer.step()
            # breakpoint()

            print(f"Step {ep}, loss: {loss}")
            # print(f"Step {ep}, loss: {loss}, pose: {pose_6d_vars}")

            camera = gen_camera(curr_pose, extra_args['focal'], W, H, z_far)
            camera.extrinsics._backend.params = torch.tensor(get_se3_pose_grad(pose_6d_vars), device='cuda', dtype=torch.float32)
            curr_im = (255 * get_image(camera)).astype(np.uint8)
            images_folder = f'{args.output}/grad/path'
            os.makedirs(images_folder, exist_ok=True)
            cv2.imwrite(f'{images_folder}/image_end_{ep+1}.png', cv2.cvtColor(curr_im, cv2.COLOR_RGB2BGR))
            
        camera.switch_backend('matrix_se3')
        cam_pose = camera.extrinsics._backend.params.detach().cpu().numpy().reshape((4,4))
        orig_pose = uncvt_pose_se3(np.linalg.inv(cam_pose))

        poses.append(orig_pose)
        losses.append(loss.item())
        cam_poses.append(cam_pose)

    best_pose_idx = np.argmin(losses)
    return losses[best_pose_idx], poses[best_pose_idx], cam_poses[best_pose_idx]


def optimize_pose(methods, tol = 1e-4):
    losses = {}
    poses = {}
    for method in methods:
        losses[method] = []
        poses[method] = []
        for i in range(len(posses)):
            start_pose = np.eye(4)
            start_pose[:3,-1] = posses[i]
            start_pose[:3,:3] = rots[i]
            new_pose_6d = get_pose_6d(cvt_pose_se3(torch.tensor(start_pose)))

            print(f'Loss at initial point {lossfn_combined_scipy(new_pose_6d)}')
            val = minimize(
                fun = lossfn_combined_scipy,
                x0 = new_pose_6d,
                method=method, 
            )
            print(f'Loss after optimization {val.fun}')
            losses[method].append(val.fun)
            poses[method].append(val.x)

    return losses, poses


c2w_estimated_dict = {} 
poses_dict = {}

c2w_file_name = f'{args.output}/c2w_estimated_dict.pkl'
if os.path.isfile(c2w_file_name):
    c2w_estimated_dict = pickle.load(open(c2w_file_name, 'rb')) 


methods = ["Nelder-Mead", "COBYLA", "Powell"]
os.makedirs(f'{args.output}', exist_ok=True)

losses, poses = optimize_pose(methods) 
pickle.dump([losses, poses], open(f'{args.output}/losses_poses.pkl', 'wb'))

# losses, poses = pickle.load(open(f'{args.output}/losses_poses.pkl', 'rb'))
total_losses = []
total_poses = []
total_c2w = []
for method in methods:
    best_pose_idx = np.argmin(losses[method])
    c2w = uncvt_pose_se3(get_se3_pose(poses[method][best_pose_idx]))
    c2w_estimated_dict[method] = c2w
    poses_dict[method] = poses[method][best_pose_idx]

    total_losses.append(losses[method][best_pose_idx])
    total_poses.append(poses[method][best_pose_idx])
    total_c2w.append(c2w)

total_best_pose_idx = np.argmin(total_losses)
c2w_estimated_dict['total'] = total_c2w[total_best_pose_idx].copy()
poses_dict['total'] = total_poses[total_best_pose_idx].copy()

loss, pose, cam_pose = optimize_pose_gd()
c2w_estimated_dict['grad'] = pose
poses_dict['grad'] = cam_pose

idx = 0
for key in poses_dict.keys():
    os.makedirs(f'{args.output}/{key}', exist_ok=True)
    rot_factor = torch.linalg.inv(target_poses[idx]) @ target_poses[0] if idx > 0 else np.eye(4)
    if key == 'grad':
        im_end = get_scipy_image(rot_factor @ poses_dict[key])
    else:
        im_end = get_scipy_image(rot_factor @ get_se3_pose(poses_dict[key]))

    im_target = cv2.cvtColor(target_images_np[idx], cv2.COLOR_RGBA2RGB)
    end_diff = cv2.absdiff(im_end, im_target)
    cv2.imwrite(f'{args.output}/{key}/image_end_{idx+1}.png', cv2.cvtColor(im_end, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{args.output}/{key}/diff_end_{idx+1}.png', cv2.cvtColor(end_diff, cv2.COLOR_RGB2BGR))

    c2w_estimated_dict[key][:3, -1] -= translation_offset.numpy()
    c2w_estimated_dict[key][:3, -1] /= translation_scale.numpy()

pickle.dump(c2w_estimated_dict, open(f'{args.output}/c2w_estimated_dict.pkl', 'wb'))
print(c2w_estimated_dict)




