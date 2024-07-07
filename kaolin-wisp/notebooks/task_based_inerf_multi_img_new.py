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
from mystic.solvers import fmin_powell
from mystic.monitors import VerboseMonitor

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
parser.add_argument('--debug', action='store_true', help='Go into breakpoints')
parser.add_argument('-o', '--object-name', type=str, help='Name of object')

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

def bp():
    if args.debug:
        breakpoint()


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

def logm2(rot):
  theta = np.arccos((np.trace(rot) - 1) / 2)
  if theta == 0:
    return np.zeros((3,3))
  if theta == np.pi:
    theta = np.pi - 1e-6
  return (theta / (2 * np.sin(theta))) * (rot - rot.T)

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
target_poses = pickle.load(open(f'{args.input}/target_poses.pkl', 'rb'))

for idx in range(num_images):
    target_images_np.append(cv2.resize(cv2.imread(f'{args.input}/rgb_{idx}.png', cv2.IMREAD_UNCHANGED), (W, H), interpolation=cv2.INTER_AREA))
    target_images_np[idx] = cv2.cvtColor(target_images_np[idx], cv2.COLOR_BGRA2RGBA)

    target_images_flatten.append(np.reshape(target_images_np[idx], [-1, 4]) / 255.0)
    target_images_flatten[idx] = torch.from_numpy(target_images_flatten[idx]).float().to(device=extra_args['device'])

    target_poses[idx] = torch.tensor(target_poses[idx])
    target_poses[idx][:3, 3] *= translation_scale.numpy()
    target_poses[idx][:3, 3] += translation_offset.numpy()


def get_image(cam):
    cam = copy.deepcopy(cam)

    rays = gen_rays_only(cam)
    rays = rays.reshape((rays.shape[0]**2, -1))

    rb = model.renderer.render(pipeline, rays, lod_idx=None)
    rgb = rb.rgb.detach().cpu().numpy().reshape((H, W, 3))
    return rgb
    
def get_rgbd_image(cam):
    cam = copy.deepcopy(cam)
    rays = gen_rays_only(cam)
    rays = rays.reshape((rays.shape[0]**2, -1))
    rb = model.renderer.render(pipeline, rays, lod_idx=None)
    rgb = rb.rgb.detach().cpu().numpy().reshape((H, W, 3))
    depth = rb.depth.detach().cpu().numpy().reshape((H, W))
    alpha = rb.alpha.detach().cpu().numpy().reshape((H, W))
    
    depth[alpha <= 0.1] = 10
    depth[alpha > 0.1] = depth[alpha > 0.1] / alpha[alpha > 0.1]
    # depth = (depth * 1000 / 6 * 1.25).astype(np.int32)
    depth = (depth / 6 * 1.25)


    return rgb, depth


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

def pose_6d2se3(p):
    pose_6d = copy.deepcopy(p)
    a,b,c = pose_6d[3:]
    skm = np.array([[0,a,b], [-a,0,c], [-b,-c,0]])
    rot = expm(skm)
    se3_pose = np.eye(4)
    se3_pose[:3,:3] = rot
    se3_pose[:3,-1] = pose_6d[:3]
    return se3_pose

# def pose_6d2se3(p):
#     pose_6d = copy.deepcopy(p)
#     # a,b,c = pose_6d[3:]
#     # skm = np.array([[0,a,b], [-a,0,c], [-b,-c,0]])
#     # rot = expm(skm)
#     se3_pose = np.eye(4)
#     # quat = np.concatenate([
#     #     pose_6d[3:], 
#     #     np.array([ np.sqrt(1-np.linalg.norm(pose_6d[3:])**2) ]) 
#     #   ])
#     # breakpoint()
#     se3_pose[:3,:3] = R.from_quat(pose_6d[3:]).as_matrix()
#     se3_pose[:3,-1] = pose_6d[:3]
#     return se3_pose

def pose_6d2se3_grad(pose_6d):
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

def get_pose_6d(s):
    se3_pose = copy.deepcopy(s)
    rot = se3_pose[:3,:3]
    skm = logm(rot)
    # skm = logm2(rot)
    pose_6d = np.zeros(6)
    pose_6d[:3] = se3_pose[:3,-1]
    pose_6d[3:] = np.array((skm[0,1], skm[0,2], skm[1,2]))
    return pose_6d

# def get_pose_6d(s):
#     se3_pose = copy.deepcopy(s)
#     # rot = se3_pose[:3,:3]
#     # skm = logm2(rot)
#     pose_6d = np.zeros(7)
#     pose_6d[:3] = se3_pose[:3,-1]
#     quat = R.from_matrix(se3_pose[:3,:3]).as_quat()
#     pose_6d[3:] = quat
#     # pose_6d[3:] = np.array((skm[0,1], skm[0,2], skm[1,2]))
#     return pose_6d

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

# def centroid(image, alpha):
#     image = (image * 255).astype(np.uint8)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     image = cv2.GaussianBlur(image, (5, 5), 0)
#     ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     breakpoint()
#     M = cv2.moments(thresh)
#     cX = int(M["m10"] / M["m00"])
#     cY = int(M["m01"] / M["m00"])
#     return np.array([cX, cY])

def find_centroid(alpha):
    mask = (alpha > 0)
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return np.array([500, 500])
    centroid_x = np.mean(indices[:, 1])
    centroid_y = np.mean(indices[:, 0])
    # breakpoint()
    return np.array([centroid_x, centroid_y])

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
        rgb_loss = torch.abs(torch.tensor(image).cuda() - target_image_flatten[:,:3]).mean()


        # target_img = target_image_flatten.cpu().numpy().reshape((H, W, 4))
        # c1 = find_centroid(alpha)
        # c2 = find_centroid(target_img[:,:,3])
        # dist_loss = np.linalg.norm(c1- c2)
        # # breakpoint()
        # loss = rgb_loss.item() + dist_loss / 1000

        # if np.isinf(loss) or np.isnan(loss):
        #     breakpoint() 
        loss = rgb_loss.item()

        return loss
    
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

def pose_ws2kaolin_orig(pose):
    pose = torch.clone(pose).detach()
    camera = gen_camera(pose, extra_args['focal'], W, H, z_far)
    pose = camera.extrinsics._backend.params.cpu().numpy().reshape((4,4))
    return pose
    # return np.linalg.inv(pose)

def pose_ws2kaolin(pose):
    blender_to_opengl = np.array([
        [ 1.,  0.,  0., 0],
        [-0., -0., -1., 0],
        [ 0.,  1.,  0., 0],
        [ 0,   0,   0,  1]
        ])
    fin_pose = np.linalg.inv(blender_to_opengl) @ pose
    return np.linalg.inv(fin_pose)

def pose_kaolin2ws(vm):
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
    # print(camera.extrinsics.parameters().dtype, camera.intrinsics.parameters().dtype)
    im = get_image(camera)
    return (255*im).astype(np.uint8) 


# def get_scipy_image(t):
#     camera = gen_camera(curr_pose, extra_args['focal'], W, H, z_far)
#     camera.extrinsics._backend.params = torch.tensor(np.linalg.inv(t), device='cuda', dtype=torch.float32)
#     # print(camera.extrinsics.parameters().dtype, camera.intrinsics.parameters().dtype)
#     im = get_image(camera)
#     return (255*im).astype(np.uint8) 

def get_scipy_image_rgbd(t):
    camera = gen_camera(curr_pose, extra_args['focal'], W, H, z_far)
    camera.extrinsics._backend.params = torch.tensor(np.linalg.inv(t), device='cuda', dtype=torch.float32)
    # print(camera.extrinsics.parameters().dtype, camera.intrinsics.parameters().dtype)
    im, depth = get_rgbd_image(camera)
    return (255*im).astype(np.uint8), depth


def lossfn_combined_scipy_old(p):
    se3_pose0 = pose_6d2se3(p)
    total_loss = 0
    for idx in range(num_images):
        se3_pose = torch.linalg.inv(target_poses[idx]) @ target_poses[0] @ se3_pose0 if idx > 0 else se3_pose0
        loss = lossfn2(se3_pose, target_images_flatten[idx], rewrite=True, opt_all=True)
        total_loss += loss
    return total_loss

def lossfn_combined_scipy(p, start_pose):
    se3_pose0 = start_pose @ pose_6d2se3(p)
    total_loss = 0
    for idx in range(num_images):
        se3_pose = torch.linalg.inv(target_poses[idx]) @ target_poses[0] @ se3_pose0 if idx > 0 else se3_pose0
        loss = lossfn2(se3_pose, target_images_flatten[idx], rewrite=True, opt_all=True)
        total_loss += loss
    return total_loss

def lossfn_combined_scipy_translation(t, r, start_pose):
    p = np.concatenate([t, r])
    se3_pose0 = start_pose @ pose_6d2se3(p)
    total_loss = 0
    for idx in range(num_images):
        se3_pose = torch.linalg.inv(target_poses[idx]) @ target_poses[0] @ se3_pose0 if idx > 0 else se3_pose0
        loss = lossfn2(se3_pose, target_images_flatten[idx], rewrite=True, opt_all=True)
        total_loss += loss
    return total_loss



def lossfn_combined_grad(p):
    se3_pose0 = pose_6d2se3_grad(p)
    total_loss = 0
    for idx in range(num_images):
        rot_factor = (torch.linalg.inv(target_poses[idx]) @ target_poses[0]).to(torch.float32).cuda()
        se3_pose = rot_factor @ se3_pose0 if idx > 0 else se3_pose0
        loss = lossfn2(se3_pose, target_images_flatten[idx], rewrite=True, opt_all=True, get_grad=True)
        total_loss += loss
    return total_loss


# base_position = target_poses[0][:3, -1].numpy()
# base_rotation = target_poses[0][:3, :3].numpy()

# r_std = 0.15
# angle_std = 4
# num_poses = args.num_starts - 1
# centers = [base_position]
# rots = [base_rotation]

# centers += [base_position + np.random.normal(scale=r_std, size=(3,)) for i in range(num_poses)]
# rots += [base_rotation @ R.from_euler('xyz', np.random.normal(scale=angle_std, size=(3,)), degrees=True).as_matrix() for i in range(num_poses)]

def estimate_camera_pose(rgb1, depth1, rgb2):
    # Step 1: Feature Extraction
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(rgb1, None)
    kp2, des2 = orb.detectAndCompute(rgb2, None)

    # Step 2: Feature Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Step 3: Depth Association
    matched_points_3d = []
    for match in matches:
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        depth = depth1[int(pt1[1]), int(pt1[0])]  # Assuming depth is in the same resolution as the RGB image
        if depth != 0:  # Filter out invalid depth values
            matched_points_3d.append((pt1[0], pt1[1], depth))

    img_matches = cv2.drawMatches(rgb1, kp1, rgb2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(f'{args.output}/before_inerf/matches.png', img_matches)
    
    if len(matched_points_3d) < 4:
        raise ValueError("Insufficient matched points for pose estimation")
    
    # Step 4: Pose Estimation
    points_3d = np.array(matched_points_3d)[:, :3].astype(np.float32)
    points_2d = np.array([
        kp2[match.trainIdx].pt for match in matches if depth1[int(kp1[match.queryIdx].pt[1]), int(kp1[match.queryIdx].pt[0])]!=0
    ]).astype(np.float32)
    
    camera_matrix = np.array([
        [extra_args['focal'], 0, W],
        [0, extra_args['focal'], H],
        [0, 0, 1]
    ])
    dist_coeffs = np.zeros((4,1))
    _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, points_2d, camera_matrix, dist_coeffs)

    Rotation_matrix_T = cv2.Rodrigues(rvec)[0]
    Rotation_matrix = Rotation_matrix_T.T
    Translate_vector = - np.dot(Rotation_matrix , tvec).squeeze()

    Transformation_matrix = np.eye(4)
    Transformation_matrix[:3, :3] = Rotation_matrix
    Transformation_matrix[:3,  3] = Translate_vector / 1000
    # Transformation_matrix[:3,  3] = Translate_vector

    
    # Step 5: Optimization (Optional)
    # You may want to refine the estimated pose using bundle adjustment or other optimization techniques
    
    return Transformation_matrix, rvec, tvec

def estimate_c2w(cx, cy, fx, fy, rgb1, depth1, c2w_1, rgb2):
    # Compute 3D points from depth image
    x = np.arange(0, depth1.shape[1])
    y = np.arange(0, depth1.shape[0])
    xx, yy = np.meshgrid(x, y)
    X = (xx - cx) * depth1 / fx
    Y = (yy - cy) * depth1 / fy
    Z = depth1

    # Flatten points
    points_3d = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

    # Apply c2w_1 transformation to points
    points_3d_c1 = np.dot(c2w_1[:3, :3], points_3d.T) + c2w_1[:3, 3].reshape(3, 1)

    # Estimate the transformation from points of the first camera to points of the second camera
    A = np.dot(points_3d_c1, points_3d.T)
    u, _, vh = np.linalg.svd(A)
    R_est = np.dot(u, vh)

    # If the determinant of the estimated rotation is -1, flip the last column of V
    if np.linalg.det(R_est) < 0:
        vh[-1, :] *= -1
        R_est = np.dot(u, vh)

    # Compute the translation component
    t_est = np.mean(points_3d.T - np.dot(R_est, points_3d_c1), axis=0)

    # Compose the transformation matrix c2w_2
    c2w_2 = np.eye(4)
    c2w_2[:3, :3] = R_est
    c2w_2[:3, 3] = t_est

    return c2w_2

WORKSPACE_CENTER = np.array((0, 0.6, 0.15))
if args.object_name == 'rubik':
    centers = [WORKSPACE_CENTER + 0.2 * (-1 if i%2==1 else 1) * np.array([1 if j == (i//2) else 0 for j in range(3)]) for i in range(6)]
else:
    centers = [WORKSPACE_CENTER + 0.3 * (-1 if i%2==1 else 1) * np.array([1 if j == (i//2) else 0 for j in range(3)]) for i in range(6)]

print(centers)

start_poses = []
start_pose_fmts = []
os.makedirs(f'{args.output}/before_inerf/', exist_ok=True)
for i in range(6):
    start_pose = get_c2w_from_pos(centers[i], WORKSPACE_CENTER).copy()
    start_pose[:3,-1] = start_pose[:3,-1] * translation_scale.numpy() + translation_offset.numpy()
    start_poses.append(start_pose)

    # start_pose_fmt = pose_6d2se3(get_pose_6d(pose_ws2kaolin(torch.tensor(start_pose))))
    # start_pose_fmt = pose_6d2se3(get_pose_6d(pose_ws2kaolin(start_pose)))
    # start_pose_fmts.append(start_pose_fmt)
    im_start = get_scipy_image(pose_ws2kaolin(start_pose))

    # breakpoint()
    # im_start = get_image(gen_camera(torch.tensor(start_pose), extra_args['focal'], W, H, z_far))
    # im_start = (im_start * 255).astype(np.uint8)
    cv2.imwrite(f'{args.output}/before_inerf/image_start_{i+1}.png', cv2.cvtColor(im_start, cv2.COLOR_RGB2BGR))

# start_poses = [start_poses[2]]

# rgb1, depth1 = get_scipy_image_rgbd(start_pose_fmts[1])
# rgb2 = target_images_np[0]
# rel_pose, rvec, tvec = estimate_camera_pose(rgb1, depth1, rgb2)
# # updated_pose = np.linalg.inv(rel_pose) @ start_poses[1]
# updated_pose = rel_pose @ start_poses[1]
# updated_pose_fmt = pose_ws2kaolin(torch.tensor(updated_pose))
# im_update = get_scipy_image(updated_pose_fmt)
# cv2.imwrite(f'{args.output}/before_inerf/image_update.png', cv2.cvtColor(im_update, cv2.COLOR_RGB2BGR))

# breakpoint()

def optimize_pose_gd():
    idx = 0
    n_steps = 100

    losses = []
    poses = []
    cam_poses = []

    for i in range(len(centers)):
        curr_pose = torch.tensor(start_poses[i])
        pose_6d_vars = torch.tensor(get_pose_6d(pose_ws2kaolin_orig(curr_pose)), device='cuda', dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam(params=[pose_6d_vars], lr=0.01)

        for ep in range(n_steps):
            optimizer.zero_grad()

            loss = lossfn_combined_grad(pose_6d_vars)
            loss.backward()
            optimizer.step()

            print(f"Step {ep}, loss: {loss}")
            # print(f"Step {ep}, loss: {loss}, pose: {pose_6d_vars}")

            camera = gen_camera(curr_pose, extra_args['focal'], W, H, z_far)
            camera.extrinsics._backend.params = torch.tensor(pose_6d2se3_grad(pose_6d_vars), device='cuda', dtype=torch.float32)
            curr_im = (255 * get_image(camera)).astype(np.uint8)
            images_folder = f'{args.output}/grad/path'
            os.makedirs(images_folder, exist_ok=True)
            # cv2.imwrite(f'{images_folder}/image_end_{ep+1}.png', cv2.cvtColor(curr_im, cv2.COLOR_RGB2BGR))
            
        camera.switch_backend('matrix_se3')
        cam_pose = camera.extrinsics._backend.params.detach().cpu().numpy().reshape((4,4))
        orig_pose = pose_kaolin2ws(np.linalg.inv(cam_pose))

        poses.append(orig_pose)
        losses.append(loss.item())
        cam_poses.append(cam_pose)

    best_pose_idx = np.argmin(losses)
    return losses[best_pose_idx], poses[best_pose_idx], cam_poses[best_pose_idx]

def constraints(p):
    q = p.copy()
    q[3:] = q[3:] / np.linalg.norm(q[3:])
    return q

# bounds = [
#     (-2.5, 0.5, 0.1, -np.pi, -np.pi, -np.pi, -np.pi),
#     (+2.5, 6, 1.5, +np.pi, +np.pi, +np.pi, +np.pi),
# ]

bounds = [
    (-2.5, 2.5),
    (-2.5, 2.5),
    (-2.5, 2.5),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
    (-np.pi, np.pi),
]

def optimize_pose(methods, tol = 1e-4):
    np.set_printoptions(suppress=True, precision=3)
    losses = {}
    poses = {}
    for method in methods:
        losses[method] = []
        poses[method] = []
        for i in range(len(start_poses)):
            start_pose = pose_ws2kaolin(start_poses[i])
            # new_pose_6d = get_pose_6d(pose_ws2kaolin(torch.tensor(start_pose)))
            # new_pose_6d = get_pose_6d(pose_ws2kaolin(start_pose))
            new_pose_6d = get_pose_6d(np.eye(4))

            # im = get_scipy_image(pose_6d2se3(start_pose))
            # cv2.imwrite(f'{args.output}/before_inerf/image_start_new_{i+1}.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

            print(f'Loss at initial point {lossfn_combined_scipy(new_pose_6d, start_pose)}')
            # captured_cam_pose = pickle.load(open('/home/saptarshi/dev/CustomComposer/task_testing2/cam_poses_rubik.pkl', 'rb'))[0]
            # gt_nerf2ws = pickle.load(open('/home/saptarshi/dev/CustomComposer/task_testing_noflip_results_inerf/rubik/pose_1/grasp_data/nerf2ws.pkl', 'rb'))
            # gt_c2w = np.linalg.inv(gt_nerf2ws) @ captured_cam_pose
            # gt_c2w[:3,-1] = gt_c2w[:3,-1] * translation_scale.numpy() + translation_offset.numpy()
            # im = get_scipy_image(pose_ws2kaolin(torch.tensor(gt_c2w)))
            # im = get_scipy_image(pose_ws2kaolin(gt_c2w))
            # gt_pose = get_pose_6d(np.linalg.inv(start_pose) @ pose_ws2kaolin(gt_c2w))
            # cv2.imwrite(f'{args.output}/gt0.png', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            # bp()

            x0 = new_pose_6d
            if args.object_name == 'rubik':
                val1 = minimize(
                    fun = lossfn_combined_scipy_translation,
                    x0 = new_pose_6d[:3],
                    # method=method, 
                    method='Nelder-Mead',
                    args=(new_pose_6d[3:], start_pose),
                    bounds=bounds[:3]
                )
                # stepmon = VerboseMonitor(1)
                # solution = fmin_powell(
                #     cost=lossfn_combined_scipy_translation,
                #     x0=new_pose_6d[:3],
                #     bounds=bounds[:3],
                #     args=(new_pose_6d[3:], start_pose),
                #     itermon=stepmon
                # )
                print(val1)
                x0[:3] = val1.x
                print(x0)
            val = minimize(
                fun = lossfn_combined_scipy,
                x0 = x0,
                method=method, 
                args=(start_pose,),
                # bounds=bounds
            )
            solution = val.x
            fn_min = val.fun

            # stepmon = VerboseMonitor(1)
            # solution = fmin_powell(
            #     cost=lossfn_combined_scipy,
            #     x0=new_pose_6d,
            #     constraints=constraints,
            #     bounds=bounds,
            #     itermon=stepmon
            # )
            # fn_min = lossfn_combined_scipy(val.x, start_pose)
            # fn_min = lossfn_combined_scipy(gt_pose, start_pose)
            # print(solution)

            print(f'Loss after optimization {fn_min}')
            losses[method].append(fn_min)
            poses[method].append(solution)

    return losses, poses



c2w_estimated_dict = {}
poses_dict = {}

c2w_file_name = f'{args.output}/c2w_estimated_dict.pkl'
if os.path.isfile(c2w_file_name):
    c2w_estimated_dict = pickle.load(open(c2w_file_name, 'rb')) 

# methods = ["Nelder-Mead", "COBYLA", "Powell"]
methods = ["Powell"]
    
os.makedirs(f'{args.output}', exist_ok=True)

losses, poses = optimize_pose(methods) 
pickle.dump([losses, poses], open(f'{args.output}/losses_poses.pkl', 'wb'))

# losses, poses = pickle.load(open(f'{args.output}/losses_poses.pkl', 'rb'))

total_losses = []
total_poses = []
total_c2w = []
for method in methods:
    best_pose_idx = np.argmin(losses[method])
    # c2w = pose_kaolin2ws(pose_6d2se3(poses[method][best_pose_idx]))
    c2w = pose_kaolin2ws(pose_ws2kaolin(start_poses[best_pose_idx]) @ pose_6d2se3(poses[method][best_pose_idx]))
    c2w_estimated_dict[method] = c2w
    poses_dict[method] = poses[method][best_pose_idx]

    total_losses.append(losses[method][best_pose_idx])
    total_poses.append(poses[method][best_pose_idx])
    total_c2w.append(c2w)

total_best_pose_idx = np.argmin(total_losses)
c2w_estimated_dict['total'] = total_c2w[total_best_pose_idx].copy()
poses_dict['total'] = total_poses[total_best_pose_idx]

# loss, pose, cam_pose = optimize_pose_gd()
# c2w_estimated_dict['grad'] = pose
# poses_dict['grad'] = cam_pose

idx = 0
for key in poses_dict.keys():
    os.makedirs(f'{args.output}/{key}', exist_ok=True)
    rot_factor = torch.linalg.inv(target_poses[idx]) @ target_poses[0] if idx > 0 else np.eye(4)
    if key == 'grad':
        im_end = get_scipy_image(rot_factor @ poses_dict[key])
    else:
        # im_end = get_scipy_image(rot_factor @ pose_6d2se3(poses_dict[key]))
        best_pose_idx = np.argmin(losses[list(poses_dict.keys())[total_best_pose_idx]])
        im_end = get_scipy_image(rot_factor @ pose_ws2kaolin(start_poses[best_pose_idx]) @ pose_6d2se3(poses_dict[key]))

    im_target = cv2.cvtColor(target_images_np[idx], cv2.COLOR_RGBA2RGB)
    end_diff = cv2.absdiff(im_end, im_target)
    cv2.imwrite(f'{args.output}/{key}/image_end_{idx+1}.png', cv2.cvtColor(im_end, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'{args.output}/{key}/diff_end_{idx+1}.png', cv2.cvtColor(end_diff, cv2.COLOR_RGB2BGR))

    c2w_estimated_dict[key][:3, -1] -= translation_offset.numpy()
    c2w_estimated_dict[key][:3, -1] /= translation_scale.numpy()

pickle.dump(c2w_estimated_dict, open(c2w_file_name, 'wb'))
print(c2w_estimated_dict)
# breakpoint() 



