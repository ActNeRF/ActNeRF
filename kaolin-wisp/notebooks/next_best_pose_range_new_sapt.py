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

def batch_forward_only(pipelines, rays, batch_size=4000):
    total_loss = 0
    for patch_rays in rays.split(batch_size):
            all_rgbs = torch.zeros((len(pipelines), batch_size, 3), dtype=float)
            for i, pipeline in enumerate(pipelines):
                rb = render_pipeline(pipeline, patch_rays, None)
                all_rgbs[i,:,:] = rb.rgb

            unc_loss = -torch.mean(torch.var(all_rgbs, axis = 0))
            total_loss += unc_loss.item()
    return total_loss


def batch_backwards(pipelines, rays:Rays, nerf_mean_unc, unc_lambda, batch_size=4000):

    total_loss = 0
    
    for patch_rays in rays.split(batch_size):
            all_rgbs = torch.zeros((len(pipelines), batch_size, 3), dtype=float)
            for i, pipeline in enumerate(pipelines):
                rb = render_pipeline(pipeline, patch_rays, None)
                all_rgbs[i,:,:] = rb.rgb

            unc_loss = unc_lambda * (-torch.mean(torch.var(all_rgbs, axis = 0)) / np.abs(nerf_mean_unc))
            total_loss += unc_loss.item()
            unc_loss.backward(retain_graph=True)
    # print(total_loss)
    return total_loss            

# OUTDATED
def batch_backwards2(pipelines, camera, batch_size=4000):

    total_loss = 0
    patch = 0
    while True:
        rays = gen_rays_only(camera)
        rays = rays.reshape((rays.shape[0]**2, -1))
        all_patch_rays = rays.split(batch_size)
        if len(all_patch_rays) == patch:
            break
        patch_rays = all_patch_rays[patch]

        all_rgbs = torch.zeros((len(pipelines), batch_size, 3), dtype=float)
        for i, pipeline in enumerate(pipelines):
            rb = render_pipeline(pipeline, patch_rays, None)
            all_rgbs[i,:,:] = rb.rgb

        unc_loss = -torch.mean(torch.var(all_rgbs, axis = 0))
        total_loss += unc_loss.item()
        unc_loss.backward()
        patch += 1
    # print(total_loss)
    return total_loss
            

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
        model_path = os.path.join(model_dir, list(sorted(os.listdir(model_dir)))[0], "model.pth")
        # print(model_path)
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

def is_flip_possible(grasp_data_dir, model_name, pipelines, iter_id=-1):
    subprocess.run(f"cd /home/saptarshi/dev/kaolin-wisp/notebooks && python generate_depth_image.py {model_name} {grasp_data_dir}/anygrasp_data/", shell=True)
    subprocess.run(f"cd /home/saptarshi/dev/anygrasp/anygrasp_sdk/grasp_detection && python demo_new.py --checkpoint_path log/checkpoint_detection.tar --data-dir {grasp_data_dir}/anygrasp_data/", shell=True)

    grasp_data = pickle.load(open(f"{grasp_data_dir}/anygrasp_data/grasp_data.pkl", 'rb'))

    
    grasp_poses, gg_pick_scores, grasp_cosine_scores = grasp_data[0], grasp_data[1], grasp_data[2]
    cam_pose = pickle.load(open(f"{grasp_data_dir}/anygrasp_data/c2w.pkl", 'rb'))
    pickle.dump(cam_pose, open(f"{grasp_data_dir}/c2w.pkl", 'wb'))
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


    # write to a csv file instead of txt
    if iter_id == 1:
        write_file = open(f'{grasp_data_dir}/../best_grasp_scores.csv', 'w')
        write_file.write('iter_id,best gg_pick_score,best grasp_cosine_score,best grasp_angle_scores,best unc_rgb,best unc_depth,best gg/u_r,best gg/u_d,best gc/u_r,best gc/u_d,best ga/u_r,best ga/u_d\n')
    else:
        write_file = open(f'{grasp_data_dir}/../best_grasp_scores.csv', 'a')
    write_file.write(f'{iter_id},{gg_best},{gc_best},{ga_best},{rgb_unc_best},{depth_unc_best},{g1_u_r_best},{g1_u_d_best},{g2_u_r_best},{g2_u_d_best},{g3_u_r_best},{g3_u_d_best}\n')

    write_file.close()

    if g3_u_d_best < 50:
        return False
    ind = np.argmax(g3_u_d)
    best_grasp = grasp_poses[ind]
    p1 = p1s[ind]
    p2 = p2s[ind]
    patched_img = cv2.rectangle(images[0].copy(), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0,0,255), 2)

    cv2.imwrite(f'{grasp_data_dir}/best_grasp_patch.png', cv2.cvtColor((patched_img*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
    pickle.dump(best_grasp, open(f'{grasp_data_dir}/grasp.pkl', 'wb'))
    return True

flip_rot_np = np.array([
    [1., 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])

flip_rot_torch_cpu = torch.tensor([
    [1., 0, 0],
    [0, -1, 0],
    [0, 0, -1]
])

flip_rot_torch_gpu = torch.tensor([
    [1., 0, 0],
    [0, -1, 0],
    [0, 0, -1]
]).cuda()


def main(output_dump_dir, ranges, pipelines, object_center, robot_poses, mean_unc, orientation, move_lambda, unc_lambda):
    print("running main at orientation : ", orientation)    
    range_x = ranges[0]
    range_y = ranges[1]
    range_z = ranges[2]
    range_x_for_zy = ranges[3]

    # if inverted:
    #     output_dump_dir = f'{output_dump_dir}/inverted'
    #     # range_z = -np.flip(range_z)
    #     # range_x_for_zy = np.flip(range_x_for_zy, axis=0)
    # else:
    #     output_dump_dir = f'{output_dump_dir}/original'
    os.makedirs(output_dump_dir, exist_ok=True)


    if robot_poses:
        cur_pose = torch.tensor(robot_poses[0])
        print(cur_pose)
        cur_pose2 = torch.tensor(robot_poses[1])
        print(cur_pose2)
    else:
        cur_pose = torch.tensor([
            [
                0.999954950938356,
                0.0019142828540652458,
                0.009296860492942136,
                0.0031295015942305326
            ],
            [
                0.009491896221019086,
                -0.20166640814931322,
                -0.9794082722390518,
                0.30402505099773407
            ],
            [
                1.1131791470998624e-17,
                0.9794523956504008,
                -0.20167549344104924,
                0.1538112252950668
            ],
            [
                0,
                0,
                0,
                1.0
            ]
        
        ])

        cur_pose2 = torch.tensor([
            [
                -0.999954950938356,
                0.0019142828540652458,
                0.009296860492942136,
                0.0031295015942305326
            ],
            [
                0.009491896221019086,
                -0.20166640814931322,
                0.9794082722390518,
                0.89402505099773407
            ],
            [
                1.1131791470998624e-17,
                0.9794523956504008,
                -0.20167549344104924,
                0.1538112252950668
            ],
            [
                0,
                0,
                0,
                1.0
            ]
        
        ])
    
    cur_pose_unscaled =  cur_pose.clone()
    cur_pose[..., :3, 3] *= translation_scale
    cur_pose[..., :3, 3] += translation_offset

    cur_pose2_unscaled =  cur_pose2.clone()
    cur_pose2[..., :3, 3] *= translation_scale
    cur_pose2[..., :3, 3] += translation_offset

    z_near, z_far = extra_args['z_near'], extra_args['z_far']
    cur_cam = gen_camera(torch.clone(cur_pose).detach(), extra_args['focal'], W, H, z_far, extra_args)
    cur_cam.switch_backend('matrix_6dof_rotation')
    current_pose = cur_cam.extrinsics.parameters().clone()
    current_pose_named = cur_cam.extrinsics.named_params()
    

    def object_close_to_img_boundary(cam, fraction=0.05):
        alphas = get_image(cam, pipeline=pipelines[0])[1].squeeze()
        y = np.arange(H)
        x = np.arange(W)
        inds = np.array(np.meshgrid(y,x)).T.reshape(H, W, 2)
        inds = (inds - np.array([H//2, W//2]))
        y_com = np.mean(alphas * inds[:, :, 0])
        x_com = np.mean(alphas * inds[:, :, 1])

        if(abs(y_com) > (0.5 - fraction)*H):
            return True

        if(abs(x_com) > (0.5 - fraction)*W):
            return True
        return False

    def in_range(x, xmin, xmax):
        return ((x >= xmin) and (x <= xmax))

    def closest_(x, y):
        from bisect import bisect_left
        k = bisect_left(x, y)
        if(k == 0):
            return 0
        right = x[k]
        left = x[k-1]
        if (right - y) < (y - left):
            return k
        else:
            return k-1

    
    def in_robot_range(c2w_scaled, forced_orient=None):
        xyz_scaled = c2w_scaled[..., :3, 3].squeeze()
        if forced_orient is not None:
            xyz_scaled = torch.matmul(forced_orient.to(xyz_scaled.device).double(), xyz_scaled.double())
        else:
            xyz_scaled = torch.matmul(orientation.to(xyz_scaled.device).double(), xyz_scaled.double())
            
        xyz = (xyz_scaled.detach().cpu() - translation_offset) / translation_scale

        if((not in_range(xyz[0], np.min(range_x), np.max(range_x))) or (not in_range(xyz[1], np.min(range_y), np.max(range_y))) or (not in_range(xyz[2], np.min(range_z), np.max(range_z)))):
            return False
        
        z_close = closest_(range_z, xyz[2])
        y_close = closest_(range_y, xyz[1])
        if(xyz[0] < 0):
            if(not in_range(xyz[0], range_x_for_zy[z_close][y_close][0], range_x_for_zy[z_close][y_close][1])):
                return False
        else:
            if(not in_range(xyz[0], range_x_for_zy[z_close][y_close][2], range_x_for_zy[z_close][y_close][3])):
                return False
        
        return True

    def early_stopping(losses, best_loss):
        if len(losses) > extra_args["wait_epochs"] and np.min(np.array(losses[(len(losses)-extra_args["wait_epochs"]):len(losses)])) > best_loss:
            return True
        return False
    
    def stopping_cond(losses, cam, best_loss):
        if(object_close_to_img_boundary(cam)):
            print("Stop Condition 1 Reached")
            return True
        
        if early_stopping(losses, best_loss):
            print("Stop Condition 2 Reached")
            return True
        
        camc = copy.deepcopy(cam)
        camc.change_coordinate_system(blender_coords())
        camc.change_coordinate_system(blender_coords())
        camc.change_coordinate_system(blender_coords())

        c2w = torch.inverse(camc.extrinsics.view_matrix())

        if(not in_robot_range(c2w)):
            print("Stop Condition 3 Reached")
            return True
        return False

    def polar2cart(polar):
        r, t, p = polar
        return r*np.array(( sin(t)*cos(p), sin(t)*sin(p), cos(t) ))
    

    def get_c2w_from_pos(camera_position: np.ndarray, lookat: np.ndarray) -> torch.tensor:
        new_z = (camera_position - lookat)
        new_z = new_z / np.linalg.norm(new_z)

        z = np.array((0,0,1))
        new_y = z - np.dot(z, new_z) * new_z
        if np.linalg.norm(new_y) == 0:
            new_y = np.array((0,1,0)) 
        new_y = new_y / np.linalg.norm(new_y)

        r, _ = R.align_vectors([(0,1,0), (0,0,1)], [new_y, new_z])
        c2w = torch.eye(4)
        c2w[:3,:3] = torch.tensor(np.linalg.inv(r.as_matrix()))
        c2w[:3,-1] = (torch.tensor(camera_position) * translation_scale) + translation_offset
        return c2w

    def camera_obj_from_c2w(c2w: np.ndarray):
        camera = gen_camera(c2w, extra_args['focal'], W, H, z_far, extra_args)
        # cv2.imwrite('abc.png', 255*get_image(camera, pipelines[0])[0])
        return camera

    def get_loss_from_pos(camera_position: np.ndarray, lookat: np.ndarray) -> float:
        c2w = get_c2w_from_pos(camera_position, lookat)
        return get_loss_from_c2w(c2w)
    
    def get_loss_polar(polar: np.ndarray, lookat: np.ndarray) -> float:
        pos = lookat + polar2cart(polar)
        return get_loss_from_pos(pos, lookat)
    
    def get_loss_from_c2w(camera_position: np.ndarray) -> float:
        camera = camera_obj_from_c2w(camera_position)
        # im = get_image(camera, pipelines[0])
        # cv2.imwrite('new.png', (im[0]*255).astype(np.uint8))

        rays = gen_rays_only(camera)
        rays = rays.reshape((rays.shape[0]**2, -1))
        cur_loss = batch_forward_only(pipelines, rays, 40000)

        return cur_loss

    def get_angle_in_deg(v1, v2):
        dot_prod = torch.dot(v1, v2)
        cosine_similarity = dot_prod / (torch.norm(v1) * torch.norm(v2))
        angle = (torch.acos(cosine_similarity) * 180)/ math.pi
        return angle

    r0 = torch.norm(cur_pose_unscaled[:3,-1] - object_center).item()

    def random_sampling(mean_unc, pop_size=300, num_candidates_to_select=10, angle_diff_th=40):
        # if flip:
        #     pop = [np.random.uniform((0.8*r0, 0, 0), (1.2*r0, np.pi/2, 2*np.pi)) for i in range(pop_size)]
        # else:
        pop = [np.random.uniform((0.8*r0, 0, 0), (1.2*r0, np.pi, 2*np.pi)) for i in range(pop_size)]

        pop = [polar2cart(elem) for elem in pop]
        pop_c2w = list(map(lambda x : get_c2w_from_pos(x + object_center.numpy(), object_center.numpy()), pop))

        filtered_pop1 = list(filter(
            lambda x :  in_robot_range(x), pop_c2w))

        filtered_pop2 = list(filter(
            lambda x : not object_close_to_img_boundary(
                camera_obj_from_c2w(x), fraction=0.1
            ), filtered_pop1))
        
        
        print(f"Remaining {len(filtered_pop2)} cands")
        filtered_pop_with_loss = [(get_loss_from_c2w(x), x) for x in filtered_pop2]
        candidates_sorted = list(sorted(filtered_pop_with_loss, key=lambda x: x[0]))

        if mean_unc is None:
            filter_pop_else1 = list(filter(
                lambda x :  in_robot_range(x, torch.matmul(flip_rot_torch_cpu.to(orientation.device).double(), orientation.double())), pop_c2w))
            
            filter_pop_else2 = list(filter(
                lambda x : not object_close_to_img_boundary(
                    camera_obj_from_c2w(x), fraction=0.1
                ), filter_pop_else1))
            
            filter_pop_else_with_loss = [(get_loss_from_c2w(x), x) for x in filter_pop_else2]

            mean_unc = np.mean([x[0] for x in filtered_pop_with_loss] + [x[0] for x in filter_pop_else_with_loss])
            open(f"{output_dump_dir}/mean_unc.txt", 'w').write(str(mean_unc))

        final_candidates = []
        ind = 0
        while len(final_candidates) < num_candidates_to_select:
            found = False
            for i in range(ind, len(candidates_sorted)):
                print(i)
                c2w = candidates_sorted[i][1]
                far_enough = True
                for sel in final_candidates:
                    v1 = c2w[..., :3, 3] - object_center
                    v2 = sel[..., :3, 3] - object_center
   
                    if(get_angle_in_deg(v1, v2) < angle_diff_th):
                        far_enough = False
                        break
                if not far_enough:
                    continue
                final_candidates.append(c2w)
                ind = i + 1
                found = True
                print("found")
                break
            if not found:
                print(f"Oh No!!! Could find only {len(final_candidates)} candidates")
                break

        # for i, c2w in enumerate(final_candidates):
        #     camera = gen_camera(c2w, extra_args['focal'], W, H, z_far, extra_args)
        #     im = get_image(camera, pipelines[0])
        #     cv2.imwrite(f'new_{i}.png', (im[0]*255).astype(np.uint8))
        return final_candidates, mean_unc
    
    from pytorch3d.transforms import matrix_to_quaternion

    def quat_dist(q1, q2):
        return 1 - torch.pow(torch.dot(q1.double(), q2.double()), 2)
    
    def move_action_loss(cam):
        blender_to_opengl = torch.tensor([
            [ 1.,  0.,  0.],
            [-0., -0., -1.],
            [ 0.,  1.,  0.]
            ]).to(device)
        c2w_blender = torch.inverse(cam.extrinsics.view_matrix())

        # return (torch.sum(c2w_blender)**2)
        xyz_scaled = torch.matmul(blender_to_opengl.double(), c2w_blender[..., :3, 3].transpose(0, 1).double()).squeeze()
        
        xyz_scaled = torch.matmul(orientation.to(xyz_scaled.device).double(), xyz_scaled.double())
        xyz = (xyz_scaled - translation_offset.to(device)) / translation_scale.to(device)
        y = xyz[1]
        if(y < object_center[1]):
            robo_xyz = cur_pose_unscaled[..., :3, 3].squeeze().to(device)
        else:
            robo_xyz = cur_pose2_unscaled[..., :3, 3].squeeze().to(device)

        rotmat = torch.matmul(orientation.to(device).double(), torch.matmul(blender_to_opengl.double(), c2w_blender[..., :3, :3].double()).squeeze())
        rot = matrix_to_quaternion(rotmat)
        if y < object_center[1]:
            robo_rot = matrix_to_quaternion(cur_pose_unscaled[..., :3, :3].squeeze().to(device))
        else:
            robo_rot = matrix_to_quaternion(cur_pose2_unscaled[..., :3, :3].squeeze().to(device))
            
        q_dist = quat_dist(rot, robo_rot)
        t_dist = torch.norm(xyz - robo_xyz)

        return (q_dist + 7*t_dist)/8
    

    def optimize(camera, optimizer, mean_unc, max_epochs=100, vid_path=None, loss_vid_path=None, merge_vid_path=None):
        imgs = []
        losses = []
        prev_best_cam = copy.deepcopy(camera)
        prev_best_loss = np.inf
        print(current_pose_named[0]['t'], camera.extrinsics.named_params()[0]['t'])
        for ep in range(max_epochs):
            optimizer.zero_grad()
            rays = gen_rays_only(camera)
            rays = rays.reshape((rays.shape[0]**2, -1))
            unc_loss = batch_backwards(pipelines, rays, mean_unc, unc_lambda, 40000)

            move_loss = move_lambda*move_action_loss(camera)
            move_loss.backward(retain_graph=True)

            cur_loss = unc_loss + move_loss.item()
            print(ep, unc_loss, move_loss.item(), cur_loss, (unc_loss*mean_unc/unc_lambda if unc_lambda != 0 else 0), (move_loss.item()/move_lambda if move_lambda != 0 else 0))
            losses.append(cur_loss)
            
            optimizer.step()
            
            img = get_image(camera, pipeline=pipelines[0])[0]
            imgs.append(img)

            if stopping_cond(losses, camera, prev_best_loss):
                break
            if(prev_best_loss > cur_loss):
                prev_best_loss = cur_loss
                prev_best_cam = copy.deepcopy(camera)

        if vid_path:
            create_vid(imgs, vid_path, W, H)
        
        if loss_vid_path:
            animate_plot(np.arange(len(losses)), losses, loss_vid_path, "epochs", "loss", "loss v/s epoch while training")
        
        if vid_path and loss_vid_path and merge_vid_path:
            merge_horizontal(vid_path, loss_vid_path, merge_vid_path)
            
        return prev_best_loss, prev_best_cam

    best_cam = None
    min_loss = np.inf
    initial_poses, mean_unc = random_sampling(mean_unc)
    
    specify_run = f"ransamp_with_rob_{extra_args['epochs']}_{extra_args['lrate']}_stop{extra_args['wait_epochs']}"
    # pickle.dump(initial_poses, open(f"{output_dump_dir}/initial_poses.pkl", 'wb'))
    for i, initial_pose in enumerate(initial_poses):
        print(f'starting search from candidate {i}')
        print(initial_pose)
        camera = gen_camera(torch.clone(initial_pose).detach(), extra_args['focal'], W, H, z_far, extra_args)
        im = get_image(camera, pipelines[0])
        cv2.imwrite(f"{output_dump_dir}/inner_loop_start_im_{i}.png", cv2.cvtColor((im[0]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))

        camera.switch_backend('matrix_6dof_rotation')
        camera.extrinsics.requires_grad = True

        optimizer = torch.optim.Adam(params=[camera.extrinsics.parameters()], lr=extra_args['lrate'])
        loss, cam  = optimize(camera, optimizer, mean_unc, extra_args["epochs"], f"{output_dump_dir}/pose{i+1}_vid.avi", f"{output_dump_dir}/pose{i+1}_vid_loss.avi", f"{output_dump_dir}/pose{i+1}_merged.avi")
        
        im = get_image(cam, pipelines[0])
        cv2.imwrite(f"{output_dump_dir}/inner_loop_end_im_{i}.png", cv2.cvtColor((im[0]*255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        
        if loss < min_loss:
            best_cam = copy.deepcopy(cam)
            min_loss = loss


    best_cam.change_coordinate_system(blender_coords())
    best_cam.change_coordinate_system(blender_coords())
    best_cam.change_coordinate_system(blender_coords())

    mat = best_cam.extrinsics.view_matrix().detach().cpu()
    mat_c2w = torch.inverse(mat)
    mat_c2w[..., :3, 3] -= translation_offset
    mat_c2w[..., :3, 3] /= translation_scale
    return torch.inverse(mat_c2w), min_loss, mean_unc

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

    # input_path = sys.argv[1]
    # range_path = sys.argv[2]
    # pkl_output_path = sys.argv[3]
    # output_dump_dir = sys.argv[4]
    # anygrasp_data_dir = sys.argv[5]

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', type=str, required=True)
    parser.add_argument('-r', '--range-path', type=str, required=True)
    parser.add_argument('-p', '--pkl-output-path', type=str, required=True)
    parser.add_argument('-d', '--output-dump-dir', type=str, required=True)
    parser.add_argument('-a', '--grasp-data-dir', type=str, required=True)
    parser.add_argument('--iter', type=int, required=False)
    parser.add_argument('-m', '--move-lambda', type=float, default=0.5)
    parser.add_argument('-u', '--unc-lambda', type=float, default=1)
    parser.add_argument('-f', '--flip_lambda', type=float, default=1)
    parser.add_argument('--no-flip', action='store_true', default=False)
    

    args = parser.parse_args()

    inputs = pickle.load(open(args.input_path, 'rb'))
    model_name, object_center, robot_poses, orientation_base = inputs
    ranges = pickle.load(open(args.range_path, 'rb'))

    orientation_base = torch.tensor(orientation_base, dtype=torch.double)

    pipelines = get_pipelines(model_name)

    flip_possible = False
    if not args.no_flip:
        flip_possible = is_flip_possible(args.grasp_data_dir, model_name, pipelines, args.iter)
        print("Flip Possible : ", flip_possible)

    output_dump_dir = f'{args.output_dump_dir}/no_flip'
    nbp, loss, mean_unc = main(output_dump_dir, ranges, pipelines, object_center, robot_poses, None, orientation_base, args.move_lambda, args.unc_lambda)

    open(f"{args.output_dump_dir}/mean_unc.txt", 'w').write(str(mean_unc))
    if not flip_possible:
        pickle.dump(nbp, open(args.pkl_output_path, 'wb'))
        pickle.dump(nbp, open(f"{args.output_dump_dir}/best_pose.pkl", 'wb'))
        pickle.dump(None, open(f"{args.output_dump_dir}/flip_rot.pkl", 'wb'))
        exit(0)
    
    output_dump_dir = f'{args.output_dump_dir}/flip'
    orientation_flip = torch.matmul(flip_rot_torch_cpu, orientation_base)
    nbp_flip, loss_flip, mean_unc_flip = main(output_dump_dir, ranges, pipelines, object_center, robot_poses, mean_unc, orientation_flip, args.move_lambda, args.unc_lambda)

    loss_flip = loss_flip + args.flip_lambda

    if loss_flip < loss:
        pickle.dump(nbp_flip, open(args.pkl_output_path, 'wb'))
        pickle.dump(nbp_flip, open(f"{args.output_dump_dir}/best_pose.pkl", 'wb'))
        pickle.dump(flip_rot_np, open(f"{args.output_dump_dir}/flip_rot.pkl", 'wb'))
    else:
        pickle.dump(nbp, open(args.pkl_output_path, 'wb'))
        pickle.dump(nbp, open(f"{args.output_dump_dir}/best_pose.pkl", 'wb'))
        pickle.dump(None, open(f"{args.output_dump_dir}/flip_rot.pkl", 'wb'))
