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


def batch_backwards(pipelines, rays:Rays, batch_size=4000):

    total_loss = 0
    
    for patch_rays in rays.split(batch_size):
            all_rgbs = torch.zeros((len(pipelines), batch_size, 3), dtype=float)
            for i, pipeline in enumerate(pipelines):
                rb = render_pipeline(pipeline, patch_rays, None)
                all_rgbs[i,:,:] = rb.rgb

            unc_loss = -torch.mean(torch.var(all_rgbs, axis = 0))
            total_loss += unc_loss.item()
            unc_loss.backward(retain_graph=True)
    # print(total_loss)
    return total_loss


            


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

def create_vid(imgs, path):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 10, (W,H))
    for i in range(len(imgs)):
        img = imgs[i]
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

# %%
def animate_plot(xs, ys, path, xlabel = "x", ylabel = "y", title = "plot"):
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)

    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        x = xs[:i]
        y = ys[:i]
        line.set_data(x, y)
        return line,

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(xs), interval=100, blit=True)

    anim.save(path, writer='ffmpeg')

# %%
argv_base = [
    '--config=../app/nerf/configs/nerf_hash.yaml',
    '--pretrained=path_to_model',
    '--valid-only'
]

base_scale = 6
base_offset = torch.tensor([0, -0.4, -0.15])
translation_scale = torch.tensor(base_scale/1.25)
translation_offset = translation_scale*base_offset
# %%
def main(ranges, model_name, object_center, robot_pose):
# model_name = "cheezit_single_side_env2_nobg_sam_scale10"
    root_dir = '/home/saptarshi/dev/kaolin-wisp/_results3/ensembles/' + model_name + '/'
    range_x = ranges[0]
    range_y = ranges[1]
    range_z = ranges[2]
    range_x_for_zy = ranges[3]
    pipelines = []
    for i in range(1,6):
        model_dir = os.path.join(root_dir, f"model_{i}")
        model_path = os.path.join(model_dir, list(sorted(os.listdir(model_dir)))[0], "model.pth")
        print(model_path)
        sys.argv[1:] = argv_base
        sys.argv[2] = sys.argv[2].replace("path_to_model", model_path)
        print(sys.argv[2])
        args, args_dict = parse_args()
        pipeline = make_model(
            args, args_dict, extra_args, None, None)
        pipelines.append(pipeline)

    if robot_pose:
        cur_pose = robot_pose
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
                0.10402505099773407
            ],
            [
                1.1131791470998624e-17,
                0.9794523956504008,
                -0.20167549344104924,
                0.07538112252950668
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        
        ])
    cur_pose_unscaled =  cur_pose.clone()
    cur_pose[..., :3, 3] *= translation_scale
    cur_pose[..., :3, 3] += translation_offset


    # %%
    z_near, z_far = extra_args['z_near'], extra_args['z_far']
    cur_cam = gen_camera(torch.clone(cur_pose).detach(), extra_args['focal'], W, H, z_far, extra_args)
    cur_cam.switch_backend('matrix_6dof_rotation')
    current_pose = cur_cam.extrinsics.parameters().clone()
    current_pose_named = cur_cam.extrinsics.named_params()

    #TODO OUTDATED: Initial orientation of object has changed, hence the angles will chnage a bit
    def calculate_initial_poses(object_center, cam_pose):
        def create_pose(rot, loc):
            pose = torch.eye(4)
            pose[..., :3, :3] = rot
            pose[..., :3, 3] = (loc*translation_scale) + translation_offset
            return pose

        print((cam_pose[..., :3, 3] - object_center)**2)
        d = torch.sqrt(torch.sum((cam_pose[..., :3, 3] - object_center)**2))
        r11 =  R.from_rotvec(np.pi/2 * np.array([1, 0, 0]))
        r12 =  R.from_rotvec(np.pi * np.array([0, 1, 0]))
        front_facing_rot = torch.tensor(r11.as_matrix() @ r12.as_matrix())

        # UP
        r2 =  R.from_rotvec(-np.pi/4 * np.array([1, 0, 0]))
        rot1 = front_facing_rot @ r2.as_matrix()
        loc1 = object_center + d/np.sqrt(2)*torch.tensor([0, 1, 1])
        
        pose1 = create_pose(rot1, loc1)
        # show_image_from_pose(pose1)

        #DOWN
        r2 =  R.from_rotvec(np.pi/4 * np.array([1, 0, 0]))
        rot2 = front_facing_rot @ r2.as_matrix()
        loc2 = object_center + d/np.sqrt(2) * torch.tensor([0, 1, -1])

        pose2 = create_pose(rot2, loc2)
        # show_image_from_pose(pose2)

        # LEFT
        r2 =  R.from_rotvec(-np.pi/4 * np.array([0, 1, 0]))
        rot3 = front_facing_rot @ r2.as_matrix()
        loc3 = object_center + d/np.sqrt(2) * torch.tensor([1, 1, 0])

        pose3 = create_pose(rot3, loc3)
        # show_image_from_pose(pose3)

        # RIGHT
        r2 =  R.from_rotvec(np.pi/4 * np.array([0, 1, 0]))
        rot4 = front_facing_rot @ r2.as_matrix()
        loc4 = object_center + d/np.sqrt(2) * torch.tensor([-1, 1, 0])

        pose4 = create_pose(rot4, loc4)
        # show_image_from_pose(pose4)

        return [pose1, pose2, pose3, pose4]
    
    def merge_horizontal(video1_path, video2_path, out_path):
        video1 = cv2.VideoCapture(video1_path)
        video2 = cv2.VideoCapture(video2_path)

            
        # Get video properties
        fps = int(video1.get(cv2.CAP_PROP_FPS))
        frame_width = int(video1.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_height_2 = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width_2 = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))

        resize_ratio = frame_height / frame_height_2

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = cv2.VideoWriter(out_path, fourcc, fps, (frame_width + int(frame_width_2 * resize_ratio) , frame_height))

        # Loop through the frames and merge them side by side
        for _ in range(num_frames):
            ret1, frame1 = video1.read()
            ret2, frame2 = video2.read()

            if ret1 and ret2:
                if resize_ratio != 1.0:
                    frame2 = cv2.resize(frame2, (0, 0), fx = resize_ratio, fy = resize_ratio)

                merged_frame = cv2.hconcat([frame1, frame2])
                output_video.write(merged_frame)
            else:
                break

        video1.release()
        video2.release()
        output_video.release()

        cv2.destroyAllWindows()

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

    
    def in_robot_range(c2w_scaled):
        xyz_scaled = c2w_scaled[..., :3, 3].squeeze()
        xyz = (xyz_scaled.detach().cpu() - translation_offset) / translation_scale
        if((not in_range(xyz[0], np.min(range_x), np.max(range_x))) or (not in_range(xyz[1], np.min(range_y), np.max(range_y))) or (not in_range(xyz[2], np.min(range_z), np.max(range_z)))):
            return False
        
        z_close = closest_(range_z, xyz[2])
        y_close = closest_(range_y, xyz[1])
        if(not in_range(xyz[0], range_x_for_zy[z_close][y_close][0], range_x_for_zy[z_close][y_close][1])):
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
        new_y = new_y / np.linalg.norm(new_y)

        r, _ = R.align_vectors([(0,1,0), (0,0,1)], [new_y, new_z])
        c2w = torch.eye(4)
        c2w[:3,:3] = torch.tensor(np.linalg.inv(r.as_matrix()))
        c2w[:3,-1] = (torch.tensor(camera_position) * translation_scale) + translation_offset
        return c2w

    def camera_obj_from_c2w(c2w: np.ndarray):
        camera = gen_camera(c2w, extra_args['focal'], W, H, z_far, extra_args)
        cv2.imwrite('abc.png', 255*get_image(camera, pipelines[0])[0])
        return camera

    def get_loss_from_pos(camera_position: np.ndarray, lookat: np.ndarray) -> float:
        c2w = get_c2w_from_pos(camera_position, lookat)
        return get_loss_from_c2w(c2w)
    
    def get_loss_polar(polar: np.ndarray, lookat: np.ndarray) -> float:
        pos = lookat + polar2cart(polar)
        return get_loss_from_pos(pos, lookat)
    
    # %%
    def get_loss_from_c2w(camera_position: np.ndarray) -> float:
        camera = camera_obj_from_c2w(camera_position)
        im = get_image(camera, pipelines[0])
        # cv2.imwrite('new.png', (im[0]*255).astype(np.uint8))

        rays = gen_rays_only(camera)
        rays = rays.reshape((rays.shape[0]**2, -1))
        cur_loss = batch_forward_only(pipelines, rays, 40000)

        return cur_loss
    # %%
    def get_angle_in_deg(v1, v2):
        dot_prod = torch.dot(v1, v2)
        cosine_similarity = dot_prod / (torch.norm(v1) * torch.norm(v2))
        angle = (torch.acos(cosine_similarity) * 180)/ math.pi
        return angle

    r0 = torch.norm(cur_pose_unscaled[:3,-1] - object_center).item()

    def random_sampling(pop_size=300, num_candidates_to_select=10, angle_diff_th=20):

        pop = [np.random.uniform((0.8*r0, 0, 0), (1.2*r0, np.pi, 2*np.pi)) for i in range(pop_size)]
        pop = [polar2cart(elem) for elem in pop]
        pop_c2w = list(map(lambda x : get_c2w_from_pos(x + object_center.numpy(), object_center.numpy()), pop))

        # for i, c2w in enumerate(pop_c2w):
        #     camera = gen_camera(c2w, extra_args['focal'], W, H, z_far, extra_args)
        #     im = get_image(camera, pipelines[0])
        #     cv2.imwrite(f'pop/pop_{i}.png', (im[0]*255).astype(np.uint8))
            
        #     rays = gen_rays_only(camera)
        #     rays = rays.reshape((rays.shape[0]**2, -1))
        #     loss = batch_forward_only(pipelines, rays, 40000)

        filtered_pop1 = list(filter(
            lambda x :  in_robot_range(x), pop_c2w))
            
        filtered_pop2 = list(filter(
            lambda x : (not object_close_to_img_boundary(
                camera_obj_from_c2w(x), fraction=0.1
            )) and in_robot_range(x), filtered_pop1))
        
        print(f"Remaining {len(filtered_pop2)} cands")
        filtered_pop_with_loss = [(get_loss_from_c2w(x), x) for x in filtered_pop2]
        candidates_sorted = list(sorted(filtered_pop_with_loss, key=lambda x: x[0]))
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
        return final_candidates
    

    # %%
    def optimize(camera, optimizer, max_epochs=100, vid_name=None, loss_vid_name=None, merge_vid_name=None):
        imgs = []
        losses = []
        prev_best_cam = copy.deepcopy(camera)
        prev_best_loss = np.inf
        print(current_pose_named[0]['t'], camera.extrinsics.named_params()[0]['t'])
        for ep in range(max_epochs):
            optimizer.zero_grad()
            rays = gen_rays_only(camera)
            rays = rays.reshape((rays.shape[0]**2, -1))
            cur_loss = batch_backwards(pipelines, rays, 40000)
            print(ep, cur_loss)
            losses.append(cur_loss)
            
            optimizer.step()
            
            img = get_image(camera, pipeline=pipelines[0])[0]
            imgs.append(img)

            if stopping_cond(losses, camera, prev_best_loss):
                break
            if(prev_best_loss > cur_loss):
                prev_best_loss = cur_loss
                prev_best_cam = copy.deepcopy(camera)

        
        if vid_name:
            create_vid(imgs, f'results/videos/{vid_name}')
        
        if loss_vid_name:
            animate_plot(np.arange(len(losses)), losses, f'results/videos/{loss_vid_name}', "epochs", "loss", "loss v/s epoch while training")
        
        if vid_name and loss_vid_name and merge_vid_name:
            merge_horizontal(f'results/videos/{vid_name}', f'results/videos/{loss_vid_name}', f'results/videos/{merge_vid_name}')
            
        return prev_best_loss, prev_best_cam



    # %%
    best_cam = None
    min_loss = np.inf
    initial_poses = random_sampling()
    
    specify_run = f"ransamp_new_{extra_args['epochs']}_{extra_args['lrate']}_stop{extra_args['wait_epochs']}"
    for i, initial_pose in enumerate(initial_poses):
        print(initial_pose)
        camera = gen_camera(torch.clone(initial_pose).detach(), extra_args['focal'], W, H, z_far, extra_args)
        camera.switch_backend('matrix_6dof_rotation')
        camera.extrinsics.requires_grad = True

        optimizer = torch.optim.Adam(params=[camera.extrinsics.parameters()], lr=extra_args['lrate'])
        # loss, cam  = optimize(camera, optimizer, extra_args["epochs"], f"pose{i+1}_{specify_run}_vid.avi", f"pose{i+1}_{specify_run}_vid_loss.avi", f"pose{i+1}_{specify_run}_merged.avi")
        loss, cam  = optimize(camera, optimizer, extra_args["epochs"])
        
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
    return torch.inverse(mat_c2w)

if __name__ == '__main__':
    input_path = sys.argv[1]
    range_path = sys.argv[2]
    output_path = sys.argv[3]
    inputs = pickle.load(open(input_path, 'rb'))
    ranges = pickle.load(open(range_path, 'rb'))
    print(inputs[0])
    print(inputs[1])
    next_best_pose = main(ranges, *inputs)
    pickle.dump(next_best_pose.detach().cpu().numpy(), open(output_path, 'wb'))