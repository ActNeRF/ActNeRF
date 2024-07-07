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
  'wait_epochs': 12,
  'boundary_th':5
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

# %%
def main(model_name, object_center, robot_pose):
# model_name = "cheezit_single_side_env2_nobg_sam_scale10"
    root_dir = '/home/saptarshi/dev/kaolin-wisp/_results_new/' + model_name + '/'

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
                -1.0, 
                -2.823936947407328e-17, 
                5.340285413173258e-16, 
                2.4463562574074363e-17
            ], 
            [
                3.5457582038357597e-16, 
                -0.04896000828076464, 
                0.9988007396819185, 
                0.3995202958290813
            ], 
            [
                -1.089239719165523e-17, 
                0.9988007396819186, 
                0.04896000828076479, 
                -0.08590599579648245
            ], 
            [
                0.0, 
                0.0, 
                0.0, 
                1.0
            ]
        ])
    cur_pose_unscaled =  cur_pose.clone()
    cur_pose[..., :3, 3] *= 10/1.25

    # %%
    z_near, z_far = extra_args['z_near'], extra_args['z_far']
    cur_cam = gen_camera(torch.clone(cur_pose).detach(), extra_args['focal'], W, H, z_far, extra_args)
    cur_cam.switch_backend('matrix_6dof_rotation')
    current_pose = cur_cam.extrinsics.parameters().clone()
    current_pose_named = cur_cam.extrinsics.named_params()


    def calculate_initial_poses(object_center, cam_pose):
        def create_pose(rot, loc):
            pose = torch.eye(4)
            pose[..., :3, :3] = rot
            pose[..., :3, 3] = loc*10/1.25
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

    def object_touching_img_boundary(cam, offset=0):
        alphas = get_image(cam, pipeline=pipelines[0])[1].squeeze()
        count = 0
        count += np.sum(alphas[offset, :])
        count += np.sum(alphas[H-1-offset, :] > 0)
        count += np.sum(alphas[:, offset] > 0)
        count += np.sum(alphas[:, W-1-offset] > 0)

        count -= ((alphas[offset, offset] > 0) + (alphas[offset, W-1-offset] > 0) + (alphas[H-1-offset, offset] > 0) + (alphas[H-1-offset, W-1-offset] > 0))

        if(count >= extra_args['boundary_th']):
            return True
        return False

    def early_stopping(losses, best_loss):
        if len(losses) > extra_args["wait_epochs"] and np.min(np.array(losses[(len(losses)-extra_args["wait_epochs"]):len(losses)])) > best_loss:
            return True
        return False
    
    def stopping_cond(losses, cam, best_loss):
        if(object_touching_img_boundary(cam)):
            print("Stop Condition 1 Reached")
            return True
        
        if early_stopping(losses, best_loss):
            print("Stop Condition 2 Reached")
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
        if (np.linalg.norm(new_y) == 0):
            new_y = np.array((0,1,0))
        # new_y = new_y / np.linalg.norm(new_y)

        r, _ = R.align_vectors([(0,1,0), (0,0,1)], [new_y, new_z])
        c2w = torch.eye(4, device='cuda')
        c2w[:3,:3] = torch.tensor(np.linalg.inv(r.as_matrix()), device='cuda')
        c2w[:3,-1] = torch.tensor(camera_position, device='cuda') * 10 / 1.25
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
        cv2.imwrite('new.png', (im[0]*255).astype(np.uint8))

        rays = gen_rays_only(camera)
        rays = rays.reshape((rays.shape[0]**2, -1))
        cur_loss = batch_forward_only(pipelines, rays, 40000)

        return cur_loss
    # %%
    # import time
    # t1 = time.time()
    # get_loss(np.array((0,2,2)), object_center)
    # t2 = time.time()
    # print(t2 - t1)

    r0 = torch.norm(cur_pose_unscaled[:3,-1] - object_center).item()

    def random_sampling(pop_size=100, num_candidates_to_select=10):

        pop = [np.random.uniform((0.8*r0, 0, 0), (1.2*r0, np.pi, 2*np.pi)) for i in range(pop_size)]
        pop = [polar2cart(elem) for elem in pop]
        pop_c2w = list(map(lambda x : get_c2w_from_pos(x + object_center.numpy(), object_center.numpy()), pop))

        for i, c2w in enumerate(pop_c2w):
            camera = gen_camera(c2w, extra_args['focal'], W, H, z_far, extra_args)
            im = get_image(camera, pipelines[0])
            cv2.imwrite(f'pop/pop_{i}.png', (im[0]*255).astype(np.uint8))
            
            rays = gen_rays_only(camera)
            rays = rays.reshape((rays.shape[0]**2, -1))
            loss = batch_forward_only(pipelines, rays, 40000)

        filtered_pop = list(filter(
            lambda x : not object_touching_img_boundary(
                camera_obj_from_c2w(x, object_center.numpy()), offset=20
            ), pop_c2w))
        
        filtered_pop_with_loss = [(get_loss_from_c2w(x), x) for x in filtered_pop]
        candidates_with_loss = list(sorted(filtered_pop_with_loss, key=lambda x: x[0]))[:num_candidates_to_select]
        candidates_c2w = [x[1] for x in candidates_with_loss]

        # candidates_c2w = list(map(lambda x : get_c2w_from_pos(x, object_center.numpy()), candidates))

        for i, c2w in enumerate(candidates_c2w):
            camera = gen_camera(c2w, extra_args['focal'], W, H, z_far, extra_args)
            im = get_image(camera, pipelines[0])
            cv2.imwrite(f'new_{i}.png', (im[0]*255).astype(np.uint8))
        return candidates_c2w
    

    # def deap_sampling(initial_population_size=100, num_candidates_to_select=10, penalty_weight=0.01):
    #     def objective_function(x):
    #         return obj_value,

    #     def euclidean_distance(ind1, ind2):
    #         return sum((a - b)**2 for a, b in zip(ind1, ind2)) ** 0.5

    #     def diversity_penalty(ind, population):
    #         return objective_function(ind) + penalty_weight * sum(euclidean_distance(ind, p) for p in population)

    #     creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    #     creator.create("Individual", list, fitness=creator.FitnessMin)

    #     toolbox = base.Toolbox()
    #     toolbox.register("attr_float", np.random.uniform, lower_bound, upper_bound)
    #     toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
    #     toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    #     # Configure the evaluation function to use the diversity penalty

    #     toolbox.register("evaluate", diversity_penalty, population=pop)

    #     # Configure and run the evolutionary algorithm

    #     pop = toolbox.population(n=initial_population_size)
    #     hof = tools.HallOfFame(maxsize=archive_size)

    #     algorithms.eaMuPlusLambda(
    #         pop,
    #         toolbox,
    #         mu=mu_value,
    #         lambda_=lambda_value,
    #         cxpb=crossover_probability,
    #         mutpb=mutation_probability,
    #         ngen=num_generations,
    #         halloffame=hof,
    #         stats=None,
    #         verbose=True
    #     )

    #     # Select diverse candidates from the Hall of Fame

    #     diverse_candidates = hof[:num_candidates_to_select]


    # def bayesian_sampling():
    #     pass
    
    val = minimize(
        fun=lambda arr: get_loss_polar(arr, object_center.numpy()),
        x0=(3,0,0), 
        method='COBYLA', 
        constraints=[
            {'type': 'ineq', 'fun': lambda arr: arr[0] - 0.8*r0},
            {'type': 'ineq', 'fun': lambda arr: 1.2*r0 - arr[0]},
            {'type': 'ineq', 'fun': lambda arr: arr[1]},
            {'type': 'ineq', 'fun': lambda arr: np.pi - arr[1]},
            {'type': 'ineq', 'fun': lambda arr: arr[2]},
            {'type': 'ineq', 'fun': lambda arr: 2*np.pi - arr[2]},
        ],
    )
    # pdb.set_trace()
    candidates_c2w = random_sampling()

    # %%
    def optimize(camera, optimizer, max_epochs=100, vid_name=None, loss_vid_name=None, merge_vid_name=None):
        imgs = []
        losses = []
        prev_best_cam = None
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
    # initial_poses = calculate_initial_poses(object_center, cur_pose_unscaled)
    specify_run = f"ransamp_{extra_args['epochs']}_{extra_args['lrate']}_stop{extra_args['wait_epochs']}_boundary{extra_args['boundary_th']}"
    initial_poses = candidates_c2w
    for i, initial_pose in enumerate(initial_poses):
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

    mat = best_cam.extrinsics.view_matrix()
    mat[..., :3, 3] /= (10/1.25)
    return mat

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    inputs = pickle.load(open(input_path, 'rb'))
    print(inputs[0])
    print(inputs[1])
    inputs[0] = 'workdir/archive/800_angle_std_2_rstd_0.01/cheezit_single_side_env2_nobg_sam_scale10_iteration_2'
    next_best_pose = main(*inputs)
    pickle.dump(next_best_pose.detach().cpu().numpy(), open(output_path, 'wb'))