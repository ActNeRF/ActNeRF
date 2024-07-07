import os
import sys
import pickle
import numpy as np

ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, ".."))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, os.path.join(ROOT_DIR, "inerf_utils"))

import json
import torch
import imageio
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt

from enum import Enum
from scipy.spatial.transform import Rotation as R

from tqdm import tqdm

from wisp.framework import WispState
from inerf_utils import *
from kaolin.render.camera import Camera, blender_coords

from wisp.ops.raygen import generate_pinhole_rays, generate_centered_pixel_coords


base_scale = 6
base_offset = torch.tensor([0, -0.6, -0.15])
translation_scale = torch.tensor(base_scale/1.25)
translation_offset = translation_scale * base_offset

parser = argparse.ArgumentParser(description='Export mesh and compute fscore')
parser.add_argument('--config', type=str, default='configs/nerf_hash.yaml', help='Path to config file.')
parser.add_argument('--valid-only', action='store_true', help='Only validate.', default=True)

parser.add_argument('-o', '--obj-name', type=str, default='', help='name of object.')
parser.add_argument('-e', '--exp-name', type=str, default='', help='name of pretrained model.')
parser.add_argument('-i', '--iter', type=int, default=-1, help='iter_id')
parser.add_argument('-w', '--workdir', type=str, default='/home/saptarshi/dev/kaolin-wisp/_results3/ensembles/workdir/', help='/workdir/')
parser.add_argument('-mt', '--mcube-threshold', type=float, default=20., help='Threshold for Marching Cube.')
parser.add_argument('-out', '--out-file', type=str, help='Path to metrics output pcd file')

args = parser.parse_args()

import open3d as o3d

# sys.argv[1:] = [
#     '--config=../app/nerf/configs/nerf_hash.yaml',
#     f'--pretrained={model_path}',
#     f'--dataset-path={dataset_path}',
#     '--valid-only'
# ]

# model_args, model_args_dict = parse_args()
# train_dataset, validation_dataset = load_dataset(args=model_args)

H, W = 200, 200
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

class Objects(Enum):
    CHEEZIT = 1
    RUBIK = 4
    SPAM = 8
    SMUG = 11
    BASKET = 12

out_name_map = {
    Objects.CHEEZIT: 'cheezit',
    Objects.RUBIK: 'rubik',
    Objects.SPAM: 'spam',
    Objects.SMUG: 'small_mug',
    Objects.BASKET: 'basket',
}

obj_scale_map = {
    Objects.CHEEZIT:(1, 1, 1),
    Objects.RUBIK:  (1.1, 1.1, 1.1),
    Objects.SPAM: (2, 2, 1.2),
    Objects.SMUG: (1, 3, 1),
    Objects.BASKET: (0.8, 0.25, 0.7),
}

def search_in_dict(d, val):
  for key, value in d.items():
    if value == val:
      return key


def compute_chamfer_distance(pcd1, pcd2):
  dist1 = pcd1.compute_point_cloud_distance(pcd2)
  dist2 = pcd2.compute_point_cloud_distance(pcd1)
  chamfer_distance = (np.array(dist1).sum() + np.array(dist2).sum()) / (len(dist1) + len(dist2))
  return chamfer_distance


def calculate_fscore(gt, pred, th=0.01):
  d1 = gt.compute_point_cloud_distance(pred)
  d2 = pred.compute_point_cloud_distance(gt)
  
  if len(d1) and len(d2):
    recall = float(sum(d < th for d in d2)) / float(len(d2))
    precision = float(sum(d < th for d in d1)) / float(len(d1))
    if recall+precision > 0:
      fscore = 2 * recall * precision / (recall + precision)
    else:
      fscore = 0
  else:
    fscore = 0
    precision = 0
    recall = 0
  return fscore, precision, recall


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


def batchify(*data, batch_size=1024, device="cpu", progress=True):
  assert all(sample is None or sample.shape[0] == data[0].shape[0] for sample in data), \
    "Sizes of tensors must match for dimension 0."

  def batchifier():
    size, batch_offset = data[0].shape[0], 0
    while batch_offset < size:
      batch_slice = slice(batch_offset, batch_offset + batch_size)
      yield [sample[batch_slice].to(device) if sample is not None else sample for sample in data]
      batch_offset += batch_size

  iterator = batchifier()
  if not progress:
      return iterator
  total = (data[0].shape[0] - 1) // batch_size + 1

  return tqdm(iterator, total=total)

def extract_mesh(pipeline):
  N = 300
  cube_side_len = 2.4
  t = np.linspace(-cube_side_len/2, cube_side_len/2, N+1)
  batch_size = 4096

  query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
  print(query_pts.shape)
  sh = query_pts.shape
  pts = torch.tensor(query_pts.reshape([-1,3]), dtype=torch.float32).to(extra_args['device'])
  dirs = torch.zeros_like(pts)
  dirs[:,2] = 1

  radiance_samples = []
  for (batch_pts, batch_dirs,) in batchify(pts, dirs, batch_size=batch_size, device='cuda'):
      with torch.no_grad():
          radiance_batch = pipeline.nef.rgba(batch_pts, batch_dirs)['density'].detach().cpu().numpy()
      radiance_samples.append(radiance_batch)
  all_radiances = np.concatenate(radiance_samples).reshape((N+1,N+1,N+1))

  import mcubes
  vertices, triangles = mcubes.marching_cubes(all_radiances, args.mcube_threshold)

  vertices = (vertices / N - 0.5) * cube_side_len
  vertices = vertices @ np.array([[0,-1,0], [0,0,-1], [1,0,0]]).T
  vertices = (vertices - translation_offset.numpy()) / translation_scale.numpy()

  mesh = o3d.geometry.TriangleMesh()
  mesh.vertices = o3d.utility.Vector3dVector(vertices)
  mesh.triangles = o3d.utility.Vector3iVector(triangles)

  return mesh

def get_pc(iter_id):
  model_dir = f'{args.workdir}/{args.exp_name}/rob0_{args.obj_name}_single_side_env_nobg_sam_iteration_{iter_id}/model_1'
  model_path = os.path.join(model_dir, list(sorted(os.listdir(model_dir)))[0], "model.pth")
  dataset_path = f'/home/saptarshi/dev/CustomComposer/workdir/{args.exp_name}/merged_images_0/'

  sys.argv[1:] = [
    '--config=../app/nerf/configs/nerf_hash.yaml',
    f'--pretrained={model_path}',
    f'--dataset-path={dataset_path}',
    '--valid-only'
  ]

  model_args, model_args_dict = parse_args()
  train_dataset, validation_dataset = load_dataset(args=model_args)
  model, pipeline = make_model(model_args, model_args_dict, extra_args, train_dataset, validation_dataset)

  num_pts = 500000
  pred_mesh = extract_mesh(pipeline)
  pred_mesh_fine = pred_mesh.subdivide_midpoint(number_of_iterations=1)
  pred_pcd = pred_mesh_fine.sample_points_uniformly(number_of_points=num_pts)
  pred_pcd, _ = pred_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

  query_pts = torch.tensor(np.array(pred_pcd.points))
  query_dirs = torch.tensor(np.zeros_like(query_pts))
  query_dirs[:,2] = 1
  batch_size = 4096

  color_samples = []
  for (batch_pts, batch_dirs,) in batchify(query_pts, query_dirs, batch_size=batch_size, device='cuda'):
      with torch.no_grad():
          color_batch = pipeline.nef.rgba(batch_pts, batch_dirs)['rgb'].detach().cpu().numpy()
      color_samples.append(color_batch)
  all_color_samples = np.concatenate(color_samples)
  pred_pcd.colors = o3d.utility.Vector3dVector(all_color_samples)

  o3d.visualization.draw_geometries([pred_pcd])
  breakpoint()

  return pred_pcd

if __name__ == '__main__':
  if args.iter != -1:
    pc = get_pc(args.iter)
    o3d.io.write_point_cloud(args.out_file, pc)
