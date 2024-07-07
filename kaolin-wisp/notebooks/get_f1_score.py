# %%
import os
import sys
import math
import torch
import pickle
import pyassimp
import numpy as np
import open3d as o3d 

from enum import Enum
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation as R

# %%
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
obj_pos_map = {
    Objects.CHEEZIT:(0, -0.66887 + 0.6, -0.033828),
    Objects.RUBIK:  (0, -0.66887 + 0.6, -0.069),
    Objects.SPAM: (0, -0.66887 + 0.6, -0.033828 + (0.07121-0.13633)),
    Objects.SMUG: (0, -0.66887 + 0.6, -0.033828),
    Objects.BASKET: (0, -0.66887 + 0.6, -0.033828),
}

obj_orient_map = {
    Objects.CHEEZIT:[math.pi/2, math.pi, 0],
    Objects.RUBIK:  [0, 0, math.pi],
    Objects.SPAM: [-math.pi/2, 0, math.pi],
    Objects.SMUG: [-math.pi/2,math.pi, -math.pi/4],
    Objects.BASKET: [math.pi, 0, 0],
}

obj_scale_map = {
    Objects.CHEEZIT:(1, 1, 1),
    Objects.RUBIK:  (1.1, 1.1, 1.1),
    Objects.SPAM: (2, 2, 1.2),
    Objects.SMUG: (1, 3, 1),
    Objects.BASKET: (0.8, 0.25, 0.7),
}

robo_pos = np.array([-0.00032, -0.66887, 0.6 - 7.70155/10])

# %%
def search_in_dict(d, val):
  for key, value in d.items():
    if value == val:
      return key


def usd_to_pointcloud(file_path):
  stage = Usd.Stage.Open(file_path)
  default_prim = stage.GetDefaultPrim()
  mesh = UsdGeom.Mesh(default_prim)
  points_attr = mesh.GetPointsAttr()
  points = points_attr.Get()
  points_array = np.array(points)
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(points_array)
  return pcd


def load_obj_to_numpy(file_path):
  scene = pyassimp.load(file_path)
  if scene is None:
    raise Exception("Error loading OBJ file:", file_path)
  mesh = scene.meshes[0]  # Assuming there's only one mesh in the scene
  vertices = np.array(mesh.vertices)
  normals = np.array(mesh.normals)
  faces = np.array(mesh.faces)
  points_with_normals = np.hstack((vertices, normals))
  return points_with_normals, faces


def compute_chamfer_distance(pcd1, pcd2):
  dist1 = pcd1.compute_point_cloud_distance(pcd2)
  dist2 = pcd2.compute_point_cloud_distance(pcd1)
  chamfer_distance = (np.asarray(dist1).sum() + np.asarray(dist2).sum()) / (len(dist1) + len(dist2))
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


# %%
obj_name = 'cheezit'
obj_type = search_in_dict(out_name_map, obj_name)

# usd_path = f'/home/saptarshi/Downloads/Dataset/{obj_name}.usd'
obj_path = f'/home/saptarshi/Downloads/Dataset_obj/{obj_name}.obj'

num_pts = 500000

pred_pts, pred_vertices = pickle.load(open('logs/examples/cheezit.pkl', 'rb'))
pred_mesh = o3d.geometry.TriangleMesh()
pred_mesh.vertices = o3d.utility.Vector3dVector(pred_pts)
pred_mesh.triangles = o3d.utility.Vector3iVector(pred_vertices)
pred_mesh_fine = pred_mesh.subdivide_midpoint(number_of_iterations=1)
pred_pcd = pred_mesh_fine.sample_points_uniformly(number_of_points=num_pts)
pred_pcd, _ = pred_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

gt_mesh = o3d.io.read_triangle_mesh(obj_path)
gt_mesh_fine = gt_mesh.subdivide_midpoint(number_of_iterations=3)
gt_pcd = gt_mesh_fine.sample_points_uniformly(number_of_points=num_pts)

gt_scale = obj_scale_map[obj_type]
gt_pos = obj_pos_map[obj_type] - robo_pos
gt_rot = obj_orient_map[obj_type]

gt_pcd.points = o3d.utility.Vector3dVector(np.asarray(gt_pcd.points) * gt_scale)
obj_act_pose_path = '/home/saptarshi/dev/CustomComposer/workdir/cheezit_5_2_val4_noflip_flip_21_new_2/pose_actual/iter_0.pkl'
obj_act_pose = pickle.load(open(obj_act_pose_path, 'rb'))
gt_pcd.transform(obj_act_pose)

# coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.array([0., 0.6, 0.15]))
# o3d.visualization.draw_geometries([coordinate_frame, gt_pcd])
# o3d.visualization.draw_geometries([coordinate_frame, pred_pcd])
# gt_pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]] * len(gt_pcd.points)))
# pred_pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * len(pred_pcd.points)))
# o3d.visualization.draw_geometries([coordinate_frame, gt_pcd.voxel_down_sample(0.01), pred_pcd.voxel_down_sample(0.01)])

ret = {
  'chamfer' : compute_chamfer_distance(pred_pcd, gt_pcd),
  'fscore' : calculate_fscore(gt_pcd, pred_pcd, 0.03)
}

print(ret)
