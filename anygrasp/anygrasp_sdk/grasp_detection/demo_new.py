import os
import torch
import pickle
import shutil
import argparse
import numpy as np
import open3d as o3d
import traceback
from PIL import Image
import pickle
import pdb    
from sklearn.cluster import DBSCAN
from collections import Counter
from matplotlib import pyplot as plt

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

from scipy.spatial.transform import Rotation as R


def bp():
    pdb.set_trace()


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max-gripper-width', type=float, default=0.08, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper-height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top-down-grasp', action='store_true', help='Output top-down grasps')
parser.add_argument('--debug', action='store_true', help='Enable visualization', default=True)
parser.add_argument('--display', action='store_true', help='Display Cloud')
parser.add_argument('--data-dir', type=str, default='./example_data/', help='Path to data directory')

cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def quat_dist(q1 : np.ndarray, q2 : np.ndarray):
    return (q1 * q2).sum()

def quaternion_angular_distance(q1, q2):
    dot_product = np.dot(q1, q2)
    angular_distance = 2.0 * np.arccos(abs(dot_product))
    return angular_distance


def create_cube_wireframe(endpoint1, endpoint2):
    size = np.abs(endpoint2 - endpoint1)

    # Create the eight vertices of the cube
    vertices = np.array([
        endpoint1,
        endpoint1 + [size[0], 0, 0],
        endpoint1 + [0, size[1], 0],
        endpoint1 + [size[0], size[1], 0],
        endpoint1 + [0, 0, size[2]],
        endpoint1 + [size[0], 0, size[2]],
        endpoint1 + [0, size[1], size[2]],
        endpoint2
    ])

    # Define the edges of the cube by specifying vertex indices
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]

    # Create a LineSet object to represent the cube edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(edges)

    return line_set

def get_bounding_box(cloud):
    bb = cloud.get_axis_aligned_bounding_box()
    point_cloud = cloud.voxel_down_sample(voxel_size=0.005) 
    # eps = 0.05
    eps = 0.1
    min_samples = 10  
    data = np.asarray(point_cloud.points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = clustering.labels_
    cluster_counts = Counter(labels)
    largest_cluster_label = max(cluster_counts, key=cluster_counts.get)
    largest_cluster_points = data[labels == largest_cluster_label]
    min_bound = largest_cluster_points.min(axis=0)
    max_bound = largest_cluster_points.max(axis=0)

    return min_bound, max_bound

def dist_val(g):
    return quaternion_angular_distance(R.from_matrix(g.rotation_matrix).as_quat(), np.array([0.5, 0.5, 0.5, -0.5]))

def convex_comb(a,b,p):
    return (1-p)*a + p*b

def demo():
    data_dir = cfgs.data_dir
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # get data
    colors = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    # depths = pickle.load(open(os.path.join(data_dir, 'depth.pkl'), 'rb'))

    # get camera intrinsics
    # fx, fy = 927.17, 927.37
    # cx, cy = 651.32, 349.62
    # scale = 1000
    
    # from Isaac Sim
    # fx, fy = 823.4666442871094, 823.4666442871094
    # cx, cy = 400, 400
    # scale = 1000

    # from Kaolin-Wisp
    # fx, fy = 1286.66, 1286.66
    fx, fy = 514.66662, 514.66662
    cx, cy = 400, 400
    scale = 1000

    # set workspace
    # xmin, xmax = -0.19, 0.12
    # ymin, ymax = 0.02, 0.15
    # zmin, zmax = 0.0, 1.0

    xmin, xmax = -10, 10
    ymin, ymax = -10, 10
    zmin, zmax = -10, 10
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    
    angles = np.arctan(np.power(np.power(xmap - cx, 2) + np.power(ymap - cy, 2), 0.5)/fx)
    points_z = depths * np.cos(angles) / scale

    # points_z = depths  / scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # remove outlier
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    # get prediction
    try:
        gg, cloud = anygrasp.get_grasp(points, colors, lims)
        print(f'Total of {len(gg)} grasps detected')

        if len(gg) == 0:
            print('No Grasp detected after collision detection!')

        # bp()
        gg = gg.nms().sort_by_score()
        # gg = gg.sort_by_score()

        bb = get_bounding_box(cloud)
        print('bounding box vertices', bb[0], bb[1])
        lower_bound = convex_comb(bb[0], bb[1], -0.2)
        upper_bound = convex_comb(bb[0], bb[1], 1.2)

        gg, cloud = anygrasp.get_grasp(points, colors, [lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1], lower_bound[2], upper_bound[2]])
        all_grasps = [g for g in gg]
        
        print(f'After re grasping, we have {len(all_grasps)} grasps left')

        z_mid = (bb[1] + bb[0])[1] / 2 # Z axis in isaac sim is Y axis in anygrasp
        z_len = (bb[1] - bb[0])[1]

        # dist_check = lambda g: abs(quat_dist(R.from_matrix(g.rotation_matrix).as_quat(), 0.5*np.array([   1,1,1,-1]))) < 0.15

        # grasps = list(filter(lambda g: np.all(g.translation > convex_comb(bb[0], bb[1], -0.2)) and np.all(g.translation < convex_comb(bb[0], bb[1], 1.2)), grasps))
        grasps_prune1 = list(filter(lambda g: abs(g.translation[1] - z_mid) < 0.1 * z_len, all_grasps)) 
        grasps_prune2 = list(filter(lambda g : dist_val(g) < 0.7, grasps_prune1)) # 20 degrees #TODO need to generalize

        grasps = grasps_prune2

        print(f'After filtering, we have {len(grasps)} grasps left')

        
        # gg_pick = grasps[0:20]
        grasp_cosine_scores = np.array([np.cos(dist_val(g)) for g in grasps])
        gg_pick_scores = [g.score for g in grasps]
        # print('gg_pick scores', gg_pick_scores)
        # print('grasp cosine scores', grasp_cosine_scores)
        # print('avg of grasp scores', np.mean(gg_pick_scores))
        # print('sd of 20 grasp scores', np.std(gg_pick_scores))
        # print('sharpe', np.mean(gg_pick_scores)/np.std(gg_pick_scores))
        grasp_poses = []
        for g in grasps:
            cam_pose = pickle.load(open(os.path.join(data_dir, 'c2w.pkl'), 'rb'))
            c2_w = np.eye(4)
            c2_w[:3,-1] = g.translation
            c2_w[:3,:3] = g.rotation_matrix @ R.as_matrix(R.from_quat([0.5, 0.5, 0.5, 0.5]))
            c1_w = np.eye(4)
            c1_w[:3,-1] = c2_w[:3,-1] * np.array([1,-1,-1])
            c1_w[:3,:3] = (R.from_rotvec(R.as_rotvec(R.from_matrix(c2_w[:3,:3])) * np.array([1,-1,-1]))).as_matrix()
            grasp_pose = cam_pose @ c1_w
            grasp_poses.append(grasp_pose)
        
        # bp()

        try:
            os.remove(f'{cfgs.data_dir}/grasp_data.pkl')
            os.remove(f'{cfgs.data_dir}/grasp_scores.txt')
            os.remove(f'{cfgs.data_dir}/grasp_cosine_scores.txt')
        except:
            pass

        with open(f'{cfgs.data_dir}/grasp_data.pkl', 'wb') as f:
            pickle.dump([grasp_poses, gg_pick_scores, grasp_cosine_scores], f)
        with open(f'{cfgs.data_dir}/grasp_score.txt', 'w') as f:
            f.write(" ".join([str(x) for x in gg_pick_scores]))
        with open(f'{cfgs.data_dir}/grasp_cosine_scores.txt', 'w') as f:
            f.write(" ".join([str(x) for x in grasp_cosine_scores]))
    
    except Exception as e:
        print(traceback.format_exc())

        with open(f'{cfgs.data_dir}/grasp_data.pkl', 'wb') as f:
            pickle.dump([[], [], []], f)
        with open(f'{cfgs.data_dir}/grasp_score.txt', 'w') as f:
            f.write(" ".join([str(x) for x in []]))
        with open(f'{cfgs.data_dir}/grasp_cosine_scores.txt', 'w') as f:
            f.write(" ".join([str(x) for x in []]))



    
    # intrinsic = o3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1]])
    # rgbd_reproj = cloud.project_to_rgbd_image(640, 480, intrinsic, depth_scale=5000.0, depth_max=10.0)
    # plt.imshow(np.asarray(rgbd_reproj.color.to_legacy()))

    # visualization
    if cfgs.display:
        trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
        cloud.transform(trans_mat)
        grippers = list(map(lambda x : x.to_open3d_geometry(), grasps))
        for gripper in grippers:
            gripper.transform(trans_mat)

        cube_line_set = create_cube_wireframe(bb[0], bb[1])
        cube_line_set.transform(trans_mat)

        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        # # sphere.translate(gg_pick[0].translation)
        # sphere.translate(np.array([0,0,0.1]))
        # sphere.transform(trans_mat)

        s1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        s1.transform(trans_mat)
        s1.paint_uniform_color([1,1,1])

        s2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        s2.translate([0.1,0,0])
        s2.transform(trans_mat)
        s2.paint_uniform_color([1,0,0])

        s3 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        s3.translate([0,0.1,0])
        s3.transform(trans_mat)
        s3.paint_uniform_color([0,1,0])

        s4 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        s4.translate([0,0,0.1])
        s4.transform(trans_mat)
        s4.paint_uniform_color([0,0,1])


        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(cube_line_set)
        o3d.visualization.draw_geometries([cube_line_set, *list(map(lambda x : x.to_open3d_geometry().transform(trans_mat), all_grasps)), cloud])
        o3d.visualization.draw_geometries([cube_line_set, *list(map(lambda x : x.to_open3d_geometry().transform(trans_mat), grasps_prune1)), cloud])
        o3d.visualization.draw_geometries([cube_line_set, *list(map(lambda x : x.to_open3d_geometry().transform(trans_mat), grasps_prune2)), cloud])

        o3d.visualization.draw_geometries([cube_line_set, *grippers, cloud])

        # o3d.visualization.draw_geometries([sphere, cube_line_set, grippers[0], cloud])
        # o3d.visualization.draw_geometries([s1, s2, s3, s4, cube_line_set, grippers[0], cloud])
        if len(grippers) > 0:
            o3d.visualization.draw_geometries([cube_line_set, grippers[0], cloud])
        else:
            o3d.visualization.draw_geometries([cube_line_set, *grippers, cloud])



        # bp()

        # vis = o3d.visualization.Visualizer()
        # vis.create_window(visible=True) #works for me with False, on some systems needs to be true
        # vis.add_geometry(cloud)
        # vis.update_geometry(grippers[0])
        # vis.poll_events()
        # vis.update_renderer()
        # vis.capture_screen_image('abc.png')
        # vis.run()
        # vis.destroy_window()

        # o3d.visualization.draw_geometries([cloud])

        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(grippers[0])
        # vis.add_geometry(cloud)
        # img = vis.capture_screen_float_buffer(True)
        # vis.destroy_window()
        # import matplotlib.pyplot as plt
        # plt.imsave("test.png", np.asarray(img))

if __name__ == '__main__':
    demo()
