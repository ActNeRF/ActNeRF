import open3d as o3d
import numpy as np
from typing import Tuple
gt_pcd = o3d.io.read_point_cloud("../notebooks_output/pc/point_cloud.pcd")
pred_pcd = o3d.io.read_point_cloud("../notebooks_output/pc/pred_point_cloud.pcd")
# o3d.visualization.draw_geometries([gt_pcd])
o3d.visualization.draw_geometries([pred_pcd])



# surface_pcd = o3d.io.read_point_cloud("../notebooks_output/surface_point_cloud.pcd")
# surface_mesh = o3d.io.read_triangle_mesh("../notebooks_output/mesh.ply")

# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries([surface_pcd])
# o3d.visualization.draw_geometries([surface_mesh])

# # create 3D surface mesh from point cloud
# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
# mesh, densities : Tuple[o3d.geometry.TriangleMesh, o3d.utility.DoubleVector] = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
# voxel_size = 0.005  # Adjust this based on your desired point density
# mesh = mesh.voxel_down_sample(voxel_size=voxel_size)
# filled_pcd = mesh.sample_points_uniformly(number_of_points=50000)  # Adjust the number of points as needed

# # visualize
# o3d.visualization.draw_geometries([filled_pcd])

# draw coordinate system
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.06, origin=[0, 0, 0])

def create_coordinate_grid(step_size = 0.1, min_bound=[-1,-1,-1], max_bound=[1, 1, 1]):
  x_grid = np.arange(min_bound[0], max_bound[0] + step_size, step_size)
  y_grid = np.arange(min_bound[1], max_bound[1] + step_size, step_size)
  z_grid = np.arange(min_bound[2], max_bound[2] + step_size, step_size)

  # Create a mesh for the grid
  mesh = o3d.geometry.LineSet()

  # Add lines for the coordinate grid
  for x in x_grid:
      lines = [[min_bound[0], x, min_bound[2]], [max_bound[0], x, min_bound[2]]]
      mesh += o3d.geometry.LineSet(
          points=o3d.utility.Vector3dVector(lines),
          lines=o3d.utility.Vector2iVector([[0, 1]])
      )

  for y in y_grid:
      lines = [[y, min_bound[1], min_bound[2]], [y, max_bound[1], min_bound[2]]]
      mesh += o3d.geometry.LineSet(
          points=o3d.utility.Vector3dVector(lines),
          lines=o3d.utility.Vector2iVector([[0, 1]])
      )

  for z in z_grid:
      lines = [[min_bound[0], min_bound[1], z], [max_bound[0], min_bound[1], z]]
      mesh += o3d.geometry.LineSet(
          points=o3d.utility.Vector3dVector(lines),
          lines=o3d.utility.Vector2iVector([[0, 1]])
      )
  return mesh

line_mesh = create_coordinate_grid()

pts = [
    [0.00290925,0.68113462,0.22875828],
    [0.07545706,0.61358093,0.1852904 ],
    [-0.01510986, 0.67424907, 0.24352436],
    [-0.08457973, 0.62132285, 0.19344727],
]
#create sphere at center c and radius r
# draw_list = [volumetric_pcd]
for idx, pt in enumerate(pts):
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
    mesh_sphere.translate(pt)
    mesh_sphere.paint_uniform_color(np.random.rand(3))
    # draw_list.append(mesh_sphere)
    # write the coordinates at the sphere
    # text = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02, origin=idx)
    # draw_list.append(text)

# o3d.visualization.draw_geometries([pcd, line_mesh, mesh_frame])
# o3d.visualization.draw_geometries([pcd])
# o3d.visualization.draw_geometries(draw_list)
  
