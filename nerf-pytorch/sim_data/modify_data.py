import os
import sys
import json
import shutil
import argparse
import numpy as np

# argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input-dir', type=str, required=True)
parser.add_argument('-o','--output-dir', type=str, required=True)
parser.add_argument('-d','--dataset-type', type=str, required=True)
parser.add_argument('--offset-y', type=float, default=-2.88, help='offset-y')
args = parser.parse_args()

input_dir, output_dir, dataset_type = args.input_dir, args.output_dir, args.dataset_type
os.makedirs(f'{output_dir}/{dataset_type}', exist_ok=True)

output_dict = {
  'camera_angle_x': None,
  'scale': 6,
  'offset': [
      0,
      args.offset_y,
      -0.72
  ],
  'frames' : []
}
depth = False
depth_files = [i for i in os.listdir(input_dir) if i.endswith('.npy')]
if len(depth_files) > 0:
  depth = True
for file in sorted(os.listdir(input_dir)):
  if file.endswith(".json"):
    fname = file.split('.')[0].split('_')[-1]

    params_path = f'{input_dir}/{file}'
    raw_image_path = f'{input_dir}/rgb_{fname}.png'
    final_image_path = f'{output_dir}/{dataset_type}/rgb_{fname}'

    raw_depth_path = f'{input_dir}/distance_to_camera_{fname}.npy'
    final_depth_path = f'{output_dir}/{dataset_type}/distance_to_camera_{fname}'

    shutil.copyfile(raw_image_path, f'{final_image_path}.png')
    if depth:
      shutil.copyfile(raw_depth_path, f'{final_depth_path}.npy')

    with open(params_path, 'r') as f:
      camera_props = json.load(f)
      focal = camera_props['cameraFocalLength']
      W, H = camera_props['renderProductResolution']
      horiz_aperture = camera_props['cameraAperture'][0]
      # horiz_aperture = camera_props['cameraAperture']
      camera_angle_x = 2 * np.arctan(horiz_aperture / (2 * focal))

      w2c = np.array(camera_props['cameraViewTransform']).reshape((4,4))
      # c2w = w2c
      # c2w = np.linalg.inv(w2c)
      c2w = np.linalg.inv(w2c.T)
      c2w = [row.tolist() for row in c2w]

      output_dict['camera_angle_x'] = camera_angle_x
      frame = {
        'file_path': f'{dataset_type}/rgb_{fname}', 
        'transform_matrix': c2w
      }
      if depth:
        frame['depth_path'] = f'{dataset_type}/distance_to_camera_{fname}'
      output_dict['frames'].append(frame)

with open(f'{output_dir}/transforms_{dataset_type}.json', 'w') as f:
  json.dump(output_dict, f)


