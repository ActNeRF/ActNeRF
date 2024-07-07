import os
import sys
import json
import shutil
from os.path import join as ospj

json1 = json.load(open(ospj(sys.argv[1], 'transforms_train.json')))
json2 = json.load(open(ospj(sys.argv[2], 'transforms_train.json')))

new_folder = sys.argv[3]
shutil.rmtree(new_folder, ignore_errors=True)
shutil.copytree(sys.argv[1], new_folder)
# os.makedirs(f'{new_folder}/train')

identifier2 = sys.argv[4] # unique identifier to add to file names from json2 to mark them different from json1
new_json = json1
new_frames = json1["frames"]

for frame in json2["frames"]:
    im_path_rel = frame['file_path']
    im_path_abs = ospj(sys.argv[2], im_path_rel) + '.png'
    im_name = im_path_rel.split('/')[-1]

    new_im_path_rel = f'train/{im_name}_{identifier2}'
    new_im_path_abs = ospj(new_folder, new_im_path_rel) + '.png'

    shutil.copyfile(im_path_abs, new_im_path_abs)
    frame['file_path'] = new_im_path_rel
    new_frames.append(frame)

new_json['frames'] = new_frames
json.dump(new_json, open(ospj(new_folder, 'transforms_train.json'), 'w'))
