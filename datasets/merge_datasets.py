import os
import sys
import json
import shutil
from os.path import join as ospj

dir1 = sys.argv[1]
dir2 = sys.argv[2]

new_folder = sys.argv[3]
shutil.rmtree(new_folder, ignore_errors=True)

shutil.copytree(sys.argv[1], new_folder)
png1 = [i for i in os.listdir(dir1) if i.endswith('.png')]
json1 = [i for i in os.listdir(dir1) if i.endswith('.json')]
npy1 = [i for i in os.listdir(dir1) if i.endswith('.npy')]

png2 = sorted([i for i in os.listdir(dir2) if i.endswith('.png')], key = lambda x: int(x.split('.')[0].split('_')[-1])) 
json2 = sorted([i for i in os.listdir(dir2) if i.endswith('.json')], key = lambda x: int(x.split('.')[0].split('_')[-1]))
npy2 = sorted([i for i in os.listdir(dir2) if i.endswith('.npy')], key = lambda x: int(x.split('.')[0].split('_')[-1]))

num_dp1 = len(png1)
num_dp2 = len(png2)
depth_available = (len(npy2) > 0)

for i in range(num_dp2):
    num_new = i + num_dp1
    x = len(str(num_new))
    png_name = png2[i]
    json_name = json2[i]

    new_png_name = f"{png_name.split('.')[0][:-x]}{num_new}.png"
    new_json_name = f"{json_name.split('.')[0][:-x]}{num_new}.json"

    shutil.copyfile(ospj(dir2, png_name), ospj(new_folder, new_png_name))
    shutil.copyfile(ospj(dir2, json_name), ospj(new_folder, new_json_name))

    if depth_available:
        npy_name = npy2[i]
        new_npy_name = f"{npy_name.split('.')[0][:-x]}{num_new}.npy"
        shutil.copyfile(ospj(dir2, npy_name), ospj(new_folder, new_npy_name))
