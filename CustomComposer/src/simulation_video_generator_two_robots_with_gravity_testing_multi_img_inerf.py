import os
import sys
import cv2
import pdb
import math
import time
import json
import pickle
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from enum import Enum
from typing import Dict
from scipy.spatial.transform import Rotation as R

def bp():
    pdb.set_trace()

from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument('-dd','--data-dir', type=str)
parser.add_argument('-ot','--obj-type', type=str)
parser.add_argument('-ooe','--nerf2ws-expected-path', type=str)
parser.add_argument('-ooa','--object-pose-actual-path', type=str)
parser.add_argument('--dont-flip', action='store_true', default=False)
parser.add_argument('--stream', action='store_true')
parser.add_argument('-cv', '--create-video', action='store_true')

args = parser.parse_args()


hmode = True
stream = args.stream
pause = False

headless = hmode
view = not hmode

CONFIG = {
    "renderer": "RayTracedLightingsaac/Props/Mounts/table.usd'", 
    "headless": headless, 
    "width": 800, 
    "height": 800, 
}

simulation_app = SimulationApp(launch_config=CONFIG)
from utils import *

import omni
import carb
import omni.usd
import omni.ui as ui
import omni.replicator.core as rep
import omni.kit.commands as commands
import omni.isaac.core.utils.numpy.rotations as rot_utils


from omni.isaac.core import World
from omni.isaac.core.utils import prims
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import get_current_stage, open_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, lookat_to_quatf
from omni.isaac.core.utils.bounds import compute_combined_aabb, create_bbox_cache
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.prims import RigidPrim


from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.sensor import Camera
from omni.isaac.core.utils import prims
from omni.isaac.core.tasks import BaseTask
from omni.isaac.franka import KinematicsSolver
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects.cuboid import VisualCuboid
from omni.isaac.core.utils.stage import get_stage_units
from pxr import UsdGeom, Usd, Gf, UsdPhysics, PhysxSchema
from omni.isaac.synthetic_utils import SyntheticDataHelper
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.numpy.rotations import rot_matrices_to_quats
from omni.isaac.franka import KinematicsSolver as FrankaKinematicsSolver
from omni.isaac.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles, lookat_to_quatf
from omni.kit.viewport.utility import get_active_viewport, create_viewport_window, get_num_viewports, get_viewport_from_window_name

from flipping_controller import FlipController

import logging
import carb

sd_helper = SyntheticDataHelper()

logging.getLogger("omni.hydra").setLevel(logging.ERROR)
logging.getLogger("omni.isaac.urdf").setLevel(logging.ERROR)
logging.getLogger("omni.physx.plugin").setLevel(logging.ERROR)

l = carb.logging.LEVEL_ERROR
carb.settings.get_settings().set("/log/level", l)
carb.settings.get_settings().set("/log/fileLogLevel", l)
carb.settings.get_settings().set("/log/outputStreamLevel", l)

if stream:
    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/livestream/proto", "ws")
    simulation_app.set_setting("/app/livestream/websocket/framerate_limit", 120)
    simulation_app.set_setting("/ngx/enabled", False)
    enable_extension("omni.kit.livestream.native")


franka_name = "fancy_franka"
franka_new_name = "fancy_franka_new"

SCOPE_NAME = "/Tabletop"
ENV_URL = "/Isaac/Environments/Grid/default_environment.usd"
ROOM_URL = '/Isaac/Environments/Simple_Room/simple_room.usd'
RUBIK_CUBE_URL = '/Isaac/Props/Rubiks_Cube/rubiks_cube.usd'
CUBE_URL = '/Isaac/Props/Blocks/basic_block.usd'
FORKLIFT_URL = "/Isaac/Props/Forklift/forklift.usd"
WOOD_URL = '/NVIDIA/Materials/vMaterials_2/Wood/Wood_Tiles_Poplar.mdl'

TABLE_URL = '/Isaac/Environments/Simple_Room/Props/table_low.usd'
TABLE_TEX = '/Isaac/Environments/Simple_Room/Materials/MI_Table.mdl'

FLOOR_URL = '/Isaac/Environments/Simple_Room/Props/Towel_Room01_floor_bottom.usd'
FLOOR_TEX = '/Isaac/Environments/Simple_Room/Materials/MI_Parquet_Floor.mdl'

FRANKA_URL = '/Isaac/Robots/FactoryFranka/factory_franka.usd'

CHEEZEIT_URL = '/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd'
CAN_URL = '/Isaac/Props/YCB/Axis_Aligned/002_master_chef_can.usd'
MUG_URL = '/Isaac/Props/YCB/Axis_Aligned/019_pitcher_base.usd'
BANANA_URL = '/Isaac/Props/YCB/Axis_Aligned/011_banana.usd'
CRATE_URL = '/Isaac/Environments/Simple_Warehouse/Props/SM_CratePlasticNote_A_01.usd'
BLOCK_URL = '/Isaac/Props/Blocks/MultiColorCube/multi_color_cube_instanceable.usd'
SPAM_URL = '/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd'
MUSTARD_URL = '/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd'
GELATIN_URL = '/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd'
SMUG_URL = '/Isaac/Props/YCB/Axis_Aligned/025_mug.usd'
WOODEN_BOX_URL = '/Isaac/Props/YCB/Axis_Aligned/036_wood_block.usd'
RUBIK_CUBE_URL = '/home/saptarshi/Downloads/rubik_final.usd'

IITD_TABLE_URL = '/home/saptarshi/Downloads/IITD_FRAME_6060_v2/IITD_TABLE_MODIFIED.usd'
HOLDER_URL = '/home/saptarshi/Downloads/IITD_FRAME_6060_v2/GRIPPER_DUMMY_2v7v2.usdc'
FRAME_URL = '/home/saptarshi/Downloads/IITD_FRAME_6060_v2/IITD_FRAME_6060_v2.usdc'
BASKET_URL = '/home/saptarshi/Downloads/basket_final.usd'

NEW_OBJECT_URL = '/home/saptarshi/Downloads/IITD_FRAME_6060_v2/model2.usd'

WORKSPACE_CENTER = np.array([0, 0.6, 0.15])


class Objects(Enum):
    CHEEZIT = 1
    CRATE = 2
    BLOCK = 3
    RUBIK = 4
    CAN = 5
    BANANA = 6
    MUG = 7
    SPAM = 8
    MUSTARD = 9
    GELATIN = 10
    SMUG = 11
    BASKET = 12

obj_prim_path_map = {
    Objects.CHEEZIT: 'Cheezit',
    Objects.CRATE: 'Crate',
    Objects.BLOCK: 'Block',
    Objects.RUBIK: 'RCube',
    Objects.CAN: 'CAN',
    Objects.BANANA: 'BANANA',
    Objects.MUG: 'MUG',
    Objects.SPAM: 'SPAM',
    Objects.MUSTARD: 'MUSTARD',
    Objects.GELATIN: 'GELATIN',
    Objects.SMUG: 'SMUG',
    Objects.BASKET: 'Basket',
}

out_name_map = {
    Objects.CHEEZIT: 'cheezit',
    Objects.CRATE: 'crate',
    Objects.BLOCK: 'block',
    Objects.RUBIK: 'rubik',
    Objects.CAN: 'can',
    Objects.BANANA: 'banana',
    Objects.MUG: 'mug',
    Objects.SPAM: 'spam',
    Objects.MUSTARD: 'mustard',
    Objects.GELATIN: 'gelatin',
    Objects.SMUG: 'small_mug',
    Objects.BASKET: 'basket',

}
obj_pos_map = {
    Objects.CRATE:  (0, -0.66887 + 0.6, -0.13973),
    Objects.BLOCK:  (0, -0.66887 + 0.6, (0.25 + 6.7 - 7.70155)/10),
    Objects.CHEEZIT:(0, -0.66887 + 0.6, -0.033828),
    Objects.RUBIK:  (0, -0.66887 + 0.6, -0.069),
    Objects.CAN: (0, -0.66887 + 0.6, -0.033828 + (0.1 - 0.13633)),
    Objects.BANANA:  (0, -0.66887 + 0.6, -0.069 + (0.0676 -0.10115)),
    Objects.MUG: (0, -0.66887 + 0.6, -0.033828 + (0.151-0.13633)),
    Objects.SPAM: (0, -0.66887 + 0.6, -0.033828 + (0.07121-0.13633)),
    Objects.MUSTARD: (0, -0.66887 + 0.6, -0.033828 + (0.125-0.13633)),
    Objects.GELATIN: (0, -0.66887 + 0.6, -0.033828 + (0.1-0.13633)),
    Objects.SMUG: (0, -0.66887 + 0.6, -0.033828),
    Objects.BASKET: (0, -0.66887 + 0.6, -0.033828),
}
train_views = False
test_dim = 2
# output_dir = f'~/dev/datasets/{out_name_map[obj_type]}_single_side_down_{"train" if train_views else "test"}_2d'

robo_pos = np.array([-0.00032, -0.66887, 0.6 - 7.70155/10])


def add_position_offsets(pos):
    robot_origin_offset = -robo_pos
    return np.array(pos) + robot_origin_offset


def prefix_with_isaac_asset_server(relative_path):
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise Exception(
            "Nucleus server not found, could not access Isaac Sim assets folder")
    path = assets_root_path + relative_path
    print('Returning path: ', path)
    return path


def place_lights():
    lights = rep.create.light(light_type="distant")
    light1 = rep.create.light(
        light_type="Sphere",
        intensity=50000,
        position=(-15, 15, 20),
        scale=1,
    )
    with rep.create.group([light1]):
        rep.modify.attribute("radius", 5)

def create_vid(imgs, path, fps=10):
    print(path)
    import cv2
    W,H = imgs[1].shape[1], imgs[1].shape[0]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (W,H))
    for i in range(1,len(imgs)):
        img = imgs[i]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)
    out.release()

def search_in_dict(d, val):
    for key, value in d.items():
        if value == val:
            return key

class FrankaPlaying(BaseTask):
    def __init__(
            self, 
            name: str, 
            object_pose_actual : np.ndarray,
            nerf2ws_expected : np.ndarray,
            camera_pose_path: str, 
            grasp_pose_path: str, 
            # final_pose_path : str,
            output_dir: str, 
            obj_type: str = "cheezit",
            dont_flip: bool = False
        ):

        super().__init__(name=name, offset=None)

        self.goal_poses = []
        self.object_pose_actual = object_pose_actual
        self.nerf2ws_expected = nerf2ws_expected
        self.dont_flip = dont_flip
        
        self.task_state = 0
        self.img_state = 0
        self.target_reaching_failed = False
        self.reached_points = []
        self.failed_points = []
        self.running_time = 0
        self.running_time_limit = 1000
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.obj_type = search_in_dict(out_name_map, obj_type)
        self.prev_posdif = 0
        self.same_dif_count = 0
        self.best_dif = np.inf
        self.best_dif_time = 0

        self.cam1_images = []
        self.cam2_images = []
        self.main_view_images = []
        # if args.create_video:
        #     path = f'{self.output_dir}/main_view.avi'
        #     self.video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), 10, (800, 800))

        c2w_cam = pickle.load(open(camera_pose_path, 'rb'))
        c2w_grasp = pickle.load(open(grasp_pose_path, 'rb'))
        # c2w_final = pickle.load(open(final_pose_path, 'rb'))

        c2w_cam = c2w_nerf2world(c2w_cam, self.nerf2ws_expected)
        c2w_grasp = c2w_nerf2world(c2w_grasp, self.nerf2ws_expected)
        # c2w_final = c2w_nerf2world(c2w_final, self.nerf2ws_expected)

        # c2w_grasp[1,3] += 0.02
        if self.dont_flip:
            self.task_state = 2

        self.setup_goals(c2w_cam, c2w_grasp)

    def setup_goals(self, c2w_cam, c2w_grasp):
        init_mat = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        source_pts = [
            [0.2, 0.45, 0.15],
            [0, 0.45, 0.3],
            [-0.2, 0.45, 0.15],
            [0.2, 0.75, 0.15],
            [0, 0.75, 0.3],
            [-0.2, 0.75, 0.15],
        ]
        if self.obj_type == Objects.SMUG:
            source_pts = [
            [0.2, 0.45, 0.15],
            [0, 0.45, 0.3],
            [-0.2, 0.45, 0.15],
            [-0.2, 0.75, 0.15],
            [0, 0.45, 0.3],
            [0, 0.45, 0.25],
        ]


        self.goal_poses = []
        
        c2w_ee = c2w_cam.copy()
        dir_look = c2w_cam[:3, :3] @ np.array([0, 0, -1])
        target_offset = 0.11
        c2w_ee[:3, -1] = c2w_cam[:3, -1] - target_offset * dir_look
        c2w_ee[:3, :3] = (R.from_matrix(c2w_cam[:3, :3]) * R.from_quat([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]) * R.inv(R.from_matrix(init_mat))).as_matrix()
        base_position = c2w_ee[:3,-1]
        base_rotation = R.from_matrix(c2w_ee[:3,:3])
        rob = int(base_position[1] > 0.6)
        self.goal_poses = [[
            base_position, 
            base_rotation.as_quat()[[3, 0, 1, 2]],  
            rob
        ]]

        target_offset = 0.06
        if self.obj_type == Objects.SMUG:
            target_offset = 0.09
        if self.obj_type == Objects.RUBIK:
            dir_look = c2w_grasp[:3, :3] @ np.array([0, 0, -1])
            angle = -np.arcsin(dir_look[2]) * (180/np.pi)
            print(angle)
            # bp()
            if angle < 12.5:
                diff = (12.5 - angle) * np.pi/180
                c2w_grasp[:3, :3] = R.as_matrix(R.from_quat([0, np.sin(-diff/2), 0, np.cos(-diff/2)])) @ c2w_grasp[:3, :3]
        c2w_ee = c2w_grasp.copy()
        dir_look = c2w_grasp[:3, :3] @ np.array([0, 0, -1])
        self.dir_look = dir_look
        c2w_ee[:3, -1] = c2w_grasp[:3, -1] - target_offset * dir_look
        c2w_ee[:3, :3] = R.as_matrix(R.inv( R.from_matrix(init_mat) *  R.inv(R.from_matrix(c2w_grasp[:3, :3]) * R.from_quat([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]))))
        base_position = c2w_ee[:3,-1]
        base_rotation = R.from_matrix(c2w_ee[:3,:3])
        self.goal_poses += [[
            base_position,
            base_rotation.as_quat()[[3, 0, 1, 2]],  
            rob
        ]]

        # for i, pt in enumerate(source_pts):
        #     init_c2w = get_c2w_from_pos(np.array(pt), WORKSPACE_CENTER)
        #     c2w_final = init_c2w.copy()
        #     # c2w_final[:3,:3] = random_rotation_matrix(5 * np.pi/180) @ c2w_final[:3,:3]
        #     # c2w_final[:3,-1] += np.random.uniform(-0.05,0.05,3) 
        #     if not args.create_video:
        #         pickle.dump(c2w_final, open(f"{args.data_dir}/c2w_final{i}.pkl", 'wb'))

        #     c2w_ee = c2w_final.copy()
        #     dir_look = c2w_final[:3, :3] @ np.array([0, 0, -1])
        #     target_offset = 0.11
        #     c2w_ee[:3, -1] = c2w_final[:3, -1] - target_offset * dir_look
        #     c2w_ee[:3, :3] = (R.from_matrix(c2w_final[:3, :3]) * R.from_quat([0, 0, 1/np.sqrt(2), 1/np.sqrt(2)]) * R.inv(R.from_matrix(init_mat))).as_matrix()
        #     base_position = c2w_ee[:3,-1]
        #     base_rotation = R.from_matrix(c2w_ee[:3,:3])
        #     rob = int(base_position[1] > 0.6)
        #     self.goal_poses.append([
        #         base_position, 
        #         base_rotation.as_quat()[[3, 0, 1, 2]],  
        #         rob
        #     ])


        print('goals')
        for goal in self.goal_poses:
            print(goal)
        # bp()

    def is_done(self):
        return self.task_state >= len(self.goal_poses)

    def set_up_scene(self, scene):
        obj_type = self.obj_type
        obj_prim_name = f'{SCOPE_NAME}/{obj_prim_path_map[obj_type]}'
        obj_pos_offset = obj_pos_map[obj_type]
        
        robot_current_pos = add_position_offsets(robo_pos)
        robot_new_current_pos = robot_current_pos.copy()
        robot_new_current_pos[1] += 1.15

        # -0.98629
        super().set_up_scene(scene)
        room1_prim_path = f"{SCOPE_NAME}/Room1"
        add_reference_to_stage(prefix_with_isaac_asset_server(ROOM_URL), room1_prim_path)
        room1 = RigidPrim(prim_path=room1_prim_path)
        room1.disable_rigid_body_physics()

        room2_prim_path = f"{SCOPE_NAME}/Room2"
        add_reference_to_stage(prefix_with_isaac_asset_server(ROOM_URL), room2_prim_path)
        room2 = RigidPrim(
          prim_path=room2_prim_path,
          orientation=euler_angles_to_quat([0, 0, math.pi])
        )
        room2.disable_rigid_body_physics()
        
        prims.get_prim_at_path(
            f"{SCOPE_NAME}/Room1/table_low_327").GetAttribute("xformOp:translate").Set((20, 20, 20))
        prims.get_prim_at_path(
            f"{SCOPE_NAME}/Room2/table_low_327").GetAttribute("xformOp:translate").Set((20, 20, 20))

        main_viewport = omni.ui.Workspace.get_window("Viewport")
        create_viewport_window('View2', visible=True, docked=True, width=CONFIG['width'], height=CONFIG['height'])
        viewport_window = get_viewport_from_window_name('View2')
        new_viewport = omni.ui.Workspace.get_window("View2")
        new_viewport.dock_in(main_viewport, omni.ui.DockPosition.RIGHT)
        viewport_window.set_active_camera(f"{SCOPE_NAME}/Franka1/panda_hand/geometry/camera")

        place_lights()

        # scene.add_default_ground_plane()
        self._franka1 = scene.add(Franka(
            prim_path=f"{SCOPE_NAME}/Franka1",
            name=franka_name,
            position=np.array(robot_current_pos),
            orientation=euler_angles_to_quat([0, 0, math.pi/2]),
            end_effector_prim_name='panda_hand'
        ))

        self._franka2 = scene.add(Franka(
            prim_path=f"{SCOPE_NAME}/Franka2",
            name=franka_new_name,
            # position=np.array(robot_new_current_pos + np.array([5, 0, 0])),
            position=np.array(robot_new_current_pos),
            orientation=euler_angles_to_quat([0, 0, -math.pi/2]),
            end_effector_prim_name='panda_hand'
        ))

        self.robo_origin = robot_current_pos
        self.camera1 = scene.add(
            Camera(
                prim_path=f"{SCOPE_NAME}/Franka1/panda_hand/geometry/camera",
                name="camera",
                translation=np.array([0, 0, 0.11]),
            )
        )
        self.camera2 = scene.add(
            Camera(
                prim_path=f"{SCOPE_NAME}/Franka2/panda_hand/geometry/camera",
                name="camera_new",
                translation=np.array([0, 0, 0.11]),
            )
        )

        self.camera3 = scene.add(
            Camera(
                prim_path=f"{SCOPE_NAME}/Camera3",
                name="camera3",
                # translation=np.array([4, 0.6, 2]),
                # orientation=euler_angles_to_quat([math.pi, 160*math.pi/180, 0])
                translation=np.array([-1.25, -0.49, 0.54]),
                orientation=euler_angles_to_quat(math.pi/180*np.array([67.9, -45.68, -16.19]))
            )
        )
   
        focal = 193
        aper = 2*np.tan(np.pi/180 * 65.5/2)*focal
        aper = 300
        # aper = 24.8

        self.camera1.prim.GetAttribute("focalLength").Set(focal)
        self.camera1.prim.GetAttribute("focusDistance").Set(100)
        self.camera1.prim.GetAttribute("horizontalAperture").Set(aper)
        self.camera1.prim.GetAttribute("verticalAperture").Set(aper)
        self.camera1.prim.GetAttribute(
            "xformOp:orient").Set(Gf.Quatd(0, 1/(2 ** 0.5), 1/(2 ** 0.5), 0))
        self.camera1.prim.GetAttribute(
            "clippingRange").Set(Gf.Vec2f(0.01, 10000))

        self.camera2.prim.GetAttribute("focalLength").Set(focal)
        self.camera2.prim.GetAttribute("focusDistance").Set(100)
        self.camera2.prim.GetAttribute("horizontalAperture").Set(aper)
        self.camera2.prim.GetAttribute("verticalAperture").Set(aper)
        self.camera2.prim.GetAttribute(
            "xformOp:orient").Set(Gf.Quatd(0, 1/(2 ** 0.5), 1/(2 ** 0.5), 0))
        self.camera2.prim.GetAttribute(
            "clippingRange").Set(Gf.Vec2f(0.01, 10000))


        # prims.create_prim(
        #     prim_path=f"{SCOPE_NAME}/Table",
        #     position=(0, 0, 0),
        #     scale=(1,1,1),
        #     usd_path=prefix_with_isaac_asset_server(TABLE_URL),
        # )

        table_prim = prims.create_prim(
            prim_path=f"{SCOPE_NAME}/RobTable",
            position=add_position_offsets((0, 0, 0.102 - 0.770155)),
            scale=(1/1000, 1/1000, 1/1000),
            orientation=euler_angles_to_quat([math.pi, 0, 0]),
            usd_path=IITD_TABLE_URL,
        )
        UsdPhysics.CollisionAPI.Apply(table_prim)

        prims.create_prim(
            prim_path=f"{SCOPE_NAME}/Holder",
            position=add_position_offsets((0, 0, 0-0.770155)),
            scale=(1/1000, 1/1000, 1/1000),
            orientation=euler_angles_to_quat([0, 0, -math.pi/2]),
            usd_path=HOLDER_URL,
        )

        prims.create_prim(
            prim_path=f"{SCOPE_NAME}/Frame",
            position=add_position_offsets(
                (-0.25132, -0.65774, -0.212883-0.770155)),
            scale=(1/1000, 1/1000, 1/1000),
            orientation=euler_angles_to_quat([0, 0, 0]),
            usd_path=FRAME_URL,
        )

        prims.create_prim(
            prim_path=f"{SCOPE_NAME}/Frame_new",
            position=add_position_offsets(
                (-0.25132, -0.65774 + 1.37, -0.212883-0.770155)),
            scale=(1/1000, 1/1000, 1/1000),
            orientation=euler_angles_to_quat([0, 0, 0]),
            usd_path=FRAME_URL,
        )

        if obj_type == Objects.CHEEZIT:
            add_reference_to_stage(prefix_with_isaac_asset_server(CHEEZEIT_URL), obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([math.pi/2, math.pi, 0]),
                # scale=(0.6, 0.8, 0.65),
                scale=(1, 1, 1),
            )

            # print('masses', object_prim.get_mass())

            # new_object_prim_name = f'{SCOPE_NAME}/new_object'
            # add_reference_to_stage(prefix_with_isaac_asset_server(NEW_OBJECT_URL), new_object_prim_name)
            # new_object_prim = RigidPrim(
            #     prim_path=new_object_prim_name,
            #     position=add_position_offsets(obj_pos_offset) + np.array([0.4, 0, 0]),
            #     orientation=euler_angles_to_quat([math.pi/2, math.pi, 0]),
            #     scale=(1, 1, 1),
            # )
            # UsdPhysics.CollisionAPI.Apply(new_object_prim.prim)
            # object_prim.set_mass(0.04)
            # print('masses', object_prim.get_mass())
            # massAPI = UsdPhysics.MassAPI(object_prim.prim)
            # print("cheezit mass", massAPI.GetDiagonalInertiaAttr().Get())
        
        elif obj_type == Objects.RUBIK:
            add_reference_to_stage(RUBIK_CUBE_URL, obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([0, 0, math.pi]),
                scale=(1.1, 1.1, 1.1),
                # usd_path=prefix_with_isaac_asset_server(RUBIK_CUBE_URL),
            )

            add_reference_to_stage(prefix_with_isaac_asset_server(WOODEN_BOX_URL), f'{SCOPE_NAME}/wooden_box')
            wooden_box = RigidPrim(
                prim_path=f'{SCOPE_NAME}/wooden_box',
                position=(0, 0.6, 0.05),
                # orientation=euler_angles_to_quat([-math.pi/2, 0, math.pi]),
                orientation=euler_angles_to_quat([0, 0, 0]),
                scale=(3, 1.5, 0.9),
                # usd_path=prefix_with_isaac_asset_server(CAN_URL),
            )

            UsdPhysics.CollisionAPI.Apply(wooden_box.prim)
            wooden_box.set_mass(4)


        elif obj_type == Objects.BLOCK:
            add_reference_to_stage(prefix_with_isaac_asset_server(BLOCK_URL), obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([math.pi, 0, math.pi]),
                scale=(2, 2, 2),
                # usd_path=prefix_with_isaac_asset_server(BLOCK_URL),
            )
        elif obj_type == Objects.CRATE:
            add_reference_to_stage(prefix_with_isaac_asset_server(BLOCK_URL), obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([0, 0, 0]),
                scale=(.3, .3, .3),
                # usd_path=prefix_with_isaac_asset_server(CRATE_URL),
            )
        elif obj_type == Objects.CAN:
            add_reference_to_stage(prefix_with_isaac_asset_server(CAN_URL), obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, math.pi]),
                # orientation=euler_angles_to_quat([0, 0, 0]),
                # scale=(0.5, 1, 0.5),
                scale=(1,1,1),
                # usd_path=prefix_with_isaac_asset_server(CAN_URL),
            )
        elif obj_type == Objects.BANANA:
            add_reference_to_stage(prefix_with_isaac_asset_server(BANANA_URL), obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, 0]),
                # orientation=euler_angles_to_quat([0, 0, 0]),
                scale=(2, 2, 2),
                # usd_path=prefix_with_isaac_asset_server(BANANA_URL),
            )
        elif obj_type == Objects.MUG:
            add_reference_to_stage(prefix_with_isaac_asset_server(MUG_URL), obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, -math.pi/4]),
                # orientation=euler_angles_to_quat([0, 0, 0]),
                scale=(1, 1, 1),
                # usd_path=prefix_with_isaac_asset_server(MUG_URL),
            )

        
        elif obj_type == Objects.MUSTARD:
            add_reference_to_stage(prefix_with_isaac_asset_server(MUSTARD_URL), obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, math.pi]),
                scale=(1, 1, 1),
                # usd_path=prefix_with_isaac_asset_server(MUSTARD_URL),
            )
        elif obj_type == Objects.SPAM:
            add_reference_to_stage(prefix_with_isaac_asset_server(SPAM_URL), obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, math.pi]),
                scale=(2, 2, 1.2),
                # usd_path=prefix_with_isaac_asset_server(SPAM_URL),
            )
        
        elif obj_type == Objects.GELATIN:
            add_reference_to_stage(prefix_with_isaac_asset_server(GELATIN_URL), obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, 0]),
                scale=(2, 2, 2),
                # usd_path=prefix_with_isaac_asset_server(GELATIN_URL),
            )
        elif obj_type == Objects.SMUG:
            add_reference_to_stage(prefix_with_isaac_asset_server(SMUG_URL), obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2,math.pi, -math.pi/4]),
                scale=(1, 3, 1),
                # usd_path=prefix_with_isaac_asset_server(GELATIN_URL),
            )
        elif obj_type == Objects.BASKET:
            # breakpoint()
            add_reference_to_stage(BASKET_URL, obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([math.pi, 0, 0]),
                scale=(0.8, 0.25, 0.7),
            )
        
        else:
            print("Invalid object")
            object_prim = None

        self.object_prim = object_prim
        UsdPhysics.CollisionAPI.Apply(object_prim.prim)
        object_prim.set_mass(0.01)

        # bp()
        # self.obj_initial_rot = R.from_quat(quat_pxr2scipy(self.object_prim.prim.GetAttribute("xformOp:orient").Get()))
        # self.obj_initial_loc = self.object_prim.prim.GetAttribute("xformOp:translate").Get()

        # print("Object ROT", self.obj_initial_rot.as_matrix())
        # print("Object LOC", self.obj_initial_loc)

        # self.obj_final_rot = R.from_matrix(self.object_pose_actual) * self.obj_initial_rot
        # self.object_prim.prim.GetAttribute("xformOp:orient").Set(quat_scipy2pxr(self.obj_final_rot.as_quat()))

        self.obj_actual_rot = self.object_pose_actual[:3, :3]
        self.obj_actual_loc = self.object_pose_actual[:3, -1]

        print("Simulation : These are the poses", self.obj_actual_rot, self.obj_actual_loc)

        self.object_prim.prim.GetAttribute("xformOp:orient").Set(quat_scipy2pxr(R.from_matrix(self.obj_actual_rot).as_quat()))
        self.object_prim.prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*self.obj_actual_loc))


        self.last_pos = None

        self.main_viewport = omni.ui.Workspace.get_window("Viewport")
        
        create_viewport_window('camera1', visible=False, docked=True,
                               width=CONFIG['width'], height=CONFIG['height'])
        viewport_window = get_viewport_from_window_name('camera1')
        viewport_window.set_active_camera(
            f"{SCOPE_NAME}/Franka1/panda_hand/geometry/camera")

        create_viewport_window('camera2', visible=False, docked=True,
                               width=CONFIG['width'], height=CONFIG['height'])
        viewport_window = get_viewport_from_window_name('camera2')
        viewport_window.set_active_camera(
            f"{SCOPE_NAME}/Franka2/panda_hand/geometry/camera")

        create_viewport_window('camera3', visible=False, docked=True,
                                 width=CONFIG['width'], height=CONFIG['height'])
        # viewport_window = get_viewport_from_window_name('camera3')
        # viewport_window.set_active_camera(f"{SCOPE_NAME}/Camera3")
        

        return

    # Information exposed to solve the task is returned from the task through get_observations
    def get_observations(self) -> Dict[str, np.ndarray]:
        current_joint_positions = self._franka1.get_joint_positions()
        observations = {
            self._franka1.name: current_joint_positions,
            self._franka2.name: current_joint_positions,
            "current_goal_pose": np.array(self.goal_poses[self.task_state]),
        }
        return observations
    
    def take_image_from_viewport(self, viewport_window, delay=0.01):
        image = sd_helper.get_groundtruth(['rgb'], viewport_window, 0.1)['rgb']
        return image
    
    def take_images(self):
        if not args.create_video:
            return
        # image1 = self.take_image_from_viewport(get_viewport_from_window_name('camera1'))
        # image2 = self.take_image_from_viewport(get_viewport_from_window_name('camera2'))
        image3 = self.take_image_from_viewport(get_viewport_from_window_name("camera3"))
        print(image3.shape)
        if image3.shape == 0:
            print("Huge Error!!!")
            breakpoint()
        # breakpoint()

        # self.cam1_images.append(image1.copy())
        # self.cam2_images.append(image2.copy())
        self.main_view_images.append(image3.copy())
        # self.video.write(image3)
        # print('images taken')

    
    def get_cam_c2w(self, cam_name):
        if cam_name == 'camera1':
            camera = self.camera1
        else:
            camera = self.camera2
        
        c2w_cam = np.linalg.inv(camera.get_view_matrix_ros())
        print(c2w_cam)
        c2w_cam[:3,:3] = R.as_matrix(R.from_matrix(c2w_cam[:3,:3]) * R.from_quat(np.array([1,0,0,0])))
        print(c2w_cam)
        return c2w_cam
    
    def create_vids(self):
        # breakpoint()
        print('creating videos')
        import imageio
        print(len(self.cam1_images))
        print(self.output_dir)
        # import pdb; pdb.set_trace()
        # imageio.mimsave(f'{self.output_dir}/cam1.avi', self.cam1_images, fps=1)
        # imageio.mimsave(f'{self.output_dir}/cam2.avi', self.cam2_images, fps=1)
        # imageio.mimsave(f'{self.output_dir}/main_view.avi', self.main_view_images, fps=1)
        # create_vid(self.cam1_images, f'{self.output_dir}/cam1.avi', fps=10)
        # create_vid(self.cam2_images, f'{self.output_dir}/cam2.avi', fps=10)
        create_vid(self.main_view_images, f'{self.output_dir}/main_view.avi', fps=10)

    def target_reached(self, ks) -> bool:
        if self.task_state == 1:
            return False
        target_pose = self.goal_poses[self.task_state]

        rob = target_pose[-1]
        
        # if(rob == 1):
        #     ee_position, ee_rot_mat = articulation_kinematics_solver_2.compute_end_effector_pose()
        # else:
        #     ee_position, ee_rot_mat = articulation_kinematics_solver_1.compute_end_effector_pose()

        ee_position, ee_rot_mat = ks.compute_end_effector_pose()
        
        ee_orientation = rot_matrices_to_quats(ee_rot_mat)
        pos_diff = np.mean(
            np.abs(np.array(ee_position) - np.array(target_pose[0])))
        # if self.task_state == 2:
        #     bp()
        orient_diff = angle_between_quats(ee_orientation, target_pose[1])
        print(ee_position, target_pose[0], pos_diff)
        print(ee_orientation, target_pose[1], orient_diff)

        if pos_diff < self.best_dif:
            self.best_dif = pos_diff
            self.best_dif_time = self.running_time

        if pos_diff < (1e-2 / get_stage_units()) and orient_diff < (4*math.pi/180):
            self.same_dif_count = 0
            self.prev_posdif = pos_diff
            return True
        else:
            if (np.abs(pos_diff - self.prev_posdif) < 1e-3):
                self.same_dif_count += 1
            else:
                self.same_dif_count = 0
            self.prev_posdif = pos_diff
            
            return False

    def update_status(self, ks):
        if task.running_time % 10 == 0:
            self.take_images()
        if self.is_done():
            print("done")
            return
        current_goal = self.goal_poses[self.task_state]
        # if self.same_dif_count > 25:
        #     self.target_reaching_failed = True
        #     print(
        #         f'Stuck with same pose diff! for {current_goal}, steps: {self.running_time}')
        
        # TODO conflicting with flipping controller code of waiting till _event_dt completes
        # if self.best_dif_time + 50 < self.running_time:
        #     self.target_reaching_failed = True
        #     print(
        #         f'Stuck with no better diff! for {current_goal}, steps: {self.running_time}')

        # TODO conflicting with flipping controller code of waiting till _event_dt completes
        # if self.running_time > self.running_time_limit:
        #     self.target_reaching_failed = True
        #     print(
        #         f'Steps limit reached for {current_goal}, steps: {self.running_time}')
            
            

        if self.target_reached(ks):
            print('reached', self.task_state,
                  f'{current_goal}, in steps {self.running_time}')
            self.reached_points.append(current_goal[0])
            if (self.running_time + 40 > self.running_time_limit):
                self.running_time_limit = self.running_time + 40
                print(
                    f"Whew! That was close! Increasing Running Time Limit to {self.running_time_limit}")
            self.running_time = 0
            self.task_state += 1
            print('now task state is ', self.task_state)
            # bp()
            self.same_dif_count = 0
            self.best_dif = np.inf
            self.best_dif_time = self.running_time

            # idx = self.goal_poses[self.task_state][-1]
            # current_viewport = get_viewport_from_window_name(f'camera{idx}')
            # image = self.take_image_from_viewport(current_viewport)
            # cv2.imwrite(f'{self.output_dir}/pre_flip_image.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if self.task_state > 2 and self.task_state not in [4, 7]:
                rob = self.goal_poses[self.task_state-1][-1] + 1
                idx = self.task_state - 2
                _ = task.take_image_from_viewport(get_viewport_from_window_name(f"camera{rob}"))
                image = task.take_image_from_viewport(get_viewport_from_window_name(f"camera{rob}"))
                fin_c2w = task.get_cam_c2w(f'camera{rob}')
                
                cv2.imwrite(f"{args.data_dir}/post_flip_image{self.img_state + 1}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
                if not args.create_video:
                    pickle.dump(fin_c2w, open(f"{args.data_dir}/post_flip_c2w_ws{self.img_state + 1}.pkl", "wb"))
                self.img_state += 1

        elif self.target_reaching_failed:
            print(f'Failed {self.task_state} {current_goal}. Steps: {self.running_time}')
            self.failed_points.append(current_goal[0])
            self.target_reaching_failed = False
            self.running_time = 0
            self.task_state += 1
            self.same_dif_count = 0
            self.best_dif = np.inf
            self.best_dif_time = self.running_time

    
def get_next_target_pos(cur_pos, target_pos, failed):
    cur_pos = np.array(cur_pos)
    target_pos = np.array(target_pos)
    next_pos = target_pos.copy()
    if(np.abs(target_pos - cur_pos)[2] > 0.025) and (not failed[2]):
        next_pos = cur_pos + np.sign(target_pos[2] - cur_pos[2])*np.array([0, 0, 0.025])
        return next_pos, 2
    elif(np.abs(target_pos - cur_pos)[1] > 0.025) and (not failed[1]):
        next_pos[0:2] = (cur_pos + np.sign(target_pos[1] - cur_pos[1])*np.array([0, 0.025, 0]))[0:2]
        return next_pos, 1
    elif(np.abs(target_pos - cur_pos)[0] > 0.025) and (not failed[0]):
        next_pos[0] = (cur_pos + np.sign(target_pos[0] - cur_pos[0])*np.array([0.025, 0, 0]))[0]
        return next_pos, 0
    else:
        return target_pos, 3


def get_c2w_from_pos(camera_position: np.ndarray, lookat: np.ndarray) -> np.ndarray:
    new_z = (camera_position - lookat)
    new_z = new_z / np.linalg.norm(new_z)

    z = np.array((0,0,1))
    new_y = z - np.dot(z, new_z) * new_z
    if (np.linalg.norm(new_y) == 0):
        new_y = np.array((0,1,0))

    r, _ = R.align_vectors([(0,1,0), (0,0,1)], [new_y, new_z])
    c2w = np.eye(4)
    c2w[:3,:3] = np.linalg.inv(r.as_matrix())
    c2w[:3,-1] = camera_position
    return c2w

def get_quat_from_cur_pos(pos):
    init_mat = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
            ])

    c2w_cam = get_c2w_from_pos(np.array(pos), WORKSPACE_CENTER)
    c2w_ee = c2w_cam.copy()

    c2w_ee[:3, :3] = R.as_matrix(R.inv( R.from_matrix(init_mat) *  R.inv(R.from_matrix(c2w_cam[:3, :3]))))
    base_rotation = R.from_matrix(c2w_ee[:3,:3])

    quat = R.as_quat(base_rotation)[[3,0,1,2]]
    return quat
    
def get_next_target_rot(cur_rot, target_rot, t=0):
    # using slerp's method
    cur_rot = np.array(cur_rot)/np.linalg.norm(cur_rot)
    target_rot = np.array(target_rot)/np.linalg.norm(target_rot)
    dot = np.dot(cur_rot, target_rot)

    theta = np.arccos(dot)
    next_rot =  (np.sin((1-t)*theta)*cur_rot + np.sin(t*theta)*target_rot)/np.sin(theta)
    return next_rot/np.linalg.norm(next_rot)
   

# json_path, output_dir, obj_type = args.json_path, args.output_dir, args.obj_type
# output_dir, obj_type = args.output_dir, args.obj_type

nerf2ws_expected = pickle.load(open(args.nerf2ws_expected_path, "rb"))
object_pose_actual = pickle.load(open(args.object_pose_actual_path, "rb"))

# json_data = json.load(open(json_path))
json_data = {}

c2w_cam = pickle.load(open(f"{args.data_dir}/c2w.pkl", 'rb'))

my_world = World(stage_units_in_meters=1.00)
task = FrankaPlaying(
    name="my_first_task", 
    nerf2ws_expected=nerf2ws_expected,
    object_pose_actual=object_pose_actual,
    camera_pose_path=f'{args.data_dir}/c2w.pkl',
    grasp_pose_path=f'{args.data_dir}/grasp.pkl',
    # final_pose_path=f'{args.data_dir}/c2w_final.pkl',
    output_dir=args.data_dir,
    obj_type=args.obj_type,
    dont_flip = args.dont_flip
)

print('look dir', task.dir_look)
my_world.add_task(task)
my_world.reset()
# bp()

end_effector_name = "panda_hand"

my_franka_1 = my_world.scene.get_object(franka_name)
articulation_kinematics_solver_1 = FrankaKinematicsSolver(
    my_franka_1, end_effector_frame_name=end_effector_name)
lula_kinematics_solver = articulation_kinematics_solver_1.get_kinematics_solver()
robot_base_translation, robot_base_orientation = my_franka_1.get_world_pose()
lula_kinematics_solver.set_robot_base_pose(
    robot_base_translation, robot_base_orientation)
rmp_controller1 = RMPFlowController("franka1_rmp", my_franka_1, 0.02, end_effector_name)
flip_controller1 = FlipController("franka1_flip", rmp_controller1, my_franka_1, articulation_kinematics_solver_1, obj_type=args.obj_type)
articulation_controller_1 = my_franka_1.get_articulation_controller()
# my_franka_1.initialize()
# print(robot_base_translation, robot_base_orientation)

my_franka_2 = my_world.scene.get_object(franka_new_name)
articulation_kinematics_solver_2 = FrankaKinematicsSolver(
    my_franka_2, end_effector_frame_name=end_effector_name)
lula_kinematics_solver = articulation_kinematics_solver_2.get_kinematics_solver()
robot_base_translation, robot_base_orientation = my_franka_2.get_world_pose()
lula_kinematics_solver.set_robot_base_pose(
    robot_base_translation, robot_base_orientation)
rmp_controller2 = RMPFlowController("franka2_rmp", my_franka_2, 0.02, end_effector_name)
flip_controller2 = FlipController("franka2_flip", rmp_controller2, my_franka_2, articulation_kinematics_solver_2, obj_type=args.obj_type)
articulation_controller_2 = my_franka_2.get_articulation_controller()
# my_franka_2.initialize()
# print(robot_base_translation, robot_base_orientation)
start_time = time.time()
my_franka_1.gripper.set_joint_positions(my_franka_1.gripper.joint_opened_positions)
my_franka_2.gripper.set_joint_positions(my_franka_2.gripper.joint_opened_positions)

if stream and pause:
    while simulation_app.is_running():
        my_world.step(render=True)

slerp_t = 0
# bp()

count_same = 0
prev_v = None
steps = 0
ks = articulation_kinematics_solver_1 if task.goal_poses[0][-1] == 0 else articulation_kinematics_solver_2

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        current_time = time.time()
        if current_time - start_time < 5.0:
            continue
        task.running_time += 1
        # bp()
        task.update_status(ks)
        if task.is_done():
            # breakpoint()
            if args.create_video:
                # task.video.release()
                task.create_vids()
            print("done2")
            break
        if my_world.current_time_step_index == 0:
            print("reset")
            my_world.reset()

        observations = my_world.get_observations()
        if observations["current_goal_pose"][2] == 0:
            controller = articulation_controller_1
            rmp_controller = rmp_controller1
            flip_controller = flip_controller1
            ks = articulation_kinematics_solver_1
        else:
            controller = articulation_controller_2
            rmp_controller = rmp_controller2
            flip_controller = flip_controller2
            ks = articulation_kinematics_solver_2
        
        if task.task_state != 1:
            actions = rmp_controller.forward(
                target_end_effector_position = observations["current_goal_pose"][0], 
                target_end_effector_orientation = observations["current_goal_pose"][1]
            )
        else:
            # task.task_state += 1
            # continue
            actions = flip_controller.forward(
                task.goal_poses[1][0],
                R.from_quat(task.goal_poses[1][1]),
                task.dir_look,
                observations[franka_name] if task.goal_poses[1][2] == 0 else observations[franka_new_name]
            )
            if flip_controller.is_done():
                # bp()
                task.running_time = 0
                task.task_state += 1
                print("FLIPPING DONE")
                task.same_dif_count = 0
                task.best_dif = np.inf
                task.best_dif_time = task.running_time
                continue
        # jv = actions.joint_velocities
        # if jv is not None:          # During stabilizing actions
        #     stuck = True
        #     for v in jv:
        #         if abs(v) > 5e-4:
        #             stuck = False
        #     if prev_v is not None:
        #         if np.linalg.norm(prev_v - jv) < 1e-4:
        #             count += 1
        #         else:
        #             count = 0
        #         if count > 15:
        #             stuck = True
        #     prev_v = jv.copy()
        #     if stuck:
        #         print("Stuck at a position. Moving to next goal")
        #         task.target_reaching_failed = True
        #         count = 0
        controller.apply_action(actions)
        steps += 1
        # if steps == 100:
        #     bp()

obj_rot_post_flip = R.from_quat(quat_pxr2scipy(task.object_prim.prim.GetAttribute("xformOp:orient").Get()))
obj_loc_post_flip = task.object_prim.prim.GetAttribute("xformOp:translate").Get()
pose = np.eye(4)
pose[:3,:3] = obj_rot_post_flip.as_matrix()
pose[:3,-1] = obj_loc_post_flip

print("Object ROT", obj_rot_post_flip.as_matrix())
print("Object LOC", obj_loc_post_flip)

# bp()
# sys.exit(0)
if not args.create_video:
    pickle.dump(pose, open(f"{args.data_dir}/object_actual_pose.pkl", "wb"))
else:
    pickle.dump(pose, open(f"{args.data_dir}/video_object_actual_pose.pkl", "wb"))
    pickle.dump(task.get_cam_c2w('camera1'), open(f"{args.data_dir}/video_cam1.pkl", "wb"))
    pickle.dump(task.get_cam_c2w('camera2'), open(f"{args.data_dir}/video_cam2.pkl", "wb"))
image = None
fin_c2w = None
# if task.goal_poses[1][-1] == 0:
#     _ = task.take_image_from_viewport(get_viewport_from_window_name("camera1"))
#     image = task.take_image_from_viewport(get_viewport_from_window_name("camera1"))
#     fin_c2w = task.get_cam_c2w('camera1')
#     cv2.imwrite(f"{args.data_dir}/post_flip_image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
#     pickle.dump(fin_c2w, open(f"{args.data_dir}/post_flip_c2w_ws.pkl", "wb"))
# else:
#     _ = task.take_image_from_viewport(get_viewport_from_window_name("camera2"))
#     image = task.take_image_from_viewport(get_viewport_from_window_name("camera2"))
#     fin_c2w = task.get_cam_c2w('camera2')
#     cv2.imwrite(f"{args.data_dir}/post_flip_image.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.uint8))
#     pickle.dump(fin_c2w, open(f"{args.data_dir}/post_flip_c2w_ws.pkl", "wb"))

if not hmode or stream:
    while simulation_app.is_running():
        my_world.step(render=True)
simulation_app.close()
