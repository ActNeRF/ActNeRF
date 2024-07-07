import os
import sys
import pdb
import math
import time
import json
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from enum import Enum
from typing import Dict
from scipy.spatial.transform import Rotation as R

from omni.isaac.kit import SimulationApp


def bp():
    pdb.set_trace()

hmode = True
stream = False

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

import logging
import carb

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
# RUBIK_CUBE_URL = '/Isaac/Props/Rubiks_Cube/rubiks_cube.usd'
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
DRILL_URL = '/Isaac/Props/YCB/Axis_Aligned/035_power_drill.usd'
MARKER_URL = '/Isaac/Props/YCB/Axis_Aligned/040_large_marker.usd'
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


class Objects(Enum):
    CHEEZIT = 1
    CRATE = 2
    BLOCK = 3
    RUBIK = 4
    CAN = 5
    BANANA = 6
    MUG = 7
    DRILL = 8
    MARKER = 9
    SPAM = 10
    MUSTARD = 11
    GELATIN = 12
    SMUG = 13
    BASKET = 14

obj_prim_path_map = {
    Objects.CHEEZIT: 'Cheezit',
    Objects.CRATE: 'Crate',
    Objects.BLOCK: 'Block',
    Objects.RUBIK: 'RCube',
    Objects.CAN: 'CAN',
    Objects.BANANA: 'BANANA',
    Objects.MUG: 'MUG',
    Objects.DRILL: 'DRILL',
    Objects.MARKER: 'MARKER',
    Objects.SPAM: 'SPAM',
    Objects.MUSTARD: 'MUSTARD',
    Objects.GELATIN: 'GELATIN',
    Objects.SMUG: 'SMUG',
    Objects.BASKET: 'BASKET',
}

out_name_map = {
    Objects.CHEEZIT: 'cheezit',
    Objects.CRATE: 'crate',
    Objects.BLOCK: 'block',
    Objects.RUBIK: 'rubik',
    Objects.CAN: 'can',
    Objects.BANANA: 'banana',
    Objects.MUG: 'mug',
    Objects.DRILL: 'drill',
    Objects.MARKER: 'marker',
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
    Objects.DRILL: (0, -0.66887 + 0.6, -0.033828 + (0.0587-0.13633)),
    Objects.MARKER: (0, -0.66887 + 0.6, -0.033828 + (0.0387-0.13633)),
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

def add_robot_offset(pos):
    add_position_offsets(pos)
    
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

def search_in_dict(d, val):
    for key, value in d.items():
        if value == val:
            return key

class FrankaPlaying(BaseTask):
    def __init__(
            self, 
            name: str, 
            object_pose_actual: np.ndarray,
            nerf2ws_expected: np.ndarray,
            w2c: np.ndarray, 
            output_dir: str, 
            obj_type: str = "cheezit", 
            num_images: int = 100, 
            r_std: int = 0.007, 
            angle_std: int = 3
        ):

        super().__init__(name=name, offset=None)

        self.goal_poses = []

        self.r1_moved = False
        self.r2_moved = False
        self.task_state = 0
        self.target_reaching_failed = False
        self.reached_points = []
        self.failed_points = []
        self.running_time = 0
        self.running_time_limit = 1000
        self.object_pose_actual = object_pose_actual
        self.nerf2ws_expected = nerf2ws_expected
        self.w2c = w2c
        self.output_dir = output_dir
        self.obj_type = search_in_dict(out_name_map, obj_type)

        os.makedirs(self.output_dir, exist_ok=True)
        self.setup_goals(num_images, r_std, angle_std)

    def setup_goals(self, num_images: int, r_std: int = 0.007, angle_std: int = 3):
        init_mat = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        c2w_cam = np.linalg.inv(self.w2c)
        # bp()
        c2w_cam = c2w_nerf2world(c2w_cam, self.nerf2ws_expected)

        c2w_ee = c2w_cam.copy()
        dir_look = c2w_cam[:3, :3] @ np.array([0, 0, -1])

        c2w_ee[:3, -1] = c2w_cam[:3, -1] - 0.11*dir_look
        c2w_ee[:3, :3] = R.as_matrix(R.inv( R.from_matrix(init_mat) *  R.inv(R.from_matrix(c2w_cam[:3, :3]))))

        base_position = c2w_ee[:3,-1]
        base_rotation = R.from_matrix(c2w_ee[:3,:3])
        # print(vec, angle)
        # base_rotation = R.from_rotvec(vec*(angle))
        posses = [base_position]
        rots = [base_rotation.as_quat()[[3, 0, 1, 2]]]
        posses += [(base_position + np.random.normal(scale=r_std, size=(3,)))
                  for i in range(num_images-1)]
        rots += [(base_rotation * R.from_euler('xyz', np.random.normal(scale=angle_std,
                 size=(3,)), degrees=True)).as_quat()[[3, 0, 1, 2]] for i in range(num_images-1)]
        # print("C2W", rot)
        print("HERE", R.as_rotvec(base_rotation))
        print(posses)
        # posses = [base_position]
        # rots_quat = [R.from_euler('xyz', base_rotation, degrees=True).as_quat()]
        idx = [int(pos[1] > 0.6) for pos in posses]
        self.goal_poses = list(zip(posses, rots, idx))

        # pdb.set_trace()
        if 0 in idx:
            self.r1_moved = True
        if 1 in idx:
            self.r2_moved = True

        # quat = R.from_matrix(rot).as_quat()
        # quat = 1/np.sqrt(2) * np.array([0, 0, 1, 1])
        # quat = 1/np.sqrt(2) * np.array([0, 0, 1, 1])

        # self.goal_poses = [[np.array([0,0.25,0.15]), quat]]
        # self.goal_poses = [[base_position, quat]]
        print('goals', self.goal_poses)
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
        room1_prim = prims.create_prim(
            prim_path=f"{SCOPE_NAME}/Room1",
            # position=(random.uniform(-20, -2), random.uniform(-1, 3), 0),
            # orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
            # scale=(10,10,10),
            usd_path=prefix_with_isaac_asset_server(ROOM_URL),
        )
        room2_prim = prims.create_prim(
            prim_path=f"{SCOPE_NAME}/Room2",
            # position=(random.uniform(-20, -2), random.uniform(-1, 3), 0),
            orientation=euler_angles_to_quat([0, 0, math.pi]),
            # scale=(10,10,10),
            usd_path=prefix_with_isaac_asset_server(ROOM_URL),
        )
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
            orientation=euler_angles_to_quat([0, 0, math.pi/2])))

        self._franka2 = scene.add(Franka(
            prim_path=f"{SCOPE_NAME}/Franka2",
            name=franka_new_name,
            position=np.array(robot_new_current_pos),
            orientation=euler_angles_to_quat([0, 0, -math.pi/2])))

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
        # self.camera.prim.GetAttribute("focalLength").Set(.0193)
        # self.camera.prim.GetAttribute("focusDistance").Set(4)
        # self.camera.prim.GetAttribute("horizontalAperture").Set(1)
        # self.camera.prim.GetAttribute("verticalAperture").Set(1)
        # self.camera.prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(0,0,1,0))
        # self.camera.prim.GetAttribute("clippingRange").Set(Gf.Vec2f(0.01, 10000))

        focal = 193
        aper = 2*np.tan(np.pi/180 * 65.5/2)*focal
        aper = 300
        # aper = 24.8

        self.camera1.prim.GetAttribute("focalLength").Set(focal)
        self.camera1.prim.GetAttribute("focusDistance").Set(100)
        self.camera1.prim.GetAttribute("horizontalAperture").Set(aper)
        self.camera1.prim.GetAttribute("verticalAperture").Set(aper)
        self.camera1.prim.GetAttribute(
            "xformOp:orient").Set(Gf.Quatd(0, 1, 0, 0))
        self.camera1.prim.GetAttribute(
            "clippingRange").Set(Gf.Vec2f(0.01, 10000))

        self.camera2.prim.GetAttribute("focalLength").Set(focal)
        self.camera2.prim.GetAttribute("focusDistance").Set(100)
        self.camera2.prim.GetAttribute("horizontalAperture").Set(aper)
        self.camera2.prim.GetAttribute("verticalAperture").Set(aper)
        self.camera2.prim.GetAttribute(
            "xformOp:orient").Set(Gf.Quatd(0, 1, 0, 0))
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
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([math.pi/2, math.pi, 0]),
                # orientation=euler_angles_to_quat([0, 0, 0]),
                scale=(1, 1, 1),
                usd_path=prefix_with_isaac_asset_server(CHEEZEIT_URL),
            )
        elif obj_type == Objects.RUBIK:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([0, 0, math.pi]),
                scale=(1.1, 1.1, 1.1),
                usd_path=RUBIK_CUBE_URL,
            )

            wooden_box = prims.create_prim(
                prim_path=f'{SCOPE_NAME}/wooden_box',
                position=(0, 0.6, 0.05),
                orientation=euler_angles_to_quat([0, 0, 0]),
                scale=(3, 1.5, 0.9),
                usd_path=prefix_with_isaac_asset_server(WOODEN_BOX_URL),
            )
        elif obj_type == Objects.BLOCK:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([math.pi, 0, math.pi]),
                scale=(2, 2, 2),
                usd_path=prefix_with_isaac_asset_server(BLOCK_URL),
            )
        elif obj_type == Objects.CRATE:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([0, 0, 0]),
                scale=(.3, .3, .3),
                usd_path=prefix_with_isaac_asset_server(CRATE_URL),
            )
        elif obj_type == Objects.CAN:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, math.pi]),
                # orientation=euler_angles_to_quat([0, 0, 0]),
                scale=(1, 1, 1),
                usd_path=prefix_with_isaac_asset_server(CAN_URL),
            )
        elif obj_type == Objects.BANANA:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, 0]),
                # orientation=euler_angles_to_quat([0, 0, 0]),
                scale=(2, 2, 2),
                usd_path=prefix_with_isaac_asset_server(BANANA_URL),
            )
        elif obj_type == Objects.MUG:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, -math.pi/4]),
                # orientation=euler_angles_to_quat([0, 0, 0]),
                scale=(1, 1, 1),
                usd_path=prefix_with_isaac_asset_server(MUG_URL),
            )
        elif obj_type == Objects.MUSTARD:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, math.pi]),
                scale=(1, 1, 1),
                usd_path=prefix_with_isaac_asset_server(MUSTARD_URL),
            )
        elif obj_type == Objects.SPAM:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, math.pi]),
                scale=(2, 2, 1.2),
                usd_path=prefix_with_isaac_asset_server(SPAM_URL),
            )
        elif obj_type == Objects.DRILL:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([0, 0, math.pi/3]),
                scale=(1, 1, 1),
                usd_path=prefix_with_isaac_asset_server(DRILL_URL),
            )
        
        elif obj_type == Objects.MARKER:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([0, 0, math.pi/3]),
                scale=(1, 1, 1),
                usd_path=prefix_with_isaac_asset_server(MARKER_URL),
            )

        elif obj_type == Objects.GELATIN:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2, 0, 0]),
                scale=(2, 2, 2),
                usd_path=prefix_with_isaac_asset_server(GELATIN_URL),
            )
        elif obj_type == Objects.SMUG:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2,math.pi, -math.pi/4]),
                scale=(1, 3, 1),
                usd_path=prefix_with_isaac_asset_server(SMUG_URL),
            )
        elif obj_type == Objects.BASKET:
            self.object_prim = prims.create_prim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([math.pi, 0, 0]),
                scale=(0.8, 0.25, 0.7),
                usd_path=BASKET_URL,
            )
        else:
            print("Invalid object")

        # block = prims.get_prim_at_path(f"{SCOPE_NAME}/Block/Cube")
        # object_prim.GetAttribute("physics:rigidBodyEnabled").Set(False)
        # object_prim.GetAttribute("physics:collisionEnabled").Set(False)
        
        self.obj_actual_rot = self.object_pose_actual[:3, :3]
        self.obj_actual_loc = self.object_pose_actual[:3, -1]

        # bp()

        print("These are the poses", self.obj_actual_rot, self.obj_actual_loc)

        self.object_prim.GetAttribute("xformOp:orient").Set(quat_scipy2pxr(R.from_matrix(self.obj_actual_rot).as_quat()))
        self.object_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*self.obj_actual_loc))

        # self.obj_initial_rot = R.from_quat(quat_pxr2scipy(self.object_prim.GetAttribute("xformOp:orient").Get()))
        # self.obj_final_rot = R.from_matrix(self.object_pose_actual) * self.obj_initial_rot


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

    def take_image(self):
        if self.goal_poses[self.task_state][2] == 0:
            camera_name = 'camera1'
            camera = self.camera1
        else:
            camera_name = 'camera2'
            camera = self.camera2
    
        viewport_window = get_viewport_from_window_name(camera_name)
        sd_helper = SyntheticDataHelper()
        print(sd_helper.get_camera_params(viewport_window))
        image = sd_helper.get_groundtruth(
            ['rgb', 'depth', 'boundingBox2DTight'], viewport_window)['rgb']
        Image.fromarray(image).save(
            f"{self.output_dir}/rgb_{self.task_state}.png")
        # bp()

        # cam_pos, cam_rot = camera.get_world_pose()
        # cam_rot_mat = R.from_quat(cam_rot).as_matrix()
        # c2w_cam = np.eye(4)
        # c2w_cam[:3, :3] = cam_rot_mat
        # c2w_cam[:3, -1] = cam_pos
        c2w_cam = np.linalg.inv(camera.get_view_matrix_ros())
        # print(c2w_cam)
        c2w_cam = c2w_world2nerf(c2w_cam, self.nerf2ws_expected)
        c2w_cam[:3,:3] = R.as_matrix(R.from_matrix(c2w_cam[:3,:3]) * R.from_quat(np.array([1,0,0,0])))
        # print(c2w_cam)

        w2c_cam = np.linalg.inv(c2w_cam)

        camera_props = {
            "cameraFocalLength": camera.get_focal_length(),
            "intrinsicsMatrix" : camera.get_intrinsics_matrix().tolist(),
            "cameraAperture": [camera.get_horizontal_aperture(), camera.get_horizontal_aperture()],
            "renderProductResolution": [CONFIG["width"], CONFIG["height"]],
            # "cameraViewTransform" : np.linalg.inv(self.camera.get_view_matrix_ros().T).tolist()
            # "cameraViewTransform": np.linalg.inv(c2w_cam).T.tolist()
            "cameraViewTransform": w2c_cam.T.tolist()
        }
        print(camera_props)
        with open(f"{self.output_dir}/camera_params_{self.task_state}.json", "w") as outfile:
            json.dump(camera_props, outfile)
        print('image taken')

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
    
    def target_reached(self) -> bool:
        target_pose = self.goal_poses[self.task_state]

        rob = target_pose[-1]
        
        if(rob == 1):
            ee_position, ee_rot_mat = articulation_kinematics_solver_2.compute_end_effector_pose()
        else:
            ee_position, ee_rot_mat = articulation_kinematics_solver_1.compute_end_effector_pose()
        
        ee_orientation = rot_matrices_to_quats(ee_rot_mat)
        pos_diff = np.mean(
            np.abs(np.array(ee_position) - np.array(target_pose[0])))
        orient_diff = angle_between_quats(ee_orientation, target_pose[1])

        if pos_diff < (1e-3 / get_stage_units()) and orient_diff < (4*math.pi/180):
            return True
        else:
            return False

    def update_status(self):
        if self.is_done():
            print("done")
            return
        current_goal = self.goal_poses[self.task_state]
        if self.running_time > self.running_time_limit:
            self.target_reaching_failed = True
            print(
                f'Steps limit reached for {current_goal}, steps: {self.running_time}')

        stage = omni.usd.get_context().get_stage()
        if self.target_reached():
            print('reached', self.task_state,
                  f'{current_goal}, in steps {self.running_time}')
            self.take_image()
            self.take_image()

            self.reached_points.append(current_goal[0])
            if (self.running_time + 40 > self.running_time_limit):
                self.running_time_limit = self.running_time + 40
                print(
                    f"Whew! That was close! Increasing Running Time Limit to {self.running_time_limit}")
            self.running_time = 0
            self.task_state += 1
            # my_world.reset()
            # self.controller.reset()
        elif self.target_reaching_failed:
            print(f'Failed {self.task_state} {current_goal}')
            self.failed_points.append(current_goal[0])
            self.target_reaching_failed = False
            self.running_time = 0
            self.task_state += 1
            # my_world.reset()

# nbv_path, output_dir, obj_type = sys.argv[1:]

parser = argparse.ArgumentParser()
parser.add_argument('-p','--nbv-path', type=str)
parser.add_argument('-o','--output-dir', type=str)
parser.add_argument('-t','--obj-type', type=str)
parser.add_argument('-rbp','--robot-poses', type=str)
parser.add_argument('-ooa','--object-pose-actual-path', type=str)
parser.add_argument('-ooe','--nerf2ws-expected-path', type=str)
parser.add_argument('-n','--num-images', type=int, default=5)
parser.add_argument('-r','--r-std', type=float, default=0.007)
parser.add_argument('-a','--angle-std', type=int, default=3)
args = parser.parse_args()

nbv_path, output_dir, obj_type = args.nbv_path, args.output_dir, args.obj_type
try:
    robot_poses = pickle.load(open(args.robot_poses, 'rb'))
except:
    # TODO remove
    robot_poses = None

nerf2ws_expected = np.eye(4)
object_pose_actual = np.eye(4)

try:
    nerf2ws_expected = pickle.load(open(args.nerf2ws_expected_path, 'rb'))
    object_pose_actual = pickle.load(open(args.object_pose_actual_path, 'rb'))
except:
    # TODO remove
    pass

next_best_pose = pickle.load(open(nbv_path, 'rb'))[0]
print(next_best_pose)

my_world = World(stage_units_in_meters=1.00)
task = FrankaPlaying(
    name="my_first_task", 
    object_pose_actual=object_pose_actual,
    nerf2ws_expected=nerf2ws_expected,
    w2c=next_best_pose, 
    output_dir=output_dir, 
    obj_type=obj_type,
    num_images=args.num_images,
    r_std=args.r_std,
    angle_std=args.angle_std
)
my_world.add_task(task)
my_world.reset()

# task.setup_goals(args.num_images, args.r_std, args.angle_std)

end_effector_name = "panda_hand"

my_franka_1 = my_world.scene.get_object(franka_name)
articulation_kinematics_solver_1 = FrankaKinematicsSolver(
    my_franka_1, end_effector_frame_name=end_effector_name)
lula_kinematics_solver = articulation_kinematics_solver_1.get_kinematics_solver()
robot_base_translation, robot_base_orientation = my_franka_1.get_world_pose()
lula_kinematics_solver.set_robot_base_pose(
    robot_base_translation, robot_base_orientation)
articulation_controller_1 = my_franka_1.get_articulation_controller()

my_franka_2 = my_world.scene.get_object(franka_new_name)
articulation_kinematics_solver_2 = FrankaKinematicsSolver(
    my_franka_2, end_effector_frame_name=end_effector_name)
lula_kinematics_solver = articulation_kinematics_solver_2.get_kinematics_solver()
robot_base_translation, robot_base_orientation = my_franka_2.get_world_pose()
lula_kinematics_solver.set_robot_base_pose(
    robot_base_translation, robot_base_orientation)
articulation_controller_2 = my_franka_2.get_articulation_controller()

start_time = time.time()

# if stream:
#     while simulation_app.is_running():
#         my_world.step(render=True)

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        current_time = time.time()
        if current_time - start_time < 5.0:
            continue
        task.running_time += 1
        task.update_status()
        if task.is_done():
            print("done2")
            if robot_poses is None:
                robot_poses = [task.get_cam_c2w('camera1'), task.get_cam_c2w('camera2')]
            else:
                if task.r1_moved:
                    robot_poses[0] = task.get_cam_c2w('camera1')
                if task.r2_moved:
                    robot_poses[1] = task.get_cam_c2w('camera2')
            
            pickle.dump(robot_poses, open(f"{output_dir}/robot_poses.pkl", 'wb'))
            break
        if my_world.current_time_step_index == 0:
            my_world.reset()

        observations = my_world.get_observations()

        if observations["current_goal_pose"][2] == 0:
            kinematics_solver = articulation_kinematics_solver_1
            controller = articulation_controller_1
        else:
            kinematics_solver = articulation_kinematics_solver_2
            controller = articulation_controller_2

        actions, succ = kinematics_solver.compute_inverse_kinematics(
            target_position=np.array(observations["current_goal_pose"][0]),
            target_orientation=np.array(observations["current_goal_pose"][1]),
        )
        if succ:
            controller.apply_action(actions)
        else:
            carb.log_warn(
                "IK did not converge to a solution.  No action is being taken.")
            task.target_reaching_failed = True

# obj_rot = R.from_quat(quat_pxr2scipy(task.object_prim.GetAttribute("xformOp:orient").Get()))


if not hmode or stream:
    while simulation_app.is_running():
        my_world.step(render=True)

simulation_app.close()
