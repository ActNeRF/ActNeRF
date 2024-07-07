import os
import sys
import pdb
import math
import time
import json
import pickle
import imageio
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from enum import Enum
from typing import Dict
from scipy.spatial.transform import Rotation as R

def bp():
    pdb.set_trace()

from omni.isaac.kit import SimulationApp

hmode = True
stream = False
pause = False

headless = hmode
view = not hmode

CONFIG = {
    "renderer": "RayTracedLightingsaac/Props/Mounts/table.usd'", 
    "headless": headless, 
    "width": 800, 
    "height": 800, 
}

class UselessObject:
    def __init__(self):
        pass


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
SOUP_URL = '/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd'
WOODEN_BOX_URL = '/Isaac/Props/YCB/Axis_Aligned/036_wood_block.usd'
RUBIK_CUBE_URL = '/home/saptarshi/Downloads/rubik_final.usd'

IITD_TABLE_URL = '/home/saptarshi/Downloads/IITD_FRAME_6060_v2/IITD_TABLE_MODIFIED.usd'
HOLDER_URL = '/home/saptarshi/Downloads/IITD_FRAME_6060_v2/GRIPPER_DUMMY_2v7v2.usdc'
FRAME_URL = '/home/saptarshi/Downloads/IITD_FRAME_6060_v2/IITD_FRAME_6060_v2.usdc'
BASKET_URL = '/home/saptarshi/Downloads/basket_final.usd'

NEW_OBJECT_URL = '/home/saptarshi/Downloads/IITD_FRAME_6060_v2/model2.usd'


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
    BASKET = 11
    SMUG = 12
    SOUP = 13

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
    Objects.BASKET: 'BASKET',
    Objects.SMUG: 'SMUG',
    Objects.SOUP: 'Soup',
    
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
    Objects.BASKET: 'basket',
    Objects.SMUG: 'small_mug',
    Objects.SOUP: 'soup',
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
    Objects.BASKET: (0, -0.66887 + 0.6, -0.033828 + (0.1-0.13633)),
    Objects.SMUG: (0, -0.66887 + 0.6, -0.033828),
    Objects.SOUP: (0, -0.66887 + 0.6, -0.033828),

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
    W,H = imgs[0].shape[1], imgs[0].shape[0]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (W,H))
    for i in range(len(imgs)):
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
            obj_type: str = "cheezit"
        ):

        super().__init__(name=name, offset=None)

        self.obj_type = search_in_dict(out_name_map, obj_type)

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
            position=np.array(robot_new_current_pos + np.array([5, 0, 0])),
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
                scale=(0.5, 1, 0.5),
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
        elif obj_type == Objects.SOUP:
            add_reference_to_stage(prefix_with_isaac_asset_server(SOUP_URL), obj_prim_name)
            object_prim = RigidPrim(
                prim_path=obj_prim_name,
                position=add_position_offsets(obj_pos_offset),
                orientation=euler_angles_to_quat([-math.pi/2,0, 0]),
                scale=(1, 1, 1),
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
            # object_prim = prims.create_prim(
            #     prim_path=obj_prim_name,
            #     position=add_position_offsets(obj_pos_offset),
            #     orientation=euler_angles_to_quat([-math.pi/2, 0, 0]),
            #     scale=(2, 2, 2),
            #     usd_path=BASKET_URL,
            # )
            # UsdPhysics.CollisionAPI.Apply(object_prim)
            # self.object_prim = UselessObject()
            # self.object_prim.prim = object_prim
            # object_prim.set_mass(0.04)

        
        else:
            print("Invalid object")
            object_prim = None

        # if obj_type != Objects.BASKET:
        self.object_prim = object_prim
        UsdPhysics.CollisionAPI.Apply(object_prim.prim)
        object_prim.set_mass(0.04)

        # bp()


        self.main_viewport = omni.ui.Workspace.get_window("Viewport")
        
        create_viewport_window('camera1', visible=False, docked=True,
                               width=CONFIG['width'], height=CONFIG['height'])
        # viewport_window = get_viewport_from_window_name('camera1')
        # viewport_window.set_active_camera(
        #     f"{SCOPE_NAME}/Franka1/panda_hand/geometry/camera")

        create_viewport_window('camera2', visible=False, docked=True,
                               width=CONFIG['width'], height=CONFIG['height'])
        # viewport_window = get_viewport_from_window_name('camera2')
        # viewport_window.set_active_camera(
        #     f"{SCOPE_NAME}/Franka2/panda_hand/geometry/camera")

        create_viewport_window('camera3', visible=False, docked=True,
                                 width=CONFIG['width'], height=CONFIG['height'])
        # viewport_window = get_viewport_from_window_name('camera3')
        # viewport_window.set_active_camera(f"{SCOPE_NAME}/Camera3")
        

        return

    
import argparse
import pickle as pkl
parser = argparse.ArgumentParser()
parser.add_argument('--out-path', type=str, required=True)
parser.add_argument('--obj-type', type=str, required=True)

args = parser.parse_args()

my_world = World(stage_units_in_meters=1.00)
task = FrankaPlaying(
    name="my_first_task", 
    obj_type= args.obj_type
)

my_world.add_task(task)
my_world.reset()
step = 1
while simulation_app.is_running():
    step = step + 1
    my_world.step(render=True)
    if step == 100:
        break

task.obj_initial_rot = R.from_quat(quat_pxr2scipy(task.object_prim.prim.GetAttribute("xformOp:orient").Get()))
task.obj_initial_loc = task.object_prim.prim.GetAttribute("xformOp:translate").Get()

print("Object ROT", task.obj_initial_rot.as_matrix())
print("Object LOC", task.obj_initial_loc)

obj_init_pos = np.eye(4)
obj_init_pos[:3, :3] = task.obj_initial_rot.as_matrix()
obj_init_pos[:3, 3] = task.obj_initial_loc
pkl.dump(obj_init_pos, open(args.out_path, 'wb'))
if not hmode or stream:
    while simulation_app.is_running():
        my_world.step(render=True)

simulation_app.close()
