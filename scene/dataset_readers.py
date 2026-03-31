
import os
import sys
from PIL import Image
from typing import NamedTuple

import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from scene.gaussian_model import BasicPointCloud
import torch
import imageio
import math
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation
import random

# ---- data structures ----
class SpectrumInfo(NamedTuple):
    R: np.array
    T_rx: np.array
    T_tx: np.array
    spectrum: np.array
    spectrum_path: str
    spectrum_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_spectrums: list
    test_spectrums: list
    nerf_normalization: dict
    ply_path: str


# ---- train/test splitting ----
def split_dataset_llffhold(datadir, train_path, test_path, ratio=0.8, seed=8371):

    llffhold_t = 8

    spectrum_dir = os.path.join(datadir, 'spectrum')
    spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
    image_names = [x.split('.')[0] for x in spt_names]

    len_image = len(image_names)

    random.seed(seed)
    np.random.seed(seed)

    test_index = np.arange(int(len_image))[:: llffhold_t]
    train_index_raw = np.array([j for j in np.arange(int(len_image)) if (j not in test_index)])
    train_len = len(train_index_raw)

    number_train = int(train_len * ratio)
    train_index = np.random.choice(train_index_raw, number_train, replace=False)

    print("\n [Llffhold split] Train: {}  Test: {}  (seed={})\n".format(
        number_train, len(test_index), seed))

    np.savetxt(train_path, train_index, fmt='%s')
    np.savetxt(test_path,  test_index,  fmt='%s')


def split_dataset_random(datadir, train_path, test_path, ratio=0.8, seed=8371):

    spectrum_dir = os.path.join(datadir, 'spectrum')
    spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])
    len_image = len(spt_names)

    random.seed(seed)
    np.random.seed(seed)

    all_indices = np.arange(len_image)
    np.random.shuffle(all_indices)

    num_train = int(len_image * ratio)
    train_index = np.sort(all_indices[:num_train])
    test_index  = np.sort(all_indices[num_train:])

    print("\n [Random split] Train: {}  Test: {}  Ratio: {:.0f}/{:.0f}  (seed={})\n".format(
        len(train_index), len(test_index), ratio*100, (1-ratio)*100, seed))

    np.savetxt(train_path, train_index, fmt='%s')
    np.savetxt(test_path,  test_index,  fmt='%s')


# ---- spectrum image loading ----
def readSpectrumImage(data_dir_path):
    data_infos = []

    tx_pos_path = os.path.join(data_dir_path, 'tx_pos.csv')
    tx_pos = pd.read_csv(tx_pos_path).values

    gateway_pos_path = os.path.join(data_dir_path, 'gateway_info.yml')
    spectrum_dir     = os.path.join(data_dir_path, 'spectrum')
    spt_names = sorted([f for f in os.listdir(spectrum_dir) if f.endswith('.png')])

    with open(gateway_pos_path) as f_loader:
        gateway_info = yaml.safe_load(f_loader)

        gateway_pos = gateway_info['gateway1']['position']
        gateway_quaternion = gateway_info['gateway1']['orientation']

    for image_idx, image_name in enumerate(spt_names):

        qvec = np.array(gateway_quaternion)
        rotation_matrix = torch.from_numpy(Rotation.from_quat(qvec).as_matrix()).float()

        tvec_rx = torch.from_numpy(np.array(gateway_pos)).float()

        tvec_tx = torch.from_numpy(np.array(tx_pos[image_idx])).float()

        image_path = os.path.join(spectrum_dir, os.path.basename(image_name))

        image_name_t = os.path.basename(image_path).split(".")[0]

        image = imageio.imread(image_path).astype(np.float32) / 255.0

        height = image.shape[0]
        width  = image.shape[1]

        resized_image = torch.from_numpy(np.array(image)).float()


        spec_info = SpectrumInfo(R=rotation_matrix,
                                 T_rx=tvec_rx,
                                 T_tx=tvec_tx,
                                 spectrum=resized_image,
                                 spectrum_path=image_path,
                                 spectrum_name=image_name_t,
                                 height=height,
                                 width=width)

        data_infos.append(spec_info)

    sys.stdout.write('\n')

    return data_infos


# ---- scene normalization ----
def getNorm_3d(specs_info, scale):

    def get_center_and_diag(gatewa_pos_t, cam_center):

        gatewa_pos_t = gatewa_pos_t.unsqueeze(1)

        cam_center = torch.stack(cam_center, dim=1)

        dists = torch.norm(cam_center - gatewa_pos_t, dim=0)
        radius = torch.max(dists) * scale

        deviations = cam_center - gatewa_pos_t

        positive_deviations = deviations.clone()
        negative_deviations = deviations.clone()

        positive_deviations[positive_deviations < 0] = 0
        negative_deviations[negative_deviations > 0] = 0

        max_positive = positive_deviations.max(dim=1).values
        max_negative = negative_deviations.min(dim=1).values.abs()

        epsilon = 1e-6
        max_positive[max_positive < epsilon] = 1.0
        max_negative[max_negative < epsilon] = 1.0

        return {"max_positive": max_positive * scale, "max_negative": max_negative * scale}, radius.item()

    cam_centers = []
    gatewa_pos  = specs_info[0].T_rx


    for cam in specs_info:
        cam_centers.append(cam.T_tx)


    diagonal, radius = get_center_and_diag(gatewa_pos, cam_centers)

    translate = -gatewa_pos

    return {"translate": translate, "radius": radius, "extent": diagonal}


def obtain_train_test_idx(args_model, len_list):

    path     = args_model.source_path
    llffhold = args_model.llffhold

    llffhold_flag = args_model.llffhold_flag

    train_index = os.path.join(path, args_model.train_index_path)
    test_index  = os.path.join(path, args_model.test_index_path)

    if llffhold_flag:

        print("\nUSING LLFFHOLD INDEX FILE\n")
        i_test = np.arange(int(len_list))[:: llffhold]
        i_train = np.array([j for j in np.arange(int(len_list)) if (j not in i_test)])

    elif "knn" in train_index:
        print("\nUSING KNN INDEX FILE\n")
        i_train = np.loadtxt(train_index, dtype=int)
        i_test  = np.loadtxt(test_index,  dtype=int)

    else:
        print("\nUSING RANDOM INDEX FILE\n")
        i_train = np.loadtxt(train_index, dtype=int)
        i_test  = np.loadtxt(test_index,  dtype=int)

    return i_train, i_test


# ---- scene info readers (per dataset) ----
def readRFSceneInfo(args_model):

    path         = args_model.source_path
    eval         = args_model.eval
    camera_scale     = args_model.camera_scale
    voxel_size_scale = args_model.voxel_size_scale

    ratio_train = args_model.ratio_train

    spectrums_infos_unsorted = readSpectrumImage(path)

    train_index_path = os.path.join(path, args_model.train_index_path)
    test_index_path  = os.path.join(path, args_model.test_index_path)

    split_method = getattr(args_model, 'split_method', 'random')
    seed = getattr(args_model, 'random_seed', 8371)
    if split_method == 'random':
        split_dataset_random(path, train_index_path, test_index_path, ratio=ratio_train, seed=seed)
    else:
        split_dataset_llffhold(path, train_index_path, test_index_path, ratio=ratio_train, seed=seed)

    i_train, i_test = obtain_train_test_idx(args_model, len(spectrums_infos_unsorted))

    spectrums_infos = sorted(spectrums_infos_unsorted.copy(), key = lambda x : int(x.spectrum_name))

    if eval:
        train_infos = [spectrums_infos[idx] for idx in i_train]
        test_infos  = [spectrums_infos[idx] for idx in i_test]

    else:
        train_infos = spectrums_infos
        test_infos = []

    nerf_normalization = getNorm_3d(spectrums_infos, camera_scale)

    ply_path = os.path.join(path, "points3D.ply")
    if ((not os.path.exists(ply_path)) or (args_model.gene_init_point)):

        receiver_pos = spectrums_infos[0].T_rx.numpy()

        frequency = float(getattr(args_model, 'frequency', 915.0e6))
        cube_size = round((3.00e8 / frequency) * voxel_size_scale, 2)

        max_points = getattr(args_model, 'max_init_points', 10000)
        num_pos = init_ply_v2(ply_path, receiver_pos, nerf_normalization["extent"], cube_size, max_points)

        print(f"\nInitialized point cloud: cube_size={cube_size}m, num_points={num_pos}\n")

    try:
        pcd = fetch_init_ply(ply_path)

    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_spectrums=train_infos,
                           test_spectrums=test_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)

    return scene_info


def readBLESceneInfo(args_model):

    path = args_model.source_path
    camera_scale = args_model.camera_scale
    voxel_size_scale = args_model.voxel_size_scale

    tx_pos = pd.read_csv(os.path.join(path, 'tx_pos.csv')).values.astype(np.float32)

    with open(os.path.join(path, 'gateway_position.yml')) as f:
        gw_dict = yaml.safe_load(f)
    gateway_names = list(gw_dict.keys())
    gateway_positions = np.array([gw_dict[name] for name in gateway_names], dtype=np.float32)

    gatewa_pos = torch.tensor(gateway_positions[0], dtype=torch.float32)

    cam_centers = [torch.tensor(tx_pos[i], dtype=torch.float32) for i in range(len(tx_pos))]

    gatewa_pos_t = gatewa_pos.unsqueeze(1)
    cam_center = torch.stack(cam_centers, dim=1)
    dists = torch.norm(cam_center - gatewa_pos_t, dim=0)
    radius = torch.max(dists) * camera_scale

    deviations = cam_center - gatewa_pos_t
    positive_deviations = deviations.clone()
    negative_deviations = deviations.clone()
    positive_deviations[positive_deviations < 0] = 0
    negative_deviations[negative_deviations > 0] = 0
    max_positive = positive_deviations.max(dim=1).values
    max_negative = negative_deviations.min(dim=1).values.abs()
    epsilon = 1e-6
    max_positive[max_positive < epsilon] = 1.0
    max_negative[max_negative < epsilon] = 1.0

    nerf_normalization = {
        "translate": -gatewa_pos,
        "radius": radius.item(),
        "extent": {"max_positive": max_positive * camera_scale,
                   "max_negative": max_negative * camera_scale},
    }

    ply_path = os.path.join(path, "points3D.ply")
    if (not os.path.exists(ply_path)) or args_model.gene_init_point:
        receiver_pos = gatewa_pos.numpy()
        frequency = float(getattr(args_model, 'frequency', 2.4e9))
        cube_size = round((3.00e8 / frequency) * voxel_size_scale, 2)
        max_points = getattr(args_model, 'max_init_points', 10000)
        num_pos = init_ply_v2(ply_path, receiver_pos, nerf_normalization["extent"], cube_size, max_points)
        print(f"\nInitialized point cloud: cube_size={cube_size}m, num_points={num_pos}\n")

    try:
        pcd = fetch_init_ply(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_spectrums=[],
                           test_spectrums=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCSISceneInfo(args_model):

    path = args_model.source_path
    camera_scale = args_model.camera_scale
    voxel_size_scale = args_model.voxel_size_scale

    with open(os.path.join(path, 'base-station.yml')) as f:
        bs = yaml.safe_load(f)
    antenna_pos = np.array(bs['base_station'], dtype=np.float32)

    gatewa_pos = torch.tensor(antenna_pos[0], dtype=torch.float32)

    max_positive = torch.tensor([1.0, 1.0, 1.0]) * camera_scale
    max_negative = torch.tensor([1.0, 1.0, 1.0]) * camera_scale
    radius = 2.0 * camera_scale

    nerf_normalization = {
        "translate": -gatewa_pos,
        "radius": radius,
        "extent": {"max_positive": max_positive, "max_negative": max_negative},
    }

    ply_path = os.path.join(path, "points3D.ply")
    if (not os.path.exists(ply_path)) or args_model.gene_init_point:
        receiver_pos = gatewa_pos.numpy()
        frequency = float(getattr(args_model, 'frequency', 2.4e9))
        cube_size = round((3.00e8 / frequency) * voxel_size_scale, 2)
        max_points = getattr(args_model, 'max_init_points', 10000)
        num_pos = init_ply_v2(ply_path, receiver_pos, nerf_normalization["extent"], cube_size, max_points)
        print(f"\nCSI point cloud: cube_size={cube_size}m, num_points={num_pos}\n")

    try:
        pcd = fetch_init_ply(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_spectrums=[],
                           test_spectrums=[],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# ---- point cloud initialization ----
def fetch_init_ply(path):

    plydata = PlyData.read(path)

    vertices = plydata['vertex']

    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    return BasicPointCloud(points=positions, attris=None, normals=normals)


# generate initial 3D grid centered at receiver, downsample if exceeding max_points
def init_ply_v2(ply_path, receiver_pos, camera_extent, cube_size, max_points=10000):

    dtype = [('x', 'f4'),  ('y', 'f4'),  ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')
            ]
    xyz = generate_cube_coordinates(receiver_pos, camera_extent, cube_size)

    # randomly subsample if grid exceeds max_points
    if max_points and xyz.shape[0] > max_points:
        indices = np.random.choice(xyz.shape[0], max_points, replace=False)
        indices.sort()
        xyz = xyz[indices]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)

    attributes = np.concatenate((xyz, normals), axis=1)

    elements[:] = list(map(tuple, attributes))

    vertex_element = PlyElement.describe(elements, 'vertex')

    ply_data = PlyData([vertex_element])

    ply_data.write(ply_path)

    return xyz.shape[0]


def generate_cube_coordinates(receiver_pos, camera_extent, cube_size):
    x_min = receiver_pos[0] - camera_extent["max_negative"][0].item()
    x_max = receiver_pos[0] + camera_extent["max_positive"][0].item()

    y_min = receiver_pos[1] - camera_extent["max_negative"][1].item()
    y_max = receiver_pos[1] + camera_extent["max_positive"][1].item()

    z_min = receiver_pos[2] - camera_extent["max_negative"][2].item()
    z_max = receiver_pos[2] + camera_extent["max_positive"][2].item()

    num_cubes_x = int(np.ceil((x_max - x_min) / cube_size))
    num_cubes_y = int(np.ceil((y_max - y_min) / cube_size))
    num_cubes_z = int(np.ceil((z_max - z_min) / cube_size))

    x_coords = np.linspace(x_min, x_max, num_cubes_x) if num_cubes_x > 1 else np.array([(x_min + x_max) / 2])
    y_coords = np.linspace(y_min, y_max, num_cubes_y) if num_cubes_y > 1 else np.array([(y_min + y_max) / 2])
    z_coords = np.linspace(z_min, z_max, num_cubes_z) if num_cubes_z > 1 else np.array([(z_min + z_max) / 2])

    x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    cube_points = np.vstack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()]).T

    return cube_points
