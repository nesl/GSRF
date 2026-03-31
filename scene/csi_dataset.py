
import os
import random
import numpy as np
import yaml
import torch
from typing import NamedTuple, List


# ---- data structures ----
class CSISample(NamedTuple):
    uplink_re: torch.Tensor
    uplink_im: torch.Tensor
    downlink_re: torch.Tensor
    downlink_im: torch.Tensor
    sample_idx: int


class CSISceneInfo(NamedTuple):
    antenna_positions: torch.Tensor
    n_antennas: int
    n_subcarriers: int
    train_samples: List[CSISample]
    test_samples: List[CSISample]
    up_re_mean: float
    up_im_mean: float
    up_std: float
    down_re_mean: float
    down_im_mean: float
    down_std: float


# ---- CSI data loading and preprocessing ----
def load_csi_data(data_dir, split_method='random', ratio_train=0.8, seed=8371):

    csi_path = os.path.join(data_dir, 'csidata.npy')
    csi_raw = np.load(csi_path)
    num_samples = csi_raw.shape[0]

    uplink = csi_raw[:, :, :26]
    downlink = csi_raw[:, :, 26:]

    up_re = np.real(uplink).astype(np.float32)
    up_im = np.imag(uplink).astype(np.float32)
    down_re = np.real(downlink).astype(np.float32)
    down_im = np.imag(downlink).astype(np.float32)

    up_re_mean, up_im_mean = up_re.mean(), up_im.mean()
    up_std = np.sqrt(up_re.var() + up_im.var())
    down_re_mean, down_im_mean = down_re.mean(), down_im.mean()
    down_std = np.sqrt(down_re.var() + down_im.var())

    # normalize complex CSI values
    up_re = (up_re - up_re_mean) / (up_std + 1e-8)
    up_im = (up_im - up_im_mean) / (up_std + 1e-8)
    down_re = (down_re - down_re_mean) / (down_std + 1e-8)
    down_im = (down_im - down_im_mean) / (down_std + 1e-8)

    bs_path = os.path.join(data_dir, 'base-station.yml')
    with open(bs_path) as f:
        bs = yaml.safe_load(f)
    antenna_positions = torch.tensor(bs['base_station'], dtype=torch.float32)

    all_samples = []
    for i in range(num_samples):
        sample = CSISample(
            uplink_re=torch.tensor(up_re[i], dtype=torch.float32),
            uplink_im=torch.tensor(up_im[i], dtype=torch.float32),
            downlink_re=torch.tensor(down_re[i], dtype=torch.float32),
            downlink_im=torch.tensor(down_im[i], dtype=torch.float32),
            sample_idx=i,
        )
        all_samples.append(sample)

    train_path = os.path.join(data_dir, 'train_index.txt')
    test_path = os.path.join(data_dir, 'test_index.txt')

    # train/test split
    if split_method == 'random':
        random.seed(seed)
        np.random.seed(seed)
        all_indices = np.arange(num_samples)
        np.random.shuffle(all_indices)
        num_train = int(num_samples * ratio_train)
        train_indices = np.sort(all_indices[:num_train])
        test_indices = np.sort(all_indices[num_train:])
        np.savetxt(train_path, train_indices, fmt='%s')
        np.savetxt(test_path, test_indices, fmt='%s')
        print(f"\n [Random split] Train: {len(train_indices)}  Test: {len(test_indices)}  (seed={seed})\n")
    else:
        train_indices = np.loadtxt(train_path, dtype=int)
        test_indices = np.loadtxt(test_path, dtype=int)
        print(f"\n [File split] Train: {len(train_indices)}  Test: {len(test_indices)}\n")

    train_samples = [all_samples[i] for i in train_indices]
    test_samples = [all_samples[i] for i in test_indices]

    return CSISceneInfo(
        antenna_positions=antenna_positions,
        n_antennas=8,
        n_subcarriers=26,
        train_samples=train_samples,
        test_samples=test_samples,
        up_re_mean=float(up_re_mean),
        up_im_mean=float(up_im_mean),
        up_std=float(up_std),
        down_re_mean=float(down_re_mean),
        down_im_mean=float(down_im_mean),
        down_std=float(down_std),
    )
