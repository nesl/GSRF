
import os
import random
import numpy as np
import pandas as pd
import yaml
import torch
from typing import NamedTuple
from scene.dataset_readers import SpectrumInfo


# ---- RSSI <-> amplitude conversion ----
def rssi_to_amplitude(rssi, floor=-100.0):
    return 1.0 - (rssi / floor)


def amplitude_to_rssi(amplitude, floor=-100.0):
    return floor * (1.0 - amplitude)


# ---- BLE spectrum construction ----
def make_ble_spectrum_info(tx_pos, gw_pos, rssi_val, sample_idx, gw_idx, n_elevation=9, n_azimuth=36):
    amplitude = rssi_to_amplitude(rssi_val)

    n_el, n_az = n_elevation, n_azimuth
    spectrum = torch.full((n_el, n_az), amplitude, dtype=torch.float32)

    R = torch.eye(3, dtype=torch.float32)

    return SpectrumInfo(
        R=R,
        T_rx=torch.tensor(gw_pos, dtype=torch.float32),
        T_tx=torch.tensor(tx_pos, dtype=torch.float32),
        spectrum=spectrum,
        spectrum_path=f"ble_{sample_idx:05d}_gw{gw_idx:02d}",
        spectrum_name=f"{sample_idx+1:05d}",
        height=n_el,
        width=n_az,
    )


# ---- per-gateway data loading and splitting ----
def load_ble_per_gateway(data_dir, split_method='random', ratio_train=0.8, seed=8371,
                         n_elevation=9, n_azimuth=36):
    rssi_df = pd.read_csv(os.path.join(data_dir, 'gateway_rssi.csv'))
    rssi_values = rssi_df.values.astype(np.float32)
    gateway_names = list(rssi_df.columns)
    num_gateways = len(gateway_names)

    with open(os.path.join(data_dir, 'gateway_position.yml')) as f:
        gw_dict = yaml.safe_load(f)
    gateway_positions = np.array([gw_dict[name] for name in gateway_names], dtype=np.float32)

    tx_pos = pd.read_csv(os.path.join(data_dir, 'tx_pos.csv')).values.astype(np.float32)
    num_samples = tx_pos.shape[0]

    train_path = os.path.join(data_dir, 'train_index.txt')
    test_path = os.path.join(data_dir, 'test_index.txt')

    if split_method == 'random':
        random.seed(seed)
        np.random.seed(seed)
        all_indices = np.arange(num_samples)
        np.random.shuffle(all_indices)
        num_train = int(num_samples * ratio_train)
        train_set = set(all_indices[:num_train].tolist())
        test_set = set(all_indices[num_train:].tolist())
        np.savetxt(train_path, np.sort(list(train_set)), fmt='%s')
        np.savetxt(test_path, np.sort(list(test_set)), fmt='%s')
        print(f"\n [Random split] Train TX: {len(train_set)}  Test TX: {len(test_set)}  (seed={seed})\n")
    else:
        train_set = set(np.loadtxt(train_path, dtype=int).tolist())
        test_set = set(np.loadtxt(test_path, dtype=int).tolist())
        print(f"\n [File split] Train TX: {len(train_set)}  Test TX: {len(test_set)}\n")

    per_gw_data = {}

    for gw_idx in range(num_gateways):
        gw_pos = gateway_positions[gw_idx]
        gw_train = []
        gw_test = []

        for sample_idx in range(num_samples):
            rssi_val = rssi_values[sample_idx, gw_idx]
            if rssi_val <= -100.0:
                continue

            info = make_ble_spectrum_info(
                tx_pos[sample_idx], gw_pos, rssi_val, sample_idx, gw_idx,
                n_elevation=n_elevation, n_azimuth=n_azimuth
            )

            if sample_idx in train_set:
                gw_train.append(info)
            elif sample_idx in test_set:
                gw_test.append(info)

        if gw_train or gw_test:
            per_gw_data[gw_idx] = (gw_train, gw_test)

        print(f"  {gateway_names[gw_idx]}: train={len(gw_train)}, test={len(gw_test)}")

    return per_gw_data, gateway_names
