import os
import torch
from random import randint
import sys
import uuid
from tqdm import tqdm
from argparse import ArgumentParser
import re
import numpy as np
import random
import time

from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from scene import Scene, GaussianModel
from gaussian_renderer import render

from scene.pos_encoder import Embedder
from utils.data_painter import paint_spectrum

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

import skimage
import lpips


def testing(model_para_args,
             optimization_para_args,
             pipeline_para_args,
             checkpointpath_inference
             ):
    
    gaussians = GaussianModel(model_para_args)

    tx_pos_encoder = Embedder(input_dims=3,
                              include_input=True,
                              max_freq_log2=9,
                              num_freqs=10,
                              log_sampling=True,
                              periodic_fns=[torch.sin, torch.cos])

    file_name = os.path.basename(checkpointpath_inference)
    match = re.search(r'(\d+)', file_name)
    extracted_number = match.group(1)

    scene = Scene(model_para_args, gaussians, load_iteration=extracted_number, shuffle=True)

    if checkpointpath_inference:
        (model_params, first_iter) = torch.load(checkpointpath_inference)
        gaussians.restore(model_params, optimization_para_args)


    bg_color = [1, 1, 1] if model_para_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    bg = torch.rand((3), device="cuda") if optimization_para_args.random_background else background
    
    viewpoint_stack = scene.getTestSpectrums().copy()
    for step_idx, viewpoint_cam in enumerate(viewpoint_stack):

        print(f"{step_idx + 1} / {len(viewpoint_stack)}")
        render_pkg = render(viewpoint_cam, gaussians, tx_pos_encoder, pipeline_para_args, bg)

        spectrum, _, _ = render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_spectrum = viewpoint_cam.spectrum.cpu().numpy()

        spectrum = spectrum.detach().cpu().numpy()

        psnr_value = skimage.metrics.peak_signal_noise_ratio(spectrum, gt_spectrum, data_range=1)

        print(f"PSNR: {psnr_value:.4f}")


def main_inference():

    checkpoint_flag = True
    random_seed_num_t = 1994

    run_testing_flag = True

    parser = ArgumentParser(description="Training script parameters")

    model_para_cls = ModelParams(parser)
    optimization_para_cls = OptimizationParams(parser)
    pipeline_para_cls = PipelineParams(parser)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true", default=False)

    parser.add_argument("--start_checkpoint", type=str, default=None)

    command_key_val = sys.argv[1: ]
    args = parser.parse_args(command_key_val)

    default_iter_inference = args.iterations

    # exp_name = args.exp_name
    exp_name = "gsrf_05212025181542"

    dataset_name = args.dataset

    log_base_folder = args.log_base_folder

    input_data_folder = args.input_data_folder

    data_dir = os.path.join(input_data_folder, dataset_name)
    args.source_path = data_dir

    basedir_out = os.path.join(log_base_folder, dataset_name)

    model_path_dir = os.path.join(basedir_out, exp_name)
    args.model_path = model_path_dir

    if checkpoint_flag:
        checkpoint_path = os.path.join(args.model_path, f"chkpnt{default_iter_inference}.pth")

        if os.path.exists(checkpoint_path):
            args.start_checkpoint = checkpoint_path

    print(f"\n\tData path: {args.source_path}\n")
    print(f"\tModel path: {args.model_path}\n")
    print(f"\tLoading checkpoint path: {args.start_checkpoint}\n")

    safe_state(args.quiet, random_seed_num_t, torch.device(args.data_device))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if run_testing_flag:

        testing(model_para_cls.extract(args), 
                optimization_para_cls.extract(args), 
                pipeline_para_cls.extract(args), 
                args.start_checkpoint
                )


if __name__ == '__main__':

    main_inference()


