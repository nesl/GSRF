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

from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.general_utils import safe_state
from scene import Scene, GaussianModel
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim, psnr, l2_loss, fourier_loss

from scene.pos_encoder import Embedder
from utils.train_utils import training_report, prepare_output_and_logger

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


def training(model_para_args,
             optimization_para_args,
             pipeline_para_args,
             testing_iterations,     # for testing
             saving_iterations,      # for saving point cloud data
             checkpoint_iterations,  # for saving training parameters
             checkpoint,             # for loading trained gaussian model
             debug_from):
        

    first_iter = 0
    tb_writer = prepare_output_and_logger(model_para_args)

    gaussians = GaussianModel(model_para_args)

    tx_pos_encoder = Embedder(input_dims=3,
                              include_input=True,
                              max_freq_log2=model_para_args.max_freq_log2,
                              num_freqs=model_para_args.num_freqs,
                              log_sampling=True,
                              periodic_fns=[torch.sin, torch.cos])
    
    if not checkpoint:  # if checkpoint is None

        scene = Scene(model_para_args, 
                      gaussians, 
                      load_iteration=None, 
                      shuffle=True)

    else:
        file_name = os.path.basename(checkpoint)
        match = re.search(r'(\d+)', file_name)
        extracted_number = match.group(1)  # for load point cloud data

        scene = Scene(model_para_args, 
                      gaussians, 
                      load_iteration=extracted_number, 
                      shuffle=True)

    gaussians.training_setup(optimization_para_args)

    if checkpoint:
        print("\nLoading saved trained model from path: {}\n".format(checkpoint))
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, optimization_para_args)

    # not used in CUDA
    bg_color = [1, 1, 1] if model_para_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if optimization_para_args.random_background else background

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, optimization_para_args.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, optimization_para_args.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainSpectrums().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipeline_para_args.debug = True

        render_pkg = render(viewpoint_cam, gaussians, tx_pos_encoder, pipeline_para_args, bg)

        spectrum, visibility_filter, radii = \
            render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        gt_spectrum = viewpoint_cam.spectrum.cuda()

        Ll1 = l2_loss(spectrum, gt_spectrum)

        pred = spectrum.unsqueeze(dim=0)
        gt   = gt_spectrum.unsqueeze(dim=0)

        ssim_loss = 1.0 - ssim(pred, gt)
        fourier_lo = fourier_loss(pred, gt)

        loss = (1.0 - optimization_para_args.lambda_dssim - optimization_para_args.lambda_dfourier) * Ll1 \
            + optimization_para_args.lambda_dssim * ssim_loss \
            + optimization_para_args.lambda_dfourier * fourier_lo
        
        # will call cuda backward
        loss.backward()

        iter_end.record()

        with torch.no_grad():

            # Exponential Moving Average (EMA) loss only for Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)

            if iteration == optimization_para_args.iterations:
                progress_bar.close()

            training_report(tb_writer,
                            iteration, 
                            Ll1, 
                            loss, 
                            l1_loss, 
                            iter_start.elapsed_time(iter_end),
                            testing_iterations,
                            scene, 
                            render, 
                            tx_pos_encoder,
                            pipeline_para_args, 
                            model_para_args, 
                            bg
                            )
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians Points".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < optimization_para_args.densify_until_iter:  # 150_000

                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], 
                                                                     radii[visibility_filter])
                
                gaussians.add_densification_stats(gaussians.get_xyz, visibility_filter)

                if iteration >= optimization_para_args.densify_from_iter \
                    and iteration % optimization_para_args.densification_interval == 0:

                    size_threshold = optimization_para_args.raddi_size_threshold \
                        if iteration > optimization_para_args.opacity_reset_interval else None

                    gaussians.densify_and_prune(optimization_para_args.densify_grad_threshold, 
                                                optimization_para_args.min_attenuation_threshold, 
                                                scene.cameras_extent, 
                                                size_threshold)
                    
                if iteration % optimization_para_args.opacity_reset_interval == 0 or \
                    (model_para_args.white_background and iteration == optimization_para_args.densify_from_iter):
                    
                    gaussians.reset_attenuation()

            # Optimizer step
            if iteration < optimization_para_args.iterations:

                # Update parameters and reset gradients
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                chkpnt_path = os.path.join(scene.model_path, f"chkpnt{str(iteration)}.pth")

                print("\n[ITER {}] Saving Checkpoint in Path: {}".format(iteration, chkpnt_path))

                torch.save((gaussians.capture(), iteration), chkpnt_path)


if __name__ == '__main__':

    checkpoint_flag = False
    random_seed_num_t = 1994

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    model_para_cls = ModelParams(parser)
    optimization_para_cls = OptimizationParams(parser)
    pipeline_para_cls = PipelineParams(parser)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true", default=False)

    """
    # The nargs argument with a value of "+" indicates that the command-line argument --test_iterations 
    #   can accept one or more values. This means when the script is run from the command line, 
    #   it can be followed by multiple integer values, like --test_iterations 1000 2000 3000
    # It will default to a list of integers [7000, 30000]
    """
    # conduting test
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])

    # saving point cloud data
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])

    # saving traning parameters and satuses
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])

    """
    # check point path for load trained gaussian model
    # the iteratino number "such as 7000" also used to load point cloud data
    # So, we need to make sure trained gaussian model and point cloud data exist both.
    """
    parser.add_argument("--start_checkpoint", type=str, default=None)

    command_key_val = sys.argv[1: ]
    args = parser.parse_args(command_key_val)

    default_iter = 7_000

    dataset_name = args.dataset

    exp_name = args.exp_name

    log_base_folder = args.log_base_folder

    input_data_folder = args.input_data_folder

    data_dir = os.path.join(input_data_folder, dataset_name)
    args.source_path = data_dir

    basedir_out = os.path.join(log_base_folder, dataset_name)
    os.makedirs(basedir_out, exist_ok=True)

    model_path_dir = os.path.join(basedir_out, exp_name)

    os.makedirs(model_path_dir, exist_ok=True)
    args.model_path = model_path_dir

    os.makedirs(os.path.join(model_path_dir, "plot_xyz"), exist_ok=True)

    # make sure to save the final point cloud data
    args.save_iterations.append(default_iter)
    args.save_iterations.append(args.iterations)           # save_iterations = [7000, 30000]

    # args.checkpoint_iterations.append(args.iterations)   # checkpoint_iterations = [7000, 30000]
    args.checkpoint_iterations = args.save_iterations

    args.test_iterations = args.save_iterations
    
    args.densify_until_iter    = args.iterations // 2
    args.position_lr_max_steps = args.iterations

    if checkpoint_flag:
        checkpoint_path = os.path.join(args.model_path, f"chkpnt{args.checkpoint_iterations[0]}.pth")
        if os.path.exists(checkpoint_path):
            args.start_checkpoint = checkpoint_path

    print(f"\n\tData path: {args.source_path}\n")
    print(f"\tModel path: {args.model_path}\n")
    print(f"\tLoading checkpoint path: {args.start_checkpoint}\n")

    # Initialize system state
    safe_state(args.quiet, random_seed_num_t, torch.device(args.data_device))

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    f_path = os.path.join(args.model_path, "config.yml")
    with open(f_path, "w") as file:
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")

    training(model_para_cls.extract(args), 
             optimization_para_cls.extract(args), 
             pipeline_para_cls.extract(args), 
             args.test_iterations, 
             args.save_iterations, 
             args.checkpoint_iterations, 
             args.start_checkpoint, 
             args.debug_from
            )

    print("\nTraining complete\n")

        