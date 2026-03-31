# standard library
import os
import sys
import re
import warnings
from random import randint
from argparse import ArgumentParser

# third-party
import torch
from tqdm import tqdm

# project
from arguments import ModelParams, PipelineParams, OptimizationParams, load_config
from utils.general_utils import safe_state
from scene import Scene, GaussianModel
from gaussian_renderer import render_rfid as render
from utils.loss_utils import l1_loss, ssim, psnr, l2_loss, fourier_loss
from utils.train_utils import training_report, prepare_output_and_logger

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")


def training(model_para_args,
             optimization_para_args,
             pipeline_para_args,
             testing_iterations,
             saving_iterations,
             checkpoint_iterations,
             checkpoint,
             debug_from):

    # initialize scene and gaussians
    first_iter = 0
    tb_writer = prepare_output_and_logger(model_para_args)

    gaussians = GaussianModel(model_para_args)

    if not checkpoint:
        scene = Scene(model_para_args,
                      gaussians,
                      load_iteration=None,
                      shuffle=True)
    else:
        file_name = os.path.basename(checkpoint)
        match = re.search(r'(\d+)', file_name)
        extracted_number = match.group(1)

        scene = Scene(model_para_args,
                      gaussians,
                      load_iteration=extracted_number,
                      shuffle=True)

    gaussians.training_setup(optimization_para_args)

    # restore from checkpoint
    if checkpoint:
        print("\nLoading saved trained model from path: {}\n".format(checkpoint))
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, optimization_para_args)

    bg_color = [1, 1, 1] if model_para_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    bg = torch.rand((3), device="cuda") if optimization_para_args.random_background else background

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end   = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0

    # training loop
    progress_bar = tqdm(range(first_iter, optimization_para_args.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, optimization_para_args.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # progressively increase FLE degree
        fle_ramp = getattr(model_para_args, '_fle_degree_ramp', 500)
        if iteration % fle_ramp == 0:
            gaussians.oneup_fle_degree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainSpectrums().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        if (iteration - 1) == debug_from:
            pipeline_para_args.debug = True

        # forward pass
        render_pkg = render(viewpoint_cam, gaussians, pipeline_para_args, bg)

        spectrum, visibility_filter, radii = \
            render_pkg["render"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_spectrum = viewpoint_cam.spectrum.cuda()

        # compute loss: L1 + SSIM + Fourier
        Ll1 = l1_loss(spectrum, gt_spectrum)

        pred = spectrum.unsqueeze(0).unsqueeze(0)
        gt   = gt_spectrum.unsqueeze(0).unsqueeze(0)

        ssim_loss = 1.0 - ssim(pred, gt)

        Lfourier = fourier_loss(spectrum, gt_spectrum)

        lambda_ssim = optimization_para_args.lambda_dssim
        lambda_fourier = optimization_para_args.lambda_dfourier
        loss = (1.0 - lambda_ssim - lambda_fourier) * Ll1 \
            + lambda_ssim * ssim_loss \
            + lambda_fourier * Lfourier

        loss.backward()

        iter_end.record()

        with torch.no_grad():

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
                            pipeline_para_args,
                            model_para_args,
                            bg
                            )

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians Points".format(iteration))
                scene.save(iteration)

            # densification and pruning
            if iteration < optimization_para_args.densify_until_iter:

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

            if iteration < optimization_para_args.iterations:

                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                chkpnt_path = os.path.join(scene.model_path, f"chkpnt{str(iteration)}.pth")

                print("\n[ITER {}] Saving Checkpoint in Path: {}".format(iteration, chkpnt_path))

                torch.save((gaussians.capture(), iteration), chkpnt_path)



if __name__ == '__main__':

    checkpoint_flag = False

    # parse config file first, then build full argument parser
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="arguments/configs/rfid/exp1.yaml", help="Path to YAML config file")
    pre_args, _ = pre_parser.parse_known_args()

    yaml_cfg = load_config(pre_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, default="arguments/configs/rfid/exp1.yaml", help="Path to YAML config file")

    model_para_cls = ModelParams(parser, yaml_cfg=yaml_cfg)
    optimization_para_cls = OptimizationParams(parser, yaml_cfg=yaml_cfg)
    pipeline_para_cls = PipelineParams(parser, yaml_cfg=yaml_cfg)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)

    command_key_val = sys.argv[1:]
    args = parser.parse_args(command_key_val)

    default_iter = 7_000

    # set up data and output paths
    dataset_name = args.dataset
    exp_name = args.exp_name
    log_base_folder = args.log_base_folder
    input_data_folder = args.input_data_folder

    data_dir = os.path.join(input_data_folder, dataset_name)
    args.source_path = data_dir

    model_path_dir = os.path.join(log_base_folder, dataset_name, exp_name)
    os.makedirs(model_path_dir, exist_ok=True)
    args.model_path = model_path_dir

    # build save iterations: 7000, then every 10k, plus final
    save_iters = [default_iter]
    for i in range(10000, args.iterations, 10000):
        if i > default_iter:
            save_iters.append(i)
    save_iters.append(args.iterations)
    save_iters = sorted(set(save_iters))

    args.save_iterations = save_iters
    args.checkpoint_iterations = save_iters
    args.test_iterations = save_iters

    args.densify_until_iter    = args.iterations // 2
    args.position_lr_max_steps = args.iterations

    # optionally resume from checkpoint
    if checkpoint_flag:
        checkpoint_path = os.path.join(args.model_path, f"chkpnt{args.checkpoint_iterations[0]}.pth")
        if os.path.exists(checkpoint_path):
            args.start_checkpoint = checkpoint_path

    print(f"\n\tData path: {args.source_path}\n")
    print(f"\tModel path: {args.model_path}\n")
    print(f"\tLoading checkpoint path: {args.start_checkpoint}\n")

    safe_state(args.quiet, random_seed, torch.device(args.data_device))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # save config to output directory
    f_path = os.path.join(args.model_path, "config.yml")
    with open(f_path, "w") as file:
        for key, value in vars(args).items():
            file.write(f"{key}: {value}\n")

    # run training
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
