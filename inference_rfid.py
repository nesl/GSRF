# standard library
import os
import sys
import re
import csv
import json
from argparse import ArgumentParser

# third-party
import numpy as np
import torch
from tqdm import tqdm
import skimage
import lpips

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

# project
from arguments import ModelParams, PipelineParams, OptimizationParams, load_config
from utils.general_utils import safe_state
from scene import Scene, GaussianModel
from gaussian_renderer import render_rfid as render
from utils.data_painter import paint_spectrum, plot_metric_bar, plot_metric_cdf


# ---- metrics ----

def compute_mse(pred, gt):
    return np.mean((pred - gt) ** 2)


def compute_psnr(pred, gt, data_range=1.0):
    return skimage.metrics.peak_signal_noise_ratio(gt, pred, data_range=data_range)


def compute_ssim(pred, gt, data_range=1.0):
    return skimage.metrics.structural_similarity(gt, pred, data_range=data_range)


def compute_lpips_value(pred_tensor, gt_tensor, lpips_fn):
    pred_t = pred_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    gt_t = gt_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    pred_t = pred_t * 2.0 - 1.0
    gt_t = gt_t * 2.0 - 1.0
    with torch.no_grad():
        val = lpips_fn(pred_t, gt_t)
    return val.item()


# ---- inference ----

def testing(model_para_args,
            optimization_para_args,
            pipeline_para_args,
            checkpointpath_inference,
            output_dir
            ):

    # load scene and restore checkpoint
    gaussians = GaussianModel(model_para_args)

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

    render_dir = os.path.join(output_dir, "rendered")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    lpips_fn = lpips.LPIPS(net='alex').cuda()

    psnr_list = []
    mse_list = []
    ssim_list = []
    lpips_list = []
    names_list = []

    viewpoint_stack = scene.getTestSpectrums().copy()

    # run inference on test set
    for step_idx, viewpoint_cam in enumerate(tqdm(viewpoint_stack, desc="Inference")):

        render_pkg = render(viewpoint_cam, gaussians, pipeline_para_args, bg)

        spectrum = render_pkg["render"]

        gt_spectrum = viewpoint_cam.spectrum.cuda()
        spec_name = viewpoint_cam.spectrum_name

        pred_np = spectrum.detach().cpu().numpy()
        gt_np = gt_spectrum.cpu().numpy()

        # compute metrics
        psnr_val = compute_psnr(pred_np, gt_np)
        mse_val = compute_mse(pred_np, gt_np)
        ssim_val = compute_ssim(pred_np, gt_np)
        lpips_val = compute_lpips_value(spectrum.detach(), gt_spectrum, lpips_fn)

        psnr_list.append(psnr_val)
        mse_list.append(mse_val)
        ssim_list.append(ssim_val)
        lpips_list.append(lpips_val)
        names_list.append(spec_name)

        if step_idx % 50 == 0 or step_idx == len(viewpoint_stack) - 1:
            save_path = os.path.join(render_dir, f"{spec_name}.png")
            paint_spectrum(gt_np, pred_np, save_path=save_path)

    psnr_arr = np.array(psnr_list)
    mse_arr = np.array(mse_list)
    ssim_arr = np.array(ssim_list)
    lpips_arr = np.array(lpips_list)

    # save per-sample CSV
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    metrics_data = {
        'psnr':       ('PSNR_dB',       psnr_arr),
        'mse':        ('MSE',           mse_arr),
        'ssim':       ('SSIM',          ssim_arr),
        'lpips':      ('LPIPS',         lpips_arr),
    }

    for key, (col_name, arr) in metrics_data.items():
        csv_path = os.path.join(csv_dir, f"{key}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'gt_name', col_name])
            for i in range(len(arr)):
                writer.writerow([i, names_list[i], f"{arr[i]:.6f}"])

    # save summary JSON
    def make_stats(arr):
        return {
            "mean":   round(float(arr.mean()), 6),
            "std":    round(float(arr.std()), 6),
            "min":    round(float(arr.min()), 6),
            "p25":    round(float(np.percentile(arr, 25)), 6),
            "p50":    round(float(np.percentile(arr, 50)), 6),
            "p75":    round(float(np.percentile(arr, 75)), 6),
            "p90":    round(float(np.percentile(arr, 90)), 6),
            "p95":    round(float(np.percentile(arr, 95)), 6),
            "max":    round(float(arr.max()), 6),
        }

    summary = {
        "checkpoint": checkpointpath_inference,
        "num_test_samples": len(psnr_arr),
        "metrics": {
            "PSNR_dB":      make_stats(psnr_arr),
            "MSE":          make_stats(mse_arr),
            "SSIM":         make_stats(ssim_arr),
            "LPIPS":        make_stats(lpips_arr),
        }
    }

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # print results
    print(f"\n{'='*60}")
    print(f"  Test Results ({len(psnr_arr)} samples)")
    print(f"{'='*60}")
    print(f"  PSNR:   {psnr_arr.mean():.4f} +/- {psnr_arr.std():.4f} dB")
    print(f"  MSE:    {mse_arr.mean():.6f} +/- {mse_arr.std():.6f}")
    print(f"  SSIM:   {ssim_arr.mean():.4f} +/- {ssim_arr.std():.4f}")
    print(f"  LPIPS:  {lpips_arr.mean():.4f} +/- {lpips_arr.std():.4f}")
    print(f"{'='*60}")
    print(f"  Results saved to: {output_dir}")
    print(f"{'='*60}\n")

    # save plots
    plot_metric_bar(psnr_arr, 'PSNR (dB)', os.path.join(plot_dir, 'psnr_bar.png'))
    plot_metric_bar(mse_arr, 'MSE', os.path.join(plot_dir, 'mse_bar.png'))
    plot_metric_bar(ssim_arr, 'SSIM', os.path.join(plot_dir, 'ssim_bar.png'))
    plot_metric_bar(lpips_arr, 'LPIPS', os.path.join(plot_dir, 'lpips_bar.png'))

    plot_metric_cdf(psnr_arr, 'PSNR (dB)', os.path.join(plot_dir, 'psnr_cdf.png'))
    plot_metric_cdf(mse_arr, 'MSE', os.path.join(plot_dir, 'mse_cdf.png'))
    plot_metric_cdf(ssim_arr, 'SSIM', os.path.join(plot_dir, 'ssim_cdf.png'))
    plot_metric_cdf(lpips_arr, 'LPIPS', os.path.join(plot_dir, 'lpips_cdf.png'))

    print(f"  Plots saved to: {plot_dir}\n")


def main_inference():

    checkpoint_flag = True

    # parse config file first, then build full argument parser
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="arguments/configs/rfid/exp1.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    yaml_cfg = load_config(pre_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    parser = ArgumentParser(description="Inference script parameters")
    parser.add_argument("--config", type=str, default="arguments/configs/rfid/exp1.yaml", help="Path to YAML config file")

    model_para_cls = ModelParams(parser, yaml_cfg=yaml_cfg)
    optimization_para_cls = OptimizationParams(parser, yaml_cfg=yaml_cfg)
    pipeline_para_cls = PipelineParams(parser, yaml_cfg=yaml_cfg)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None, help="Override model path (default: logs/<dataset>/<exp_name>/)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")

    command_key_val = sys.argv[1:]
    args = parser.parse_args(command_key_val)

    default_iter_inference = args.iterations

    # set up data and model paths
    exp_name = args.exp_name
    dataset_name = args.dataset
    log_base_folder = args.log_base_folder
    input_data_folder = args.input_data_folder

    data_dir = os.path.join(input_data_folder, dataset_name)
    args.source_path = data_dir

    if args.model_path:
        model_path_dir = args.model_path
    else:
        model_path_dir = os.path.join(log_base_folder, dataset_name, exp_name)

    args.model_path = model_path_dir

    # locate checkpoint
    if checkpoint_flag:
        checkpoint_path = os.path.join(args.model_path, f"chkpnt{default_iter_inference}.pth")
        if os.path.exists(checkpoint_path):
            args.start_checkpoint = checkpoint_path

    print(f"\n\tData path: {args.source_path}\n")
    print(f"\tModel path: {args.model_path}\n")
    print(f"\tLoading checkpoint path: {args.start_checkpoint}\n")

    safe_state(args.quiet, random_seed, torch.device(args.data_device))
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.model_path, f"inference_iter{default_iter_inference}")

    # run inference
    testing(model_para_cls.extract(args),
            optimization_para_cls.extract(args),
            pipeline_para_cls.extract(args),
            args.start_checkpoint,
            output_dir
            )


if __name__ == '__main__':
    main_inference()
