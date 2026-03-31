
# standard library
import os
import sys
import json
import csv
import time
from random import randint
from argparse import ArgumentParser

# third-party
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# project
from arguments import ModelParams, PipelineParams, OptimizationParams, load_config
from utils.general_utils import safe_state
from scene import Scene, GaussianModel
from gaussian_renderer import render_rfid as render
from scene.ble_dataset import load_ble_per_gateway, amplitude_to_rssi
from utils.train_utils import setup_fle_only_optimizer, init_gaussians_from_reference


# ---- per-gateway evaluation ----

def evaluate_gateway(gaussians, pipe_args, bg, test_samples,
                     gw_output_dir, gw_name, gw_idx, iteration):

    all_mae, all_pred, all_gt = [], [], []

    with torch.no_grad():
        for viewpoint in test_samples:
            render_pkg = render(viewpoint, gaussians, pipe_args, bg)
            spectrum = render_pkg["render"]
            pred_amp = spectrum.mean().cpu().item()
            gt_amp = viewpoint.spectrum.mean().item()

            pred_rssi = amplitude_to_rssi(pred_amp)
            gt_rssi = amplitude_to_rssi(gt_amp)
            mae = abs(pred_rssi - gt_rssi)

            all_mae.append(mae)
            all_pred.append(pred_rssi)
            all_gt.append(gt_rssi)

    mae_arr = np.array(all_mae)
    pred_arr = np.array(all_pred)
    gt_arr = np.array(all_gt)

    result = {
        "gateway": gw_name,
        "gateway_idx": gw_idx,
        "iterations": iteration,
        "num_test": len(test_samples),
        "num_gaussians": gaussians.get_xyz.shape[0],
        "MAE_dBm": {
            "mean": round(float(mae_arr.mean()), 4),
            "std": round(float(mae_arr.std()), 4),
            "min": round(float(mae_arr.min()), 4),
            "p25": round(float(np.percentile(mae_arr, 25)), 4),
            "p50": round(float(np.percentile(mae_arr, 50)), 4),
            "p75": round(float(np.percentile(mae_arr, 75)), 4),
            "p90": round(float(np.percentile(mae_arr, 90)), 4),
            "p95": round(float(np.percentile(mae_arr, 95)), 4),
            "max": round(float(mae_arr.max()), 4),
        },
    }

    with open(os.path.join(gw_output_dir, f"result_iter{iteration}.json"), 'w') as f:
        json.dump(result, f, indent=2)

    csv_path = os.path.join(gw_output_dir, f"predictions_iter{iteration}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["index", "gt_rssi", "pred_rssi", "mae_dBm"])
        for i in range(len(mae_arr)):
            writer.writerow([i, f"{gt_arr[i]:.2f}", f"{pred_arr[i]:.2f}", f"{mae_arr[i]:.2f}"])

    return result


# ---- per-gateway training ----

def train_one_gateway(gw_idx, gw_name, train_samples, test_samples,
                      model_para_args, opt_args, pipe_args, gw_output_dir,
                      test_iterations, ref_gaussians=None):
    """
    Train one gateway model.
    - ref_gaussians=None (first gateway): train all parameters from scratch
    - ref_gaussians provided (subsequent gateways): reuse geometry, only train FLE coefficients
    """

    freeze_geometry = ref_gaussians is not None

    if freeze_geometry:
        model_para_args.gene_init_point = False

    gaussians = GaussianModel(model_para_args)
    scene = Scene(model_para_args, gaussians, load_iteration=None, shuffle=True)
    scene.train_set = train_samples
    scene.test_set = test_samples

    if freeze_geometry:
        init_gaussians_from_reference(gaussians, ref_gaussians, model_para_args)
        setup_fle_only_optimizer(gaussians, opt_args)
        print(f"    (reusing geometry from first gateway, training FLE only)")
    else:
        gaussians.training_setup(opt_args)

    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    viewpoint_stack = None
    ema_loss = 0.0
    results_per_iter = {}

    progress_bar = tqdm(range(opt_args.iterations),
                        desc=f"  {gw_name}", leave=False)

    # training loop
    for iteration in range(1, opt_args.iterations + 1):

        if not freeze_geometry:
            gaussians.update_learning_rate(iteration)

        # progressively increase FLE degree
        fle_ramp = getattr(model_para_args, '_fle_degree_ramp', 500)
        if iteration % fle_ramp == 0:
            gaussians.oneup_fle_degree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainSpectrums().copy()

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # forward pass and MSE loss on mean amplitude
        render_pkg = render(viewpoint_cam, gaussians, pipe_args, bg)
        spectrum = render_pkg["render"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        pred_amp = spectrum.mean()
        gt_amp = viewpoint_cam.spectrum.cuda().mean()

        loss = (pred_amp - gt_amp) ** 2
        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.6f}"})
                progress_bar.update(10)

            if iteration == opt_args.iterations:
                progress_bar.close()

            # densification only for first gateway (full training)
            if not freeze_geometry:
                if iteration < opt_args.densify_until_iter:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(gaussians.get_xyz, visibility_filter)

                    if iteration >= opt_args.densify_from_iter \
                            and iteration % opt_args.densification_interval == 0:
                        size_threshold = opt_args.raddi_size_threshold \
                            if iteration > opt_args.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt_args.densify_grad_threshold,
                                                   opt_args.min_attenuation_threshold,
                                                   scene.cameras_extent,
                                                   size_threshold)

                    if iteration % opt_args.opacity_reset_interval == 0:
                        gaussians.reset_attenuation()

            # save checkpoint and evaluate
            if iteration in test_iterations:
                scene.save(iteration)
                ckpt_path = os.path.join(gw_output_dir, f"chkpnt{iteration}.pth")
                torch.save((gaussians.capture(), iteration), ckpt_path)

                result = evaluate_gateway(gaussians, pipe_args, bg,
                                         test_samples, gw_output_dir, gw_name, gw_idx, iteration)
                results_per_iter[iteration] = result

            if iteration < opt_args.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    return gaussians, results_per_iter


# ---- main: train all gateways ----

def main():
    # parse config file first, then build full argument parser
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="arguments/configs/ble/exp1.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    yaml_cfg = load_config(pre_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    parser = ArgumentParser(description="BLE RSSI Training (per-gateway)")
    parser.add_argument("--config", type=str, default="arguments/configs/ble/exp1.yaml")

    model_para_cls = ModelParams(parser, yaml_cfg=yaml_cfg)
    optimization_para_cls = OptimizationParams(parser, yaml_cfg=yaml_cfg)
    pipeline_para_cls = PipelineParams(parser, yaml_cfg=yaml_cfg)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true", default=False)

    args = parser.parse_args()

    # set up data and output paths
    dataset_name = args.dataset
    exp_name = args.exp_name
    data_dir = os.path.join(args.input_data_folder, dataset_name)
    args.source_path = data_dir

    # logs/<dataset>/<exp_name>/
    model_path = os.path.join(args.log_base_folder, dataset_name, exp_name)
    os.makedirs(model_path, exist_ok=True)
    args.model_path = model_path

    args.densify_until_iter = args.iterations // 2
    args.position_lr_max_steps = args.iterations

    # build test iterations: 7000, then every 10k, plus final
    iters = args.iterations
    default_iter = 7000
    test_iters = [default_iter]
    for i in range(10000, iters, 10000):
        if i > default_iter:
            test_iters.append(i)
    test_iters.append(iters)
    test_iters = sorted(set(test_iters))

    safe_state(args.quiet, random_seed, torch.device(args.data_device))

    # load per-gateway BLE data
    split_method = getattr(args, 'split_method', 'random')
    ratio_train = getattr(args, 'ratio_train', 0.8)
    n_azimuth = getattr(args, 'n_azimuth', 36)
    n_elevation = getattr(args, 'n_elevation', 9)
    per_gw_data, gateway_names = load_ble_per_gateway(
        data_dir, split_method=split_method, ratio_train=ratio_train, seed=random_seed,
        n_elevation=n_elevation, n_azimuth=n_azimuth
    )

    print(f"\n{'='*60}")
    print(f"  BLE RSSI Training (per-gateway models)")
    print(f"  Data: {data_dir}")
    print(f"  Output: {model_path}")
    print(f"  Gateways: {len(per_gw_data)}")
    print(f"  Iterations per gateway: {args.iterations}")
    print(f"  Test at iterations: {test_iters}")
    print(f"{'='*60}\n")

    with open(os.path.join(model_path, "config.json"), 'w') as f:
        config_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        json.dump(config_dict, f, indent=2, default=str)

    model_args = model_para_cls.extract(args)
    opt_args = optimization_para_cls.extract(args)
    pipe_args = pipeline_para_cls.extract(args)

    # train one model per gateway
    # first valid gateway: full training (geometry + FLE)
    # subsequent gateways: reuse geometry, only train FLE coefficients
    all_gw_results = {}
    ref_gaussians = None

    for gw_idx, gw_name in enumerate(gateway_names):
        if gw_idx not in per_gw_data:
            print(f"\n  Skipping {gw_name} — no valid samples")
            continue

        train_samples, test_samples = per_gw_data[gw_idx]

        if len(train_samples) < 10 or len(test_samples) < 5:
            print(f"\n  Skipping {gw_name} — too few samples "
                  f"(train={len(train_samples)}, test={len(test_samples)})")
            continue

        mode = "full" if ref_gaussians is None else "FLE-only"
        print(f"\n[{gw_idx+1}/{len(gateway_names)}] Training {gw_name} [{mode}] "
              f"(train={len(train_samples)}, test={len(test_samples)})")

        gw_output_dir = os.path.join(model_path, gw_name)
        os.makedirs(gw_output_dir, exist_ok=True)

        model_args.model_path = gw_output_dir
        model_args.gene_init_point = True

        gaussians, results_per_iter = train_one_gateway(
            gw_idx, gw_name, train_samples, test_samples,
            model_args, opt_args, pipe_args, gw_output_dir, test_iters,
            ref_gaussians=ref_gaussians
        )

        # save first gateway's trained Gaussians as reference for subsequent gateways
        if ref_gaussians is None:
            ref_gaussians = gaussians

        final_iter = max(results_per_iter.keys())
        final_result = results_per_iter[final_iter]
        gw_mae = final_result['MAE_dBm']
        print(f"  {gw_name}: MAE = {gw_mae['mean']:.2f} dBm "
              f"(median: {gw_mae['p50']:.2f}, p90: {gw_mae['p90']:.2f}, #G: {final_result['num_gaussians']})")

        all_gw_results[gw_name] = results_per_iter

    # aggregate results across all gateways
    if all_gw_results:
        for test_iter in test_iters:
            all_sample_mae = []
            iter_gw_results = []

            for gw_name, results_per_iter in all_gw_results.items():
                if test_iter not in results_per_iter:
                    continue
                iter_gw_results.append(results_per_iter[test_iter])

                gw_dir = os.path.join(model_path, gw_name)
                csv_path = os.path.join(gw_dir, f"predictions_iter{test_iter}.csv")
                if os.path.exists(csv_path):
                    rows = list(csv.DictReader(open(csv_path)))
                    all_sample_mae.extend([float(row['mae_dBm']) for row in rows])

            if not all_sample_mae:
                continue

            all_mae_arr = np.array(all_sample_mae)

            summary = {
                "iterations": test_iter,
                "num_gateways_trained": len(iter_gw_results),
                "num_test_samples_total": len(all_sample_mae),
                "overall_MAE_dBm": {
                    "mean": round(float(all_mae_arr.mean()), 4),
                    "std": round(float(all_mae_arr.std()), 4),
                    "min": round(float(all_mae_arr.min()), 4),
                    "p25": round(float(np.percentile(all_mae_arr, 25)), 4),
                    "p50": round(float(np.percentile(all_mae_arr, 50)), 4),
                    "p75": round(float(np.percentile(all_mae_arr, 75)), 4),
                    "p90": round(float(np.percentile(all_mae_arr, 90)), 4),
                    "p95": round(float(np.percentile(all_mae_arr, 95)), 4),
                    "max": round(float(all_mae_arr.max()), 4),
                },
                "per_gateway": iter_gw_results,
            }

            with open(os.path.join(model_path, f"summary_iter{test_iter}.json"), 'w') as f:
                json.dump(summary, f, indent=2)

            all_csv_path = os.path.join(model_path, f"all_predictions_iter{test_iter}.csv")
            with open(all_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["gateway", "index", "gt_rssi", "pred_rssi", "mae_dBm"])
                for r in iter_gw_results:
                    gw_dir = os.path.join(model_path, r['gateway'])
                    gw_csv = os.path.join(gw_dir, f"predictions_iter{test_iter}.csv")
                    if os.path.exists(gw_csv):
                        for row in csv.DictReader(open(gw_csv)):
                            writer.writerow([r['gateway'], row['index'],
                                            row['gt_rssi'], row['pred_rssi'], row['mae_dBm']])

            print(f"\n{'='*60}")
            print(f"  BLE Results @ iter {test_iter} — {len(iter_gw_results)} gateways, {len(all_sample_mae)} samples")
            print(f"{'='*60}")
            print(f"  MAE:    {all_mae_arr.mean():.2f} +/- {all_mae_arr.std():.2f} dBm")
            print(f"  Median: {np.median(all_mae_arr):.2f} dBm")
            print(f"  P90:    {np.percentile(all_mae_arr, 90):.2f} dBm")

        print(f"\n  Results: {model_path}\n")


if __name__ == '__main__':
    main()
