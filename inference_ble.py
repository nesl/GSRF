
# standard library
import os
import json
import csv
import time
from argparse import ArgumentParser

# third-party
import numpy as np
import torch
from tqdm import tqdm

# project
from arguments import ModelParams, PipelineParams, OptimizationParams, load_config
from utils.general_utils import safe_state
from scene import Scene, GaussianModel
from gaussian_renderer import render_rfid as render
from scene.ble_dataset import load_ble_per_gateway, amplitude_to_rssi
from utils.train_utils import load_gaussians_from_checkpoint
from utils.data_painter import plot_metric_cdf, plot_metric_histogram


# ---- per-gateway evaluation ----

def evaluate_gateway(gw_idx, gw_name, test_samples, model_args, opt_args, pipe_args, ckpt_path):
    """Load checkpoint and evaluate on test samples for one gateway."""

    # load model from checkpoint
    gaussians = GaussianModel(model_args)
    load_gaussians_from_checkpoint(gaussians, ckpt_path)

    all_mae = []
    all_pred = []
    all_gt = []
    infer_times = []

    with torch.no_grad():
        for viewpoint in test_samples:
            t0 = time.time()

            render_pkg = render(viewpoint, gaussians, pipe_args)
            pred_amp = render_pkg["render"].mean().cpu().item()

            infer_times.append((time.time() - t0) * 1000)

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
    infer_arr = np.array(infer_times)

    result = {
        "gateway": gw_name,
        "gateway_idx": gw_idx,
        "checkpoint": ckpt_path,
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
        "Infer_ms": {
            "mean": round(float(infer_arr.mean()), 2),
            "std": round(float(infer_arr.std()), 2),
        },
    }

    return result, pred_arr, gt_arr, mae_arr


# ---- main: standalone inference on saved checkpoints ----

def main():
    # parse args: config + optional model_path override
    parser = ArgumentParser(description="BLE RSSI Inference")
    parser.add_argument("--config", type=str, default="arguments/configs/ble/exp1.yaml")
    parser.add_argument("--model_path", type=str, default=None, help="Override model path (default: logs/<dataset>/<exp_name>/)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--iter", type=int, default=None, help="Checkpoint iteration (default: latest)")

    cmd_args = parser.parse_args()

    yaml_cfg = load_config(cmd_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    base_parser = ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=cmd_args.config)
    model_cls = ModelParams(base_parser, yaml_cfg=yaml_cfg)
    opt_cls = OptimizationParams(base_parser, yaml_cfg=yaml_cfg)
    pipe_cls = PipelineParams(base_parser, yaml_cfg=yaml_cfg)
    base_args, _ = base_parser.parse_known_args(["--config", cmd_args.config])

    safe_state(False, random_seed, torch.device(base_args.data_device))

    data_dir = os.path.join(base_args.input_data_folder, base_args.dataset)
    base_args.source_path = data_dir

    n_az = getattr(base_args, 'n_azimuth', 360)
    n_el = getattr(base_args, 'n_elevation', 90)
    per_gw, gw_names = load_ble_per_gateway(data_dir, seed=random_seed,
                                             n_elevation=n_el, n_azimuth=n_az)

    # auto-resolve model path or use override
    if cmd_args.model_path:
        run_dir = cmd_args.model_path
    else:
        run_dir = os.path.join(base_args.log_base_folder, base_args.dataset, base_args.exp_name)

    if cmd_args.output_dir:
        output_dir = cmd_args.output_dir
    else:
        output_dir = os.path.join(run_dir, "inference")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  BLE RSSI Inference")
    print(f"  Model path: {run_dir}")
    print(f"{'='*60}\n")

    # discover gateway subdirectories (e.g. gateway1/, gateway2/)
    gw_dirs = sorted([d for d in os.listdir(run_dir)
                      if d.startswith('g') and os.path.isdir(os.path.join(run_dir, d))])

    all_results = []
    all_sample_mae = []

    # evaluate each gateway
    for gw_dir_name in gw_dirs:
        gw_path = os.path.join(run_dir, gw_dir_name)

        gw_idx = gw_names.index(gw_dir_name) if gw_dir_name in gw_names else None
        if gw_idx is None or gw_idx not in per_gw:
            continue

        _, test_samples = per_gw[gw_idx]
        if len(test_samples) < 5:
            continue

        # find checkpoint: specific iteration or latest
        if cmd_args.iter:
            ckpt_path = os.path.join(gw_path, f"chkpnt{cmd_args.iter}.pth")
        else:
            ckpts = sorted([f for f in os.listdir(gw_path) if f.startswith("chkpnt")])
            if not ckpts:
                print(f"  {gw_dir_name}: no checkpoints found, skipping")
                continue
            ckpt_path = os.path.join(gw_path, ckpts[-1])

        if not os.path.exists(ckpt_path):
            print(f"  {gw_dir_name}: {ckpt_path} not found, skipping")
            continue

        model_args = model_cls.extract(base_args)
        model_args.model_path = gw_path
        model_args.source_path = data_dir
        opt_args = opt_cls.extract(base_args)
        pipe_args = pipe_cls.extract(base_args)

        print(f"  [{gw_dir_name}] Loading {os.path.basename(ckpt_path)}...")

        result, pred_arr, gt_arr, mae_arr = evaluate_gateway(
            gw_idx, gw_dir_name, test_samples, model_args, opt_args, pipe_args, ckpt_path
        )

        print(f"  [{gw_dir_name}] Median MAE = {result['MAE_dBm']['p50']:.2f} dBm\n\n")

        all_results.append(result)
        all_sample_mae.extend(mae_arr.tolist())

        gw_output = os.path.join(output_dir, gw_dir_name)
        os.makedirs(gw_output, exist_ok=True)

        with open(os.path.join(gw_output, "result.json"), 'w') as f:
            json.dump(result, f, indent=2)

        with open(os.path.join(gw_output, "predictions.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "gt_rssi", "pred_rssi", "mae_dBm"])
            for i in range(len(mae_arr)):
                writer.writerow([i, f"{gt_arr[i]:.2f}", f"{pred_arr[i]:.2f}", f"{mae_arr[i]:.2f}"])

    # aggregate results and save summary/plots
    if all_results and all_sample_mae:
        all_mae_arr = np.array(all_sample_mae)

        summary = {
            "run_dir": run_dir,
            "num_gateways": len(all_results),
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
            "per_gateway": all_results,
        }

        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        with open(os.path.join(output_dir, "all_predictions.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["gateway", "index", "gt_rssi", "pred_rssi", "mae_dBm"])
            for r in all_results:
                gw_csv = os.path.join(output_dir, r['gateway'], "predictions.csv")
                if os.path.exists(gw_csv):
                    for row in csv.DictReader(open(gw_csv)):
                        writer.writerow([r['gateway'], row['index'],
                                        row['gt_rssi'], row['pred_rssi'], row['mae_dBm']])

        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        plot_metric_histogram(all_mae_arr, 'MAE (dBm)', os.path.join(plots_dir, "mae_histogram.png"))
        plot_metric_cdf(all_mae_arr, 'MAE (dBm)', os.path.join(plots_dir, "mae_cdf.png"))

        print(f"\n{'='*60}")
        print(f"  BLE Inference — {len(all_results)} gateways, {len(all_sample_mae)} samples")
        print(f"{'='*60}")
        print(f"  MAE:    {all_mae_arr.mean():.2f} +/- {all_mae_arr.std():.2f} dBm")
        print(f"  Median: {np.median(all_mae_arr):.2f} dBm")
        print(f"  P90:    {np.percentile(all_mae_arr, 90):.2f} dBm")
        print(f"  Per gateway:")
        for r in all_results:
            print(f"    {r['gateway']}: mean={r['MAE_dBm']['mean']:.2f}, median={r['MAE_dBm']['p50']:.2f} dBm")
        print(f"{'='*60}")
        print(f"  Results: {output_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
