# standard library
import os
import json
from argparse import ArgumentParser, Namespace

# third-party
import numpy as np
import torch

# project
from arguments import ModelParams, PipelineParams, load_config
from utils.general_utils import safe_state
from scene import GaussianModel
from gaussian_renderer.render_csi import render_csi
from scene.csi_dataset import load_csi_data
from scene.csi_model import CSIEncoder
from scene.dataset_readers import SpectrumInfo
from utils.train_utils import load_gaussians_from_checkpoint
from utils.data_painter import plot_metric_cdf


# ---- viewpoint helper ----

def make_viewpoint(tx_pos, rx_pos, n_elevation=9, n_azimuth=36):
    R = torch.eye(3, dtype=torch.float32)
    dummy = torch.zeros(n_elevation, n_azimuth, dtype=torch.float32)
    return SpectrumInfo(R=R, T_rx=rx_pos, T_tx=tx_pos,
                        spectrum=dummy, spectrum_path="csi",
                        spectrum_name="00001", height=n_elevation, width=n_azimuth)


# ---- per-antenna evaluation ----

def evaluate_one_antenna(ant_idx, gaussians, encoder, scene_info, pipe_args,
                         n_azimuth, n_elevation, output_dir):
    device = next(encoder.parameters()).device
    rx_pos = scene_info.antenna_positions[ant_idx].to(device)
    test_samples = scene_info.test_samples

    all_snr = []
    all_pred_re = []
    all_pred_im = []
    all_gt_re = []
    all_gt_im = []

    with torch.no_grad():
        for sample in test_samples:
            tx_pos = encoder(sample.uplink_re.to(device), sample.uplink_im.to(device))
            viewpoint = make_viewpoint(tx_pos, rx_pos, n_elevation, n_azimuth)
            render_pkg = render_csi(viewpoint, gaussians, pipe_args,
                                   n_azimuth=n_azimuth, n_elevation=n_elevation)
            csi_52 = render_pkg["render"].mean(dim=(1, 2))
            pred_re = csi_52[0::2]
            pred_im = csi_52[1::2]

            gt_re = sample.downlink_re[ant_idx].to(device)
            gt_im = sample.downlink_im[ant_idx].to(device)

            all_pred_re.append(pred_re.cpu().numpy())
            all_pred_im.append(pred_im.cpu().numpy())
            all_gt_re.append(gt_re.cpu().numpy())
            all_gt_im.append(gt_im.cpu().numpy())

            err = ((pred_re - gt_re)**2 + (pred_im - gt_im)**2).sum().item()
            gt_pwr = (gt_re**2 + gt_im**2).sum().item()
            all_snr.append(-10 * np.log10(err / (gt_pwr + 1e-8) + 1e-10))

    snr_arr = np.array(all_snr)

    result = {
        "antenna": ant_idx,
        "num_test": len(test_samples),
        "num_gaussians": gaussians.get_xyz.shape[0],
        "SNR_dB_mean": round(float(snr_arr.mean()), 2),
        "SNR_dB_std": round(float(snr_arr.std()), 2),
        "SNR_dB_min": round(float(snr_arr.min()), 2),
        "SNR_dB_p25": round(float(np.percentile(snr_arr, 25)), 2),
        "SNR_dB_p50": round(float(np.percentile(snr_arr, 50)), 2),
        "SNR_dB_p90": round(float(np.percentile(snr_arr, 90)), 2),
        "SNR_dB_p95": round(float(np.percentile(snr_arr, 95)), 2),
        "SNR_dB_max": round(float(snr_arr.max()), 2),
    }

    # save per-antenna results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=2)

    np.savez(os.path.join(output_dir, "csi_results.npz"),
             pred=np.array(all_pred_re) + 1j * np.array(all_pred_im),
             gt=np.array(all_gt_re) + 1j * np.array(all_gt_im),
             snr_db=snr_arr)

    return result, snr_arr


# ---- main: standalone inference on saved checkpoints ----

def main():
    # parse args: config + optional model_path override
    parser = ArgumentParser(description="CSI Inference")
    parser.add_argument("--config", type=str, default="arguments/configs/csi/exp1.yaml")
    parser.add_argument("--model_path", type=str, default=None, help="Override model path (default: logs/<dataset>/<exp_name>/)")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--iter", type=int, default=None, help="Checkpoint iteration (default: latest)")
    parser.add_argument("--pretrained_encoder", type=str, default=None, help="Path to pretrained encoder")

    cmd_args = parser.parse_args()

    yaml_cfg = load_config(cmd_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    base_parser = ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=cmd_args.config)
    model_cls = ModelParams(base_parser, yaml_cfg=yaml_cfg)
    pipe_cls = PipelineParams(base_parser, yaml_cfg=yaml_cfg)
    base_args, _ = base_parser.parse_known_args(["--config", cmd_args.config])

    safe_state(False, random_seed, torch.device(base_args.data_device))

    dataset_name = base_args.dataset
    data_dir = os.path.join(base_args.input_data_folder, dataset_name)
    base_args.source_path = data_dir

    # load CSI data
    split_method = getattr(base_args, 'split_method', 'random')
    ratio_train = getattr(base_args, 'ratio_train', 0.8)
    scene_info = load_csi_data(data_dir, split_method=split_method,
                               ratio_train=ratio_train, seed=random_seed)

    # auto-resolve model path or use override
    if cmd_args.model_path:
        run_dir = cmd_args.model_path
    else:
        run_dir = os.path.join(base_args.log_base_folder, dataset_name, base_args.exp_name)

    if cmd_args.output_dir:
        output_dir = cmd_args.output_dir
    else:
        output_dir = os.path.join(run_dir, "inference")
    os.makedirs(output_dir, exist_ok=True)

    # load pretrained encoder: check --pretrained_encoder, then model_path/, then logs/<dataset>/
    device = torch.device(base_args.data_device)
    encoder_path = cmd_args.pretrained_encoder
    if encoder_path is None:
        for candidate in [
            os.path.join(run_dir, "pretrained_encoder.pth"),
            os.path.join(base_args.log_base_folder, dataset_name, "pretrained_encoder.pth"),
        ]:
            if os.path.exists(candidate):
                encoder_path = candidate
                break

    if not encoder_path or not os.path.exists(encoder_path):
        print(f"Error: pretrained encoder not found. Provide --pretrained_encoder or place it in "
              f"{run_dir}/ or logs/{dataset_name}/.")
        return

    encoder = CSIEncoder(n_antennas=scene_info.n_antennas,
                         n_subcarriers=scene_info.n_subcarriers).to(device)
    ckpt = torch.load(encoder_path)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    n_azimuth = getattr(base_args, 'n_azimuth', 36)
    n_elevation = getattr(base_args, 'n_elevation', 9)

    print(f"\n{'='*60}")
    print(f"  CSI Inference")
    print(f"  Model path: {run_dir}")
    print(f"  Encoder: {encoder_path}")
    print(f"  Antennas: {scene_info.n_antennas}")
    print(f"{'='*60}\n")

    # discover antenna subdirectories and evaluate each
    all_results = []
    all_snr_samples = []

    for ant_idx in range(scene_info.n_antennas):
        ant_dir = os.path.join(run_dir, f"antenna_{ant_idx}")
        if not os.path.isdir(ant_dir):
            print(f"  antenna_{ant_idx}: directory not found, skipping")
            continue

        # find checkpoint: specific iteration or latest
        if cmd_args.iter:
            ckpt_path = os.path.join(ant_dir, f"chkpnt{cmd_args.iter}.pth")
        else:
            ckpts = sorted([f for f in os.listdir(ant_dir) if f.startswith("chkpnt")])
            if not ckpts:
                print(f"  antenna_{ant_idx}: no checkpoints found, skipping")
                continue
            ckpt_path = os.path.join(ant_dir, ckpts[-1])

        if not os.path.exists(ckpt_path):
            print(f"  antenna_{ant_idx}: {ckpt_path} not found, skipping")
            continue

        # load model
        model_args = model_cls.extract(base_args)
        model_args.model_path = ant_dir
        model_args.source_path = data_dir
        model_args.num_channels_override = 52

        pipe_args_ns = Namespace(**{k: getattr(base_args, k) for k in
                                    ['convert_SHs_python', 'compute_cov3D_python', 'debug', 'radius_rx']
                                    if hasattr(base_args, k)})

        gaussians = GaussianModel(model_args)
        load_gaussians_from_checkpoint(gaussians, ckpt_path)

        print(f"  [antenna_{ant_idx}] Loading {os.path.basename(ckpt_path)}...")

        ant_output = os.path.join(output_dir, f"antenna_{ant_idx}")
        result, snr_arr = evaluate_one_antenna(
            ant_idx, gaussians, encoder, scene_info, pipe_args_ns,
            n_azimuth, n_elevation, ant_output
        )

        print(f"  [antenna_{ant_idx}] SNR = {result['SNR_dB_mean']:.2f} +/- {result['SNR_dB_std']:.2f} dB")

        all_results.append(result)
        all_snr_samples.append(snr_arr)

    # aggregate results across all antennas
    if all_results:
        all_snr_flat = np.concatenate(all_snr_samples)

        summary = {
            "model_path": run_dir,
            "num_antennas": len(all_results),
            "num_test_samples_total": len(all_snr_flat),
            "overall_SNR_dB": {
                "mean": round(float(all_snr_flat.mean()), 4),
                "std": round(float(all_snr_flat.std()), 4),
                "min": round(float(all_snr_flat.min()), 4),
                "p25": round(float(np.percentile(all_snr_flat, 25)), 4),
                "p50": round(float(np.percentile(all_snr_flat, 50)), 4),
                "p75": round(float(np.percentile(all_snr_flat, 75)), 4),
                "p90": round(float(np.percentile(all_snr_flat, 90)), 4),
                "p95": round(float(np.percentile(all_snr_flat, 95)), 4),
                "max": round(float(all_snr_flat.max()), 4),
            },
            "per_antenna": all_results,
        }

        with open(os.path.join(output_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  CSI Inference — {len(all_results)} antennas, {len(all_snr_flat)} samples")
        print(f"{'='*60}")
        print(f"  SNR:    {all_snr_flat.mean():.2f} +/- {all_snr_flat.std():.2f} dB")
        print(f"  Median: {np.median(all_snr_flat):.2f} dB")
        print(f"  P90:    {np.percentile(all_snr_flat, 90):.2f} dB")
        print(f"  Per antenna:")
        for r in all_results:
            print(f"    antenna_{r['antenna']}: mean={r['SNR_dB_mean']:.2f}, median={r['SNR_dB_p50']:.2f} dB")
        print(f"{'='*60}")
        print(f"  Results: {output_dir}")
        print(f"{'='*60}\n")

        # CDF plot
        plot_metric_cdf(all_snr_flat, 'SNR (dB)', os.path.join(output_dir, "snr_cdf.png"))
        print(f"  CDF plot saved to: {os.path.join(output_dir, 'snr_cdf.png')}\n")


if __name__ == "__main__":
    main()
