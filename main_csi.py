
# standard library
import os
import json
from random import randint
from argparse import ArgumentParser, Namespace

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
from gaussian_renderer.render_csi import render_csi
from scene.csi_dataset import load_csi_data
from scene.csi_model import CSIEncoder, CSIAutoDecoder
from scene.dataset_readers import SpectrumInfo
from utils.train_utils import setup_fle_only_optimizer, init_gaussians_from_reference


# ---- viewpoint helper ----

def make_viewpoint(tx_pos, rx_pos, n_elevation=9, n_azimuth=36):
    R = torch.eye(3, dtype=torch.float32)
    dummy = torch.zeros(n_elevation, n_azimuth, dtype=torch.float32)
    return SpectrumInfo(R=R, T_rx=rx_pos, T_tx=tx_pos,
                        spectrum=dummy, spectrum_path="csi",
                        spectrum_name="00001", height=n_elevation, width=n_azimuth)


# ---- phase 1: autoencoder pretraining (uplink CSI -> TX position) ----

def pretrain_autoencoder(scene_info, args, model_path):
    device = torch.device(args.data_device)
    pretrain_iters = getattr(args, 'pretrain_iters', 10000)

    encoder = CSIEncoder(n_antennas=scene_info.n_antennas,
                         n_subcarriers=scene_info.n_subcarriers).to(device)
    auto_decoder = CSIAutoDecoder(n_antennas=scene_info.n_antennas,
                                  n_subcarriers=scene_info.n_subcarriers).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(auto_decoder.parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, pretrain_iters, eta_min=1e-5)

    train_samples = scene_info.train_samples
    loss_log = []
    batch_size = 32

    all_up_re = torch.stack([s.uplink_re for s in train_samples]).to(device)
    all_up_im = torch.stack([s.uplink_im for s in train_samples]).to(device)
    n_train = all_up_re.shape[0]

    print(f"\n  Phase 1: Autoencoder pretraining ({pretrain_iters} iters)\n")

    progress_bar = tqdm(range(pretrain_iters), desc="  Pretrain")
    for iteration in range(1, pretrain_iters + 1):
        idx = torch.randint(0, n_train, (batch_size,), device=device)
        batch_re = all_up_re[idx]
        batch_im = all_up_im[idx]

        positions = encoder(batch_re, batch_im)
        pred_re, pred_im = auto_decoder(positions)

        recon_loss = torch.nn.functional.mse_loss(pred_re, batch_re) \
                   + torch.nn.functional.mse_loss(pred_im, batch_im)

        dists = torch.cdist(positions, positions)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=device)
        spread_loss = torch.relu(0.1 - dists[mask].view(batch_size, -1).min(dim=1)[0]).mean()

        loss = recon_loss + 0.01 * spread_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()

        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"}); progress_bar.update(10)
        if iteration % 500 == 0:
            loss_log.append((iteration, loss.item()))

    progress_bar.close()

    encoder.eval()
    test_re = torch.stack([s.uplink_re for s in scene_info.test_samples]).to(device)
    test_im = torch.stack([s.uplink_im for s in scene_info.test_samples]).to(device)
    with torch.no_grad():
        test_pos = encoder(test_re, test_im)
        test_pred_re, test_pred_im = auto_decoder(test_pos)
        test_loss = (torch.nn.functional.mse_loss(test_pred_re, test_re)
                    + torch.nn.functional.mse_loss(test_pred_im, test_im)).item()
    print(f"\n  Pretrain done: test_loss={test_loss:.6f}")

    fixed_path = os.path.join(os.path.dirname(model_path), "pretrained_encoder.pth")
    torch.save({'encoder': encoder.state_dict()}, fixed_path)
    print(f"  Saved: {fixed_path}")

    if loss_log:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(*zip(*loss_log)); ax.set_xlabel("Iter"); ax.set_ylabel("Loss")
        ax.set_title("Autoencoder Pretrain"); ax.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(model_path, "pretrain_loss.png"), dpi=200); plt.close()

    encoder.train()
    return encoder


# ---- per-antenna evaluation ----

def evaluate_one_antenna(ant_idx, gaussians, encoder, scene_info, pipe_args,
                         n_azimuth, n_elevation, ant_output_dir, iteration):
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

            # denormalize for fair SNR comparison
            down_std = scene_info.down_std
            pred_re_dn = pred_re * down_std + scene_info.down_re_mean
            pred_im_dn = pred_im * down_std + scene_info.down_im_mean
            gt_re_dn = gt_re * down_std + scene_info.down_re_mean
            gt_im_dn = gt_im * down_std + scene_info.down_im_mean

            all_pred_re.append(pred_re_dn.cpu().numpy())
            all_pred_im.append(pred_im_dn.cpu().numpy())
            all_gt_re.append(gt_re_dn.cpu().numpy())
            all_gt_im.append(gt_im_dn.cpu().numpy())

            err = ((pred_re_dn - gt_re_dn)**2 + (pred_im_dn - gt_im_dn)**2).sum().item()
            gt_pwr = (gt_re_dn**2 + gt_im_dn**2).sum().item()
            all_snr.append(-10 * np.log10(err / (gt_pwr + 1e-8) + 1e-10))

    snr_arr = np.array(all_snr)
    result = {
        "antenna": ant_idx, "iteration": iteration,
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

    out_dir = os.path.join(ant_output_dir, f"eval_iter{iteration}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), 'w') as f:
        json.dump(result, f, indent=2)

    np.savez(os.path.join(out_dir, "csi_results.npz"),
             pred=np.array(all_pred_re) + 1j * np.array(all_pred_im),
             gt=np.array(all_gt_re) + 1j * np.array(all_gt_im),
             snr_db=snr_arr)

    print(f"    Ant{ant_idx} iter{iteration}: SNR={snr_arr.mean():.2f}±{snr_arr.std():.2f} dB "
          f"(#G={gaussians.get_xyz.shape[0]})")
    return result


# ---- phase 2: per-antenna Gaussian splatting training ----

def train_one_antenna(ant_idx, encoder, scene_info, args, ant_output_dir, test_iterations,
                      ref_gaussians=None):
    """
    Train one antenna model.
    - ref_gaussians=None (antenna 0): train all parameters from scratch
    - ref_gaussians provided (antenna 1+): reuse geometry, only train FLE coefficients
    """

    device = torch.device(args.data_device)
    freeze_geometry = ref_gaussians is not None

    args.num_channels_override = 52
    gaussians = GaussianModel(args)

    if freeze_geometry:
        # reuse geometry from antenna 0, reinitialize FLE coefficients
        args.gene_init_point = False
        scene = Scene(args, gaussians, load_iteration=None, shuffle=True)
        init_gaussians_from_reference(gaussians, ref_gaussians, args)
        setup_fle_only_optimizer(gaussians, args)
        print(f"    (reusing geometry from antenna 0, training FLE only)")
    else:
        # train everything from scratch
        scene = Scene(args, gaussians, load_iteration=None, shuffle=True)
        gaussians.training_setup(args)

    rx_pos = scene_info.antenna_positions[ant_idx].to(device)
    n_azimuth = getattr(args, 'n_azimuth', 36)
    n_elevation = getattr(args, 'n_elevation', 9)

    pipe_args = Namespace(**{k: getattr(args, k) for k in
                            ['convert_SHs_python', 'compute_cov3D_python', 'debug', 'radius_rx']
                            if hasattr(args, k)})

    train_samples = scene_info.train_samples
    iters = args.iterations

    progress_bar = tqdm(range(iters), desc=f"  Ant{ant_idx}", leave=False)

    # training loop
    for iteration in range(1, iters + 1):
        if not freeze_geometry:
            gaussians.update_learning_rate(iteration)

        # progressively increase FLE degree
        fle_ramp = getattr(args, '_fle_degree_ramp', 500)
        if iteration % fle_ramp == 0:
            gaussians.oneup_fle_degree()

        sample = train_samples[randint(0, len(train_samples) - 1)]

        with torch.no_grad():
            tx_pos = encoder(sample.uplink_re.to(device), sample.uplink_im.to(device))
        viewpoint = make_viewpoint(tx_pos, rx_pos, n_elevation, n_azimuth)

        render_pkg = render_csi(viewpoint, gaussians, pipe_args,
                               n_azimuth=n_azimuth, n_elevation=n_elevation)
        rendered = render_pkg["render"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        csi_52 = rendered.mean(dim=(1, 2))
        pred_re = csi_52[0::2]
        pred_im = csi_52[1::2]

        gt_re = sample.downlink_re[ant_idx].to(device)
        gt_im = sample.downlink_im[ant_idx].to(device)

        loss = torch.nn.functional.mse_loss(pred_re, gt_re) \
             + torch.nn.functional.mse_loss(pred_im, gt_im)

        loss.backward()

        with torch.no_grad():
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.6f}"}); progress_bar.update(10)
            if iteration == iters:
                progress_bar.close()

            if iteration in test_iterations:
                scene.save(iteration)
                torch.save({
                    'gaussians': gaussians.capture(),
                    'iteration': iteration,
                }, os.path.join(ant_output_dir, f"chkpnt{iteration}.pth"))

                evaluate_one_antenna(ant_idx, gaussians, encoder, scene_info,
                                     pipe_args, n_azimuth, n_elevation,
                                     ant_output_dir, iteration)

            # densification only for antenna 0 (full training)
            if not freeze_geometry:
                densify_until = getattr(args, 'densify_until_iter', iters // 2)
                if iteration < densify_until:
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                    gaussians.add_densification_stats(gaussians.get_xyz, visibility_filter)

                    densify_from = getattr(args, 'densify_from_iter', 500)
                    densify_interval = getattr(args, 'densification_interval', 100)
                    if iteration >= densify_from and iteration % densify_interval == 0:
                        size_threshold = getattr(args, 'raddi_size_threshold', 10) \
                            if iteration > getattr(args, 'opacity_reset_interval', 3000) else None
                        gaussians.densify_and_prune(
                            getattr(args, 'densify_grad_threshold', 0.0002),
                            getattr(args, 'min_attenuation_threshold', 0.004),
                            scene.cameras_extent, size_threshold)

                    if iteration % getattr(args, 'opacity_reset_interval', 3000) == 0:
                        gaussians.reset_attenuation()

            if iteration < iters:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

    return gaussians, pipe_args


# ---- main: train all antennas ----

if __name__ == '__main__':

    # parse config file first, then build full argument parser
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="arguments/configs/csi/exp1.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    yaml_cfg = load_config(pre_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    parser = ArgumentParser(description="CSI Training (per-antenna models)")
    parser.add_argument("--config", type=str, default="arguments/configs/csi/exp1.yaml")

    model_para_cls = ModelParams(parser, yaml_cfg=yaml_cfg)
    optimization_para_cls = OptimizationParams(parser, yaml_cfg=yaml_cfg)
    pipeline_para_cls = PipelineParams(parser, yaml_cfg=yaml_cfg)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--pretrain_iters", type=int, default=10000)
    parser.add_argument("--pretrained_encoder", type=str, default=None)

    cmd_args = parser.parse_args()

    # set up data and output paths
    dataset_name = cmd_args.dataset
    exp_name = cmd_args.exp_name
    data_dir = os.path.join(cmd_args.input_data_folder, dataset_name)
    cmd_args.source_path = data_dir

    # logs/<dataset>/<exp_name>/
    model_path = os.path.join(cmd_args.log_base_folder, dataset_name, exp_name)
    os.makedirs(model_path, exist_ok=True)
    cmd_args.model_path = model_path

    cmd_args.densify_until_iter = cmd_args.iterations // 2
    cmd_args.position_lr_max_steps = cmd_args.iterations

    safe_state(cmd_args.quiet, random_seed, torch.device(cmd_args.data_device))

    # load CSI data
    split_method = getattr(cmd_args, 'split_method', 'random')
    ratio_train = getattr(cmd_args, 'ratio_train', 0.8)
    scene_info = load_csi_data(data_dir, split_method=split_method,
                               ratio_train=ratio_train, seed=random_seed)

    print(f"\n{'='*60}")
    print(f"  CSI Training (per-antenna models)")
    print(f"  Data: {data_dir}")
    print(f"  Output: {model_path}")
    print(f"  Samples: train={len(scene_info.train_samples)}, test={len(scene_info.test_samples)}")
    print(f"  Antennas: {scene_info.n_antennas} (one model each)")
    print(f"  Phase 1: {cmd_args.pretrain_iters} iters")
    print(f"  Phase 2: {cmd_args.iterations} iters × {scene_info.n_antennas} antennas")
    print(f"{'='*60}")

    with open(os.path.join(model_path, "config.json"), 'w') as f:
        config_dict = {k: v for k, v in vars(cmd_args).items() if not k.startswith('_')}
        json.dump(config_dict, f, indent=2, default=str)

    args = model_para_cls.extract(cmd_args)
    for k, v in vars(optimization_para_cls.extract(cmd_args)).items():
        setattr(args, k, v)
    for k, v in vars(pipeline_para_cls.extract(cmd_args)).items():
        setattr(args, k, v)
    for k in ['model_path', 'densify_until_iter', 'position_lr_max_steps',
              'data_device', 'pretrain_iters']:
        if hasattr(cmd_args, k):
            setattr(args, k, getattr(cmd_args, k))

    n_azimuth = getattr(args, 'n_azimuth', 36)
    n_elevation = getattr(args, 'n_elevation', 9)

    # load or pretrain encoder
    device = torch.device(cmd_args.data_device)
    encoder_path = cmd_args.pretrained_encoder
    if encoder_path is None:
        default_path = os.path.join(cmd_args.log_base_folder, dataset_name, "pretrained_encoder.pth")
        if os.path.exists(default_path):
            encoder_path = default_path

    if encoder_path and os.path.exists(encoder_path):
        print(f"\n  Loading pretrained encoder: {encoder_path}")
        encoder = CSIEncoder(n_antennas=scene_info.n_antennas,
                             n_subcarriers=scene_info.n_subcarriers).to(device)
        ckpt = torch.load(encoder_path)
        encoder.load_state_dict(ckpt['encoder'])
        print(f"  Loaded. Skipping Phase 1.\n")
    else:
        encoder = pretrain_autoencoder(scene_info, args, model_path)

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    iters = args.iterations
    test_iters_cfg = getattr(args, '_test_iterations', None)
    if test_iters_cfg:
        test_iters = sorted(set(test_iters_cfg + [iters]))
    else:
        test_step = max(iters // 5, 5)
        test_iters = sorted(set([i for i in range(test_step, iters + 1, test_step)] + [iters]))

    # train one model per antenna
    # antenna 0: full training (geometry + FLE)
    # antenna 1+: reuse geometry from antenna 0, only train FLE coefficients
    ref_gaussians = None

    for ant_idx in range(scene_info.n_antennas):
        ant_dir = os.path.join(model_path, f"antenna_{ant_idx}")
        os.makedirs(ant_dir, exist_ok=True)

        args.model_path = ant_dir

        mode = "full" if ant_idx == 0 else "FLE-only"
        print(f"\n  Training antenna {ant_idx} [{mode}] (RX={scene_info.antenna_positions[ant_idx].tolist()})")

        gaussians, _ = train_one_antenna(
            ant_idx, encoder, scene_info, args, ant_dir, test_iters,
            ref_gaussians=ref_gaussians
        )

        # save antenna 0's trained Gaussians as reference for subsequent antennas
        if ant_idx == 0:
            ref_gaussians = gaussians

    # aggregate results across all antennas
    print(f"\n  Collecting joint summaries...")
    for test_iter in test_iters:
        ant_results = []
        all_found = True
        for ant_idx in range(scene_info.n_antennas):
            result_path = os.path.join(model_path, f"antenna_{ant_idx}",
                                       f"eval_iter{test_iter}", "result.json")
            if not os.path.exists(result_path):
                all_found = False
                break
            with open(result_path) as f:
                ant_results.append(json.load(f))

        if not all_found:
            continue

        all_snr = []
        per_ant_means = []
        for ant_idx in range(scene_info.n_antennas):
            eval_dir = os.path.join(model_path, f"antenna_{ant_idx}", f"eval_iter{test_iter}")
            data = np.load(os.path.join(eval_dir, "csi_results.npz"))
            pred_re, pred_im = data['pred'].real, data['pred'].imag
            gt_re, gt_im = data['gt'].real, data['gt'].imag

            if ant_idx == 0:
                n_test = pred_re.shape[0]
                total_err = np.zeros(n_test)
                total_gt = np.zeros(n_test)

            err = ((pred_re - gt_re)**2 + (pred_im - gt_im)**2).sum(axis=1)
            gt_pwr = (gt_re**2 + gt_im**2).sum(axis=1)
            total_err += err
            total_gt += gt_pwr
            per_ant_means.append(ant_results[ant_idx]["SNR_dB_mean"])

        all_snr = -10 * np.log10(total_err / (total_gt + 1e-8) + 1e-10)

        summary = {
            "iteration": test_iter,
            "num_test": len(all_snr),
            "SNR_dB": {
                "mean": round(float(all_snr.mean()), 2),
                "std": round(float(all_snr.std()), 2),
                "min": round(float(all_snr.min()), 2),
                "p25": round(float(np.percentile(all_snr, 25)), 2),
                "p50": round(float(np.percentile(all_snr, 50)), 2),
                "p90": round(float(np.percentile(all_snr, 90)), 2),
                "p95": round(float(np.percentile(all_snr, 95)), 2),
                "max": round(float(all_snr.max()), 2),
            },
            "per_antenna_SNR_dB": {f"ant{i}": m for i, m in enumerate(per_ant_means)},
        }

        out_path = os.path.join(model_path, f"summary_iter{test_iter}.json")
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  iter{test_iter}: SNR={all_snr.mean():.2f}±{all_snr.std():.2f} dB  "
              f"per-ant={['%.1f' % m for m in per_ant_means]}")

    print(f"\n  CSI Training complete. Results in: {model_path}\n")
