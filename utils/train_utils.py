
from argparse import Namespace
import os
import torch
import uuid
import random

from .data_painter import paint_spectrum
from .loss_utils import psnr
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ---- tensorboard logging ----
def training_report(tb_writer,
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    elapsed,
                    testing_iterations,
                    scene,
                    renderFunc,
                    pipe_args,
                    dataset_args
                    ):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    image_path = os.path.join(dataset_args.model_path, "spectrums")
    os.makedirs(image_path, exist_ok=True)

    number_of_samples = 5
    if iteration in testing_iterations:

        torch.cuda.empty_cache()

        validation_configs = ({'name': 'test',  'spectrums': random.sample(scene.getTestSpectrums(),  number_of_samples)},
                              {'name': 'train', 'spectrums': random.sample(scene.getTrainSpectrums(), number_of_samples)}
                             )

        for config in validation_configs:
            if config['spectrums'] and len(config['spectrums']) > 0:

                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['spectrums']):

                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, \
                                                   pipe_args)["render"], 0.0, 1.0)

                    gt_image = torch.clamp(viewpoint.spectrum.to("cuda"), 0.0, 1.0)

                    image = image.unsqueeze(0)
                    gt_image = gt_image.unsqueeze(0)

                    if tb_writer:

                        file_name_test = config['name'] + "_view_{}/render".format(viewpoint.spectrum_name)

                        tb_writer.add_images(file_name_test, image[None], global_step=iteration)

                        if iteration == testing_iterations[0]:

                            file_name_gt = config['name'] + "_view_{}/ground_truth".format(viewpoint.spectrum_name)
                            tb_writer.add_images(file_name_gt, gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    filename = os.path.join(image_path, f"ite_{iteration:06d}_{config['name']}_{idx:06d}.png")
                    paint_spectrum(gt_image.cpu().squeeze().numpy(),
                                                  image.cpu().squeeze().numpy(),
                                                  save_path=filename)

                l1_test /= len(config['spectrums'])
                psnr_test /= len(config['spectrums'])

                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)


        if tb_writer:

            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_attenuation, iteration)

            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

        torch.cuda.empty_cache()


# ---- output directory and logger setup ----
def prepare_output_and_logger(args):

    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')

        else:
            unique_str = str(uuid.uuid4())

        args.model_path = os.path.join("./output/", unique_str[0: 10])

    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("\nTensorboard not available: not logging progress!\n")

    return tb_writer


# ---- shared geometry reuse (BLE per-gateway, CSI per-antenna) ----

def setup_fle_only_optimizer(gaussians, opt_args):
    """Set up optimizer that only trains FLE coefficients (freeze geometry)."""
    from .general_utils import get_expon_lr_func

    gaussians.percent_dense = opt_args.percent_dense
    gaussians.xyz_gradient_accum = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")
    gaussians.denom = torch.zeros((gaussians.get_xyz.shape[0], 1), device="cuda")

    l = [
        {'params': [gaussians._features_dc],   'lr': opt_args.feature_lr,   "name": "f_dc"},
        {'params': [gaussians._features_rest], 'lr': opt_args.feature_lr * getattr(opt_args, '_rest_lr_ratio', 1.0), "name": "f_rest"},
    ]

    gaussians.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    gaussians.xyz_scheduler_args = get_expon_lr_func(
        lr_init=opt_args.position_lr_init * gaussians.spatial_lr_scale,
        lr_final=opt_args.position_lr_final * gaussians.spatial_lr_scale,
        lr_delay_mult=opt_args.position_lr_delay_mult,
        max_steps=opt_args.position_lr_max_steps)


def init_gaussians_from_reference(gaussians, ref_gaussians, args):
    """Copy frozen geometry from reference Gaussians, reinitialize FLE coefficients."""
    gaussians._xyz = nn.Parameter(ref_gaussians._xyz.clone(), requires_grad=False)
    gaussians._scaling = nn.Parameter(ref_gaussians._scaling.clone(), requires_grad=False)
    gaussians._rotation = nn.Parameter(ref_gaussians._rotation.clone(), requires_grad=False)
    gaussians._attenuation = nn.Parameter(ref_gaussians._attenuation.clone(), requires_grad=False)
    gaussians.spatial_lr_scale = ref_gaussians.spatial_lr_scale

    fle_init_scale = getattr(args, '_fle_init_scale', 0.1)
    n_points = gaussians._xyz.shape[0]
    n_channels = gaussians.num_channels
    max_deg = gaussians.max_fle_degree

    features = torch.randn(n_points, n_channels, (max_deg + 1) ** 2, device="cuda") * fle_init_scale
    gaussians._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous())
    gaussians._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous())

    gaussians.max_radii2D = torch.zeros((n_points,), device="cuda")
    gaussians.active_fle_degree = 0


def load_gaussians_from_checkpoint(gaussians, ckpt_path):
    """Load Gaussian parameters from checkpoint without optimizer restore (works for both full and FLE-only checkpoints)."""
    ckpt_data = torch.load(ckpt_path, map_location='cuda')

    # handle both formats: (capture_tuple, iter) and {'gaussians': capture_tuple, 'iteration': iter}
    if isinstance(ckpt_data, dict):
        model_params = ckpt_data['gaussians']
    else:
        model_params = ckpt_data[0]

    (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
     gaussians._attenuation, gaussians._scaling, gaussians._rotation,
     gaussians.max_radii2D, _, _, _, gaussians.spatial_lr_scale) = model_params
    gaussians.active_fle_degree = gaussians.max_fle_degree
