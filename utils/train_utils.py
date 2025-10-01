
from argparse import Namespace
import os
import torch
from random import randint
import uuid
import random

from .data_painter import paint_spectrum
from .loss_utils import psnr
import torch.nn as nn

import lpips
loss_lpips_t = lpips.LPIPS(net='alex')


try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training_report(tb_writer,
                    iteration,
                    Ll1,
                    loss,
                    l1_loss,
                    elapsed,
                    testing_iterations,
                    scene,
                    renderFunc, 
                    tx_pos_encoder_func,
                    pipe_args, 
                    dataset_args, background
                    ):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    image_path = os.path.join(dataset_args.model_path, "spectrums")
    os.makedirs(image_path, exist_ok=True)

    number_of_samples = 5
    # Report test and samples of training set
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
                                                   tx_pos_encoder_func, pipe_args, background)["render"], 0.0, 1.0)
                    
                    gt_image = torch.clamp(viewpoint.spectrum.to("cuda"), 0.0, 1.0)

                    image = image.unsqueeze(0)
                    gt_image = gt_image.unsqueeze(0)

                    # if tb_writer and (idx < 5):
                    if tb_writer:
                        
                        file_name_test = config['name'] + "_view_{}/render".format(viewpoint.spectrum_name)

                        # add_images function expects the input tensor to be in the N-C-H-W format, after image[None]: (1, 1, 90, 360) 
                        tb_writer.add_images(file_name_test, image[None], global_step=iteration)
                        
                        if iteration == testing_iterations[0]:  # testing_iterations = [7000, 30000]

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

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("\nTensorboard not available: not logging progress!\n")

    return tb_writer


def initialize_weights(module):

    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)






