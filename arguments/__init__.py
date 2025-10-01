#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from argparse import ArgumentParser, Namespace
import sys
import os
from datetime import datetime


class GroupParams:
    pass


# The ParamGroup class is designed to add a group of parameters to an ArgumentParser object
class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none=False):

        group = parser.add_argument_group(name)

        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1: ]

            t = type(value)
            value = value if not fill_none else None

            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0: 1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0: 1]), default=value, type=t)

            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()

        for arg in vars(args).items():  # arg: ('sh_degree', 3)
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                
                setattr(group, arg[0], arg[1])

        return group


class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):

        # current file's directory
        curr_dir = os.path.dirname(os.path.abspath(__file__))  
        base_dir = os.path.dirname(curr_dir)

        current_time = datetime.now().strftime("%m%d%Y%H%M%S")

        self.input_data_folder = os.path.join(base_dir, "data")

        self.dataset = "rfid"
        self.exp_name = f"gsrf_{current_time}"
        self.log_base_folder = os.path.join(base_dir, "logs")

        self.llffhold = 8
        self.llffhold_flag = False
        self.ratio_train = 0.1

        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._white_background = False
        self.data_device = "cuda:0"
        self.eval = True
        
        self.gene_init_point = False
        self.camera_scale = 1.5
        self.voxel_size_scale = 1.0

        self.train_index_path = "train_index.txt"  
        self.test_index_path = "test_index.txt"

        self.max_freq_log2 = 9
        self.num_freqs = 10

        self.hidden_dim_1 = 256
        self.hidden_dim_2 = 64
        self.output_dim = 3

        self.input_dim_emd  = (3 + 2 * 3 * self.num_freqs) * 2

        super().__init__(parser, "Loading Parameters", sentinel)


    def extract(self, args):
        g = super().extract(args)

        g.source_path = os.path.abspath(g.source_path)

        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = True
        self.compute_cov3D_python = True
        self.debug = False
        self.radius_rx = 1.1

        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    
    def __init__(self, parser):
        max_iter = 30_000

        self.iterations = max_iter
        self.position_lr_init = 0.00016
        self.position_lr_final = 1.6e-06
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = max_iter
        self.feature_lr = 0.0025
        self.opacity_lr = 0.01
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.02
        self.lambda_dssim = 0.4
        self.lambda_dfourier = 0.0

        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = max_iter // 2
        self.densify_grad_threshold = 0.0001

        self.min_attenuation_threshold = 0.004

        self.raddi_size_threshold = 10

        self.random_background = False
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser : ArgumentParser):

    cmdlne_string = sys.argv[1: ]
    args_cmdline = parser.parse_args(cmdlne_string)

    cfgfile_string = "Namespace()"
    
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)

        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()

    except TypeError:
        print("Config file not found at")
        pass

    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()

    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    return Namespace(**merged_dict)


