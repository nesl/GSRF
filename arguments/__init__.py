# standard library
from argparse import ArgumentParser, Namespace
import sys
import os
import yaml
from datetime import datetime


# project root paths
_CURR_DIR = os.path.dirname(__file__)           # arguments/
_BASE_DIR = os.path.dirname(_CURR_DIR)          # project root
_CONFIG_DIR = os.path.join(_CURR_DIR, "configs") # arguments/configs/


# ---- config loading ----

def load_config(config_path):
    if config_path is None:
        available = [f for f in os.listdir(_CONFIG_DIR) if f.endswith('.yaml')]
        raise ValueError(
            f"--config is required. Available configs in {_CONFIG_DIR}/:\n"
            + "\n".join(f"  - {f}" for f in sorted(available))
        )

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    print(f"  Loaded config: {config_path}")
    return cfg


# ---- parameter groups ----

class GroupParams:
    pass


def _register_args(group, params_dict):
    for key, value in params_dict.items():
        shorthand = False
        if key.startswith("_"):
            continue

        t = type(value)
        if t == bool:
            group.add_argument("--" + key, default=value, action="store_true")
        else:
            group.add_argument("--" + key, default=value, type=t)


class ParamGroup:

    def _init_from_yaml(self, yaml_section):
        if yaml_section is None:
            return
        for key, value in yaml_section.items():
            setattr(self, key, value)

    def _register(self, parser, group_name):
        group = parser.add_argument_group(group_name)
        _register_args(group, vars(self))

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        for key, value in vars(self).items():
            if key.startswith("_") and not hasattr(group, key) and not hasattr(group, key[1:]):
                setattr(group, key, value)
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False, yaml_cfg=None):

        # default data/log paths relative to project root
        self.input_data_folder = os.path.join(_BASE_DIR, "data")
        self.log_base_folder = os.path.join(_BASE_DIR, "logs")
        self._source_path = ""
        self._model_path = ""

        self.exp_name = (yaml_cfg or {}).get("exp_name", "unnamed")

        section = (yaml_cfg or {}).get("model", {})
        self._init_from_yaml(section)

        self._register(parser, "Loading Parameters")

    def extract(self, args):
        g = super().extract(args)
        g.source_path = g.source_path
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser, yaml_cfg=None):

        section = (yaml_cfg or {}).get("pipeline", {})
        self._init_from_yaml(section)

        self._register(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser, yaml_cfg=None):

        section = (yaml_cfg or {}).get("optimization", {})
        self._init_from_yaml(section)

        # set defaults if not specified in config
        iterations = getattr(self, 'iterations', 30000)
        if not hasattr(self, 'position_lr_max_steps'):
            self.position_lr_max_steps = iterations
        if not hasattr(self, 'densify_until_iter'):
            self.densify_until_iter = iterations // 2

        self._register(parser, "Optimization Parameters")


# ---- utility: merge CLI args with saved config ----

def get_combined_args(parser: ArgumentParser):

    cmdlne_string = sys.argv[1:]
    args_cmdline = parser.parse_args(cmdlne_string)

    cfgfile_string = "Namespace()"

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        with open(cfgfilepath) as cfg_file:
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        pass

    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v is not None:
            merged_dict[k] = v

    return Namespace(**merged_dict)
