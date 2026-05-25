"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch

# Side-effect imports for custom trainers that don't auto-register via
# pointcept.models. The config file does the same import as a side effect,
# but DDP spawn workers receive the already-parsed cfg dict over the
# multiprocessing pipe and never re-import the config — so registration
# must happen at this top level (which runs in every spawn worker).
# Each module's @TRAINERS.register_module() decorator runs exactly once
# (modules are cached in sys.modules).
import pointcept.models.shower_clustering.trainer  # noqa: F401

#import torch
#torch.autograd.set_detect_anomaly(True)


def main_worker(cfg):
    #import torch
    #torch.autograd.set_detect_anomaly(True)
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
