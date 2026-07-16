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
import pointcept.models.LArFormer.trainer  # noqa: F401
import pointcept.models.LArFormer.evaluator  # noqa: F401

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

    # SLURM's pre-timeout --signal=USR1 is delivered to EVERY process in the
    # step cgroup. The trainer ranks install handlers (SignalCheckpointHook),
    # but with num_gpus > 1 this parent process only sits in mp.spawn join()
    # and would die from the unhandled signal (observed as exit 138 on
    # P05B.1), tearing down the ranks before they can checkpoint. Ignore it
    # here; children override the inherited disposition when the hook
    # registers its own handler.
    import signal
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)

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
