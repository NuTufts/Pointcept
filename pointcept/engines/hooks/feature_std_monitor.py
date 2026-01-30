import torch

from .default import HookBase
from .builder import HOOKS

@HOOKS.register_module()
class FeatureStdMonitor(HookBase):
    """
    Hook to monitor the standard deviation of feature vectors in student and teacher models.
    
    This is useful for tracking feature collapse and ensuring features remain diverse
    during training. The hook uses forward hooks to compute stats directly during
    the forward pass, avoiding storing large feature tensors in memory.
    
    Args:
        log_frequency (int): How often to log feature statistics (default: 10)
        prefix (str): Prefix for logging keys (default: "feature_std")
        monitor_student (bool): Whether to monitor student model features (default: True)
        monitor_teacher (bool): Whether to monitor teacher model features (default: True)
        track_channels (bool): Whether to track per-channel statistics (default: False)
    """
    
    def __init__(
        self,
        log_frequency=10,
        prefix="feature_std",
        monitor_student=True,
        monitor_teacher=True,
        track_channels=False
    ):
        self.log_frequency = log_frequency
        self.prefix = prefix
        self.monitor_student = monitor_student
        self.monitor_teacher = monitor_teacher
        self.track_channels = track_channels
        self.step_count = 0
        self.hook_handles = []
    
    def before_train(self):
        """Register forward hooks to capture feature statistics."""
        self.trainer.logger.info(f"Monitoring feature statistics with prefix '{self.prefix}'")
        
        # Access the model (unwrap DDP if needed)
        if hasattr(self.trainer.model, 'module'):
            model = self.trainer.model.module
        else:
            model = self.trainer.model
            
        # Find Sonata modules to monitor
        self._register_sonata_hooks(model)
    
    def _register_sonata_hooks(self, model):
        """Register hooks on student and teacher modules to capture feature stats."""
        # Clear previous hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # Debug: print model class name
        self.trainer.logger.info(f"[FeatureStdMonitor] Model class: {model.__class__.__name__}")

        # Check if model is Sonata or contains Sonata
        sonata_module = None
        if hasattr(model, 'sonata'):
            sonata_module = model.sonata
            self.trainer.logger.info("[FeatureStdMonitor] Found sonata via model.sonata")
        elif isinstance(model, torch.nn.ModuleDict) and 'sonata' in model:
            sonata_module = model['sonata']
            self.trainer.logger.info("[FeatureStdMonitor] Found sonata in ModuleDict")
        elif hasattr(model, '__class__') and 'Sonata' in model.__class__.__name__:
            sonata_module = model
            self.trainer.logger.info("[FeatureStdMonitor] Model is Sonata class directly")

        if not sonata_module:
            self.trainer.logger.warning("[FeatureStdMonitor] Could not find Sonata module for feature monitoring")
            return

        # Debug: print what attributes are available
        self.trainer.logger.info(f"[FeatureStdMonitor] Sonata has teacher: {hasattr(sonata_module, 'teacher')}")
        self.trainer.logger.info(f"[FeatureStdMonitor] Sonata has student: {hasattr(sonata_module, 'student')}")

        # Register hooks on teacher backbone
        if self.monitor_teacher and hasattr(sonata_module, 'teacher'):
            if 'backbone' in sonata_module.teacher:
                self.trainer.logger.info("[FeatureStdMonitor] Registering feature monitor on teacher backbone")
                hook = sonata_module.teacher['backbone'].register_forward_hook(
                    self._feature_stats_hook('teacher')
                )
                self.hook_handles.append(hook)
            else:
                self.trainer.logger.warning(f"[FeatureStdMonitor] teacher has no 'backbone', keys: {list(sonata_module.teacher.keys())}")

        # Register hooks on student backbone
        if self.monitor_student and hasattr(sonata_module, 'student'):
            if 'backbone' in sonata_module.student:
                self.trainer.logger.info("[FeatureStdMonitor] Registering feature monitor on student backbone")
                hook = sonata_module.student['backbone'].register_forward_hook(
                    self._feature_stats_hook('student')
                )
                self.hook_handles.append(hook)
            else:
                self.trainer.logger.warning(f"[FeatureStdMonitor] student has no 'backbone', keys: {list(sonata_module.student.keys())}")

        self.trainer.logger.info(f"[FeatureStdMonitor] Total hooks registered: {len(self.hook_handles)}")
    
    def _feature_stats_hook(self, module_name):
        """Create a forward hook function that captures feature statistics."""
        def hook_fn(module, input, output):
            # Only process on certain steps
            if not hasattr(self, '_step_counter'):
                self._step_counter = {}
            if module_name not in self._step_counter:
                self._step_counter[module_name] = 0
            
            self._step_counter[module_name] += 1
            if self._step_counter[module_name] % self.log_frequency != 0:
                return
            
            # Get features from output (assuming Point structure or tensor)
            if hasattr(output, 'feat'):
                features = output.feat
            elif isinstance(output, torch.Tensor):
                features = output
            else:
                return
                
            # Calculate statistics with proper distributed synchronization
            with torch.no_grad():
                import torch.distributed as dist
                from pointcept.utils.comm import get_world_size
                
                features_flat = features.float()
                local_n = torch.tensor([features_flat.numel()], device=features.device, dtype=torch.float64)
                local_sum = features_flat.sum().to(torch.float64)
                local_sum_sq = (features_flat ** 2).sum().to(torch.float64)
                
                # Synchronize across GPUs for global std
                if get_world_size() > 1:
                    dist.all_reduce(local_n)
                    dist.all_reduce(local_sum)
                    dist.all_reduce(local_sum_sq)
                
                global_mean = local_sum / local_n
                global_var = (local_sum_sq / local_n) - (global_mean ** 2)
                global_std = torch.sqrt(global_var.clamp(min=0)).item()
                
                # Batch-wise std (local is fine for this metric)
                batch_std = torch.std(features, dim=1).mean().item()
                
                # Channel-wise std with distributed sync
                # Each GPU: compute local sum and sum_sq per channel
                local_channel_n = torch.tensor([features.shape[0]], device=features.device, dtype=torch.float64)
                local_channel_sum = features.sum(dim=0).to(torch.float64)  # (channels,)
                local_channel_sum_sq = (features ** 2).sum(dim=0).to(torch.float64)
                
                if get_world_size() > 1:
                    dist.all_reduce(local_channel_n)
                    dist.all_reduce(local_channel_sum)
                    dist.all_reduce(local_channel_sum_sq)
                
                channel_mean = local_channel_sum / local_channel_n
                channel_var = (local_channel_sum_sq / local_channel_n) - (channel_mean ** 2)
                channel_std = torch.sqrt(channel_var.clamp(min=0))
                
                channel_mean_std = channel_std.mean().item()
                channel_min_std = channel_std.min().item()
                channel_max_std = channel_std.max().item()
                
                stats = {
                    "global_std": global_std,
                    "batch_std": batch_std,
                    "channel_mean_std": channel_mean_std,
                    "channel_min_std": channel_min_std,
                    "channel_max_std": channel_max_std
                }
                        
            # Log to console (brief summary)
            self.trainer.logger.info(
                f"[{self.prefix}/{module_name}] global_std={global_std:.4f} batch_std={batch_std:.4f} "
                f"channel_std=(mean={channel_mean_std:.4f}, min={channel_min_std:.4f}, max={channel_max_std:.4f})"
            )

            # Log to tensorboard if available
            if hasattr(self.trainer, 'writer') and self.trainer.writer is not None:
                global_step = self.trainer.comm_info.get("iter", 0)

                # Log metrics to tensorboard
                for stat_name, stat_value in stats.items():
                    self.trainer.writer.add_scalar(
                        f"{self.prefix}/{module_name}/{stat_name}",
                        stat_value,
                        global_step
                    )

                # Log per-channel std if requested
                if self.track_channels:
                    for i, std_val in enumerate(channel_std):
                        self.trainer.writer.add_scalar(
                            f"{self.prefix}/{module_name}/channel_{i}_std",
                            std_val.item(),
                            global_step
                        )

            # Log to wandb if enabled
            if getattr(self.trainer.cfg, 'enable_wandb', False):
                import wandb
                if wandb.run is not None:
                    wandb_metrics = {
                        f"{self.prefix}/{module_name}/{stat_name}": stat_value
                        for stat_name, stat_value in stats.items()
                    }
                    wandb.log(wandb_metrics, step=wandb.run.step)
        
        return hook_fn
    
    def after_train(self):
        """Clean up hooks."""
        for handle in self.hook_handles:
            handle.remove()
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"log_frequency={self.log_frequency}, "
                f"prefix='{self.prefix}')")