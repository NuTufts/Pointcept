import torch

from .default import HookBase
from .builder import HOOKS

@HOOKS.register_module()
class PrototypeUsageLogger(HookBase):
    """
    Hook to monitor prototype utilization in clustering/tokenization models like Sonata.
    
    This hook tracks:
    1. How many prototypes are actually being used (by looking at argmax assignments)
    2. What percentage of prototypes are unused
    3. How many tokens are assigned to each active prototype on average
    
    Args:
        log_frequency (int): How often to log prototype usage (default: 10)
        prefix (str): Prefix for logging keys (default: "prototypes")
    """
    
    def __init__(
        self,
        log_frequency=10,
        prefix="prototypes"
    ):
        self.log_frequency = log_frequency
        self.prefix = prefix
        self.hook_handles = []
        self._step_counters = {}
    
    def before_train(self):
        """Register hooks on Sonata heads to capture prototype usage."""
        self.trainer.logger.info(f"Monitoring prototype usage with prefix '{self.prefix}'")
        
        # Access the model (unwrap DDP if needed)
        if hasattr(self.trainer.model, 'module'):
            model = self.trainer.model.module
        else:
            model = self.trainer.model
            
        # Register hooks on the model
        self._register_hooks(model)
    
    def _register_hooks(self, model):
        """Register hooks on Sonata heads to capture prototype usage."""
        # Clear previous hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # Debug: print model class name
        self.trainer.logger.info(f"[PrototypeUsageLogger] Model class: {model.__class__.__name__}")

        # Find Sonata module
        sonata_module = None
        if hasattr(model, 'sonata'):
            sonata_module = model.sonata
            self.trainer.logger.info("[PrototypeUsageLogger] Found sonata via model.sonata")
        elif isinstance(model, torch.nn.ModuleDict) and 'sonata' in model:
            sonata_module = model['sonata']
            self.trainer.logger.info("[PrototypeUsageLogger] Found sonata in ModuleDict")
        elif hasattr(model, '__class__') and 'Sonata' in model.__class__.__name__:
            sonata_module = model
            self.trainer.logger.info("[PrototypeUsageLogger] Model is Sonata class directly")

        if not sonata_module:
            self.trainer.logger.warning("[PrototypeUsageLogger] Could not find Sonata module for prototype monitoring")
            return

        # Debug: print what attributes are available
        self.trainer.logger.info(f"[PrototypeUsageLogger] Sonata has teacher: {hasattr(sonata_module, 'teacher')}")
        self.trainer.logger.info(f"[PrototypeUsageLogger] Sonata has student: {hasattr(sonata_module, 'student')}")

        # Register hooks on teacher heads
        if hasattr(sonata_module, 'teacher') and isinstance(sonata_module.teacher, torch.nn.ModuleDict):
            self.trainer.logger.info(f"[PrototypeUsageLogger] Teacher keys: {list(sonata_module.teacher.keys())}")
            for head_name, head in sonata_module.teacher.items():
                if 'head' in head_name.lower():
                    self.trainer.logger.info(f"[PrototypeUsageLogger] Registering prototype monitor on teacher/{head_name}")
                    hook = head.register_forward_hook(self._prototype_stats_hook(f"teacher/{head_name}"))
                    self.hook_handles.append(hook)

        # Register hooks on student heads
        if hasattr(sonata_module, 'student') and isinstance(sonata_module.student, torch.nn.ModuleDict):
            self.trainer.logger.info(f"[PrototypeUsageLogger] Student keys: {list(sonata_module.student.keys())}")
            for head_name, head in sonata_module.student.items():
                if 'head' in head_name.lower():
                    self.trainer.logger.info(f"[PrototypeUsageLogger] Registering prototype monitor on student/{head_name}")
                    hook = head.register_forward_hook(self._prototype_stats_hook(f"student/{head_name}"))
                    self.hook_handles.append(hook)

        self.trainer.logger.info(f"[PrototypeUsageLogger] Total hooks registered: {len(self.hook_handles)}")
    
    def _prototype_stats_hook(self, name):
        """Create a forward hook that calculates prototype statistics."""
        def hook_fn(module, input, output):
            # Skip if no output
            if output is None:
                return
                
            # Initialize counter for this module if it doesn't exist
            if name not in self._step_counters:
                self._step_counters[name] = 0
                
            # Increment counter
            self._step_counters[name] += 1
            
            # Only process on certain steps
            if self._step_counters[name] % self.log_frequency != 0:
                return
            
            # Calculate statistics
            if isinstance(output, tuple):
                stats = {}
                for i, o in enumerate(output):
                    stats[f"output_{i}"] = self._get_stats(o)
            else:
                stats = self._get_stats(output)

            # Log to console (brief summary)
            if isinstance(stats, dict) and 'used_count' in stats:
                self.trainer.logger.info(
                    f"[{self.prefix}/{name}] used={stats['used_count']} "
                    f"unused={stats['unused_percent']:.1f}% "
                    f"tokens/proto={stats['tokens_per_prototype']:.1f} "
                    f"entropy={stats['assignment_entropy']:.2f}"
                )

            # Log to tensorboard if available
            if hasattr(self.trainer, 'writer') and self.trainer.writer is not None:
                global_step = self.trainer.comm_info.get("iter", 0)

                # Log metrics to tensorboard
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, dict):
                        for k, v in stat_value.items():
                            self.trainer.writer.add_scalar(
                                f"{self.prefix}/{name}/{stat_name}/{k}",
                                v,
                                global_step
                            )
                    else:
                        self.trainer.writer.add_scalar(
                            f"{self.prefix}/{name}/{stat_name}",
                            stat_value,
                            global_step
                        )

            # Log to wandb if enabled
            if getattr(self.trainer.cfg, 'enable_wandb', False):
                import wandb
                if wandb.run is not None:
                    wandb_metrics = {}
                    for stat_name, stat_value in stats.items():
                        if isinstance(stat_value, dict):
                            for k, v in stat_value.items():
                                wandb_metrics[f"{self.prefix}/{name}/{stat_name}/{k}"] = v
                        else:
                            wandb_metrics[f"{self.prefix}/{name}/{stat_name}"] = stat_value
                    wandb.log(wandb_metrics, step=wandb.run.step)

        return hook_fn

    def _get_stats(self, output):
        """Calculate statistics from output with proper distributed synchronization."""
        import torch.distributed as dist
        from pointcept.utils.comm import get_world_size
        
        with torch.no_grad():
            # Get assignments by taking argmax of logits
            assignments = output.argmax(dim=-1)  # (tokens,)
            
            # Total number of prototypes
            total_prototypes = output.shape[-1]
            
            # Count tokens per prototype locally
            local_counts = torch.bincount(assignments, minlength=total_prototypes).float()
            
            # Synchronize counts across all GPUs
            if get_world_size() > 1:
                dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
            
            global_counts = local_counts
            total_tokens = global_counts.sum().item()
            
            # Calculate global usage metrics
            used_mask = global_counts > 0
            used_count = used_mask.sum().item()
            unused_count = total_prototypes - used_count
            unused_percent = (unused_count / total_prototypes) * 100
            
            # Calculate tokens per prototype (global average)
            tokens_per_prototype = total_tokens / used_count if used_count > 0 else 0
            
            # Calculate entropy of assignment distribution (global)
            probs = global_counts / total_tokens if total_tokens > 0 else global_counts
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            
            # Create stats dictionary
            stats = {
                "used_count": used_count,
                "unused_percent": unused_percent,
                "tokens_per_prototype": tokens_per_prototype,
                "assignment_entropy": entropy.item()
            }
        return stats
    
    def after_train(self):
        """Clean up hooks when training is done."""
        for handle in self.hook_handles:
            handle.remove()
    
    def __repr__(self):
        return f"{self.__class__.__name__}(log_frequency={self.log_frequency}, prefix='{self.prefix}')"
