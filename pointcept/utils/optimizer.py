"""
Optimizer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import copy
import torch
from pointcept.utils.logger import get_root_logger
from pointcept.utils.registry import Registry

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.SGD, name="SGD")
OPTIMIZERS.register_module(module=torch.optim.Adam, name="Adam")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")


# Default substring patterns identifying learned query / positional / embedding
# parameters that should be EXEMPT from weight decay even though they are >=2D
# (so the ndim<=1 rule below would not otherwise catch them).
DEFAULT_NO_DECAY_KEYWORDS = (
    "query_content",
    "query_pos",
    "pool_query",
    "query_embed",
    "pos_embed",
    "embed",  # catches nn.Embedding weights / positional embeddings generally
)


def _is_no_decay_param(name, param, no_decay_keywords):
    """A parameter is exempt from weight decay if it is 1D (norm gains/biases,
    all biases, scalar gains) or its name matches a learned query/positional/
    embedding pattern."""
    if param.ndim <= 1:
        return True
    for kw in no_decay_keywords:
        if kw in name:
            return True
    return False


def build_optimizer(cfg, model, param_dicts=None):
    cfg = copy.deepcopy(cfg)

    # Opt-in: split parameters so weight decay is NOT applied to 1D params
    # (norm gains/biases, all biases) and learned query/positional/embedding
    # parameters. `no_decay_on_1d_and_embeddings` may be True (use defaults) or
    # a list of substring patterns. It is popped here because it is not a valid
    # torch optimizer kwarg.
    no_decay_spec = cfg.pop("no_decay_on_1d_and_embeddings", False)
    if no_decay_spec:
        if param_dicts is not None:
            raise NotImplementedError(
                "no_decay_on_1d_and_embeddings does not currently compose with "
                "param_dicts. Use one or the other."
            )
        no_decay_keywords = (
            DEFAULT_NO_DECAY_KEYWORDS
            if no_decay_spec is True
            else tuple(no_decay_spec)
        )
        base_wd = cfg.get("weight_decay", 0.0)
        decay = dict(names=[], params=[], weight_decay=base_wd)
        no_decay = dict(names=[], params=[], weight_decay=0.0)
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            group = no_decay if _is_no_decay_param(n, p, no_decay_keywords) else decay
            group["names"].append(n)
            group["params"].append(p)
        cfg.params = [decay, no_decay]

        logger = get_root_logger()
        for grp_name, grp in (("decay", decay), ("no_decay", no_decay)):
            n_params = sum(p.numel() for p in grp["params"])
            logger.info(
                f"WeightDecay split '{grp_name}': weight_decay={grp['weight_decay']}; "
                f"{len(grp['params'])} tensors, {n_params} elements; "
                f"params: {grp['names']}."
            )
        # Strip the per-group 'names' bookkeeping before handing to the optimizer.
        for grp in cfg.params:
            grp.pop("names")
        return OPTIMIZERS.build(cfg=cfg)

    if param_dicts is None:
        cfg.params = model.parameters()
    else:
        cfg.params = [dict(names=[], params=[], lr=cfg.lr)]
        for i in range(len(param_dicts)):
            param_group = dict(names=[], params=[])
            if "lr" in param_dicts[i].keys():
                param_group["lr"] = param_dicts[i].lr
            if "momentum" in param_dicts[i].keys():
                param_group["momentum"] = param_dicts[i].momentum
            if "weight_decay" in param_dicts[i].keys():
                param_group["weight_decay"] = param_dicts[i].weight_decay
            cfg.params.append(param_group)

        for n, p in model.named_parameters():
            flag = False
            for i in range(len(param_dicts)):
                if param_dicts[i].keyword in n:
                    cfg.params[i + 1]["names"].append(n)
                    cfg.params[i + 1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                cfg.params[0]["names"].append(n)
                cfg.params[0]["params"].append(p)

        logger = get_root_logger()
        for i in range(len(cfg.params)):
            param_names = cfg.params[i].pop("names")
            message = ""
            for key in cfg.params[i].keys():
                if key != "params":
                    message += f" {key}: {cfg.params[i][key]};"
            logger.info(f"Params Group {i+1} -{message} Params: {param_names}.")
    return OPTIMIZERS.build(cfg=cfg)
