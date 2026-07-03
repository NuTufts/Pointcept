#!/bin/bash
# Quick GPU health + CUDA smoke test. Run this on a fresh node BEFORE the full
# inference job so you don't waste time on a sick GPU.
#
#   bash gpu_healthcheck.sh
#
# PASS = the bottom line prints "CUDA SMOKE TEST OK".
# Any infoROM/ECC warning, Pending retirement, non-Default compute mode, or a
# failed smoke test means: try a different node and report this one to HPC.

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/pointcept_cuml.sif
module load apptainer 2>/dev/null || true

echo "===== node: $(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} ====="

echo "----- GPU health (look for infoROM/ECC/Pending/non-Default) -----"
nvidia-smi -q | grep -iE "Product Name|Bus Id|infoROM|Pending|Compute Mode|Uncorrectable|Pending Page"

echo "----- CUDA smoke test inside container -----"
apptainer exec --nv -B /cluster:/cluster "$container" python3 -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
assert torch.cuda.is_available(), 'torch.cuda.is_available() == False'
x = torch.randn(1024, 1024, device='cuda')
y = (x @ x).sum().item()
torch.cuda.synchronize()
print('device:', torch.cuda.get_device_name(0))
print('matmul result (finite):', y == y)
print('CUDA SMOKE TEST OK')
"
