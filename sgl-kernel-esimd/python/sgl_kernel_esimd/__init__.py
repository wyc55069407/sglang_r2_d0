import ctypes
import os

import torch

# if os.path.exists("/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12"):
#     ctypes.CDLL(
#         "/usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.12",
#         mode=ctypes.RTLD_GLOBAL,
#     )

from sgl_kernel_esimd import common_ops
from sgl_kernel_esimd import common_ops_lgrf

from sgl_kernel_esimd.gemm import (
    awq_dequantize,
)
from sgl_kernel_esimd.esimd_ops import (
    esimd_mul_scale_factor_and_add,
    esimd_kernel_uni,
    esimd_mul_lgrf,
    esimd_kernel_uni_lgrf,
    esimd_kernel_uni_large_params,
)

from sgl_kernel_esimd.version import __version__

build_tree_kernel = (
    None  # TODO(ying): remove this after updating the sglang python code.
)
