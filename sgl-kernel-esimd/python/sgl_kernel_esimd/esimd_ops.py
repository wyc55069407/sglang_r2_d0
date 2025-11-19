from typing import List, Optional, Tuple

import torch


def esimd_mul_scale_factor_and_add(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, len: int, factor: float
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel_esimd.esimd_add(a, b, c, len, factor)

def esimd_kernel_uni(
    t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, t4: torch.Tensor, t5: torch.Tensor, t6: torch.Tensor, t7: torch.Tensor, t8: torch.Tensor, t9: torch.Tensor,
    i0: int, i1: int, i2: int, i3: int, i4: int, i5: int, i6: int, i7: int, i8: int, i9: int, 
    f0: float, f1: float, f2: float, f3: float, f4: float,
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel_esimd.esimd_kernel_uni(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, f0, f1, f2, f3, f4)

def esimd_kernel_uni_large_params(
    t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, t4: torch.Tensor, t5: torch.Tensor, t6: torch.Tensor, t7: torch.Tensor, t8: torch.Tensor, t9: torch.Tensor, t10: torch.Tensor, 
    t11: torch.Tensor, t12: torch.Tensor, t13: torch.Tensor, t14: torch.Tensor,t15: torch.Tensor, t16: torch.Tensor, t17: torch.Tensor, t18: torch.Tensor,t19: torch.Tensor,
    i0: int, i1: int, i2: int, i3: int, i4: int, i5: int, i6: int, i7: int, i8: int, i9: int, i10: int, i11: int, i12: int, i13: int, i14: int,i15: int, i16: int, i17: int, i18: int, i19: int,
    f0: float, f1: float, f2: float, f3: float, f4: float,
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel_esimd.esimd_kernel_uni_large_params(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, 
                                                                    i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, f0, f1, f2, f3, f4)

def esimd_kernel_uni_huge_params(
    t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, t4: torch.Tensor, t5: torch.Tensor, t6: torch.Tensor, t7: torch.Tensor, t8: torch.Tensor, t9: torch.Tensor, t10: torch.Tensor, 
    t11: torch.Tensor, t12: torch.Tensor, t13: torch.Tensor, t14: torch.Tensor,t15: torch.Tensor, t16: torch.Tensor, t17: torch.Tensor, t18: torch.Tensor,t19: torch.Tensor,t20: torch.Tensor,
    t21: torch.Tensor, t22: torch.Tensor, t23: torch.Tensor, t24: torch.Tensor,t25: torch.Tensor,
    i0: int, i1: int, i2: int, i3: int, i4: int, i5: int, i6: int, i7: int, i8: int, i9: int,
    f0: float, f1: float, f2: float, f3: float, f4: float,
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel_esimd.esimd_kernel_uni_huge_params(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25,
                                                                    i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, f0, f1, f2, f3, f4)


def esimd_mul_lgrf(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, flag: int, len: int
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel_esimd.esimd_mul_lgrf(a, b, c, flag, len)

def esimd_kernel_uni_lgrf(
    t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, t3: torch.Tensor, t4: torch.Tensor, t5: torch.Tensor, t6: torch.Tensor, t7: torch.Tensor, t8: torch.Tensor, t9: torch.Tensor,
    i0: int, i1: int, i2: int, i3: int, i4: int, i5: int, i6: int, i7: int, i8: int, i9: int, 
    f0: float, f1: float, f2: float, f3: float, f4: float,
) -> torch.ByteTensor:
    return torch.ops.sgl_kernel_esimd.esimd_kernel_uni_lgrf(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, f0, f1, f2, f3, f4)
