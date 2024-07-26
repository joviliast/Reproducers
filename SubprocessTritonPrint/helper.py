import torch
import triton
import triton.language as tl

@triton.jit
def kernel_device(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.device_print("x: ", x)
    tl.store(Y + tl.arange(0, BLOCK), x)

x = torch.arange(0, 128, dtype=torch.int32, device='cuda').to(getattr(torch, 'int32'))
y = torch.zeros((128, ), dtype=x.dtype, device='cuda')
kernel_device[(1, )](x, y, num_warps=4, BLOCK=128)
