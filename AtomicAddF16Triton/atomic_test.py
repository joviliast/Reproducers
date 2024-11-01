import torch

import triton  # @manual=//triton:tritonn
import triton.language as tl  # @manual=//triton:tritonl


@triton.jit
def atomic_kernel(x_ptr, y_ptr, BLOCK_SIZE: tl.constexpr):
   pid = tl.program_id(0)
   offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
   x = tl.load(x_ptr + offsets)
   tl.atomic_add(y_ptr + tl.arange(0, BLOCK_SIZE), x)


def run_once(x, y, BLOCK_SIZE):
   n = x.shape[0]
   assert n % BLOCK_SIZE == 0
   tile_num = n // BLOCK_SIZE
   atomic_kernel[(tile_num,)](x, y, BLOCK_SIZE=BLOCK_SIZE)


def test(n, BLOCK_SIZE=32, dtype=torch.bfloat16):
   x = torch.rand(n, dtype=dtype, device="cuda")
   y = torch.rand(BLOCK_SIZE, dtype=dtype, device="cuda")
   y_ref = y.clone().detach()
   run_once(x, y, BLOCK_SIZE)
   y_ref += x.reshape(-1, BLOCK_SIZE).sum(dim=0)
   torch.testing.assert_close(y, y_ref)


@triton.testing.perf_report(
   triton.testing.Benchmark(
       x_names=["size"],
       x_vals=[1024 * i for i in range(1, 28, 1)],
       line_arg="dtype",
       line_vals=[torch.bfloat16, torch.float16],
       line_names=["bf16", "fp16"],
       styles=[("blue", "-"), ("green", "-")],
       ylabel="us",
       plot_name="atomicadd",
       args={},
   )
)
def benchmark(size, dtype):
   BLOCK_SIZE = 32
   quantiles = [0.5, 0.2, 0.8]
   x = torch.rand(size, dtype=dtype, device="cuda")
   y = torch.rand(BLOCK_SIZE, dtype=dtype, device="cuda")
   ms, min_ms, max_ms = triton.testing.do_bench(
       lambda: run_once(x, y, BLOCK_SIZE), quantiles=quantiles
   )
   ms2us = lambda ms: round(ms * 1000.0, 3)
   return ms2us(ms), ms2us(max_ms), ms2us(min_ms)


def main():
   benchmark.run(print_data=True, show_plots=True, save_path=".")
   # test(1024, dtype=torch.float16)

if __name__ == "__main__":
   main()

