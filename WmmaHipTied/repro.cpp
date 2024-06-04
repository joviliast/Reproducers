// Wave Matrix Multiply Accumulate (WMMA) using HIP compiler intrinsic
// Does a matrix multiplication of two 16x16, fp16 matrices, and stores them into a 16x16 fp16 result matrix

#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <chrono>
using namespace std::chrono;
using namespace std;

// Use half16 as an alias of the internal clang vector type of 16 fp16 values
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));

__global__ void wmma_matmul(__half* a, __half* b, __half* c)
{
    const int gIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lIdx = threadIdx.x;

    // a and b fragments are stored in 8 VGPRs each, in packed format, so 16 elements each for a and b
    // a_frag will store one column of the 16x16 matrix A tile
    // b_frag will store one row of the 16x16 matrix B tile
    half16 a_frag0 = {};
    half16 a_frag1 = {};
    half16 a_frag2 = {};
    half16 a_frag3 = {};
    half16 a_frag4 = {};
    half16 a_frag5 = {};
    half16 a_frag6 = {};
    half16 a_frag7 = {};
    half16 b_frag;
    // initialize c fragment to 0
    half16 c_frag0 = {};
    half16 c_frag1 = {};
    half16 c_frag2 = {};
    half16 c_frag3 = {};

    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    const int lane = lIdx % 16;

    for (int ele = 0; ele < 16; ++ele)
    {
        b_frag[ele] = b[16*ele + lane];
    }

    for (int ele = 0; ele < 16; ++ele)
    {
        a_frag0[ele] = a[16 * lane + ele];
        a_frag1[ele] = a[16 * lane + ele + 16 * 16];
        a_frag2[ele] = a[16 * lane + ele + 2 * 16 * 16];
        a_frag3[ele] = a[16 * lane + ele + 3 * 16 * 16];
        a_frag4[ele] = a[16 * lane + ele + 4 * 16 * 16];
        a_frag5[ele] = a[16 * lane + ele + 5 * 16 * 16];
        a_frag6[ele] = a[16 * lane + ele + 6 * 16 * 16];
        a_frag7[ele] = a[16 * lane + ele + 7 * 16 * 16];
    }

    // call the WMMA intrinsic with OPSEL set to "false"
    c_frag0 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag0, b_frag, c_frag0, false);
    c_frag0 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag1, b_frag, c_frag0, true);
    c_frag1 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag2, b_frag, c_frag1, false);
    c_frag1 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag3, b_frag, c_frag1, true);
    c_frag2 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag4, b_frag, c_frag2, false);
    c_frag2 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag5, b_frag, c_frag2, true);
    c_frag3 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag6, b_frag, c_frag3, false);
    c_frag3 = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag7, b_frag, c_frag3, true);

    for (int ele = 0; ele < 8; ++ele)
    {
        const int r = ele * 2 + (lIdx / 16);
        c[16 * r + lane] = c_frag0[ele*2];
        c[16 * r + lane + 16 * 16] = c_frag0[ele*2 + 1];
        c[16 * r + lane + 2 * 16 * 16] = c_frag1[ele*2];
        c[16 * r + lane + 3 * 16 * 16] = c_frag1[ele*2 + 1];
        c[16 * r + lane + 4 * 16 * 16] = c_frag2[ele*2];
        c[16 * r + lane + 5 * 16 * 16] = c_frag2[ele*2 + 1];
        c[16 * r + lane + 6 * 16 * 16] = c_frag3[ele*2];
        c[16 * r + lane + 7 * 16 * 16] = c_frag3[ele*2 + 1];
    }

}

int main(int argc, char* argv[])

{
    __half a[128 * 16] = {};
    __half b[16 * 16] = {};
    __half c[128 * 16] = {};
    __half *a_gpu, *b_gpu, *c_gpu;
    hipMalloc(&a_gpu, 128*16 * sizeof(__half));
    hipMalloc(&b_gpu, 16*16 * sizeof(__half));
    hipMalloc(&c_gpu, 128*16 * sizeof(__half));


    // fill in some data into matrices A and B
    for (int i = 0; i < 16; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            b[i * 16 + j] = (__half)0.f;
        }
    }
    for (int i = 0; i < 128; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            a[i * 16 + j] = (__half)1.f;
        }
    }

    hipMemcpy(a_gpu, a, (128*16) * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(b_gpu, b, (16*16) * sizeof(__half), hipMemcpyHostToDevice);
    hipMemcpy(c_gpu, c, (128*16) * sizeof(__half), hipMemcpyHostToDevice);

    auto start = high_resolution_clock::now();
    for (int i = 0; i < 100; ++i){
    wmma_matmul<<<dim3(1), dim3(128, 1, 1), 0, 0>>>(a_gpu, b_gpu, c_gpu);
    hipMemcpy(c, c_gpu, (128 * 16) * sizeof(__half), hipMemcpyDeviceToHost);
    }
    auto stop = high_resolution_clock::now();
    hipFree(a_gpu);
    hipFree(b_gpu);
    hipFree(c_gpu);
    for (int i = 0; i < 128; ++i)
    {
        for (int j = 0; j < 16; ++j)
        {
            printf("%f ", (float)c[i * 16 + j]);
        }
        printf("\n");
    }

    auto duration = duration_cast<microseconds>((stop - start)/100);

// To get the value of duration use the count()
// member function on the duration object
cout << "time: " << duration.count() << endl;
//42
//135440

    return 0;
}
