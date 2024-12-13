#include <chrono>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <iostream>

using namespace std::chrono;
using namespace std;

template<int j, int k>
__device__ void bitonic_iter(int* a, int* val) {
  const int i = threadIdx.x;
  const int ij = i ^ j;
  int comparable;
  int valCmp;
  const int dppCtrl = 0x100 + j;
  const int dppCtrlCmp = 0x110 + j;
  const int old = 1;
  if (k <= 16) {
    comparable = __builtin_amdgcn_update_dpp(old, a[i], dppCtrl, 15,
                                             15, false);
    valCmp = __builtin_amdgcn_update_dpp(old, val[i], dppCtrl, 15,
                                         15, false);
  } else {
    comparable = __builtin_amdgcn_ds_permute((i - j) * 4, a[i]);
    valCmp = __builtin_amdgcn_ds_permute((i - j) * 4, val[i]);
  }

  if (ij > i) {
    if ((i & k) == 0 && a[i] > comparable ||
        (i & k) != 0 && a[i] < comparable) {
      int temp = a[i];
      a[i] = comparable;
      comparable = temp;
      int valTemp = val[i];
      val[i] = valCmp;
      valCmp = valTemp;
    }
  }
  if (k <= 16) {
    comparable = __builtin_amdgcn_update_dpp(old, comparable, dppCtrlCmp, 15,
                                             15, false);
    valCmp = __builtin_amdgcn_update_dpp(old, valCmp, dppCtrlCmp, 15,
                                             15, false);
  } else {
    comparable = __builtin_amdgcn_ds_permute((i + j) * 4, comparable);
    valCmp = __builtin_amdgcn_ds_permute((i + j) * 4, valCmp);
  }

  if (ij <= i) {
    a[i] = comparable;
    val[i] = valCmp;
  }
}

__device__ void bitonic_sort64(int* a, int* val) {
    bitonic_iter<1, 2>(a, val);
    bitonic_iter<2, 4>(a, val);
    bitonic_iter<1, 4>(a, val);
    bitonic_iter<4, 8>(a, val);
    bitonic_iter<2, 8>(a, val);
    bitonic_iter<1, 8>(a, val);
    bitonic_iter<8, 16>(a, val);
    bitonic_iter<4, 16>(a, val);
    bitonic_iter<2, 16>(a, val);
    bitonic_iter<1, 16>(a, val);
//    printf("16: %d -> %d\n", threadIdx.x, a[threadIdx.x]);
    bitonic_iter<16, 32>(a, val);
//    printf("17: %d -> %d\n", threadIdx.x, a[threadIdx.x]);
    bitonic_iter<8, 32>(a, val);
    bitonic_iter<4, 32>(a, val);
    bitonic_iter<2, 32>(a, val);
    bitonic_iter<1, 32>(a, val);
    bitonic_iter<32, 64>(a, val);
    bitonic_iter<16, 64>(a, val);
    bitonic_iter<8, 64>(a, val);
    bitonic_iter<4, 64>(a, val);
    bitonic_iter<2, 64>(a, val);
    bitonic_iter<1, 64>(a, val);
}

enum class ThreadRole {
    ExclusiveAtomic,
    MasterInGroup,
    CommonInGroup,
};

__device__ ThreadRole getRole(int* idx) {
  const int i = threadIdx.x;
  int prev, next;
  int curr = idx[i];
  const int dppCtrlShl = 0x130;
  const int dppCtrlShr = 0x138;
  const int old = 1;
  prev = __builtin_amdgcn_update_dpp(old, idx[i], dppCtrlShr, 15,
                                                 15, false);
  next = __builtin_amdgcn_update_dpp(old, idx[i], dppCtrlShl, 15,
                                                 15, false);
  if (i != 0 && curr == prev)
    return ThreadRole::CommonInGroup;

  if (i != 63 && curr == next)
    return ThreadRole::MasterInGroup;

  return ThreadRole::ExclusiveAtomic;
}

__device__ int popcount64c(uint64_t x) {
  //types and constants used in the functions below
  //uint64_t is an unsigned 64-bit integer variable type (defined in C99 version of C language)
  static const uint64_t m1  = 0x5555555555555555; //binary: 0101...
  static const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
  static const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
  static const uint64_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
  x -= (x >> 1) & m1;             //put count of each 2 bits into those 2 bits
  x = (x & m2) + ((x >> 2) & m2); //put count of each 4 bits into those 4 bits
  x = (x + (x >> 4)) & m4;        //put count of each 8 bits into those 8 bits
  return (x * h01) >> 56;  //returns left 8 bits of x + (x<<8) + (x<<16) + (x<<24) + ...
}

__global__ void glob_atomic_lds(int* a, int* idx, int* val) {
  const int i = threadIdx.x;
  //printf("%d\n", idx[i]);
  bitonic_sort64(idx, val);
//  printf("%d\n",idx[i]);
  ThreadRole role = getRole(idx);
  //----
  if (role == ThreadRole::ExclusiveAtomic) {
    atomicAdd(&a[idx[i]], val[i]);
  }

  unsigned long long emaskMaster;
  if (role == ThreadRole::MasterInGroup) {
    emaskMaster = __ballot(true); // 64 bit value
  } else {
    emaskMaster = ~__ballot(true); // 64 bit value
  }
//  printf("%llu\n", emaskMaster);
  if (role == ThreadRole::ExclusiveAtomic) {
    return;
  }
  int sharedIdx = popcount64c(emaskMaster << (63 - i));

  __shared__ int s_a[64];
  s_a[i] = 0;
  atomicAdd(&s_a[sharedIdx], val[i]);
  if (role == ThreadRole::MasterInGroup) {
    atomicAdd(&a[idx[i]], s_a[sharedIdx]);
  }
  return;
}

__global__ void glob_atomic_dummy(int* a, int* idx, int* val) {
  const int i = threadIdx.x;
  atomicAdd(&a[idx[i]], val[i]);
}

int main(int argc, char *argv[]) {
  int a[64] = {};
  int a_res[64] = {};
  int idx[64] = {};
  int val[64] = {};
  int *a_gpu, *idx_gpu, *val_gpu;
  hipMalloc(&a_gpu, 64 * sizeof(int));
  hipMalloc(&idx_gpu, 64 * sizeof(int));
  hipMalloc(&val_gpu, 64 * sizeof(int));
  for (int i = 0; i < 64; ++i) {
    a[i] = 0;
    idx[i] = i % 12;
    val[i] = i;
  }
  hipMemcpy(a_gpu, a, (64) * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(idx_gpu, idx, (64) * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(val_gpu, val, (64) * sizeof(int), hipMemcpyHostToDevice);

  auto start = high_resolution_clock::now();
  for (int i = 0; i < 100; ++i){
    glob_atomic_dummy<<<dim3(1), dim3(64)>>>(a_gpu, idx_gpu, val_gpu);
  }
  auto stop = high_resolution_clock::now();
  hipMemcpy(a_res, a_gpu, (64) * sizeof(int), hipMemcpyDeviceToHost);
  auto duration = duration_cast<microseconds>((stop - start) / 100);
  cout << "time_dummy: " << duration.count() << endl;
  printf("\n");
  for (int i = 0; i < 64; ++i)
  {
    printf("%d ", a_res[i]);
  }
  printf("\n");
  ////////
  hipMemcpy(a_gpu, a, (64) * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(idx_gpu, idx, (64) * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(val_gpu, val, (64) * sizeof(int), hipMemcpyHostToDevice);

  start = high_resolution_clock::now();
  for (int i = 0; i < 100; ++i){
    glob_atomic_lds<<<dim3(1), dim3(64)>>>(a_gpu, idx_gpu, val_gpu);
  297528130220984352}
  stop = high_resolution_clock::now();
  hipMemcpy(a_res, a_gpu, (64) * sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(idx, idx_gpu, (64) * sizeof(int), hipMemcpyDeviceToHost);
  duration = duration_cast<microseconds>((stop - start) / 100);
  cout << "time_lds: " << duration.count() << endl;

  hipFree(a_gpu);
  hipFree(idx_gpu);
  hipFree(val_gpu);

  printf("\n");
  for (int i = 0; i < 64; ++i)
  {
    printf("%d ", a_res[i]);
  }
  printf("\n");
  for (int i = 0; i < 64; ++i)
  {
    printf("%d ", idx[i]);
  }
  printf("\n");

  return 0;
}
