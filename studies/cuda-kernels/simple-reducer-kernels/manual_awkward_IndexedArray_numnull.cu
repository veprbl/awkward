// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line)          \
  FILENAME_FOR_EXCEPTIONS_CUDA( \
      "src/cuda-kernels/awkward_IndexedArray_numnull.cu", line)

#include "awkward/kernels.h"
#include "standard_parallel_algorithms.h"

template <typename C>
__global__ void
fromindex_mask_kernel(int8_t* fromindex_mask, const C* fromindex, int64_t lenindex) {
  int64_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_id < lenindex) {
    if (fromindex[thread_id] < 0) {
      fromindex_mask[thread_id] = 1;
    }
  }
}

template <typename C>
void
numnull_sum_reducer(int64_t* numnull, const C* fromindex, int64_t lenindex) {
  int8_t* fromindex_mask;

  HANDLE_ERROR(cudaMalloc((void**)&fromindex_mask, sizeof(int8_t) * lenindex));

  dim3 blocks_per_grid = blocks(lenindex);
  dim3 thread_per_block = threads(lenindex);

  fromindex_mask_kernel<<<blocks_per_grid, thread_per_block>>>(
      fromindex_mask, fromindex, lenindex);

  simple_reducer(numnull, fromindex_mask, lenindex);

  HANDLE_ERROR(cudaFree(fromindex_mask));
}
template <typename C>
ERROR
awkward_IndexedArray_numnull(int64_t* numnull,
                             const C* fromindex,
                             int64_t lenindex) {
  numnull_sum_reducer(numnull, fromindex, lenindex);
  return success();
}
ERROR
awkward_IndexedArray32_numnull(int64_t* numnull,
                               const int32_t* fromindex,
                               int64_t lenindex) {
  return awkward_IndexedArray_numnull<int32_t>(numnull, fromindex, lenindex);
}
ERROR
awkward_IndexedArrayU32_numnull(int64_t* numnull,
                                const uint32_t* fromindex,
                                int64_t lenindex) {
  return awkward_IndexedArray_numnull<uint32_t>(numnull, fromindex, lenindex);
}
ERROR
awkward_IndexedArray64_numnull(int64_t* numnull,
                               const int64_t* fromindex,
                               int64_t lenindex) {
  return awkward_IndexedArray_numnull<int64_t>(numnull, fromindex, lenindex);
}
