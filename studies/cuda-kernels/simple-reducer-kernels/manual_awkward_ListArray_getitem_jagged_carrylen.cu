// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line)          \
  FILENAME_FOR_EXCEPTIONS_CUDA( \
      "src/cuda-kernels/awkward_ListArray_getitem_jagged_carrylen.cu", line)

#include "awkward/kernels.h"
#include "standard_parallel_algorithms.h"

template <typename T>
__global__ void
compute_offsets_kernel(int64_t* offsets,
                       const T* slicestarts,
                       const T* slicestops,
                       int64_t sliceouterlen) {
  int64_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_id < sliceouterlen) {
    offsets[thread_id] =
        (int64_t)(slicestops[thread_id] - slicestarts[thread_id]);
  }
}

template <typename T>
void
carrylen_sum_reducer(int64_t* carrylen,
                     const T* slicestarts,
                     const T* slicestops,
                     int64_t sliceouterlen) {
  int64_t* offsets;

  HANDLE_ERROR(cudaMalloc((void**)&offsets, sizeof(int64_t) * sliceouterlen));

  dim3 blocks_per_grid = blocks(sliceouterlen);
  dim3 thread_per_block = threads(sliceouterlen);

  compute_offsets_kernel<<<blocks_per_grid, thread_per_block>>>(
      offsets, slicestarts, slicestops, sliceouterlen);

  simple_reducer(carrylen, offsets, sliceouterlen);

  HANDLE_ERROR(cudaFree(offsets));
}

template <typename T>
ERROR
awkward_ListArray_getitem_jagged_carrylen(int64_t* carrylen,
                                          const T* slicestarts,
                                          const T* slicestops,
                                          int64_t sliceouterlen) {
  carrylen_sum_reducer(carrylen,
                       slicestarts,
                       slicestops,
                       sliceouterlen);
  return success();
}
ERROR
awkward_ListArray_getitem_jagged_carrylen_64(int64_t* carrylen,
                                             const int64_t* slicestarts,
                                             const int64_t* slicestops,
                                             int64_t sliceouterlen) {
  return awkward_ListArray_getitem_jagged_carrylen<int64_t>(
      carrylen, slicestarts, slicestops, sliceouterlen);
}
