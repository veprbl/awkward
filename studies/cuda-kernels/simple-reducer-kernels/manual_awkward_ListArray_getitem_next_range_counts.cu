// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line)          \
  FILENAME_FOR_EXCEPTIONS_CUDA( \
      "src/cuda-kernels/awkward_ListArray_getitem_next_range_counts.cu", line)

#include "awkward/kernels.h"
#include "standard_parallel_algorithms.h"

template <typename C>
__global__ void
fromoffsets_offsets_kernel(int64_t* fromoffsets_range_mask,
                           const C* fromoffsets,
                           int64_t lenstarts) {
  int64_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_id < lenstarts) {
    fromoffsets_range_mask[thread_id] =
        static_cast<int64_t>(fromoffsets[thread_id + 1]) -
        static_cast<int64_t>(fromoffsets[thread_id]);
  }
}

template <typename C>
void
total_sum_reducer(int64_t* total, const C* fromoffsets, int64_t lenstarts) {
  int64_t* fromoffsets_range_mask;  // Couldn't figure out a better variable name

  HANDLE_ERROR(
      cudaMalloc((void**)&fromoffsets_range_mask, sizeof(int64_t) * lenstarts));

  dim3 blocks_per_grid = blocks(lenstarts);
  dim3 thread_per_block = threads(lenstarts);

  fromoffsets_offsets_kernel<<<blocks_per_grid, thread_per_block>>>(
      fromoffsets_range_mask, fromoffsets, lenstarts);

  simple_reducer(total, fromoffsets_range_mask, lenstarts);

  HANDLE_ERROR(cudaFree(fromoffsets_range_mask));
}

template <typename C>
ERROR awkward_ListArray_getitem_next_range_counts(
  int64_t* total,
  const C* fromoffsets,
  int64_t lenstarts) {
	total_sum_reducer(total, fromoffsets, lenstarts);
  return success();
}
ERROR awkward_ListArray32_getitem_next_range_counts_64(
  int64_t* total,
  const int32_t* fromoffsets,
  int64_t lenstarts) {
  return awkward_ListArray_getitem_next_range_counts<int32_t>(
    total,
    fromoffsets,
    lenstarts);
}
ERROR awkward_ListArrayU32_getitem_next_range_counts_64(
  int64_t* total,
  const uint32_t* fromoffsets,
  int64_t lenstarts) {
  return awkward_ListArray_getitem_next_range_counts<uint32_t>(
    total,
    fromoffsets,
    lenstarts);
}
ERROR awkward_ListArray64_getitem_next_range_counts_64(
  int64_t* total,
  const int64_t* fromoffsets,
  int64_t lenstarts) {
  return awkward_ListArray_getitem_next_range_counts<int64_t>(
    total,
    fromoffsets,
    lenstarts);
}
