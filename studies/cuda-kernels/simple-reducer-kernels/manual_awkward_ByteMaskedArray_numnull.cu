// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line)          \
  FILENAME_FOR_EXCEPTIONS_CUDA( \
      "src/cuda-kernels/awkward_ByteMaskedArray_numnull.cu", line)

#include "awkward/kernels.h"
#include "standard_parallel_algorithms.h"

__global__ void
form_validwhen_mask(int8_t* validwhen_mask,
                    const int8_t* mask,
                    int64_t length,
                    bool validwhen) {
  int64_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_id < length) {
    if ((mask[thread_id] != 0) != validwhen) {
      validwhen_mask[thread_id] = 1;
    }
  }
}

void
numnull_sum_reducer(int64_t* numnull,
                    const int8_t* mask,
                    int64_t length,
                    bool validwhen) {
  int8_t* validwhen_mask;

  HANDLE_ERROR(cudaMalloc((void**)&validwhen_mask, sizeof(int8_t) * length));

  dim3 blocks_per_grid = blocks(length);
  dim3 thread_per_block = threads(length);

  form_validwhen_mask<<<blocks_per_grid, thread_per_block>>>(
      validwhen_mask, mask, length, validwhen);

  simple_reducer(numnull, validwhen_mask, length);

  HANDLE_ERROR(cudaFree(validwhen_mask));
}

ERROR
awkward_ByteMaskedArray_numnull(int64_t* numnull,
                                const int8_t* mask,
                                int64_t length,
                                bool validwhen) {
  numnull_sum_reducer(numnull, mask, length, validwhen);
  return success();
}
