// BSD 3-Clause License; see https://github.com/scikit-hep/awkward-1.0/blob/main/LICENSE

#define FILENAME(line) FILENAME_FOR_EXCEPTIONS_C("src/cuda-kernels/awkward_NumpyArray_getitem_boolean_numtrue.cu", line)

#include "awkward/kernels.h"
#include "standard_parallel_algorithms.h"
#include "stdio.h"

__global__ void
form_fromptr_mask(int8_t* fromptr_mask,
                  const int8_t* fromptr,
                  int64_t length,
                  int64_t stride) {
  int64_t thread_id = blockDim.x * blockIdx.x + threadIdx.x;

  if (thread_id < length && (thread_id % stride) == 0) {
    if (fromptr[thread_id] != 0) {
    	fromptr_mask[(thread_id / stride)] = 1;
    }
		else {
    	fromptr_mask[(thread_id / stride)] = 0;
		}
  }
}

void
numtrue_sum_reducer(int64_t* numtrue,
                    const int8_t* fromptr,
                    int64_t length,
                    int64_t stride) {
  int8_t* fromptr_mask;

  HANDLE_ERROR(cudaMalloc((void**)&fromptr_mask, sizeof(int8_t) * (length / stride)));

  dim3 blocks_per_grid = blocks(length);
  dim3 thread_per_block = threads(length);

  form_fromptr_mask<<<blocks_per_grid, thread_per_block>>>(
      fromptr_mask, fromptr, length, stride);

  simple_reducer(numtrue, fromptr_mask, (length / stride));

  HANDLE_ERROR(cudaFree(fromptr_mask));
}
ERROR awkward_NumpyArray_getitem_boolean_numtrue(
  int64_t* numtrue,
  const int8_t* fromptr,
  int64_t length,
  int64_t stride) {
	numtrue_sum_reducer(numtrue, fromptr, length, stride);
  return success();
}

