#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

#pragma once

static void
HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    int aa = 0;
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    scanf("%d", &aa);
    exit(EXIT_FAILURE);
  }
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

inline dim3
threads(int64_t length) {
  if (length > 1024) {
    return dim3(1024);
  }
  return dim3(length);
}

inline dim3
blocks(int64_t length) {
  if (length > 1024) {
    return dim3(ceil((length) / 1024.0));
  }
  return dim3(1);
}

inline dim3
threads_2d(int64_t length_x, int64_t length_y) {
  if (length_x > 32 && length_y > 32) {
    return dim3(32, 32);
  } else if (length_x > 32 && length_y <= 32) {
    return dim3(32, length_y);
  } else if (length_x <= 32 && length_y > 32) {
    return dim3(length_x, 32);
  } else {
    return dim3(length_x, length_y);
  }
}

inline dim3
blocks_2d(int64_t length_x, int64_t length_y) {
  if (length_x > 32 && length_y > 32) {
    return dim3(ceil(length_x / 32.0), ceil(length_y / 32.0));
  } else if (length_x > 32 && length_y <= 32) {
    return dim3(ceil(length_x / 32.0), 1);
  } else if (length_x <= 32 && length_y > 32) {
    return dim3(1, ceil(length_y / 32.0));
  } else {
    return dim3(1, 1);
  }
}

template <typename T>
__global__ void
exclusive_scan_kernel(T* d_in,
                      T* d_out,
                      T* d_final,
                      int64_t curr_step,
                      int64_t total_steps,
                      int64_t stride,
                      bool in_out_flag,
                      int64_t length) {
  int64_t block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

  int64_t thread_id = block_id * blockDim.x + threadIdx.x;
  int64_t sum = 0;

  if (thread_id < length) {
    if (!in_out_flag) {
      if (thread_id < stride) {
        sum = d_out[thread_id];
        d_in[thread_id] = sum;
      } else {
        sum = d_out[thread_id] + d_out[thread_id - stride];
        d_in[thread_id] = sum;
      }
    } else {
      if (thread_id < stride) {
        sum = d_in[thread_id];
        d_out[thread_id] = sum;
      } else {
        sum = d_in[thread_id] + d_in[thread_id - stride];
        d_out[thread_id] = sum;
      }
    }

    if (curr_step == total_steps) {
      d_final[thread_id] = sum;
    }
  }
}

template <typename T, typename F>
__global__ void
copy(T* out, F* in, int64_t length) {
  int64_t block_id =
      blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;

  int64_t thread_id = block_id * blockDim.x + threadIdx.x;

  if (thread_id < length) {
    out[thread_id] = (T)(in[thread_id]);
  }
}

template <typename T, typename F>
void
exclusive_scan(T* out, const F* in, int64_t length) {
  T* d_in;
  T* d_out;

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  HANDLE_ERROR(cudaMalloc((void**)&d_in, length * sizeof(T)));
  HANDLE_ERROR(cudaMalloc((void**)&d_out, length * sizeof(T)));

  copy<<<blocks_per_grid, threads_per_block>>>(d_in, in, length);

  int64_t stride = 1;
  int64_t total_steps = ceil(log2(static_cast<float>(length)));

  for (int64_t curr_step = 1; curr_step <= total_steps; curr_step++) {
    bool in_out_flag = (curr_step % 2) != 0;
    exclusive_scan_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_in, d_out, out, curr_step, total_steps, stride, in_out_flag, length);
    stride = stride * 2;
  }
}

template <typename C>
__global__ void
simple_reducer_kernel(int64_t* d_out, const C* in, int64_t length) {
  __shared__ int64_t arr[1024];
  int64_t tid = threadIdx.x;
  int64_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  if(thread_id < length) {
    arr[tid] = in[thread_id];
  } else {
    arr[tid] = 0;
  }
  __syncthreads();

  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    int index = 2 * s * tid;
    if (index < blockDim.x) {
      if (index + s < length) {
        arr[index] += arr[index + s];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = arr[0];
  }
}

void inline simple_reducer_recurse(int64_t* d_out,
                                   int64_t* in,
                                   int64_t length) {
  if (length == 1) {
    return;
  }

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  simple_reducer_kernel<<<blocks_per_grid, threads_per_block>>>(
      d_out, in, length);

  length = ceil(static_cast<float>(length) / 1024.0);

  simple_reducer_recurse(d_out, d_out, length);
}

template <typename C>
void
simple_reducer(int64_t* out, const C* in, int64_t length) {
  int64_t* d_out;

  HANDLE_ERROR(
      cudaMalloc((void**)&d_out,
                 sizeof(int64_t) * ceil(static_cast<float>(length) / 1024.0)));
  HANDLE_ERROR(cudaMemset(
      d_out, 0, sizeof(int64_t) * ceil(static_cast<float>(length) / 1024.0)));

  dim3 blocks_per_grid = blocks(length);
  dim3 threads_per_block = threads(length);

  simple_reducer_kernel<<<blocks_per_grid, threads_per_block>>>(
      d_out, in, length);

  length = ceil(static_cast<float>(length) / 1024.0);
  if (length > 1) {
    simple_reducer_recurse(d_out, d_out, length);
  }
  HANDLE_ERROR(cudaMemcpy(out, d_out, sizeof(int64_t), cudaMemcpyDefault));
  HANDLE_ERROR(cudaFree(d_out));
}

