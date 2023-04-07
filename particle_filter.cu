#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <float.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <curand_kernel.h>

#define THREADSIZE 4
#define BLOCKSIZE 4

__device__ void propagate(
    int index, float* X1_prev, float* X2_prev, const float* L, float dt) 
{

    float g = 9.81;
    curandState state;
    curand_init(index, 0, 0, &state);

    float w1 = curand_normal(&state);
    float w2 = curand_normal(&state);

    X1_prev[index] = X1_prev[index] + X2_prev[index] * dt + w1 * L[0] + w2 * L[2];
    X2_prev[index] = X2_prev[index] - g * sin(X1_prev[index]) * dt + w1 * L[1] + w2 * L[3];
}

__device__ void log_likehood_pendulum(
    int index, float* x1, float* W, float y, float stdev, int N)
{

    float var = stdev * stdev; // sigma^2

    float log_likehood = -0.5 * N * log(2 * M_PI) - 0.5 * N * log(var);
    float diff = y - sin(x1[index]);
    log_likehood += -0.5 / var * (diff * diff);
    W[index] = log_likehood;

}

__device__ void max_ws(int index, float* ws, float* result, int size) {

    extern __shared__ float shared_ws[THREADSIZE];
    int tx = threadIdx.x;
    shared_ws[tx] = (index < size) ? ws[index] : -FLT_MAX;
    __syncthreads();

    for (int i = blockDim.x / 2; i > 0; i >>= 1) {

        if (tx < i) {
            shared_ws[tx] = (shared_ws[tx] < shared_ws[tx + i]) ? shared_ws[tx + i] : shared_ws[tx];
        }
        __syncthreads();


    }

    if (threadIdx.x == 0)
        result[blockIdx.x] = shared_ws[0];
}


__device__ void max_value(
    int index, float* W, int N, float* result)
{
    int size = N;
    while (size > 1) {
        max_ws(index, W, result, size);
        size = ceil(float(size) / THREADSIZE);
    }
}


__global__ void particleFilter(float* X1, float* X2, float dt, int N,int J, float* L,float* x1_prev, float* x2_prev, float* W, float* y, float* max_val, float stdev) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    for (int n = 0; n < N; n++) {
        int matrix_ind = index + n * N;

        propagate(index, x1_prev, x2_prev, L, dt);
        __syncthreads();
        log_likehood_pendulum(index, x1_prev, W, y[n], stdev, N);
        __syncthreads();
        max_value(index, W, N, max_val);
        __syncthreads();
        X1[matrix_ind] = max_val[0];
        X2[matrix_ind] = max_val[0];
        
        
	}
}