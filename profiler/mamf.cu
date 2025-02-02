#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h> // For FP16 data type
#include <chrono>

#define CUDA_CHECK(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(status) \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Kernel to initialize FP16 matrices
__global__ void init_matrix_fp16(__half* matrix, int size, __half value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        matrix[idx] = value;
    }
}

int main() {
    // Matrix dimensions
    const int N = 4096*8; // Matrix size N x N
    const int repeat_count = 20; // Number of repetitions
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);

    // Calculate the number of floating-point operations
    long long flops_per_matrix = 2LL * N * N * N;
    long long total_flops = flops_per_matrix * repeat_count;

    // Allocate device memory for FP16 matrices
    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * N * sizeof(__half)));

    // Initialize matrices on device
    int threads_per_block = 256;
    int blocks = (N * N + threads_per_block - 1) / threads_per_block;
    init_matrix_fp16<<<blocks, threads_per_block>>>(d_A, N * N, __float2half(1.0f));
    init_matrix_fp16<<<blocks, threads_per_block>>>(d_B, N * N, __float2half(1.0f));
    init_matrix_fp16<<<blocks, threads_per_block>>>(d_C, N * N, __float2half(0.0f));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Create cuBLAS handle and enable Tensor Core
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Measure time for repeated matrix multiplications
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < repeat_count; ++i) {
        CUBLAS_CHECK(cublasGemmEx(handle, 
                                  CUBLAS_OP_N, CUBLAS_OP_N, 
                                  N, N, N, 
                                  &alpha, 
                                  d_A, CUDA_R_16F, N, 
                                  d_B, CUDA_R_16F, N, 
                                  &beta, 
                                  d_C, CUDA_R_16F, N, 
                                  CUDA_R_16F, // Compute type: FP16
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Calculate performance in TFLOPS
    double tflops = (total_flops / elapsed.count()) / 1e12;

    // Print results
    std::cout << "Matrix size: " << N << " x " << N << std::endl;
    std::cout << "Repetitions: " << repeat_count << std::endl;
    std::cout << "Time elapsed: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;

    // Cleanup
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
