#include <stdio.h>
#include <array>
#include <numeric>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "cuda_runtime.h"

#include "Matrix.h"
#include "CudaHelpers.h"

const unsigned int TILE_SIZE_X = 16;
const unsigned int TILE_SIZE_Y = TILE_SIZE_X;
const unsigned int DIM_X = 1024;
const unsigned int DIM_Y = DIM_X;
const unsigned int NUM_REPS = 100;

template <typename T>
__global__ void baseLineCopy(const T * const in, T * const out, unsigned int width, unsigned int height) {
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idxIn = yIndex * width + xIndex;

	out[idxIn] = in[idxIn];
}

template <typename T>
__global__ void baseLineCopyShared(const T * const in, T * const out, unsigned int width, unsigned int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X];
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idxIn = yIndex * width + xIndex;

	cache[threadIdx.y][threadIdx.x] = in[idxIn];
	__syncthreads();

	out[idxIn] = cache[threadIdx.y][threadIdx.x];
}

template <typename T>
__global__ void naiveTranspose(const T * const in, T * const out, unsigned int width, unsigned int height) {
	for (unsigned int j = 0; j < height; j++) {
		for (unsigned int i = 0; i < width; i++) {
			auto idxIn = j * width + i;
			auto idxOut = i * height + j;
			out[idxOut] = in[idxIn];
		}
	}
}

template <typename T>
__global__ void naiveParallelTranspose(const T * const in, T * const out, unsigned int width, unsigned int height) {
	unsigned int idxIn = blockIdx.x * width + threadIdx.x;
	unsigned int idxOut = threadIdx.x * height + blockIdx.x;

	out[idxOut] = in[idxIn];
}

template <typename T>
__global__ void naiveBlockWiseParallelTranspose(const T * const in, T * const out, unsigned int width, unsigned int height) {
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idxIn = yIndex * width + xIndex;
	unsigned int idxOut = xIndex * height + yIndex;

	out[idxOut] = in[idxIn];
}

template <typename T>
__global__ void naiveSharedBlockWiseParallelTranspose(const T * const in, T * const out, unsigned int width, unsigned int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X];
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idxIn = yIndex * width + xIndex;
	unsigned int idxOut = xIndex * height + yIndex;

	cache[threadIdx.y][threadIdx.x] = in[idxIn];
	__syncthreads();

	out[idxOut] = cache[threadIdx.y][threadIdx.x];
}

template <typename T>
__global__ void coalescedSharedBlockWiseParallelTranspose(const T * const in, T * const out, unsigned int width, unsigned int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X];
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idxIn = yIndex * width + xIndex;

	// write out consecutively and do transposing by swapping x and y when reading from shared memory
	unsigned int xIndexOut = blockIdx.y * blockDim.y + threadIdx.x;
	unsigned int yIndexOut = blockIdx.x * blockDim.x + threadIdx.y;
	unsigned int idxOut = yIndexOut * height + xIndexOut;

	cache[threadIdx.y][threadIdx.x] = in[idxIn];
	__syncthreads();

	out[idxOut] = cache[threadIdx.x][threadIdx.y];
}

template <typename T>
__global__ void coalescedSharedBlockWiseParallelTransposeNoBankConflicts(const T * const in, T * const out, unsigned int width, unsigned int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X + 1];
	unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idxIn = yIndex * width + xIndex;

	// write out consecutively and do transposing by swapping x and y when reading from shared memory
	unsigned int xIndexOut = blockIdx.y * blockDim.y + threadIdx.x;
	unsigned int yIndexOut = blockIdx.x * blockDim.x + threadIdx.y;
	unsigned int idxOut = yIndexOut * height + xIndexOut;

	cache[threadIdx.y][threadIdx.x] = in[idxIn];
	__syncthreads();

	out[idxOut] = cache[threadIdx.x][threadIdx.y];
}

template <typename T, unsigned int WIDTH, unsigned int HEIGHT>
void transposeCpu(const Matrix<T, WIDTH, HEIGHT>& in, Matrix<T, HEIGHT, WIDTH>& out) {
	for (unsigned int j = 0; j < HEIGHT; j++) {
		for (unsigned int i = 0; i < WIDTH; i++) {
			auto inIndex = j * WIDTH + i;
			auto outIndex = i * HEIGHT + j;
			out[outIndex] = in[inIndex];
		}
	}
}

template<typename T>
struct KernelParams {
	const T * const d_in;
	T * const d_out;
	unsigned int width;
	unsigned int height;
	dim3 gridDim;
	dim3 blockDim;
};

template <typename T>
float averageTime(void(*kernel)(const T * const, T * const, unsigned int, unsigned int),
				  const T * const in, T * const out, unsigned int width, unsigned int height,
				  dim3 grid, dim3 block, unsigned int numReps) {
	float kernelTime = -1.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	try {
		checkCudaStatus(cudaEventCreate(&start));
		checkCudaStatus(cudaEventCreate(&stop));
		// Clear error status
		checkCudaStatus(cudaGetLastError());

		// warmup to avoid timing startup
		kernel << <grid, block >> > (in, out, width, height);

		checkCudaStatus(cudaEventRecord(start, 0));

		for (unsigned int i = 0; i < numReps; i++)
		{
			kernel << <grid, block >> > (in, out, width, height);
			// Ensure no launch failure
			checkCudaStatus(cudaGetLastError());
		}

		checkCudaStatus(cudaEventRecord(stop, 0));
		checkCudaStatus(cudaEventSynchronize(stop));
		checkCudaStatus(cudaEventElapsedTime(&kernelTime, start, stop));
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	catch (CudaException e) {
		std::cerr << "Kernel launch failed: " << e.what() << std::endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	return kernelTime / numReps;
}

template <typename T>
float averageTime(void(*kernel)(const T * const, T * const, unsigned int, unsigned int),
				  const KernelParams<T>& params, unsigned int numReps) {
	float kernelTime = -1.0f;
	cudaEvent_t start;
	cudaEvent_t stop;
	try {
		checkCudaStatus(cudaEventCreate(&start));
		checkCudaStatus(cudaEventCreate(&stop));
		// Clear error status
		checkCudaStatus(cudaGetLastError());

		// warmup to avoid timing startup
		kernel << <params.gridDim, params.blockDim >> > (params.d_in, params.d_out, params.width, params.height);

		checkCudaStatus(cudaEventRecord(start, 0));

		for (unsigned int i = 0; i < numReps; i++)
		{
			kernel << <params.gridDim, params.blockDim >> > (params.d_in, params.d_out, params.width, params.height);
			// Ensure no launch failure
			checkCudaStatus(cudaGetLastError());
		}

		checkCudaStatus(cudaEventRecord(stop, 0));
		checkCudaStatus(cudaEventSynchronize(stop));
		checkCudaStatus(cudaEventElapsedTime(&kernelTime, start, stop));
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}
	catch (CudaException e) {
		std::cerr << "Kernel launch failed: " << e.what() << std::endl;
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	return kernelTime / numReps;
}

void printStatistics(float avgTime, size_t mem_size) {
	static const unsigned int BYTES_PER_GIGABYTE = 1024 * 1024 * 1024;
	float kernelBandwidth = 2.0f * 1000.0f * mem_size / (BYTES_PER_GIGABYTE) / (avgTime);
	std::cout << avgTime << " " << mem_size << " " << kernelBandwidth << std::endl;
}

template <typename T, unsigned int WIDTH, unsigned int HEIGHT>
void compare(const Matrix<T, HEIGHT, WIDTH>& toCompare, const Matrix<T, HEIGHT, WIDTH>& groundTruth) {
	std::cout << (toCompare == groundTruth ? "Test Passed" : "Test Failed") << std::endl;
}

template <typename T>
void testKernel(void(*kernel)(const T * const, T * const, unsigned int, unsigned int),
				const KernelParams<T>& params, unsigned int numReps) {
	float avgTime = averageTime(kernel, params, numReps);
	printStatistics(avgTime, params.width * params.height * sizeof(T));
}

template <typename T, unsigned int WIDTH, unsigned int HEIGHT >
void testKernel(void(*kernel)(const T * const, T * const, unsigned int, unsigned int),
				const KernelParams<T>& params, unsigned int numReps,
				Matrix<T, WIDTH, HEIGHT>& out, const Matrix<T, WIDTH, HEIGHT>& groundTruth) {
	testKernel(kernel, params, numReps);
	cudaMemcpy(out.getData(), params.d_out, WIDTH * HEIGHT * sizeof(T), cudaMemcpyDefault);
	compare(out, groundTruth);
}

template <typename T, unsigned int WIDTH, unsigned int HEIGHT>
void transposeGpu(const Matrix<T, WIDTH, HEIGHT>& in, Matrix<T, HEIGHT, WIDTH>& out, const Matrix<T, HEIGHT, WIDTH>& groundTruth) {
	cudaSetDevice(0);
	T* d_in;
	T* d_out;
	const size_t byteSize = WIDTH * HEIGHT * sizeof(T);
	cudaMalloc(&d_in, byteSize);
	cudaMalloc(&d_out, byteSize);
	cudaMemcpy(d_in, in.getData(), byteSize, cudaMemcpyDefault);

	dim3 gridTiled(WIDTH / TILE_SIZE_X, HEIGHT / TILE_SIZE_Y);
	dim3 blockTiled(TILE_SIZE_X, TILE_SIZE_Y);

	KernelParams<T> kernelParams = { d_in, d_out, WIDTH, HEIGHT, gridTiled, blockTiled };

	testKernel(baseLineCopy, kernelParams, NUM_REPS);
	testKernel(baseLineCopyShared, kernelParams, NUM_REPS);

	if (WIDTH * HEIGHT < 4096) {
		dim3 gridOne(1);
		dim3 blockOne(1);
		kernelParams.gridDim = gridOne;
		kernelParams.blockDim = blockOne;
		testKernel(naiveTranspose, kernelParams, NUM_REPS, out, groundTruth);
	}

	dim3 gridRowWise(HEIGHT);
	dim3 blockRowWise(WIDTH);
	kernelParams.gridDim = gridRowWise;
	kernelParams.blockDim = blockRowWise;
	testKernel(naiveParallelTranspose, kernelParams, NUM_REPS, out, groundTruth);

	kernelParams.gridDim = gridTiled;
	kernelParams.blockDim = blockTiled;
	testKernel(naiveBlockWiseParallelTranspose, kernelParams, NUM_REPS, out, groundTruth);
	testKernel(naiveSharedBlockWiseParallelTranspose, kernelParams, NUM_REPS, out, groundTruth);
	testKernel(coalescedSharedBlockWiseParallelTranspose, kernelParams, NUM_REPS, out, groundTruth);
	testKernel(coalescedSharedBlockWiseParallelTransposeNoBankConflicts, kernelParams, NUM_REPS, out, groundTruth);

	cudaFree(d_in);
	cudaFree(d_out);

	cudaDeviceReset();
}

int main() {
	Matrix<int, DIM_X, DIM_Y> matrix;
	std::iota(&matrix[0], &matrix[DIM_X * DIM_Y], 0);

	//std::cout << matrix << std::endl;

	Matrix<int, DIM_Y, DIM_X> groundTruth;
	transposeCpu(matrix, groundTruth);
	//std::cout << groundTruth << std::endl;

	Matrix<int, DIM_Y, DIM_X> gpuTransposed;
	transposeGpu(matrix, gpuTransposed, groundTruth);
	//std::cout << gpuTransposed << std::endl;

	return 0;
}