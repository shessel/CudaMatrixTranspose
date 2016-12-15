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

const int TILE_SIZE_X = 16;
const int TILE_SIZE_Y = TILE_SIZE_X;
const int DIM_X = 1024;
const int DIM_Y = DIM_X;
const int NUM_REPS = 100;

template <typename T>
__global__ void baseLineCopy(const T * const in, T * const out, int width, int height) {
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int idxIn = yIndex * width + xIndex;

	if (idxIn < width * height) {
		out[idxIn] = in[idxIn];
	}
}

template <typename T>
__global__ void baseLineCopyShared(const T * const in, T * const out, int width, int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X];
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int idxIn = yIndex * width + xIndex;

	if (idxIn < width * height) {
		cache[threadIdx.y][threadIdx.x] = in[idxIn];
		__syncthreads();

		out[idxIn] = cache[threadIdx.y][threadIdx.x];
	}
}

template <typename T>
__global__ void naiveTranspose(const T * const in, T * const out, int width, int height) {
	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			auto idxIn = j * width + i;
			auto idxOut = i * height + j;
			out[idxOut] = in[idxIn];
		}
	}
}

template <typename T>
__global__ void naiveParallelTranspose(const T * const in, T * const out, int width, int height) {
	int idxIn = blockIdx.x * width + threadIdx.x;

	if (idxIn < width * height) {
		int idxOut = threadIdx.x * height + blockIdx.x;
		out[idxOut] = in[idxIn];
	}
}

template <typename T>
__global__ void naiveBlockWiseParallelTranspose(const T * const in, T * const out, int width, int height) {
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int idxIn = yIndex * width + xIndex;

	if (xIndex < width && yIndex < height) {
		int idxOut = xIndex * height + yIndex;
		out[idxOut] = in[idxIn];
	}
}

template <typename T>
__global__ void naiveSharedBlockWiseParallelTranspose(const T * const in, T * const out, int width, int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X];
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int idxIn = yIndex * width + xIndex;

	if (xIndex < width && yIndex < height) {
		cache[threadIdx.y][threadIdx.x] = in[idxIn];
		__syncthreads();

		int idxOut = xIndex * height + yIndex;
		out[idxOut] = cache[threadIdx.y][threadIdx.x];
	}
}

template <typename T>
__global__ void coalescedSharedBlockWiseParallelTranspose(const T * const in, T * const out, int width, int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X];
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int idxIn = yIndex * width + xIndex;

	// write out consecutively (with increasing threadIdx.x) and do
	// transposing by swapping x and y when reading from shared memory
	int xIndexOut = blockIdx.y * blockDim.y + threadIdx.x;
	int yIndexOut = blockIdx.x * blockDim.x + threadIdx.y;
	int idxOut = yIndexOut * height + xIndexOut;

	if (xIndex < width && yIndex < height) {
		cache[threadIdx.y][threadIdx.x] = in[idxIn];
	}
	__syncthreads();

	if (xIndexOut < width && yIndexOut < height) {
		out[idxOut] = cache[threadIdx.x][threadIdx.y];
	}
}

template <typename T>
__global__ void coalescedSharedBlockWiseParallelTransposeNoBankConflicts(const T * const in, T * const out, int width, int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X + 1];
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int idxIn = yIndex * width + xIndex;

	// write out consecutively (with increasing threadIdx.x) and do
	// transposing by swapping x and y when reading from shared memory
	int xIndexOut = blockIdx.y * blockDim.y + threadIdx.x;
	int yIndexOut = blockIdx.x * blockDim.x + threadIdx.y;
	int idxOut = yIndexOut * height + xIndexOut;

	if (xIndex < width && yIndex < height) {
		cache[threadIdx.y][threadIdx.x] = in[idxIn];
	}
	__syncthreads();

	if (xIndexOut < width && yIndexOut < height) {
		out[idxOut] = cache[threadIdx.x][threadIdx.y];
	}
}

template <typename T, int WIDTH, int HEIGHT>
void transposeCpu(const Matrix<T, WIDTH, HEIGHT>& in, Matrix<T, HEIGHT, WIDTH>& out) {
	for (int j = 0; j < HEIGHT; j++) {
		for (int i = 0; i < WIDTH; i++) {
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
	int width;
	int height;
	dim3 gridDim;
	dim3 blockDim;
};

template <typename T>
float averageTime(void(*kernel)(const T * const, T * const, int, int),
				  const T * const in, T * const out, int width, int height,
				  dim3 grid, dim3 block, int numReps) {
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

		for (int i = 0; i < numReps; i++)
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
float averageTime(void(*kernel)(const T * const, T * const, int, int),
				  const KernelParams<T>& params, int numReps) {
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

		for (int i = 0; i < numReps; i++)
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
	static const int BYTES_PER_GIGABYTE = 1024 * 1024 * 1024;
	// (1 read + 1 write) * (size in GB) / (time in sec)
	float kernelBandwidth = (2.0f * mem_size / (BYTES_PER_GIGABYTE)) / (avgTime / 1000.0f);
	std::cout << avgTime << " " << mem_size << " " << kernelBandwidth << std::endl;
}

template <typename T, int WIDTH, int HEIGHT>
void compare(const Matrix<T, HEIGHT, WIDTH>& toCompare, const Matrix<T, HEIGHT, WIDTH>& groundTruth) {
	std::cout << (toCompare == groundTruth ? "Test Passed" : "Test Failed") << std::endl;
}

template <typename T>
void testKernel(const char * const name, void(*kernel)(const T * const, T * const, int, int),
				const KernelParams<T>& params, int numReps) {
	std::cout << name << ": ";
	float avgTime = averageTime(kernel, params, numReps);
	printStatistics(avgTime, params.width * params.height * sizeof(T));
}

template <typename T, int WIDTH, int HEIGHT >
void testKernel(const char * const name, void(*kernel)(const T * const, T * const, int, int),
				const KernelParams<T>& params, int numReps,
				Matrix<T, WIDTH, HEIGHT>& out, const Matrix<T, WIDTH, HEIGHT>& groundTruth) {
	testKernel(name, kernel, params, numReps);
	cudaMemcpy(out.getData(), params.d_out, WIDTH * HEIGHT * sizeof(T), cudaMemcpyDefault);
	compare(out, groundTruth);
}

template <typename T, int WIDTH, int HEIGHT>
void transposeGpu(const Matrix<T, WIDTH, HEIGHT>& in, Matrix<T, HEIGHT, WIDTH>& out, const Matrix<T, HEIGHT, WIDTH>& groundTruth) {
	cudaSetDevice(0);
	T* d_in;
	T* d_out;
	const size_t byteSize = WIDTH * HEIGHT * sizeof(T);
	cudaMalloc(&d_in, byteSize);
	cudaMalloc(&d_out, byteSize);
	cudaMemcpy(d_in, in.getData(), byteSize, cudaMemcpyDefault);

	dim3 gridTiled((WIDTH - 1) / TILE_SIZE_X + 1, (HEIGHT - 1) / TILE_SIZE_Y + 1);
	dim3 blockTiled(TILE_SIZE_X, TILE_SIZE_Y);

	KernelParams<T> kernelParams = { d_in, d_out, WIDTH, HEIGHT, gridTiled, blockTiled };

	testKernel("baseLineCopy", baseLineCopy, kernelParams, NUM_REPS);
	testKernel("baseLineCopyShared", baseLineCopyShared, kernelParams, NUM_REPS);

	if (WIDTH * HEIGHT < 4096) {
		dim3 gridOne(1);
		dim3 blockOne(1);
		kernelParams.gridDim = gridOne;
		kernelParams.blockDim = blockOne;
		testKernel("naiveTranspose", naiveTranspose, kernelParams, NUM_REPS, out, groundTruth);
	}

	dim3 gridRowWise(HEIGHT);
	dim3 blockRowWise(WIDTH);
	kernelParams.gridDim = gridRowWise;
	kernelParams.blockDim = blockRowWise;
	testKernel("naiveParallelTranspose", naiveParallelTranspose, kernelParams, NUM_REPS, out, groundTruth);

	kernelParams.gridDim = gridTiled;
	kernelParams.blockDim = blockTiled;
	testKernel("naiveBlockWiseParallelTranspose", naiveBlockWiseParallelTranspose, kernelParams, NUM_REPS, out, groundTruth);
	testKernel("naiveSharedBlockWiseParallelTranspose", naiveSharedBlockWiseParallelTranspose, kernelParams, NUM_REPS, out, groundTruth);
	testKernel("coalescedSharedBlockWiseParallelTranspose", coalescedSharedBlockWiseParallelTranspose, kernelParams, NUM_REPS, out, groundTruth);
	testKernel("coalescedSharedBlockWiseParallelTransposeNoBankConflicts", coalescedSharedBlockWiseParallelTransposeNoBankConflicts, kernelParams, NUM_REPS, out, groundTruth);

	cudaFree(d_in);
	cudaFree(d_out);

	cudaDeviceReset();
}

int main() {
	using DataType = int;
	Matrix<DataType, DIM_X, DIM_Y> matrix;
	std::iota(&matrix[0], &matrix[DIM_X * DIM_Y], static_cast<DataType>(0));

	//std::cout << matrix << std::endl;

	Matrix<DataType, DIM_Y, DIM_X> groundTruth;
	transposeCpu(matrix, groundTruth);
	//std::cout << groundTruth << std::endl;

	Matrix<DataType, DIM_Y, DIM_X> gpuTransposed;
	transposeGpu(matrix, gpuTransposed, groundTruth);
	//std::cout << gpuTransposed << std::endl;

	return 0;
}