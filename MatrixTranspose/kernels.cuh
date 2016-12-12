#pragma once
#include "cuda_runtime.h"

const unsigned int TILE_SIZE_X = 16;
const unsigned int TILE_SIZE_Y = TILE_SIZE_X;
const unsigned int DIM_X = 1024;
const unsigned int DIM_Y = DIM_X;
const unsigned int NUM_REPS = 100;

template <typename T>
__global__ void baseLineCopy(const T * const in, T * const out, unsigned int width, unsigned int height) {
	unsigned int bIdxIn = blockIdx.y * blockDim.y * width + blockIdx.x * blockDim.x;
	unsigned int tIdxIn = threadIdx.y * width + threadIdx.x;
	unsigned int idxIn = bIdxIn + tIdxIn;

	out[idxIn] = in[idxIn];
}

template <typename T>
__global__ void baseLineCopyShared(const T * const in, T * const out, unsigned int width, unsigned int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X];
	unsigned int bIdxIn = blockIdx.y * blockDim.y * width + blockIdx.x * blockDim.x;
	unsigned int tIdxIn = threadIdx.y * width + threadIdx.x;
	unsigned int idxIn = bIdxIn + tIdxIn;

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
	unsigned idxOut = threadIdx.x * height + blockIdx.x;

	out[idxOut] = in[idxIn];
}

template <typename T>
__global__ void naiveBlockWiseParallelTranspose(const T * const in, T * const out, unsigned int width, unsigned int height) {
	unsigned int bIdxIn = blockIdx.y * blockDim.y * width + blockIdx.x * blockDim.x;
	unsigned int tIdxIn = threadIdx.y * width + threadIdx.x;
	unsigned int idxIn = bIdxIn + tIdxIn;

	unsigned bIdxOut = blockIdx.x * blockDim.x * height + blockIdx.y * blockDim.y;
	unsigned tIdxOut = threadIdx.x * height + threadIdx.y;
	unsigned idxOut = bIdxOut + tIdxOut;

	out[idxOut] = in[idxIn];
}

template <typename T>
__global__ void naiveSharedBlockWiseParallelTranspose(const T * const in, T * const out, unsigned int width, unsigned int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X];
	unsigned int bIdxIn = blockIdx.y * blockDim.y * width + blockIdx.x * blockDim.x;
	unsigned int tIdxIn = threadIdx.y * width + threadIdx.x;
	unsigned int idxIn = bIdxIn + tIdxIn;

	unsigned bIdxOut = blockIdx.x * blockDim.x * height + blockIdx.y * blockDim.y;
	unsigned tIdxOut = threadIdx.x * height + threadIdx.y;
	unsigned idxOut = bIdxOut + tIdxOut;

	cache[threadIdx.y][threadIdx.x] = in[idxIn];
	__syncthreads();

	out[idxOut] = cache[threadIdx.y][threadIdx.x];
}

template <typename T>
__global__ void coalescedSharedBlockWiseParallelTranspose(const T * const in, T * const out, unsigned int width, unsigned int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X];
	unsigned int bIdxIn = blockIdx.y * blockDim.y * width + blockIdx.x * blockDim.x;
	unsigned int tIdxIn = threadIdx.y * width + threadIdx.x;
	unsigned int idxIn = bIdxIn + tIdxIn;

	unsigned bIdxOut = blockIdx.x * blockDim.x * height + blockIdx.y * blockDim.y;
	unsigned tIdxOut = threadIdx.y * height + threadIdx.x;
	unsigned idxOut = bIdxOut + tIdxOut;

	cache[threadIdx.y][threadIdx.x] = in[idxIn];
	__syncthreads();

	out[idxOut] = cache[threadIdx.x][threadIdx.y];
}

template <typename T>
__global__ void coalescedSharedBlockWiseParallelTransposeNoBankConflicts(const T * const in, T * const out, unsigned int width, unsigned int height) {
	__shared__ T cache[TILE_SIZE_Y][TILE_SIZE_X + 1];
	unsigned int bIdxIn = blockIdx.y * blockDim.y * width + blockIdx.x * blockDim.x;
	unsigned int tIdxIn = threadIdx.y * width + threadIdx.x;
	unsigned int idxIn = bIdxIn + tIdxIn;

	unsigned bIdxOut = blockIdx.x * blockDim.x * height + blockIdx.y * blockDim.y;
	unsigned tIdxOut = threadIdx.y * height + threadIdx.x;
	unsigned idxOut = bIdxOut + tIdxOut;

	cache[threadIdx.y][threadIdx.x] = in[idxIn];
	__syncthreads();

	out[idxOut] = cache[threadIdx.x][threadIdx.y];
}