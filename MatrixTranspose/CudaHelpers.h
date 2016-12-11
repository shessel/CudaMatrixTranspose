#pragma once
#include <stdexcept>
#include "cuda_runtime.h"

struct CudaException : public std::runtime_error {
	CudaException(const char * message) : std::runtime_error(message) {}
	CudaException(std::string &message) : std::runtime_error(message) {}
};

void checkCudaStatus(cudaError_t status, std::string message = { "Cuda Error" }) {
#if defined(DEBUG) || defined(_DEBUG)
	if (status != cudaSuccess) {
		std::string cudaErrorString(cudaGetErrorString(status));
		if (cudaErrorString.compare("") != 0) {
			message += ": " + cudaErrorString;
		}
		throw CudaException(message);
	}
#endif
}