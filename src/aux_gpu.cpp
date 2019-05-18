#include "aux_gpu.h"


bool checkHW(char *name, char *gpuType, int dev) {
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	strcpy_s(name, 256, deviceProp.name);

	if (!_strnicmp(deviceProp.name, gpuType, strlen(gpuType))) {
		return true;
	} else {
		return false;
	}
}

int findGraphicsGPU(char *name, bool show_verbose) {
	int nGraphicsGPU = 0;
	int deviceCount = 0;
	cudaDeviceProp prop;
	bool bFoundGraphics = false;
	char firstGraphicsName[256], temp[256];
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		PrintFormat("> cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.,\n\nFAILED\n");
		return false;
	} else {
		if (show_verbose) {
			PrintFormat("> Found %d CUDA Capable Device(s)\n", deviceCount);
		}
	}
	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0) {
		PrintFormat("> There are no device(s) supporting CUDA\n");
		return false;
	}
	for (int dev = 0; dev < deviceCount; ++dev) {
		bool bGraphics = !checkHW(temp, "Tesla", dev);
		cudaGetDeviceProperties(&prop, dev);
		if (show_verbose) {
			PrintFormat("> %s\t\tGPU %d: %s,Architecture Version:  %d.%d\n", (bGraphics ? "Graphics" : "Compute"), dev, temp, prop.major, prop.minor);
		}
		if (bGraphics) {
			if (!bFoundGraphics) {
				strcpy_s(firstGraphicsName, temp);
			}
			nGraphicsGPU++;
		}
	}
	if (nGraphicsGPU) {
		strcpy_s(name, 256, firstGraphicsName);
	} else {
		strcpy_s(name, 256, "this hardware");
	}
	return nGraphicsGPU;
}

