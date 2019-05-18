#include "mex_global_resources.h"

MexResources::MexResources() {
	PrintFormat("Loading MexResources...\n");
	initDevice();
	params = new CParams(1);
	
}
void MexResources::initDevice() {
	//std::cout << "Loaded Parameters..." << std::endl;
	cudaSetDevice(0);
	//std::cout << "Loading cudaSetDevice" << std::endl;
	cudaGetDeviceProperties(&deviceProp, 0);
	//std::cout << "Loading cudaGetDeviceProperties..." << std::endl;
	char device_name[256];
	if (!findGraphicsGPU(device_name, false)) {
		PrintFormat(">SDK not supported on \"%s\" exiting...   PASSED\n" , device_name);
		cutilExit(argc, argv);
	}
	GSolver = new CSolver(0);
	//std::cout << "Loading findGraphicsGPU..." << std::endl;
}
MexResources::~MexResources() {
	PrintFormat("Destroying MexResources...\n");
	GSolver->clearSolver();
	delete GSolver;
	delete params;
}

int MexResources::updateRunTimes() {
	PrintFormat("Running updateRunTimes... %d\n",counter);
	counter++;
	mainlog.clearLog();
	return checkRunTimes();
}
