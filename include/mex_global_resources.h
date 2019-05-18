#pragma once
#include "params.h"
#include "Log.h"
#include "aux_gpu.h"
#include "solver.h"

class MexResources {
	int counter = 0;
public:
	cudaDeviceProp deviceProp;
	CParams *params;
	CSolver *GSolver;
	Log mainlog; // for logging purposes
	MexResources();
	void initDevice(); // init gpu device
	~MexResources();
	inline int checkRunTimes() { return counter; }
	int updateRunTimes();
};
