// AudioLabCM.h : main header file for the AudioLabCM DLL
//

#pragma once

#ifndef __AFXWIN_H__
	#error "include 'stdafx.h' before including this file for PCH"
#endif

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#include <Windows.h>
#endif
#include "resource.h"		// main symbols
#include "mex.h"
#include "mxstreambuf.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// includes, project
#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cufft.h>
#include <iomanip>
#include <cuda_d3d9_interop.h>
#include <device_double_functions.h>
#include <cassert>
#include <ctime>
#include "mutual.h"
#include "cvector.h"

#include "stdafx.h"
#include "solver.h"

#include "const.h"
#include "cochlea_common.h"
//#include "cochlea_ref.h"
#include "model.h"
#include "Log.h"
#include "aux_gpu.h"
#include "mex_global_resources.h"




#ifndef gpuAssert
#define gpuAssert( condition ) { if( (condition) != 0 ) { fprintf( stderr, "\n FAILURE %s in %s, line %d\n", cudaGetErrorString(condition), __FILE__, __LINE__ ); exit(1); } }
#endif
// CAudioLabCMApp
// See AudioLabCM.cpp for the implementation of this class
//

class CAudioLabCMApp : public CWinApp
{
public:
	CAudioLabCMApp();

// Overrides
public:
	virtual BOOL InitInstance();

	DECLARE_MESSAGE_MAP()
};

MexResources *mex_handler = NULL;
void destroyGlobalResources();