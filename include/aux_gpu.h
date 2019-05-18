
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "cochlea_utils.h"
#include "const.h"
bool checkHW(char *name, char *gpuType, int dev);
int findGraphicsGPU(char *name, bool show_verbose);