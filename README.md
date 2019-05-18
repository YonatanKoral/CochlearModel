# AudioLab
## Efficent Tool to solve cochlear model in the time domain

## Requirements
	- CUDA Toolkit installed + Samples
	- Matlab 
	- Eigen3 (Linear Algebra Library, free software)
	- cmake either seperatly if you are using Visual Studio 2015 or installed with visual studio 2017
	- ensure that the envirnment variable NVCUDASAMPLES_ROOT will contains the nvidia samles directory, example
	`C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.2`
## Installation

This instructions are for windows with Visual Studio 2015, Visual Studio 2017 supports cmake and can be adjusted differently without cmake gui
	- open the zip file, it contains all necessary code for the project
	- after installing necessary requirements, add build dirctory inside the code directory or near it
	- run cmake gui 
		* choose the source code directory as the directory contains the CMakelists.txt file
		* choose Build binaries directory the build directory you created
		* press configure
	- you will have dialog that asks you to generator
		* choose Visual Studio 14 2015 Win64
		* or Visual Studio 15 2017 Win64, depending on your VS studio
	- in optional toolset flag (-T) write host=x64. this ensures that the entire toolchain is 64bit and press finish
	- after initial configuring you can change the options in the cache
		* mark grouped and advanced to view all options grouped
		* in CMAKE_INSTALL_PREFIX option set the target folder that you run the mexw64 file should be (this can be set also in the CMakelists.txt file before launching the cmake gui)
		* if Eigen3 directory not found set it manually
		* set the CUDA_SDK_ROOT_DIR to the samples directory if your NVCUDASAMPLES_ROOT environment variable was not set
		* press configure again
		* you should have Matlab subcategory with directories contains necessary lib files positions and also in ungrouped the Eigen3_DIR
		* press generate
		* press open project to open the sln file that the cmake created
	- change Mode to Release
	- build the AudioLabCM project, you can ignore the warnings
	- build the INSTALL project, this will copy the the mexw64 file to your target directory where you can call it from matlab
	- since the code sent toy you originally contained the AudioLab15.mexw64, go to ProcessAudioLab.m in your target directory and change AudioLab15 to AudioLabCM to use the new version
	- this necessary only for further development of the project
	- you can use AudioLab15 if no updates are necessary for the cuda project
## Project important files
there are several files contain algorithm main classes, I added description for the most important classes/functions
	- AudioLabCM - enterance file for the project, contains mexFunction which called when the mexw64 file executed
	- params - contains parsed params structure, constains values for the program to define inputs, outputs, ear physical parameters etc.
	- solver - container class to execute stages in cochlear equations solution, look in Run_Cuda code for sub stages executed
	- cochlea.cuh - contains diffrent versions of BM velocity calculation kernel, use BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_triple_aggregations (kernel version controlled by Run_Fast_BM_Calculation parmeter, use Run_Fast_BM_Calculation=17)
		since BMOHC_FAST_Pre_fmaf_No_Constants_kernel_4B_Optimized_triple_aggregations is most advanced, comments for BM velociity written in there 
	- cochlea.cu - contain rest of cuda kernels to be executed (descriptions are in cochlea_common.h - functions exposed to host side algorithms)
	- VirtualHandler - abstract class for Input/Output Operations
	- ConfigFileHandler - implementation of VirtualHandler for I/O to hardrive
	- MexHandler - implementation of VirtualHandler for I/O to matlab
	- mexplus - C++ envelope auxiliary library I/O from matlab, used by MexHandler
	- firpm - auxiliary library to handle FIR filters generation
	- IowaHillsFilters - library to generate IIR filters
	- AudioGramCreator - interface to handle AN response and JND calculations, CPU version is outdated, use only GPU version
	- ComplexJNDProfile - has profile of JND calculation, include range of ANR results to read from and signal profile (noise type and power + signal type and power)
## Project important flags
the project has several important flags that will be noted here
	* set Run_Fast_BM_Calculation to 17 to gain additional 80% speed
	* IHC_Vector damage profile for Inner hair cells, can be vector of 256 values or scalar for constant profile accross the entire cochlea (8 IHC healthy, 5 IHC unresponsive)
	* OHC_Vector damage profile for Outer hair cells, can be vector of 256 values or scalar for constant profile accross the entire cochlea (0.5 OHC healthy, 0 OHC unresponsive)
	* Allowed_Outputs is flag array for AN response and BM activity
		- 1 BM velocity (Eq 10, sigma BM from Proffesor Furst Article "Cochlear Model for Hearing Loss")
		- 2 Lambda high (Eq 13, from Proffesor Furst Article "Cochlear Model for Hearing Loss")
		- 4 Lambda Medium (Eq 13, from Proffesor Furst Article "Cochlear Model for Hearing Loss")
		- 8 Lambda Low (Eq 13, from Proffesor Furst Article "Cochlear Model for Hearing Loss")
	* Calculate_JND set one to evaluate JND results(Eq 24, from Proffesor Furst Article "Cochlear Model for Hearing Loss") (will add noise signal to reference)
	* testedPowerLevels array of powers in dB spl to test on input signal
	* testedNoises array of powers in dB spl to test noise power (1111 is no noise)
	* SPLRef is the base value of SPL reference, physical value is 2e-5 NM, but this parameter can be tuned
	* testedFrequencies array of frequencies in HZ, can be used to test multiple single pitvh signals
	* Decouple_Filter - number of cuda blocks dedictated for each signal, for pure pitch will generate signal at length Time_Block_Length*Decouple_Filter seconds, for signal read from fill, will trunc if signal is longer than this value
	* Time_Block_Length - length of time in seconds (default 0.02) that each cuda block calculate such that each signal handeled has length of Time_Block_Length*Decouple_Filter seconds
