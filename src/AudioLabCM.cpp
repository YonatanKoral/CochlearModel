// AudioLabCM.cpp : Defines the initialization routines for the DLL.
//

#include "stdafx.h"
#include "AudioLabCM.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//
//TODO: If this DLL is dynamically linked against the MFC DLLs,
//		any functions exported from this DLL which call into
//		MFC must have the AFX_MANAGE_STATE macro added at the
//		very beginning of the function.
//
//		For example:
//
//		extern "C" BOOL PASCAL EXPORT ExportedFunction()
//		{
//			AFX_MANAGE_STATE(AfxGetStaticModuleState());
//			// normal function body here
//		}
//
//		It is very important that this macro appear in each
//		function, prior to any calls into MFC.  This means that
//		it must appear as the first statement within the 
//		function, even before any object variable declarations
//		as their constructors may generate calls into the MFC
//		DLL.
//
//		Please see MFC Technical Notes 33 and 58 for additional
//		details.
//

// CAudioLab15App

BEGIN_MESSAGE_MAP(CAudioLabCMApp, CWinApp)
END_MESSAGE_MAP()


// CAudioLabCMApp construction

CAudioLabCMApp::CAudioLabCMApp()
{
	// TODO: add construction code here,
	// Place all significant initialization in InitInstance
}


// The one and only CAudioLabCMApp object

int global_counter = 0;
CAudioLabCMApp theApp;

void destroyGlobalResources() {
	mxstreambuf mout;
	// delete all resources on request
	PrintFormat("Terminating AudioLab after %d times\n",mex_handler->checkRunTimes());
	if (mex_handler != NULL) {
		delete mex_handler;
		mex_handler = NULL;


		gpuAssert(cudaDeviceReset());
	}
	PrintFormat("AudioLab terminated successfully \n");
}

// CAudioLabCMApp initialization

BOOL CAudioLabCMApp::InitInstance()
{
	CWinApp::InitInstance();

	return TRUE;
}



/**
* exported function to matlab
* @nlhs[int] - number of outputs parametrs
* @plhs[**MaxArrays] - pointer to array of outputs, all matlab parametrs are MxArrays
* @nrhs[int] - number of inputs parametrs
* @prhs[**MaxArrays] - pointer to array of inputs, all matlab parametrs are MxArrays
* procedure parse inputs with MEXHandler and executes the solver while logging run times
* if runs first time creates the resource handler and creates handler to release program memory
*/
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[]) {
	mxstreambuf mout;
	if (global_counter == 0) {
		PrintFormat("running Audiolab emulation started global_counter== %d\n",global_counter);
		mex_handler = new MexResources();	  // creating handler for global resources
		mexAtExit(destroyGlobalResources); // creating terminating sequence for the clearing of audio lab program
	}
	else {
		//mex_handler->initDevice();
	}
	global_counter++;
	// obtain control over function memory resources, blokcs matlab delete it from memory
	mexLock();
	try {
		mex_handler->updateRunTimes();
		PrintFormat("running Audiolab emulation update run times\n");
		mex_handler->mainlog.markTime(0);


		mex_handler->mainlog.markTime(1);

		//parsing inputs and create inner params class
		mex_handler->params->parseMexFile(prhs, plhs);
		//std::cout << "running Audiolab emulation post parsing" << std::endl;
		std::string exp_filename(REF_FILENAME);
		std::string generic_extention_filename(DEFAULT_EXTENTION_FILENAME);		// use default input file name
		std::string data_files_path(OUTPUT_PATH);
		mex_handler->mainlog.markTime(3);
		//CSolver *GSolver = new CSolver(0);
		// create necessary inner data structures for the solver
		mex_handler->GSolver->updateStatus(mex_handler->params, 1);
		//clock_t solver_init_time = clock();
		mex_handler->mainlog.markTime(4);

		// defining number of signals to process and their duration
		mex_handler->GSolver->init_params(0); // initializing the first set of parameters

		mex_handler->mainlog.markTime(5);
		// solving cochlear equation + an response + JND (if needed)
		mex_handler->GSolver->Run(0.0);
		
		mex_handler->mainlog.markTime(6);
		// assert device synchronized
		gpuAssert(cudaThreadSynchronize());
		mex_handler->mainlog.markTime(7);
		
		mex_handler->mainlog.markTime(8);
		int Show_CPU_Run_Time = mex_handler->params->Show_CPU_Run_Time;
		// show run times for different steps
		if (Show_CPU_Run_Time & 1) {
			std::cout << "running Audiolab cpu run time" << std::endl;
			mex_handler->mainlog.elapsedTimeView("run", 0, 8);
			mex_handler->mainlog.elapsedTimeView("Solver constructor", 3, 4);
			mex_handler->mainlog.elapsedTimeView("Solver init", 4, 5);
			mex_handler->mainlog.elapsedTimeView("Solver run", 5, 6);
			mex_handler->mainlog.elapsedTimeView("Solver normal exit", 6, 7);
			mex_handler->mainlog.elapsedTimeView("Solver delete", 7, 8);
			mex_handler->params->vhout->flushToIOHandler(mex_handler->mainlog, "main_log");
		}
		// release input/output memory control without delete it
		mex_handler->params->vhout->clearPrimaryMajor(0);
		// release inner definitions of ear physical state
		mex_handler->GSolver->clearModel();
	}
	catch (const std::bad_array_new_length& exc) {
		mexErrMsgIdAndTxt("AudioLab:ArrayFail", "Error: AudioLab array new length fail '%s'", exc.what());
	}
	catch (const std::bad_alloc& ba) {
		mexErrMsgIdAndTxt("AudioLab:MemoryFail", "Error: AudioLab momory fail due to '%s'", ba.what());

	}
	catch (const std::out_of_range& exc) {
		mexErrMsgIdAndTxt("AudioLab:RangeFail", "Error: AudioLab out of range fail due to '%s'", exc.what());
	}
	catch (const std::bad_exception& exc) {
		mexErrMsgIdAndTxt("AudioLab:Exception", "Error: AudioLab exception '%s'", exc.what());
	}
	catch (const std::exception& exc) {
		mexErrMsgIdAndTxt("AudioLab:GeneralException", "Error: AudioLab exception failure '%s'", exc.what());
	}
	catch (const std::system_error& exc)
	{
		mexErrMsgIdAndTxt("AudioLab:SystemException", "Error: AudioLab system exception failure '%s'", exc.what());
	}
	catch (const std::runtime_error& exc) {
		mexErrMsgIdAndTxt("AudioLab:DataFail", "Error: AudioLab failed due to '%s'", exc.what());
	}
	// mex function executed release control over function memory, allows matlab to delete it if necessary
	mexUnlock();
	return;
}
