/******************************************************************\

	File Name : 
	===========

	Classes Defined :	
	=================

	Description	:
	=============
	here we define constants that are fixed at compile time.

\******************************************************************/
#pragma once

#ifndef __CONST
#define __CONST

#define SECTIONS			256			// Cochlea spatial resolution (sections)
#define LOG_SECTIONS		8			// for faster BM calculations
#define SECTIONS_PER_IHC_FILTER_BLOCK 16
#define IHC_FILTER_BLOCK ((uint)(SECTIONS/SECTIONS_PER_IHC_FILTER_BLOCK))
#define THREADS_PER_IHC_FILTER_SECTION 64
# define TIME_SECTION       21000L //20000L       // unit of microseconds 
# define TIME_SECTION_SHORT       20000L //20000L       // unit of microseconds 
# define TIME_OFFSET        5000L //5000L 
# define TIME_OFFSET_NORMALIZED        0.005f //5000L 
# define TOTAL_DEFAULT_RUN_TIME     0.5 //0.5 //0.5
# define TIME_SECTION_OVERLLAPED (TIME_SECTION+TIME_OFFSET)//(((float)TIME_SECTION*5.0)/4.0)
# define TIME_SECTIONS      8L //15L //15L // 60L
# define FULL_TIME_WRITE	((float)TIME_SECTIONS*TIME_SECTION_SHORT/(1000000L)) 
# define CUDA_BLOCKS        TIME_SECTIONS
# define SAMPLE_RATE        20000 //44100 //50000
# define MIN_SAMPLE_RATE	2000 // sample rate cant be below this level
# define VOICE_SIGNAL       0 
# define FORCE_GAMMA_0      0
# define VOICE_AMP          1.0 //1.0
# define INPUT_FILENAME		"andugottahelpus" //"bor" //"khet" //"pan" //"bor" //"in"
# define SIN_PERFORMENCE_TEST   0
# define SIN_PERFORMENCE_CHECK_METHOD 1 // 0 - first iteration 1 - 2nd iteration
# define SIN_PERFORMENCE_TEST_2 0 // for matlab
# define MATLAB_VERSION         1
# define DEFAULT_TEST_DB	10.0f
# define DEFAULT_TEST_FREQ	1000.0f
# define START_DB			-10.0f //-10.0
# define END_DB				100.0f //100.0
# define START_FREQ			1000
# define END_FREQ 		    10000
# define MAX_OUTPUT_BUFFER_LENGTH	(2560*1024*1024) // 64M since it will be single precision buffer will take max 256MB memory
# define INPUT_SIG_FREQ     4000
# define INPUT_SIGNAL_dB    40.0f
# define INPUT_SIG_AMP      (10.0f*(20.0e-6f)*pow(10.0f,(INPUT_SIGNAL_dB)/20.0f))
# define CONV_dB_TO_AMP(a)  (10.0f*(20.0e-6f)*pow(10.0f,(a)/20.0f))
//# define INPUT_SIG_AMP      1e-4 // 1e-7
# define OUTPUT_NORM_FACTOR (1.0e-10f) //(INPUT_SIG_AMP/100.0)
# define OUTPUT_NORM_LOG_FACTOR (10.0f)
# define OUT_TIME_STEP      (1.0f/static_cast<float>(SAMPLE_RATE))
# define OLD_CUDA_VERSION     0
# if  OLD_CUDA_VERSION
# else
#define cutilSafeCall(a)  (a)
#define cutilExit(a,b) throw std::runtime_error("cutilExit - 1")
#define cutilCheckMsg(a) 
# endif

# if SIN_PERFORMENCE_TEST
# define SIN_PERFORMENCE_TEST_ 1
# else
# if SIN_PERFORMENCE_TEST_2
# define SIN_PERFORMENCE_TEST_ 1
# else
# define SIN_PERFORMENCE_TEST_ 0
# endif
# endif

// X DB -> Amp = 10*20e-6*10^(X/20)

# define OUTPUT_SCALING_FACTOR 8
# define OUTPUT_OFFSET 1500
# define START_PLOT_TIME 0.07f
# define END_PLOT_TIME   0.075f

# define PRINT_AMPLITUDES 1
# define PRINT_FULL_OUTPUTS 1

# define __tmax(a,b) (a>b ? a : b)
# define __tmin(a,b) (a<b ? a : b)

# define SAMPLES_BUFFER_LEN __tmax((int)((TIME_SECTION*TIME_SECTIONS*(long long)SAMPLE_RATE)/(1000000)),10)
# define SAMPLES_BUFFER_LEN_SHORT __tmax((int)((TIME_SECTION_SHORT*TIME_SECTIONS*(long long)SAMPLE_RATE)/(1000000)),10)
# define SAMPLES_BUFFER_LEN_P __tmax((int)((TIME_SECTION*(TIME_SECTIONS+1L)*(long long)SAMPLE_RATE)/(1000000)),20)
 

#define T_WIDTH             SECTIONS //512 //256
#define T_BLOCKS            (SECTIONS/T_WIDTH)
#define LAST_BLOCK          (T_BLOCKS-1)
#define FIRST_STAGE_WIDTH       SECTIONS //512 //256
#define SECTIONSM2			(SECTIONS-2)
#define FIRST_STAGE_MEMSIZE     FIRST_STAGE_WIDTH
#define FIRST_STAGE_BLOCKS     (SECTIONS/FIRST_STAGE_WIDTH)
#define CONST_TIME_STEP_VAL  (1e-6f) //(1e-6)

#define _CUDA_JACOBBY_LOOPS  10
#define _CUDA_JACOBBY_LOOPS1 20 //20
#define _CUDA_JACOBBY_LOOPS2 2 //2
#define _CUDA_OUTERN_LOOPS 10 
#define LIPSCHITS_BIG_GAP 10  
#define LIPSCHITS_SMALL_GAP 3 
#define LIPSCHITS_THR 4  
#define CUDA_AUX_TM_LOOPS   2 //2 <- may work


# define USE_CUDA           1
# if USE_CUDA
#define CUDA_MAX_TIME_STEP (1e-6f)
#define CUDA_MIN_TIME_STEP (1e-18f) 
# else
#define CUDA_MAX_TIME_STEP (1e-6)
#define CUDA_MIN_TIME_STEP (1e-18)
# endif

#define CUDA_UPDATE_STEP    2
#define REF_RUN_TIME 0.026f

#define _CUDA_ITERATIONS_PER_KERNEL static_cast<int>(1.0f/(SAMPLE_RATE*CONST_TIME_STEP_VAL))
#define MAX_CHUNK_SIZE   15360   

# define VERIFY_RESULTS     0
# define JACCOBY_VERSION    1
# define DORON_FIX			0
# define DORON_DEBUG		0
# define DORON_FAST_DEBUG	0
# define DEBUG_BUG          0
# define DEBUG_ITERATIONS   0
# define DEBUG_ITERATION_PRINTS 6
# define CONST_TIME_STEP    0
# define DIFF_THR           (1e-10f)
# define ERR_THR (0.01f)

/////////////////////////////////////////////////////

#define PI					(3.141592653589793f)
#define LINE_LENGTH			256			// parstack.h - Number of characters in one line
#define MAX_BUF_LENGTH		200			// maximum buffer size
#define PREFIX_MAX_SIZE		100
#define MAX_BUF_SIZE		100

// Functionality of the Model:
// ---------------------------
#define OW_FLAG					true		// Enable\Disable the OW boundary conditions.
#define OHC_NL_FLAG				true		// Enable\Disable the use of the nonlinear OHC function. If set to <false>, the algorithm will use the linear case tanh(x)~x. 
#define ACTIVE_GAMMA_FLAG		false		// Enable\Disable the change of gamma at float time, according to <psi>.

// Save Data Flags:
#define SAVE_BM_DISP_FLAG		false		// save\don't save
#define SAVE_BM_SP_FLAG			false		// save\don't save
#define SAVE_TM_DISP_FLAG		false		// save\don't save
#define SAVE_TM_SP_FLAG			false		// save\don't save
#define SAVE_OW_DISP_FLAG		false		// save\don't save (stapes speed)
#define SAVE_OW_SP_FLAG			false		// save\don't save (stapes speed)
#define SAVE_OAE_FLAG			false		// save\don't save
#define SHRQATest				false

// Paths:
// ------ 


#define LOG_PATH "Logs\\"
#define DEBUG_BIN_PATH LOG_PATH
#define GAMMA_SHORT_FILENAME	"Data\\gamma.bin"
#define SHORT_OUTPUT_FILENAME	"Data\\bm_sp_out_cpp.bin"
#define SHORT_DATA_PATH			"Data\\"
#define SHORT_REF_NAME			"Data\\ref_3k.bin"
//#define EXPECTED_FILENAME	DEBUG_BIN_PATH "Data\\expected.bin"
#define OUTPUT_FILENAME		"Logs\\Data\\bm_sp_out_cpp.bin"
#define OUTPUT_PATH			"Logs\\Data\\"
#define REF_FILENAME 	    "Logs\\Data\\ref_3k.bin"
//#define LOG_PATH			".\\"
//#define DEBUG_BIN_PATH		".\\"
//#define GAMMA_FULL_FILENAME	".\\Data\\gamma.bin" 
//#define EXPECTED_FILENAME	".\\Data\\expected.bin"
//#define OUTPUT_FILENAME		".\\Data\\bm_sp_out_cpp.bin"
//#define OUTPUT_PATH			".\\Data\\"

// Data file format:
#define DATA_FILE_FORMAT				".bin"

// Generic file names. The user can add extentions to all of these files through the main function.
#define DEFAULT_EXTENTION_FILENAME		""			// The default is an empty string
#define BM_DISP_FILENAME				"BM_disp"
#define BM_SP_FILENAME					"BM_sp"
#define TM_DISP_FILENAME				"TM_disp"
#define TM_SP_FILENAME					"TM_sp"
#define OW_DISP_FILENAME				"OW_disp"
#define OW_SP_FILENAME					"OW_sp"
#define OAE_FILENAME					"OAE"
#define INPUT_MAX_SIZE					3000000L // this is 3e6 places if input too long will be read in stages
// ODE time steps:
// ---------------
# if USE_CUDA
#define INIT_TIME_STEP		CONST_TIME_STEP_VAL
# else
#define INIT_TIME_STEP		CONST_TIME_STEP_VAL //(CONST_TIME_STEP ? (CONST_TIME_STEP_VAL) : (1e-18))
# endif
#define MIN_TIME_STEP		CUDA_MIN_TIME_STEP	
#define MAX_TIME_STEP		(1e-6f)		// 1e-6 
#define MAX_M1_SP_ERROR		(1e-15f)
#define MAX_TOLERANCE		(1e-10f)

// Output Data:
// ------------
#define SAVE_RATE			1			// [smp] Save data each time 0 == mod( step_counter, SAVE_RATE )
#define SAVE_START_TIME		0.0f		    // [sec] Starts to save after <step_counter> >= SAVE_START_TIME


// Model's parameters:
// ------------------- 

// BM:
#define COCHLEA_AREA		(0.5f)			// [cm^2] cochlear cross section
#define COCHLEA_LEN			(3.5f)			// [cm] cochlear length
#define LIQUID_RHO			(1.0f)			// [gr/cm^3] liquid density (Density of perilymph)
#define BM_WIDTH			(0.003f)		// [cm] BM width

#define M0					(1.286e-6f)	// [g/cm^2] Const factor of the Basilar Membrane mass density per unit area
#define M1					(1.5f)			// [g/cm^2] Exponent factor of the Basilar Membrane mass density per unit area
#define R0					(0.25f)		// [g/(cm^2*sec)] Const factor of the Basilar Membrane mass resistance per unit area
#define R1					(-0.06f)		// [g/(cm^2*sec)] Exponent factor of the Basilar Membrane mass resistance per unit area
#define S0					(1.282e4f)		// [g/(cm^2*sec^2)] (elasticity) ExpConstonent factor of the Basilar Membrane mass stiffness per unit area
#define S1Cochlear					(-1.5f)		// [g/(cm^2*sec^2)] (elasticity) Exponent factor of the Basilar Membrane mass stiffness per unit area

// OW (oval window <-> middle ear):
#define W_OW				2*PI*1500		// [rad] Middle ear angular frequency
#define SIGMA_OW			(0.5f)				//ME1: 0.35	 //ME2: 0.35	//ME0: 1.85		// [gr/cm^2] Oval window aerial density	
#define GAMMA_OW			(20e3f)			//ME1: 50e3	 //ME2: 500.0	//ME0: 500		// [1/sec] Middle ear damping constant
#define C_ME				(2*PI*1340)*(2*PI*1340)*0.059f/(0.49f*1.4f)	// (~6e6) Mechanical gain of the ossicles
#define G_ME				(21.4f)			// Coupling of oval window displacement to ear canal pressure
#define C_OW				(6e-3f)			//0.032/0.011			// (~2.9091) Coupling of oval window to basilar membrane

// BC boundary conditions:
#define BC_0				((2.0f*LIQUID_RHO*C_OW)/SIGMA_OW)				// <--> _a0
#define BC_1				( SIGMA_OW*(W_OW*W_OW) + C_ME*G_ME )	// <--> _a1
#define BC_2				SIGMA_OW*GAMMA_OW						// <--> _a2

// OHC
#define W_OHC				2*PI*1000.0f			// [rad] OHC angular frequency (Hz)
#define OHC_PSI_0			70e-3f				// [V] resting OHC potential
#define RATIO_K				1e-3f				// [] ratio between the TM stiffness and the OHC stifness		
#define ETA_1				((1.0f)/ALPHA_L*W_OHC)
#define ETA_2				0

// OHC Nonlinear
#define	BM_ALPHA_R			(0.0f)		//0.0		// Converts the BM into a van der pol ascilator. P_BM = ... + (1 + alpha_r*csi_BM^2)*csi_BM + ... (control <_alpha_r> in the model class).
#define ALPHA_S				(1e-6f)	//1e-4		// _alpha_s & _alpha_l are two arbitrary constants (found by Dani Mackrants, 2007)
#define ALPHA_L				(2e-6f)	//1e-2		// _alpha_s & _alpha_l are two arbitrary constants (found by Dani Mackrants, 2007)

// lambda calculation special 
#define JND_Sat_Value_DB	(150.0f)	
#define DEFAULT_M1_SP_Fix_Factor (0.0f)
#define DEFAULT_Tolerance_Fix_Factor (0.0f)
#define CONVERT_CMPerSeconds_To_MetersPerSecond 1 // (0.01f)
#define Scale_BM_Velocity_For_Lambda_Calculation 1 // 1 for use cm/s, 0.01 for use of m/s
#define LAMBDA_DEFAULT_PARAMS 1
#define SOFTENING_FACTOR			1e-30 // prevents underflow in DC calculations
#define CALC_LAMBDA_FLAG    1
#define CUFFT_FLAG			1
#define DEVICE_MAIN_PARAMS 64
#define HIGH_FREQ_NERVE		(70.0)
#define MEDIUM_FREQ_NERVE	(50.0)
#define LOW_FREQ_NERVE		(30.0)
#define SPONT_HIGH_RATE		(60.0)
#define SPONT_MEDIUM_RATE	(3.0)
#define SPONT_LOW_RATE		(0.1)
#define FIBER_HIGH_DIST		(0.61)
#define FIBER_MEDIUM_DIST	(0.23)
#define FIBER_LOW_DIST		(0.16)
#define AN_FIBERS			30000
#define ACTIVE_FIBERS_LOW	234.375f
#define ACTIVE_FIBERS_HIGH	2f
#define ACTIVE_FIBERS_BARRIER 110
#define CALC_FISHER_AI		1
#define PASS_BAND_RIPPLE    3 // in dB
#define STOP_BAND_ATTENUATION 30 // in dB
#define LPASSPF				800 // HZ
#define STOPPASSPF			1200 // Hz
#define LPF_ARRAY_LENGTH	10
#define Tdc					2e-3f
#define AMP_DC				(100.0)
#define AMP_AC				(1.0)
#define	RSAT				(500.0)
#define IHC_FACTOR			(1e+8)
#define DOUBLE_ACCURACY		(1e-16)
#define EPS					(1e-20)
#define MIN_LAST_SAVED_TIME	0.01
#define RATE_FREQUENCY	20000.0f // rate frequency gives additional rate for last save nodes for freqs above 20khz
#define SPLRef				(2e-5f)
#define SYNAPSE_B_NORMALIZER 1e-6 // 1e-6 for Fs=20KHz and 1e-9 for Fs=44.1KHz
#define DSECTIONS			2*SECTIONS
#define SAMPLE_BUFFER_LEN __tmax((int)((TIME_SECTION*(long long)SAMPLE_RATE)/(1000000)),10) // a single sample buffer length
#define SAMPLE_BUFFER_LEN_SHORT 400
#define SAMPLE_BUFFER_LEN_SHORT_INTERVAL 0.02f
#define LAMBDA_BLOCK_OFFSET (0.0096f) //in seconds
#define SAMPLE_BUFFER_LEN_POW2 ((int)(SAMPLE_BUFFER_LEN_SHORT+(int)((SAMPLE_RATE*LAMBDA_BLOCK_OFFSET)/TIME_SECTIONS)))
#define FFT_ZERO_SECTION	2
#define FFT_ZERO_SECTION_DOUBLE __tmax((int)(FFT_ZERO_SECTION*2),2)
#define LAMBDA_COUNT		3
#define LAST_BLOCK_SAVED (TIME_SECTIONS*(SAMPLE_BUFFER_LEN_POW2-SAMPLE_BUFFER_LEN_SHORT)) // notice this number should be divisble by TIME_SECTIONS for easy writing
#define BACKUP_NODES_LEN (SECTIONS*LAST_BLOCK_SAVED)
#define OFFSET_LENGTH	   302
#define SAMPLE_BUFFER_LEN_PADDED __tmax((int)(SAMPLE_BUFFER_LEN_POW2),10)
#define SAMPLES_BUFFER_LEN_PADDED __tmax((int)(SAMPLE_BUFFER_LEN_PADDED*TIME_SECTIONS),10)
#define SAMPLE_BUFFER_LEN_POW_COMPLEX 256
#define SAMPLE_BUFFER_LEN_PADDED_COMPLEX __tmax((int)(SAMPLE_BUFFER_LEN_POW_COMPLEX),10)
#define SAMPLES_BUFFER_LEN_PADDED_COMPLEX __tmax((int)(1+SAMPLE_BUFFER_LEN_PADDED_COMPLEX*TIME_SECTIONS),10)
#define SAMPLE_FREQUENCY_BLOCK_SIZE SAMPLE_BUFFER_LEN_PADDED_COMPLEX
#define HIGH_LAMBDA_FILE    "lambda_high.bin"
#define MEDIUM_LAMBDA_FILE  "lambda_medium.bin"
#define LOW_LAMBDA_FILE		"lambda_low.bin"
#define LAMBDA_PATH			"Data\\%s"
#define SAVED_SPEEDS_FILE			"Data\\bm_few_ms_buffer.bin"
#define OUTPUT_RESULTS_FILE			"Data\\output_results.bin"
#define CUDA_TEST_RESULTS_FILE			"Data\\cuda_test.bin"
#define CPP_TO_CUDA_TEST_RESULTS_FILE			"Data\\cpp_to_cuda_test.bin"
#define AC_TIME_FILTER_FILE			"Data\\AC_time_filter.bin"
#define FILE_NAME_LENGTH_MAX		200
#define DEVICE_MAX_PARAMS			64 // max number of params for the IHC calculation
#define DEVICE_MAX_FILTER_ORDER			1024 // I doubt i will need higher order fir filters
//#define checkCudaErrors(val)           check ( (val), #val, __FILE__, __LINE__ )
#define CALC_FILTER 1
#define DELTA_RESPONSE 0
#define HEALTY_IHC  (8.0f)
#define HEALTY_OHC  (0.5f)
#define PARAM_SET_NOT_EXIST (-1) // if no param set match criteria this index will be returned
#define IS_NO_PARAM_SET(index)	(index==PARAM_SET_NOT_EXIST)
#define HAS_PARAM_SET(index)	(index>PARAM_SET_NOT_EXIST)
#define DELTA_RESPONSE 0
# define SIM_TYPE_VOICE  0
# define SIM_TYPE_SIN  1
# define SIM_TYPE_PERF  2
# define SIM_TYPE_PROFILE_GENEARATING  3
# define SIM_TYPE_JND_COMPLEX_CALCULATIONS  4 // oded starvitsky method
# define MIN_INF_POWER_LEVEL 1111.0f
#define TIME_FILTER 1
#define MAX_PARK_WIDTH 0.3
#define MIN_PARK_WIDTH 0.01
#define MIN_SUM_REF (2e-4f)
#define JND_INTERVAL_SIZE 0.02f
#define JND_BLOCK_HEAD 0.014f
#define JND_BLOCK_TAIL 0.002f
#define JND_DOUBLE_LOW_BARRIER 1e-15
#define JND_MAX_BUFFER_SIZE 1000000
#define MAX_LONG_INPUT_LENGTH 2048
#define Const_JND_Delta_Alpha_Time_Factor 0.2
#define MODEL_FLOATS_CONSTANTS_SIZE 64
#define MODEL_INTEGERS_CONSTANTS_SIZE 16   
#define MODEL_INTEGERS_CONSTANTS_SIZE_P1 (MODEL_INTEGERS_CONSTANTS_SIZE+1) 
#define EXTENDED_MODEL_INTEGERS_CONSTANTS_SIZE (MODEL_FLOATS_CONSTANTS_SIZE+1024)
#define MODEL_LONGS_CONSTANTS_SIZE 8
#define SX_SIZE 21
#define COMMON_INTS_SIZE 8
#define WARP_SIZE 32
#define MAX_WARPS_PER_BM_BLOCK 32
#define MAX_NUMBER_OF_BLOCKS 65536
#define INTERRUPTS_NUMBER 64
#define ACTIVE_JND_FLAGS 0x3
typedef float lambdaFloat; // uses to calculate lambda process in order to switch easily between double and float
typedef float JNDFloat; // floating number for jnd calculations
#endif
// Check windows
#if _WIN32 || _WIN64
#if _WIN64
#define COMPILE64BITS 1
#else
#define COMPILE64BITS 0
#endif
#endif
// Check GCC
#if __GNUC__
#if __x86_64__ || __ppc64__
#define COMPILE64BITS 1
#else
#define COMPILE64BITS 0
#endif
#endif
//#if COMPILE64BITS == 1
//typedef double real;
//#define FILTERS64BITS 1
//#else
//typedef float real;
//#endif

#if defined(FILTERS64BITS)
typedef double filterfloat;
#else
typedef float filterfloat;
#endif
#define CUDA_ERROR_CHECK 1
//#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )
#define CudaClearError()    __cudaClearError( __FILE__, __LINE__ )
#define SHOW_DEBUG 0

#define _USE_MATH_DEFINES
// iowa filters



// fir filter const


#define MAX_NUMTAPS 256
#define M_2PI  6.28318530717958647692
#define NUM_FREQ_ERR_PTS  1000    // these are only used in the FIRFreqError function.
#define dNUM_FREQ_ERR_PTS 1000.0


// from debug class that was removed
#define __RELEASE				// Show notes for the release version.
#define CMD_STATUS_RATE					1		// Show data on cmd each time 0 == mod(conter, CMD_STATUS_RATE).
#define CMD_STATUS_RATE_RELEASE			1000	// Same as CMD_STATUS_RATE but for the release version.



// if defined debug define memory allocation follow up
/*
#ifdef _DEBUG
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
// debug new command for proper follow up
	#ifndef DBG_NEW
		#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
		#define new DBG_NEW
	#endif
#endif
	*/

// end defined debug follow up


#ifdef __INTELLISENSE__
#define M_LN2       0.693147180559945309417 // log(2) = ln(2)
#define LOG_OF_TWO  0.301029995663981184    // log10(2)
#define ARC_DB_HALF 0.316227766016837952
#define DBL_MIN     2.22507385850720E-308  // from float.h
#define DBL_MAX     1.79769313486232E+308
#define FLT_MIN     1.175494E-38
#define FLT_MAX     3.402823E+38

#define ZERO_PLUS   8.88178419700125232E-16   // 2^-50 = 4*DBL_EPSILON
#define ZERO_MINUS -8.88178419700125232E-16

// from float.h  Epsilon is the smallest value that can be added to 1.0 so that 1.0 + Epsilon != 1.0
#define LDBL_EPSILON   1.084202172485504434E-019L // = 2^-63
#define DBL_EPSILON    2.2204460492503131E-16     // = 2^-52
#define FLT_EPSILON    1.19209290E-07F            // = 2^-23

#define M_E         2.71828182845904523536      // natural e
#define M_PI        3.14159265358979323846      // Pi
#define M_2PI	    6.28318530717958647692      // 2*Pi
#define M_PI_2      1.57079632679489661923      // Pi/2
#define M_PI_4      0.785398163397448309616     // Pi/4
#define M_1_PI      0.318309886183790671538     // 0.1 * Pi
#define M_SQRT2     1.41421356237309504880      // sqrt(2)
#define M_SQRT_2    0.707106781186547524401     // sqrt(2)/2
#define M_SQRT3     1.7320508075688772935274463 // sqrt(3) from Wikipedia
#define M_SQRT3_2   0.8660254037844386467637231 // sqrt(3)/2



#endif
