//
//Initialize MATLAB's auxiliary libraries.
//
//
// Oded, 18-11-2009.

#include "debug.h"
//#include "main.h"

#ifdef __DEBUG_MATLAB

bool Init_MATLAB(void)	
{
	if ( !mclInitializeApplication(NULL,0) )
	{
		printf("Could not initialize the application.\n");
		printf("Press any key to continue...");
		return false;
	}

	if ( !M_AuxInitialize() )
	{
		printf("Could not initialize MainMATLAB_DLL_Initialize library.\n");
		printf("Press any key to continue...");
		return false;
	}

	return true;
}

//Clear MATLAB's auxiliary libraries.
void Dest_MATLAB(void)
{
	M_AuxTerminate();
	mclTerminateApplication();

}

void Plot_On_Screen( CState* now, CModel model, long counter )	{

		if ( !(counter % MATLAB_SHOW_ON_SCREAN_COUNTER) )	{

			mxArray *fig_num = mxCreateDoubleScalar( 1 );		// Figure number 1.

			const int sections = SECTIONS;
			mxArray *mx_xxx = mxCreateDoubleMatrix(sections, 1, mxfloat);			// x axis (length).		

			// --- Plot now->_BM_sp ---
			// ---------------------------
			mxArray *mx_BM_speed = mxCreateDoubleMatrix(sections, 1, mxfloat);		// MATLAB's temporary vector.

			// Assign a value to the mxArray:
			for (int i = 0 ; i < sections ; i++)	{
				*( mxGetPr(mx_xxx)+i ) = i * model._len / sections;
				*( mxGetPr(mx_BM_speed)+i ) = now->_BM_sp[i];
			}

			mlfM_Hold_Figure(fig_num, mxCreateString("on"));
			mlfM_Plot_1(fig_num, mx_xxx, mx_BM_speed, mxCreateString("BM_{speed}"));


			//// --- Plot now->_pressure ---
			//// ---------------------------
			//*( mxGetPr(fig_num) ) = 2;							// Figure number 2.
			//mxArray *mx_pressure = mxCreateDoubleMatrix(sections, 1, mxfloat);		// MATLAB's temporary vector.

			//// Assign a value to the mxArray:
			//for (int i = 0 ; i < sections ; i++)
			//	*( mxGetPr(mx_pressure)+i ) = now->_pressure[i];
			//
			//mlfM_Hold_Figure(fig_num, mxCreateString("on"));
			//mlfM_Plot_1(fig_num, mx_xxx, mx_pressure, mxCreateString("P(x,t) - Pressure"));

			//// --- Plot now->_p_ohc ---
			//// -------------------------------
			//*( mxGetPr(fig_num) ) = 3;							// Figure number 2.
			//mxArray *mx_ohc_pressure = mxCreateDoubleMatrix(sections, 1, mxfloat);		// MATLAB's temporary vector.

			//// Assign a value to the mxArray:
			//for (int i = 0 ; i < sections ; i++)
			//	*( mxGetPr(mx_ohc_pressure)+i ) = now->_p_ohc[i];
			//
			//mlfM_Hold_Figure(fig_num, mxCreateString("on"));
			//mlfM_Plot_1(fig_num, mx_xxx, mx_ohc_pressure, mxCreateString("OHC_{pressure}"));


			//// --- Plot now->_deltaL_disp ---
			//// ------------------------------
			//*( mxGetPr(fig_num) ) = 4;							// Figure number 2.
			//mxArray *mx_delta_l_ohc = mxCreateDoubleMatrix(sections, 1, mxfloat);		// MATLAB's temporary vector.

			//// Assign a value to the mxArray:
			//for (int i = 0 ; i < sections ; i++)
			//	*( mxGetPr(mx_delta_l_ohc)+i ) = now->_deltaL_disp[i];
			//
			//mlfM_Hold_Figure(fig_num, mxCreateString("on"));
			//mlfM_Plot_1(fig_num, mx_xxx, mx_delta_l_ohc, mxCreateString("\\Delta l_{ohc}"));


			//// --- CF ---
			//// ------------------------------
			//*( mxGetPr(fig_num) ) = 5;							// Figure number 2.
			//CVector CF(SECTIONS, 0);
			//CF = sqrt( model._S_bm / model._M_bm );

			//mxArray *mx_CF = mxCreateDoubleMatrix(sections, 1, mxfloat);		// MATLAB's temporary vector.

			//// Assign a value to the mxArray:
			//for (int i = 0 ; i < sections ; i++)
			//	*( mxGetPr(mx_CF)+i ) = CF[i];
			//
			//mlfM_Hold_Figure(fig_num, mxCreateString("on"));
			//mlfM_Plot_1(fig_num, mx_xxx, mx_CF, mxCreateString("CF(x) = sqrt( S(x)/C(s) )"));


			/*
			// --- Plot now->_p_ohc ---
			*( mxGetPr(fig_num) ) = 4;							// Figure number 2.
			mxArray *mx_OW_speed = mxCreateDoubleMatrix(sections, 1, mxfloat);		// MATLAB's temporary vector.

			// Assign a value to the mxArray:
			for (int i = 0 ; i < sections ; i++)
				*( mxGetPr(mx_OW_speed)+i ) = now->_OW_sp;
			
			mlfM_Hold_Figure(fig_num, mxCreateString("on"));
			mlfM_Plot_1(fig_num, mxCreateDoubleScalar(counter/MATLAB_SHOW_ON_SCREAN_COUNTER), mx_OW_speed, mxCreateString("OW_{speed}"));
			*/

			//pause();

			// Clean Memory :
			// -----------------
			mxDestroyArray(fig_num);
			mxDestroyArray(mx_xxx);
			mxDestroyArray(mx_BM_speed);
			//mxDestroyArray(mx_pressure);

		}


}

#endif
