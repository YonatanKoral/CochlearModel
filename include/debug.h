
// ------------------------ MATLAB's Procedures ------------------------
#pragma once

// --------------------------------------------------
// DEBUG Switches:
// ---------------

//#define M_AUX_DLL			// MATLAB Auxiliary DLL.
//#define _DEBUG_01			// debug switch

//#define __DEBUG_MATLAB			// Define MATLAB debug flag.
//#define __DEBUG_CMD_STATUS			// Writes remarks to the cmd.
//#define __DEBUG_CMD_NOTES				// Writes remarks to the cmd.
//#define __DEBUG_LOG					// Writes to the log.
//#define __DEBUG_BIN					// Writes stuff to a bin file.
#define __RELEASE				// Show notes for the release version.
// --------------------------------------------------

#ifndef __DEBUG_H
#define __DEBUG_H

#define CMD_STATUS_RATE					1		// Show data on cmd each time 0 == mod(conter, CMD_STATUS_RATE).
#define CMD_STATUS_RATE_RELEASE			1000	// Same as CMD_STATUS_RATE but for the release version.

#ifdef __DEBUG_MATLAB

#define MATLAB_SHOW_ON_SCREAN_COUNTER	100

#pragma comment( lib, "mclmcrrt" )
#pragma comment( lib, "M_Aux" )

#include "model.h"
#include "state.h"

#include "matrix.h"
#include "mat.h"
//#include "mclmcrrt.h"
//#include "mclmcr.h"
#include "M_Aux.h"

//#pragma comment( lib, "mclmcrrt" )
//#pragma comment( lib, "M_Aux" )

bool Init_MATLAB(void);												//Initialize MATLAB's auxiliary libraries.
void Dest_MATLAB(void);												//Clear MATLAB's auxiliary libraries.
void Plot_On_Screen( CState* now, CModel model, long counter );		// Plot stuff on screen.

#endif

#endif
