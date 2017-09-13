// Enable from host when using double inside kernel
#ifdef USE_FP64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

// Include custom types and finite diff functions
#include <QGripper_Kernels_Debug.cl>
#include <QGripper_Kernels_IDs.cl>
#include <QGripper_Kernels_Types.cl>
#include <QGripper_Kernels_Indexers.cl>
#include <QGripper_Kernels_LoadSave.cl>
#include <QGripper_Kernels_FiniteDifference.cl>
#include <QGripper_Kernels_KGParams.cl>


//////////////////////////////////////////////////////////////////
//                                                              //
// Kernel to enforce interpolation of second radial co-ordinate //
//                                                              //
//////////////////////////////////////////////////////////////////

__kernel void kgInterpolate(
                               __global REAL * y1,    // Dynamical variables
                               __global REAL * y2,    // Dynamical variables
                               __global REAL * y3     // Dynamical variables
                           )
{
    KGEquations my_y   = loadEquationsToPrivate(MOD_GID + (int2)(0,0), y1, y2, y3);
    KGEquations my_yp1 = loadEquationsToPrivate(MOD_GID + (int2)(1,0), y1, y2, y3);
	
    KGEquations result;
    INTVEC y_coord = shiftThreadIndexY(MOD_GID.y);
    
    result.Psi_t = CONVERT_INTEGRALVEC(y_coord % 2) ? (CONVERT_INTEGRALVEC(y_coord != 0) ? my_yp1.Psi_t / 16 : my_yp1.Psi_t / 8) : my_y.Psi_t;
    result.Psi   = CONVERT_INTEGRALVEC(y_coord % 2) ? (CONVERT_INTEGRALVEC(y_coord != 0) ? my_yp1.Psi   / 16 : my_yp1.Psi   / 8) : my_y.Psi;
    result.Psi_r = CONVERT_INTEGRALVEC(y_coord % 2) ? (CONVERT_INTEGRALVEC(y_coord != 0) ? my_yp1.Psi_r / 16 : my_yp1.Psi_r / 8) : my_y.Psi_r;

    saveEquationsFromPrivate(MOD_GID, result, y1, y2, y3);
}


///////////////////////////////////////
//                                   //
// Kernel to update display vertices //
//                                   //
///////////////////////////////////////

__kernel void kgUpdVtx(
                          int graphLength,
                          __global REAL * y,
                          __global float4* vtx
                        )
{
	int myIndex = RK4GlobalIndexer(GID);

    // Read from global to private
	float data = y[ myIndex ];

	float4 out; out.x = GID.x; out.y = GID.y; out.z = data; out.w = 1.0f;

    // Write from __private to __global
	vtx[ myIndex ] = out;
}


//////////////////////////////////////////
//                                      //
// Kernel to update dynamical variables //
//                                      //
//////////////////////////////////////////

__kernel void kgUpdDynVar(
                             int stepNum,
					         __global REAL * y,
					         __global REAL * dyn0
					       )
{
}