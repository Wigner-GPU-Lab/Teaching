#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void vecAdd(double a,
                     __global double* x,
                     __global double* y)
{
	int gid = get_global_id(0);
	
	y[gid] = a * x[gid] + y[gid];
}
