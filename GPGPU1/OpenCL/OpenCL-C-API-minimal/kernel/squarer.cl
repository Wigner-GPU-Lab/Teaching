__kernel void squarer(__global float* in, __global float* out)
{
	int idx = get_global_id(0);
	out[idx] = in[idx] * in[idx];
}
