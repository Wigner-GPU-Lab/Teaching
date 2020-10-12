__kernel void reduce0(__global float* dst, __global float* src, __local float* tmp, int n) 
{
    // each thread loads one element from global to shared mem
    unsigned int tid = get_local_id(0);
    unsigned int i   = get_global_id(0);
    
    tmp[tid] = src[i];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // do reduction in shared mem
    for(unsigned int s=1; s < get_local_size(0); s *= 2)
    {
        if (tid % (2*s) == 0)
        {
            tmp[tid] += tmp[tid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // write result for this block to global mem
    if(tid == 0){ dst[get_group_id(0)] = tmp[0]; }
}