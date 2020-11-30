__kernel void gpu_histo_global_atomics( __global unsigned int* output, __global uchar4* input, int W )
{
    // linear block index within 2D grid
    int B = get_group_id(0) + get_group_id(1) * get_num_groups(0);

    //Output index start for this block's histogram:
    int I = B*(3*256);
    __global unsigned int* H = output + I;
    
    // process pixel blocks horizontally
    // updates our block's partial histogram in global memory
    int y = get_group_id(1) * get_local_size(1) + get_local_id(1);
    for (int x = get_local_id(0); x < W; x += get_local_size(0))
    {
        uchar4 pixels = input[y * W + x];
        atomic_add(&H[0 * 256 + pixels.x], 1);
        atomic_add(&H[1 * 256 + pixels.y], 1);
        atomic_add(&H[2 * 256 + pixels.z], 1);
    }
}

__kernel void gpu_histo_shared_atomics( __global unsigned int* output, __global uchar4* input, int W )
{
    //histograms are in shared memory:
    __local unsigned int histo[3 * 256];

    //Number of threads in the block:
    int Nthreads = get_local_size(0) * get_local_size(1);
    //Linear thread idx:
    int LinID = get_local_id(0) + get_local_id(1) * get_local_size(0);
    //zero histogram:
    for (int i = LinID; i < 3*256; i += Nthreads){ histo[i] = 0; }
    __syncthreads();

    // linear block index within 2D grid
    int B = get_group_id(0) + get_group_id(1) * get_num_groups(0);

    // process pixel blocks horizontally
    // updates the partial histogram in shared memory
    int y = get_group_id(1) * get_local_size(1) + get_local_id(1);
    for (int x = get_local_id(0); x < W; x += get_local_size(0))
    {
        uchar4 pixels = input[y * W + x];
        atomic_add(&histo[0 * 256 + pixels.x], 1);
        atomic_add(&histo[1 * 256 + pixels.y], 1);
        atomic_add(&histo[2 * 256 + pixels.z], 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //Output index start for this block's histogram:
    int I = B*(3*256);
    __global unsigned int* H = output + I;

    //Copy shared memory histograms to globl memory:
    for (int i = LinID; i < 256; i += Nthreads)
    {
        H[0*256 + i] = histo[0*256 + i];
        H[1*256 + i] = histo[1*256 + i];
        H[2*256 + i] = histo[2*256 + i];
    }
}

__kernel void gpu_histo_accumulate(__global unsigned int* out, __global const unsigned int* in, int nBlocks)
{
    //each thread sums one shade of the r, g, b histograms
    int i = get_global_id(0);
    if(i < 3 * 256)
    {
        unsigned int sum = 0;
        for(int j = 0; j < nBlocks; j++)
        {
            sum += in[i + (3*256) * j];
        }            
        out[i] = sum;
    }
}