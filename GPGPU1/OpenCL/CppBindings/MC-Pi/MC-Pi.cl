#define CLRNG_SINGLE_PRECISION
#include <clRNG/mrg31k3p.clh>

__kernel void count(unsigned int gens_per_item,
                    __global clrngMrg31k3pHostStream *streams,
                    __global unsigned int *out)
{
    size_t gid = get_global_id(0);

    clrngMrg31k3pStream workItemStream;
    clrngMrg31k3pCopyOverStreamsFromGlobal(1, &workItemStream, &streams[gid]);

    unsigned int count = 0;
    for(int i = 0 ; i < gens_per_item ; ++i)
    {
        float2 pos = (float2)(
            clrngMrg31k3pRandomU01(&workItemStream),
            clrngMrg31k3pRandomU01(&workItemStream)
        );

        if (pos.x * pos.x + pos.y * pos.y < 1) count++;
    }

    out[gid] = count;
}

unsigned int op(unsigned int lhs, unsigned int rhs);

unsigned int read_local(local unsigned int* shared, size_t count, unsigned int zero, size_t i)
{
    return i < count ? shared[i] : zero;
}

kernel void reduce(
    global unsigned int* front,
    global unsigned int* back,
    local unsigned int* shared,
    unsigned int length,
    unsigned int zero_elem
)
{
    const size_t lid = get_local_id(0),
                 lsi = get_local_size(0),
                 wid = get_group_id(0),
                 wsi = get_num_groups(0);

    const size_t wg_stride = lsi * 2,
                 valid_count = wid != wsi - 1 ? // If not last group
                    wg_stride :                 // as much as possible
                    length - wid * wg_stride;   // only the remaining

    // Copy real data to local
    event_t read;
    async_work_group_copy(
        shared,
        front + wid * wg_stride,
        valid_count,
        read);
    wait_group_events(1, &read);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lsi; i != 0; i /= 2)
    {
        if (lid < i)
            shared[lid] =
                op(
                    read_local(shared, valid_count, zero_elem, lid),
                    read_local(shared, valid_count, zero_elem, lid + i)
                );
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) back[wid] = shared[0];
}
