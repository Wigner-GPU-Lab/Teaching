kernel void reduce(
    unsigned int length,
    local float* shared,
    global float* front,
    global float* back
)
{
    const size_t gid = get_global_id(0),
                 lid = get_local_id(0),
                 lsi = get_local_size(0),
                 wid = get_group_id(0),
                 wsi = get_num_groups(0);

    const size_t wg_stride = lsi * 2;

    // Copy real data to local
    event_t read;
    async_work_group_copy(
        shared,
        front + wid * wg_stride,
        wid != wsi - 1 ?        // If not last group
            wg_stride :         // fill local as normal
            length % wg_stride, // otherwise copy only available
        read);
    wait_group_events(1, &read);

    // Fill rest with dummy zero elements of the group
    if (wid == (wsi - 1))
    {
        if ((lsi + lid) >= (length % wg_stride))
            shared[lsi + lid] = MAXFLOAT;
        if (lid >= (length % wg_stride))
            shared[lid] = MAXFLOAT;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lsi; i != 0; i /= 2)
    {
        if (lid < i)
            shared[lid] = min(shared[lid], shared[lid + i]);
            //shared[lid] = shared[lid] < shared[lid + i] ? shared[lid] : shared[lid + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) back[wid] = shared[0];
}