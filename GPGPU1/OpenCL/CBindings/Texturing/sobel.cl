__constant sampler_t sampler =
      CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP_TO_EDGE
    | CLK_FILTER_NEAREST;

__kernel void sobel(read_only image2d_t src, write_only image2d_t dst)
{
	int x = get_global_id(0);
    int y = get_global_id(1);

    int w = get_global_size(0);
    int h = get_global_size(1);
    	
    if(x > 0 && x < w && y > 0 && y < h)
    {
        //pixels:
        float4 D[9];
		D[0] = read_imagef(src, sampler, (int2)(x-1, y-1));
		D[1] = read_imagef(src, sampler, (int2)(x+0, y-1));
		D[2] = read_imagef(src, sampler, (int2)(x+1, y-1));
		D[3] = read_imagef(src, sampler, (int2)(x-1, y+0));
        D[4] = read_imagef(src, sampler, (int2)(x+0, y+0));
        D[5] = read_imagef(src, sampler, (int2)(x+1, y+0));
		D[6] = read_imagef(src, sampler, (int2)(x-1, y+1));
        D[7] = read_imagef(src, sampler, (int2)(x+0, y+1));
        D[8] = read_imagef(src, sampler, (int2)(x+1, y+1));
        

        float4 resx = D[0] - D[2] + 2.0f * (D[3] - D[5]) + D[6] - D[8];
        float4 resy = D[0] + D[2] + 2.0f * (D[1] - D[7]) - D[6] - D[8];

        //write_imagef(dst, (int2)(x, y), D[4] * (0.25f + 0.75f * sqrt(dot(resx, resx)+dot(resy, resy))));
        write_imagef(dst, (int2)(x, y), clamp(D[4] * (sqrt(dot(resx, resx)+dot(resy, resy))), 0.0f, 1.0f));
    }

}