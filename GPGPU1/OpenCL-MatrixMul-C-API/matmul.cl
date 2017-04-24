__kernel void matmul0(__global double* A, 
                      __global double* B, 
                      __global double* C, 
                      int size)
{
  
   int thx = get_global_id(0); 
   int thy = get_global_id(1);

   double acc = 0.0;
   for (int i = 0; i < size; ++i)
   {
      acc += A[thy * size + i] * B[i * size + thx];
   }
 
   C[thy * size + thx] = acc;
}


__kernel void matmul1(__global double* A,
                      __global double* B,
                      __global double* C,
					           int    size,
                               int    blocksize,
                      __local  double* Ablock,
					  __local  double* Bblock)
{
	int lx = get_local_id(0);
	int ly = get_local_id(1);

	int gx = get_global_id(0);
	int gy = get_global_id(1);

	int steps = size / blocksize;
	double acc = 0.0;
	for( int s=0; s<steps; s=s+1)
	{
		int Ablockoffset = ly * blocksize + lx;
		int Bblockoffset = lx * blocksize + ly;
		int Aoffset      = gy * size + s * blocksize + lx;
		int Boffset      = (s * blocksize + ly) * size + gx;

		Ablock[Ablockoffset] = A[Aoffset];
		Bblock[Bblockoffset] = B[Boffset];

		barrier(CLK_LOCAL_MEM_FENCE);

		for (int i = 0; i < blocksize; ++i)
		{
			double fA = Ablock[ly*blocksize+i];
			double fB = Bblock[lx*blocksize+i];
			acc += fA * fB;
		}

        barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	C[gy * size + gx] = acc;
}