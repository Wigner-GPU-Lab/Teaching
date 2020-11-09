// Manybody includes
#include <particle.cl>

__kernel void interaction( __global particle* particles, __local particle* share )
{
	int gid = get_global_id(0);
	int gsi = get_global_size(0);
	int lid = get_local_id(0);
	int lsi = get_local_size(0);
	int Gsi = get_num_groups(0);

	particle my_particle = particles[gid];

	float3 force = (float3)(0.0, 0.0, 0.0);

	// Loop over batches of shared particles
	for (int I = 0 ; I < Gsi ; ++I)
	{
		// Copy individually
		share[lid] = particles[I * lsi + lid];
		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop over particles inside a batch of shared particles
		for (int i = 0 ; i < lsi ; ++i)
		{
			particle temp = share[i];

			if (gid != (I * lsi + i))
			{
				force += calculate_force0(&my_particle, &temp);
			}
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	particles[gid].f = force;
}