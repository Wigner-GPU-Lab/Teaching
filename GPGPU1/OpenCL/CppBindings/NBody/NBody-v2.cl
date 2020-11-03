// Manybody includes
#include <particle.cl>

__kernel void interaction( __global particle* particles )
{
	int gid = get_global_id(0);
	int gsi = get_global_size(0);

	particle my_particle = particles[gid];

	float3 force = (float3)(0.0, 0.0, 0.0);

	for (int i = 0 ; i < gsi ; ++i)
	{
		particle temp = particles[i];

		if (gid != i)
		{
			force += calculate_force(&my_particle, &temp);
		}
	}

	particles[gid].f = force;
}

__kernel void forward_euler( __global particle* particles, float dt )
{
	int gid = get_global_id(0);

	particle my_particle = particles[gid];

	euler_helper(&my_particle, dt);

	particles[gid] = my_particle;
}