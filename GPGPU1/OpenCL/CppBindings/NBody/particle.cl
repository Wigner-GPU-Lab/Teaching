//#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define G 1.0f
//6.67384e-11f

typedef struct __attribute__ ((aligned(16)))
{
    float3 pos;
	float3 v;
	float3 f;	
	float mass;
} particle;

float3 calculate_force0(__private particle* first, __private particle* second)
{
	return -G * first->mass * second->mass * (first->pos - second->pos) / pown(sqrt(pown(first->pos.x - second->pos.x, 2) + pown(first->pos.y - second->pos.y, 2) + pown(first->pos.z - second->pos.z, 2)), 3);
}

float3 calculate_force1(__private particle* first, __private particle* second)
{
	return -G * first->mass * second->mass * (first->pos - second->pos) * pown(distance(first->pos, second->pos), -3);
}

float3 calculate_force2(__private particle* first, __private particle* second)
{
    float3 d = first->pos - second->pos;
    float  q = sqrt(dot(d, d));
	return -G * first->mass * second->mass * d / ( q*q*q );
}

void euler_helper( __private particle* part, float dt )
{
	float dt_per_m = dt / part->mass;
    float dt_sq = dt*dt;
    float half_dt_sq_per_m = 0.5f * dt_sq / part->mass;

    part->pos += part->v * dt + part->f * half_dt_sq_per_m;

    part->v += part->f * dt_per_m;

	part->f = (float3)(0.0, 0.0, 0.0);
}

__kernel void forward_euler( __global particle* particles, float dt )
{
    int gid = get_global_id(0);

    particle my_particle = particles[gid];

    euler_helper(&my_particle, dt);

    particles[gid] = my_particle;
}