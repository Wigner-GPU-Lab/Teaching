#define G 6.67384e-11

typedef struct __attribute__ ((aligned(16)))
{
    float3 pos;
    float3 v;
    float3 f;	
    float mass;
} particle;

float3 burning_calculate_force(__private particle* first, __private particle* second)
{
    return -G * first->mass * second->mass * (first->pos - second->pos) / pown(sqrt(pown(first->pos.x - second->pos.x, 2) + pown(first->pos.y - second->pos.y, 2) + pown(first->pos.z - second->pos.z, 2)), 3);
}

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
            force += burning_calculate_force(&my_particle, &temp);
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