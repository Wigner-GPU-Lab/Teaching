float sq(float x){ return x*x; }
float cube(float x){ return x*x*x; }

__kernel void nbody_step(__global float3* V, __global float4* P, float G, float dt) 
{
    unsigned int N = get_global_size(0);
    unsigned int i = get_global_id(0);
    float3 sum = {0.0f, 0.0f, 0.0f};
    float4 Pi = P[i];
    float x = Pi.x;
    float y = Pi.y;
	float z = Pi.z;
	
	for(int j=0; j<N; ++j)
    {
        float4 Pj = P[j];
        float dx = Pj.x - x;
        float dy = Pj.y - y;
        float dz = Pj.z - z;
        float q = i != j ? sqrt( sq(dx) + sq(dy) + sq(dz) ) : 1.0f;
        float rec = Pj.w / cube(q);
        sum.x += dx * rec;
        sum.y += dy * rec;
        sum.z += dz * rec;
    }
    
    //a = F/m = -G * sum
    //pos = pos + vel * dt + acc / 2 *dt^2
    float3 Vi = V[i];
    float vx = Vi.x + G*sum.x * dt;
    float vy = Vi.y + G*sum.y * dt;
    float vz = Vi.z + G*sum.z * dt;

    x = x + (Vi.x + G/2*sum.x * dt)*dt;
    y = y + (Vi.y + G/2*sum.y * dt)*dt;
    z = z + (Vi.z + G/2*sum.z * dt)*dt;

    P[i] = (float4)(x, y, z, Pi.w);
    V[i] = (float3)(vx, vy, vz);

}