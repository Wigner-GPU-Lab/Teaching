/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * Each work-item invocation of this kernel, calculates the position for 
 * one particle
 *
 */

// Enable from host when using double inside kernel
#ifdef USE_FP64
#pragma OPENCL EXTENSION cl_khr_fp64: enable
#endif

#define UNROLL_FACTOR  8
__kernel void nbody_sim(__global real4* pos,
                        __global real4* newPosition,
                        __global real4* vel,
                        __global real4* newVelocity,
                        unsigned int numBodies,
                        real deltaTime,
                        real epsSqr)
{
    size_t gid = get_global_id(0);
    real4 myPos = pos[gid];
    real4 acc = (real4)0;

	// NOTE 1:
	//
	// This loop construct unrolls the outer loop in chunks of UNROLL_FACTOR
	// up to a point that still fits into numBodies. After the unrolled part
	// the remainder os particles are accounted for. (NOTE 1.1: the loop variable)
	// 'j' is not used anywhere in the body. It's only used as a trivial
	// unroll construct. NOTE 1.2: 'i' is left intact after the unrolled loops.
	// The finishing loop picks up where the other left off.
	//
	// NOTE 2:
	//
	// epsSqr is used to omit self interaction checks alltogether by introducing
	// a minimal skew in the deistance calculation. Thus, the almost infinity
	// in invDistCube is suppressed by the ideally 0 distance calulated by
	// r.xyz = p.xyz - myPos.xyz where the right-hand side hold identical values.
	//
    unsigned int i = 0;
    for (; (i+UNROLL_FACTOR) < numBodies; )
	{
#pragma unroll UNROLL_FACTOR
        for(int j = 0; j < UNROLL_FACTOR; j++,i++)
		{
            real4 p = pos[i];
            real4 r;
            r.xyz = p.xyz - myPos.xyz;
            real distSqr = r.x * r.x  +  r.y * r.y  +  r.z * r.z;

            real invDist = 1.0 / sqrt(distSqr + epsSqr);
            real invDistCube = invDist * invDist * invDist;
            real s = p.w * invDistCube;

            // accumulate effect of all particles
            acc.xyz += s * r.xyz;
        }
    }
    for (; i < numBodies; i++) {
        real4 p = pos[i];
        real4 r;
        r.xyz = p.xyz - myPos.xyz;
        real distSqr = r.x * r.x  +  r.y * r.y  +  r.z * r.z;

        real invDist = 1.0 / sqrt(distSqr + epsSqr);
        real invDistCube = invDist * invDist * invDist;
        real s = p.w * invDistCube;

        // accumulate effect of all particles
        acc.xyz += s * r.xyz;
    }

    real4 oldVel = vel[gid];

    // updated position and velocity
    real4 newPos;
    newPos.xyz = myPos.xyz + oldVel.xyz * deltaTime + acc.xyz * 0.5 * deltaTime * deltaTime;
    newPos.w = myPos.w;

    real4 newVel;
    newVel.xyz = oldVel.xyz + acc.xyz * deltaTime;
    newVel.w = oldVel.w;

    // write to global memory
    newPosition[gid] = newPos;
    newVelocity[gid] = newVel;
}
