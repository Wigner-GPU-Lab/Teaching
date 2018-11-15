// NBody includes
#include <NBodyStep.hpp>

// C++ includes
#include <utility>  // std::forward
#include <limits>   // std::numeric_limits


struct NBodyStepKernel;

template <typename T, std::size_t J>
struct unroll_step
{
    unroll_step(std::size_t i, T&& t)
    {
        unroll_step<T, J - 1>(i, std::forward<T>(t));
        t(i + J);
    }
};

template <typename T>
struct unroll_step<T, (std::size_t)0>
{
    unroll_step(std::size_t i, T&& t) { t(i); }
};

template <std::size_t N, typename T>
void unroll_loop(std::size_t first, std::size_t last, T&& t)
{
    std::size_t i = first;

    for (; i + N < last; i += N)
        unroll_step<T, N - 1>(i, std::forward<T>(t));

    for (; i < last; ++i)
        t(i);
}

void NBodyStep(cl::sycl::queue sycl_queue,
               std::vector<cl::Memory>& gl_resources,
               std::array<cl::sycl::buffer<real4>, 2> pos_double_buf,
               std::array<cl::sycl::buffer<real4>, 2> vel_double_buf,
               std::size_t particle_count,
               bool fast_interop)
{
    // NOTE 1: When cl_khr_gl_event is NOT supported, then clFinish() is the only portable
    //         sync method and hence that will be called.
    //
    // NOTE 2.1: When cl_khr_gl_event IS supported AND the possibly conflicting OpenGL
    //           context is current to the thread, then it is sufficient to wait for events
    //           of clEnqueueAcquireGLObjects, as the spec guarantees that all OpenGL
    //           operations involving the acquired memory objects have finished. It also
    //           guarantees that any OpenGL commands issued after clEnqueueReleaseGLObjects
    //           will not execute until the release is complete.
    //         
    //           See: opencl-1.2-extensions.pdf (Rev. 15. Chapter 9.8.5)

    cl::Event acquire, release;
    
    cl::CommandQueue{ sycl_queue.get() }.enqueueAcquireGLObjects(&gl_resources, nullptr, &acquire);
    acquire.wait();

    sycl_queue.submit([&](cl::sycl::handler& cgh)
    {
        auto pos = pos_double_buf[DoubleBuffer::Front].get_access<cl::sycl::access::mode::read>();
        auto new_pos = pos_double_buf[DoubleBuffer::Back].get_access<cl::sycl::access::mode::discard_write>();
        auto vel = vel_double_buf[DoubleBuffer::Front].get_access<cl::sycl::access::mode::read>();
        auto new_vel = vel_double_buf[DoubleBuffer::Back].get_access<cl::sycl::access::mode::discard_write>();

        cgh.parallel_for<NBodyStepKernel>(cl::sycl::range<1>{ particle_count }, [=](const cl::sycl::item<1> item)
        {
            /*
            real4 myPos = pos[item];
            real4 acc = { 0.f, 0.f, 0.f, 0.f };
            real epsSqr = std::numeric_limits<real>::epsilon();
            real deltaTime = real(0.005);

            auto xyz = [](real4& vec) { return vec.swizzle<0, 1, 2>(); };
            auto interact = [&](const size_t i)
            {
                real4 p = pos[i];
                real4 r;
                xyz(r) = xyz(p) - xyz(myPos);
                real distSqr = r.x() * r.x() + r.y() * r.y() + r.z() * r.z();

                real invDist = 1.0f / sqrt(distSqr + epsSqr);
                real invDistCube = invDist * invDist * invDist;
                real s = p.w() * invDistCube;

                // accumulate effect of all particles
                xyz(acc) += s * xyz(r);
            };

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
            constexpr size_t factor = 8; // loop unroll depth
            unroll_loop<factor>(0, particle_count, interact);

            real4 oldVel = vel[item];

            // updated position and velocity
            real4 newPos;
            xyz(newPos) = xyz(myPos) + xyz(oldVel) * deltaTime + xyz(acc) * real(0.5) * deltaTime * deltaTime;
            newPos.w() = myPos.w();

            real4 newVel;
            xyz(newVel) = xyz(oldVel) + xyz(acc) * deltaTime;
            newVel.w() = oldVel.w();

            // write to global memory
            new_pos[item] = newPos;
            new_vel[item] = newVel;
            */
            /*
            new_pos[item] = pos[item];
            new_vel[item] = vel[item];
            */
            real4 myPos = pos[item];
            real4 acc = { 0.f, 0.f, 0.f, 0.f };
            real epsSqr = std::numeric_limits<real>::epsilon();
            real deltaTime = real(0.005);

            auto xyz = [](real4& vec) { return vec.swizzle<0, 1, 2>(); };
            for (int i = 0; i < particle_count; ++i)
            {
                real4 p = pos[i];
                real4 r;
                xyz(r) = xyz(p) - xyz(myPos);
                real distSqr = r.x() * r.x() + r.y() * r.y() + r.z() * r.z();

                real invDist = 1.0f / sqrt(distSqr + epsSqr);
                real invDistCube = invDist * invDist * invDist;
                real s = p.w() * invDistCube;

                // accumulate effect of all particles
                auto eff = s * xyz(r);
                acc += real4{ eff.x(), eff.y(), eff.z(), 0.0f };
            }

            real4 oldVel = vel[item];

            // updated position and velocity
            real4 newPos;
            xyz(newPos) = xyz(myPos) + xyz(oldVel) * deltaTime + xyz(acc) * real(0.5) * deltaTime * deltaTime;
            newPos.w() = myPos.w();

            real4 newVel;
            xyz(newVel) = xyz(oldVel) + xyz(acc) * deltaTime;
            newVel.w() = oldVel.w();
        });
    });
    
    cl::CommandQueue{ sycl_queue.get() }.enqueueReleaseGLObjects(&gl_resources, nullptr, &release);
    
    // Wait for all OpenCL commands to finish
    if (!fast_interop)
        cl::finish();
    else
        release.wait();
}
