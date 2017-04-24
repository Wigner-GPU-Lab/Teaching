#include <CL/sycl.hpp>
#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode sycl_read_write = cl::sycl::access::mode::read_write;

template <typename T> class MatulKernel;

template <typename T>
void matmul_kernel(int size, int blocksize, std::vector<T> const& mA, std::vector<T> const& mB, std::vector<T>& mC)
{
   cl::sycl::queue deviceQueue{cl::sycl::gpu_selector()};

  cl::sycl::buffer<T, 1> ba(mA.data(), size*size);
  cl::sycl::buffer<T, 1> bb(mB.data(), size*size);
  cl::sycl::buffer<T, 1> bc(mC.data(), size*size);

  {
      deviceQueue.submit([&](cl::sycl::handler &cgh)
      {
        auto A = ba.template get_access<sycl_read>(cgh);
        auto B = bb.template get_access<sycl_read>(cgh);
        auto C = bc.template get_access<sycl_write>(cgh);

        auto local_range = cl::sycl::range<1>(blocksize * blocksize);
        auto global_range = cl::sycl::nd_range<2>{cl::sycl::range<2>(size, size), cl::sycl::range<2>(blocksize, blocksize)};
        
        cl::sycl::accessor<T, 1, sycl_read_write, cl::sycl::access::target::local> Ablock(local_range, cgh);
        cl::sycl::accessor<T, 1, sycl_read_write, cl::sycl::access::target::local> Bblock(local_range, cgh);

        cgh.parallel_for<class MatulKernel<T>>(global_range, [=](cl::sycl::nd_item<2> i)
        {
            int lx = i.get_local(0);
	    int ly = i.get_local(1);

	    int gx = i.get_global(0);
	    int gy = i.get_global(1);

	    int steps = size / blocksize;
            T acc = 0.0;

            for( int s=0; s<steps; s=s+1)
	    {
               int Ablockoffset = ly * blocksize + lx;
               int Bblockoffset = lx * blocksize + ly;
               int Aoffset      = gy * size + s * blocksize + lx;
               int Boffset      = (s * blocksize + ly) * size + gx;

               Ablock[Ablockoffset] = A[Aoffset];
               Bblock[Bblockoffset] = B[Boffset];

               i.barrier(cl::sycl::access::fence_space::local_space);

               for (int i = 0; i < blocksize; ++i)
               {
                   T fA = Ablock[ly*blocksize+i];
                   T fB = Bblock[lx*blocksize+i];
                   acc += fA * fB;
                }
                
                i.barrier(cl::sycl::access::fence_space::local_space);
             }
             C[gy * size + gx] = acc;
        });
      });
  }
}

int main()
{
    static const int size = 1024;
    static const int blocksize = 8;
    std::vector<double> A(size*size), B(size*size), C(size*size), D(size*size);

    std::random_device rnd_device;
    std::mt19937 mersenne_engine(rnd_device());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto gen = [&]() { return dist(mersenne_engine); };
    std::generate(A.begin(), A.end(), gen );
    std::generate(B.begin(), B.end(), gen );
    std::fill(C.begin(), C.end(), 0.0f);
    std::fill(D.begin(), D.end(), 0.0f);

    //naive implementation:
    for(int i=0; i<size; ++i)
    {
        for(int j=0; j<size; ++j)
        {
            auto acc = 0.0;
            for(int k=0; k<size; ++k)
            {
                acc += A[i*size+k] * B[k*size+j];
            }
            C[i*size+j] = acc;
        }
    }

    matmul_kernel(size, blocksize, A, B, D);

    auto checker = [&](std::vector<double> const& u, std::vector<double> const& v)
    {
        return std::inner_product(u.cbegin(), u.cend(), v.cbegin(), true, [](bool const& b, double const& x){ return b && x < 1e-10; }, [](double const& ref, double const& x){ return std::abs(ref-x); } );
    };

    auto res = checker(C, D);

    std::cout << "Result: " << std::string(res ? "PASSED" : "FAILED") << "\n";
    
    return 0;
}
