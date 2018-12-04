// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <numeric>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>

static const double pi = 3.141592653589793238462643383279502884197169399375105820974;

template<typename T>
auto sq(T const& x){ return x*x; }

double uniform_rnd(long& state)
{
	const long A = 48271;      /* multiplier*/
	const long M = 2147483647; /* modulus */
	const long Q = M / A;      /* quotient */
	const long R = M % A;      /* remainder */
	long t = A * (state % Q) - R * (state / Q);
	if (t > 0){ state = t;     }
	else      { state = t + M; }
	return ((double) state / M);
}

template<typename Fi, typename Fm, typename T>
auto MCI(cl::sycl::queue& queue, size_t nth, size_t N, Fi f, Fm mask, T x0, T x1, T y0, T y1)
{
	size_t local_count = 256;

	size_t thcount = N / nth;

	std::vector<T> tmp( (size_t)std::ceil(nth/local_count) );

	//printf("Number of samples: %zi\n", N);
	//printf("Number of threads: %zi\n", nth);
	//printf("Number of blocks: %zi\n", tmp.size());
	//printf("Try limit per thread: %zi\n", thcount);

	T res = (T)0;
	auto t0 = std::chrono::high_resolution_clock::now();
	{
		cl::sycl::buffer<T, 1> b_tmp(tmp.data(), tmp.size());
		cl::sycl::buffer<T, 1> b_res(&res, 1); 

		cl::sycl::nd_range<1> r(nth, local_count);

		queue.submit([&](cl::sycl::handler& cgh)
		{
			auto a_tmp = b_tmp.template get_access<cl::sycl::access::mode::write>(cgh);

			cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> loc(cl::sycl::range<1>(local_count), cgh);
			t0 = std::chrono::high_resolution_clock::now();
			cgh.parallel_for<class MCIKernel>(r, [=](cl::sycl::nd_item<1> id)
			{
				auto g = id.get_group().get_id(0);
				auto bs = id.get_local_range().get(0);
				auto l = id.get_local_id().get(0);

				long q0 = (long)(15348919 ^ (7+g));
				long state = (long)(2147483647 * uniform_rnd(q0));

				double t = 0.0;
				size_t n = 0;
				while(n < thcount)
				{
					double x = uniform_rnd(state) * (x1-x0) + x0;
					double y = uniform_rnd(state) * (y1-y0) + y0;

					if(mask(x, y))
					{
						t += f(x, y);
					}
					n += 1;
				}
				loc[l] = t / N;

				id.barrier(cl::sycl::access::fence_space::local_space);

				// do reduction in shared mem
				for(auto s=bs/2; s > 0; s >>= 1)
				{
					if(l < s)
					{
						loc[l] = loc[l] + loc[l + s];
					}
					id.barrier(cl::sycl::access::fence_space::local_space);
				}

				// write result for this block to global mem
				if(l == 0){ a_tmp[g] = loc[0]; }
			});
			t0 = std::chrono::high_resolution_clock::now();
		});

		queue.wait();
	}

	if(tmp.size() == 1){ res = tmp[0]; }
	else{ res = std::accumulate( tmp.cbegin(), tmp.cend(), 0.0); }

	res = (res)*(x1-x0)*(y1-y0);

	auto t1 = std::chrono::high_resolution_clock::now();

	return std::make_pair(res, std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count());
}

int main()
{
	try
	{
		cl::sycl::queue queue{ cl::sycl::gpu_selector() };
		std::cout << "Selected platform: " << queue.get_context().get_platform().get_info<cl::sycl::info::platform::name>() << "\n";
		std::cout << "Selected device:   " << queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";

		std::ofstream file("MCscaling.txt");
		for(size_t n=1; n<16; ++n)
		{
			auto res = MCI(queue, 256*n, (size_t)1 << (size_t)30, [](auto x, auto y){ return 1.0; }, [](auto x, auto y){ return sq(x)+sq(y) <= 1.0; }, -1.0, 1.0, -1.0, 1.0);
			//printf("GPU result = %16.16f\n", res.first);
			//printf("ref result = %16.16f\n", pi);
			printf("result ratio = %16.16f\n", res.first / pi);
			printf("[%zi] time = %f ms\n", n, res.second/1000./1000.);
			file << n << "   " << res.second/1000./1000. << "   " << abs(1.0 - res.first / pi) << "\n";
		}
	}
	catch (cl::sycl::exception e){ std::cout << "Exception encountered in SYCL: " << e.what() << "\n"; return -1; }

	return 0;
}
