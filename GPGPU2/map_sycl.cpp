// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <vector>

template<typename F, typename RA, typename WA>
void map(F f, RA src, WA dst, cl::sycl::id<1> id)
{
	dst[id] = f(src[id]); 
}

int main()
{
	using T = double;

	// Size of vectors
	size_t n = 16;

	// Host vectors
	std::vector<T> h_src(n);
	std::vector<T> h_dst(n);

	// Initialize vectors on host
	for(int i = 0; i < n; i++ )
	{
		h_src[i] = i*1.0;
		h_dst[i] = 0;
	}

	try
	{
		cl::sycl::queue queue{cl::sycl::intel_selector()};
		std::cout << "Selected platform: " << queue.get_context().get_platform().get_info<cl::sycl::info::platform::name>() << "\n";
		std::cout << "Selected device:   " << queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";

		cl::sycl::buffer<T, 1> b_src(h_src.data(), n);
		cl::sycl::buffer<T, 1> b_dst(h_dst.data(), n);

		cl::sycl::range<1> r(n);

		auto sq = [](auto const& x){ return x*x; };

		queue.submit([&](cl::sycl::handler& cgh)
		{
			auto a_src = b_src.get_access<cl::sycl::access::mode::read>(cgh);
			auto a_dst = b_dst.get_access<cl::sycl::access::mode::discard_write>(cgh);

			cgh.parallel_for<class Map>(r, [=](cl::sycl::id<1> i)
			{
				map(sq, a_src, a_dst, i);
			});
		});

		queue.wait();
	}
	catch (cl::sycl::exception e){ std::cout << "Exception encountered in SYCL: " << e.what() << "\n"; return -1; }

	for(int i=0; i<n; i++)
	{
		std::cout << "result[" << i << "] = " << h_dst[i] << "\n";
	}

	return 0;
}
