// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <vector>

template<typename F, typename RA, typename WA>
void zip(F f, RA src1, RA src2, WA dst, cl::sycl::id<1> id)
{
	dst[id] = f(src1[id], src2[id]); 
}

int main()
{
	using T = double;

	// Size of vectors
	size_t n = 16;

	// Host vectors
	std::vector<T> h_src1(n);
	std::vector<T> h_src2(n);
	std::vector<T> h_dst(n);

	// Initialize vectors on host
	for(int i = 0; i < n; i++ )
	{
		h_src1[i] = i*1.0;
		h_src2[i] = (i+1)*1.0;
		h_dst[i] = 0;
	}

	{
		cl::sycl::queue queue{ cl::sycl::gpu_selector() };

		cl::sycl::buffer<T, 1> b_src1(h_src1.data(), n);
		cl::sycl::buffer<T, 1> b_src2(h_src2.data(), n);
		cl::sycl::buffer<T, 1> b_dst (h_dst.data(),  n);

		cl::sycl::range<1> r(n);

		auto sqsum = [](auto const& x, auto const& y){ return x*x + y*y; };

		queue.submit([&](cl::sycl::handler& cgh)
		{
			auto a_src1 = b_src1.get_access<cl::sycl::access::mode::read>(cgh);
			auto a_src2 = b_src2.get_access<cl::sycl::access::mode::read>(cgh);
			auto a_dst  = b_dst.get_access<cl::sycl::access::mode::write>(cgh);

			cgh.parallel_for<class Zip>(r, [=](cl::sycl::id<1> i)
			{
				zip(sqsum, a_src1, a_src2, a_dst, i);
			});
		});
	}

	for(int i=0; i<n; i++)
	{
		std::cout << "result[" << i << "] = " << h_dst[i] << "\n";
	}

	return 0;
}
