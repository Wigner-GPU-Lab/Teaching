// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <vector>

template<typename F, typename RA, typename RWA, typename WA>
void reduce(F f, RA src, RWA tmp, WA dst, cl::sycl::nd_item<1> id)
{
	auto g = id.get_group().get(0);
	auto bs = id.get_local_range().get(0);
	auto l = id.get_local().get(0);

	auto i = g * bs * 2 + l;

	tmp[l] = f( src[i], src[i+bs] );

	id.barrier(cl::sycl::access::fence_space::local_space);

	// do reduction in shared mem
	for(auto s=bs/2; s > 0; s >>= 1)
	{
		if(l < s)
		{
			tmp[l] = f(tmp[l], tmp[l + s]);
		}
		id.barrier(cl::sycl::access::fence_space::local_space);
	}

	// write result for this block to global mem
	if(l == 0){ dst[g] = tmp[0]; }
}

int main()
{
	using T = double;

	// Size of vectors
	size_t n = 256;
	//block size
	size_t local_count = 16;

	// Host vectors
	std::vector<T> h_src(n);
	std::vector<T> h_dst(n/local_count);

	// Initialize vectors on host
	for(size_t i = 0; i < n; i++ )
	{
		h_src[i] = i*0.001;
	}

	for(size_t i = 0; i < h_dst.size(); i++ )
	{
		h_dst[i] = 0;
	}

	auto sum = [](auto const& x, auto const& y){ return x + y; };

	{
		cl::sycl::queue queue{ cl::sycl::gpu_selector() };

		cl::sycl::buffer<T, 1> b_src(h_src.data(), n);
		cl::sycl::buffer<T, 1> b_dst(h_dst.data(), n);

		cl::sycl::nd_range<1> r(n/2, local_count);
	
		queue.submit([&](cl::sycl::handler& cgh)
		{
			auto a_src = b_src.get_access<cl::sycl::access::mode::read>(cgh);
			
			cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
				                     cl::sycl::access::target::local> a_tmp(cl::sycl::range<1>(local_count), cgh);

			auto a_dst  = b_dst.get_access<cl::sycl::access::mode::write>(cgh);

			cgh.parallel_for<class Reduce>(r, [=](cl::sycl::nd_item<1> i)
			{
				reduce(sum, a_src, a_tmp, a_dst, i);
			});
		});
	}


	T res = 0.0;
	for(size_t i=0; i<h_dst.size(); i++)
	{
		res = sum(res, h_dst[i]);
	}

	std::cout << "result = " << res << "\n";

	return 0;
}