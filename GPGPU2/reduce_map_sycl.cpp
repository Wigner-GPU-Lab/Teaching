// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <numeric>
#include <vector>

template<typename Fr, typename Fm, typename T, typename R = std::result_of_t<Fm(T)>>
R reduce_map(Fr fr, Fm fm, std::vector<T> const& inp)
{
	//block size, assumes input > local_count
	size_t local_count = 16;

	size_t n = inp.size();
	std::vector<R> res( (size_t)std::ceil(n/2.0/local_count) );

	{
		cl::sycl::queue queue{ cl::sycl::gpu_selector() };

		cl::sycl::buffer<T, 1> b_src(inp.data(), n);
		cl::sycl::buffer<R, 1> b_dst(res.data(), res.size()); 

		cl::sycl::nd_range<1> r(n/2, local_count);

		queue.submit([&](cl::sycl::handler& cgh)
		{
			auto src = b_src.template get_access<cl::sycl::access::mode::read>(cgh);

			cl::sycl::accessor<R, 1, cl::sycl::access::mode::read_write,
				cl::sycl::access::target::local> tmp(cl::sycl::range<1>(local_count), cgh);

			auto dst  = b_dst.template get_access<cl::sycl::access::mode::write>(cgh);

			cgh.parallel_for<class ReduceMap>(r, [=](cl::sycl::nd_item<1> id)
			{
				auto g = id.get_group().get(0);
				auto bs = id.get_local_range().get(0);
				auto l = id.get_local().get(0);

				auto i = g * bs * 2 + l;

				tmp[l] = fr( fm(src[i]), fm(src[i+bs]) );

				id.barrier(cl::sycl::access::fence_space::local_space);

				// do reduction in shared mem
				for(auto s=bs/2; s > 0; s >>= 1)
				{
					if(l < s)
					{
						tmp[l] = fr(tmp[l], tmp[l + s]);
					}
					id.barrier(cl::sycl::access::fence_space::local_space);
				}

				// write result for this block to global mem
				if(l == 0){ dst[g] = tmp[0]; }
			});
		});
	}

	if(res.size() == 1){ return res[0]; }
	else{ return std::accumulate( res.cbegin()+1, res.cend(), res[0], fr ); }

}

int main()
{
	using T = int;

	// Size of vector
	size_t n = 1024;
	
	// Host vector
	std::vector<T> v(n);

	// Initialize vectors on host
	for(int i = 0; i < n; i++ )
	{
		v[i] = i+1;
	}

	auto sum = [](auto const& x, auto const& y){ return x + y; };
	auto rec_sq = [](auto const& x){ return 1.0 / (1.0 * x*x); };
	
	auto res = reduce_map(sum, rec_sq, v);

	//std::cout << "result = " << res << "\n";
	printf("result = %16.16f\n", res);
	return 0;
}