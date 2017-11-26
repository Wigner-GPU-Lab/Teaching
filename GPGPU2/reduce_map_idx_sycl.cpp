// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <numeric>
#include <vector>

static const double pi = 3.141592653589793238462643383279502884197169399375105820974;

template<typename T>
auto sq(T const& x){ return x*x; }

template<typename Fr, typename Fm, typename R = std::result_of_t<Fm(size_t)>>
R reduce_map_idx(Fr fr, Fm fm, size_t n)
{
	//block size, assumes input > n
	size_t local_count = 16;

	std::vector<R> res( (size_t)std::ceil(n/2.0/local_count) );

	{
		cl::sycl::queue queue{ cl::sycl::gpu_selector() };

		cl::sycl::buffer<R, 1> b_dst(res.data(), res.size()); 

		cl::sycl::nd_range<1> r(n/2, local_count);

		queue.submit([&](cl::sycl::handler& cgh)
		{
			cl::sycl::accessor<R, 1, cl::sycl::access::mode::read_write,
				cl::sycl::access::target::local> tmp(cl::sycl::range<1>(local_count), cgh);

			auto dst  = b_dst.template get_access<cl::sycl::access::mode::write>(cgh);

			cgh.parallel_for<class ReduceMapIdx>(r, [=](cl::sycl::nd_item<1> id)
			{
				auto g = id.get_group().get(0);
				auto bs = id.get_local_range().get(0);
				auto l = id.get_local().get(0);

				auto i = 2 * (g * bs + l);

				tmp[l] = fr( fm(i), fm(i+1) );

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

template<typename F>
auto integrate(F f, double L, size_t n)
{
	auto f0 = [=](auto const& i0)
	{
		auto i = i0+1;
		//w_i * f( x_i )
		auto recnp1 = 1.0 / (n+1);
		auto w = L*pi * recnp1 / sq( cl::sycl::sin(i*pi * recnp1) );
		auto x = L / cl::sycl::tan( pi * i  * recnp1);
		return w * f(x);
	};

	auto sum = [](auto const& x, auto const& y){ return x + y; };

	return reduce_map_idx(sum, f0, n);
}

int main()
{
	auto f = [](auto const& x){ return cl::sycl::exp(-x*x); };

	auto res = integrate(f, 1.5, 1024);

	//std::cout << "result = " << res << "\n";
	printf("result    = %16.16f\n", res);
	printf("reference = %16.16f\n", sqrt(pi));
	return 0;
}