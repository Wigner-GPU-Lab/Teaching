// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <iomanip>
#include <numeric>
#include <vector>

//Maybe functor:
struct Just{};
struct Nothing{};
template<typename T> struct Maybe
{
	T value;
	bool nothing;

	Maybe(Nothing            ):value{},         nothing{true}{}
	Maybe(Just,    T value_in):value{value_in}, nothing{false}{}
	operator bool() const { return nothing ? false : true; }
};

template<typename F>
auto maybe(bool pred, F f){ using R = decltype(f()); return pred ? Maybe<R>(Just(), f()) : Maybe<R>(Nothing()); }

//Minimal Pair implementation
template<typename L, typename R> struct Pair
{
    L l; R r;
};

template<typename L, typename R>
auto makePair(L const& l, R const& r){ return Pair<L, R>{ l, r }; }

template<typename UF, typename FF, typename S, typename Z>
auto hylo( UF uf, FF ff, S seed, Z const& zero )
{
    auto maybe_val_and_seed = uf( seed );
    Z acc = zero;
    while( maybe_val_and_seed )
    {
        acc = ff( acc, maybe_val_and_seed.value.l );
        maybe_val_and_seed = uf( maybe_val_and_seed.value.r );
    }
    return acc;
}

template<typename F, typename dF, typename S, typename T>
auto NewtonIterator( F f, dF df, S const& start, T const& tolerance, int nmaxit )
{
    return hylo(    [=](Pair<T, int> const& xn_n)
                    {
                        auto xnn = xn_n.l - f(xn_n.l)/df(xn_n.l);
						int n = xn_n.r + 1;
						return maybe(cl::sycl::fabs(xnn-xn_n.l)*2 > tolerance && n < nmaxit, [=]{ return makePair(makePair(xnn, n), makePair(xnn, n)); });
                    },
                    [](auto xn, auto xnn){ return xnn; },
                    start, makePair(0.0, 0) );
}

template<typename F, typename DF, typename T>
auto roots(F f, DF df, T x00, T x10, size_t n, T tol)
{
	auto x0 = std::min(x00, x10);
	auto x1 = std::max(x00, x10);
	std::vector<T> res0( n );
	auto dx = (x1 - x0) / (n-1.0);
	{
		cl::sycl::queue queue{ cl::sycl::gpu_selector() };

		cl::sycl::buffer<T, 1> b_dst(res0.data(), res0.size()); 

		cl::sycl::range<1> r(n);

		queue.submit([&](cl::sycl::handler& cgh)
		{
			auto dst  = b_dst.template get_access<cl::sycl::access::mode::write>(cgh);

			cgh.parallel_for<class Newton>(r, [=](cl::sycl::item<1> id)
			{
				auto i = id.get(0);
				auto xlo = x0 + i * dx;
				auto xhi = xlo + dx;

				/*do
				{
					auto x2 = x = x - f(x) / df(x);
					if( cl::sycl::fabs(x-x2)/2 < tol ){ x = x2; break; }
					++n;
				}while(n<20000);*/

				auto res = NewtonIterator(f, df, makePair((xhi + xlo)/2.0, 0), tol, 20000);

				double x = res.l;
				int n = res.r;
				if(n >= 20000)
				{
					dst[i] = xhi*2;
				}
				else
				{
					dst[i] = x;
				}
			});
		});
	}

	//filter roots:
	std::sort(res0.begin(), res0.end());
	res0.erase(std::remove_if(res0.begin(), res0.end(), [=](auto x){ return x > x1 || x < x0; }), res0.end());
	std::vector<T> res;
	if(res0.size() == 0){ return res0; }
	{
		T last = res0[0];
		int last_idx = 0;
		for(int i=1; i<res0.size(); ++i)
		{
			auto diff = res0[i] - last;
			if(diff > tol)
			{
				res.push_back(res0[(i + last_idx)/2]);
				last_idx = i;
				last = res0[i];
			}
		}
		res.push_back(last);
	}
	return res;
}

int main()
{
	std::cout << std::setprecision(16);
	{
		//Calculate square root os 612:
		auto f = [](double x){ return x*x - 612.; };
		auto df = [](double x){ return 2. * x; };

		auto x0 = 24.0;

		//Newton iteration with tracing:
		auto res = hylo([=](auto const& xn_n)
						 {
							auto xnn = xn_n.l - f(xn_n.l)/df(xn_n.l);
							int n = xn_n.r + 1;
							return maybe(cl::sycl::fabs(xnn-xn_n.l)*2 > 1e-14 && n < 5000, [=]{ return makePair(makePair(xnn, n), makePair(xnn, n)); });
						 },
						 [](auto xn, auto xnn){ std::cout << xnn.r << "   " << xn.l << " -> " << xnn.l << "\n"; return xnn; }, makePair(x0, 0), makePair(x0, 0) );
		std::cout << "Result = " << res.l << "\n"
			      << "Exact  = " << sqrt(612.) << "\n";
	}

	//12th Hermite polinomial:
	auto f = [](auto const& x)
	{
		auto sqx = x*x;
		return cl::sycl::exp(-sqx/2) * ((((((4096* sqx - 135168)*sqx + 1520640)*sqx - 7096320)*sqx + 13305600)*sqx - 7983360)*sqx + 665280);
	};

	auto df = [](auto const& x)
	{
		auto sqx = x*x;
		return -x*64*cl::sycl::exp(-sqx/2)*((((((64*sqx - 2880)*sqx + 44880)*sqx - 300960)*sqx + 873180)*sqx - 956340)*sqx + 259875);
	};

	auto res = roots(f, df, -5.0, 5.0, 2048, 1e-9);

	//Compare: http://www.wolframalpha.com/input/?i=12th+Hermite+polynomial+*+exp(-x*x%2F2)
	std::cout << "Roots of the 12th Hermite polinomial:\n";
	for(int i=0; i<(int)res.size(); ++i)
	{
		std::cout << i << " = " << res[i] << "\n";
	}
	return 0;
}