// SYCL include
#include <CL/sycl.hpp>

#define COMPUTECPP_ABACUS_DETAIL_VECTOR_TWO_ARGS(category, func, x, y) \
  {                                                                    \
    return abacus::detail::category::func<                             \
        typename abacus::convert_abacus_sycl<                          \
            cl::sycl::vec<T, width>>::abacus_type>(                    \
        *(reinterpret_cast<typename abacus::convert_abacus_sycl<       \
              cl::sycl::vec<T, width>>::abacus_type *>(&x)),           \
        *(reinterpret_cast<typename abacus::convert_abacus_sycl<       \
              cl::sycl::vec<T, width>>::abacus_type *>(&y)));          \
  }

namespace cl {
	namespace sycl {

		template <typename T, int width = 2>
			T dot(cl::sycl::vec<T, 2> x, cl::sycl::vec<T, 2> y) {
			COMPUTECPP_ABACUS_DETAIL_VECTOR_TWO_ARGS(geometric, dot, x, y)
		}
	}
}

// Standard C++ includes
#include <iostream>
#include <vector>
#include <chrono>

template<int i> struct Int{};

using Time = decltype(std::chrono::high_resolution_clock::now());
auto ms(Time const& t0, Time const& t1){ return std::chrono::duration_cast<std::chrono::microseconds>( std::max(t1, t0) - std::min(t1, t0) ).count()/1000.0; }

template<typename T> using Ra = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>;
template<typename T> using Wa = cl::sycl::accessor<T, 1, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer>;
template<typename T> using RWla = cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;

//method0: textbook
template<typename T>
void method_0(Ra<T> M, Ra<T> V, Wa<T> U, cl::sycl::item<1> id, size_t n)
{
	auto i = id.get(0);

	auto tmp = (T)0;
	for(size_t k=0; k<n; ++k)
	{
		tmp += M[i*n+k] * V[k];
	}
	U[i] = tmp;
}

//method1:
template<typename T>
void method(Int<0>, Ra<T> M, Ra<T> V, Wa<T> U, RWla<T> t, cl::sycl::nd_item<1> id, size_t n, size_t bs)
{
	auto i = id.get_global().get(0);
	auto b = id.get_group().get(0);
	auto l = id.get_local().get(0);

	auto sum = (T)0;
	auto Blim = (int)n/bs;
	for(int B = 0; B<Blim; ++B)
	{
		t[l] = V[B*bs+l];
		id.barrier(cl::sycl::access::fence_space::local_space);

		for(int k=0; k<bs; ++k)
		{
			sum += M[i*n+B*bs+k] * t[k];
		}
	}
	U[i] = sum;
}

//method2:
template<typename T>
void method(Int<1>, Ra<T> M, Ra<T> V, Wa<T> U, RWla<T> t, cl::sycl::nd_item<1> id, size_t n, size_t bs)
{
	auto i = id.get_global().get(0);
	auto b = id.get_group().get(0);
	auto l = id.get_local().get(0);

	static const size_t N = 8;

	auto tmp0 = (T)0;
	auto tmp1 = (T)0;
	auto tmp2 = (T)0;
	auto tmp3 = (T)0;
	auto tmp4 = (T)0;
	auto tmp5 = (T)0;
	auto tmp6 = (T)0;
	auto tmp7 = (T)0;
	for(size_t k=0; k<n; ++k)
	{
		auto v = V[k];
		tmp0 += M[(N*i+0)*n+k] * v;
		tmp1 += M[(N*i+1)*n+k] * v;
		tmp2 += M[(N*i+2)*n+k] * v;
		tmp3 += M[(N*i+3)*n+k] * v;
		tmp4 += M[(N*i+4)*n+k] * v;
		tmp5 += M[(N*i+5)*n+k] * v;
		tmp6 += M[(N*i+6)*n+k] * v;
		tmp7 += M[(N*i+7)*n+k] * v;
	}
	U[N*i+0] = tmp0;
	U[N*i+1] = tmp1;
	U[N*i+2] = tmp2;
	U[N*i+3] = tmp3;
	U[N*i+4] = tmp4;
	U[N*i+5] = tmp5;
	U[N*i+6] = tmp6;
	U[N*i+7] = tmp7;
}

//method3:
template<typename T>
void method3(Ra<T> M, Ra<T> V, Wa<T> U, RWla<T> t, cl::sycl::nd_item<2> id, size_t n, size_t bs, size_t hs)
{
	auto i = id.get_global().get(0);
	auto b = id.get_group().get(0);
	auto lx = id.get_local().get(0);
	auto ly = id.get_local().get(1);

	auto rbs = bs/hs;

	auto sum = (T)0;
	auto Blim = (int)n/bs;
	for(int B = 0; B<Blim; ++B)
	{
		if(id.get_local().get(1) == 0)
		{
			t[lx] = V[B*bs+lx];
		}
		id.barrier(cl::sycl::access::fence_space::local_space);

		for(int k=0; k<rbs; ++k)
		{
			sum += M[i*n+B*bs+hs*k+ly] * t[hs*k+ly];
		}
		id.barrier(cl::sycl::access::fence_space::local_space);
	}
	//id.barrier(cl::sycl::access::fence_space::local_space);
	t[lx*hs+ly] = sum;
	id.barrier(cl::sycl::access::fence_space::local_space);
	
	if(id.get_local().get(1) == 0)
	{
		sum = (T)0;
		for(int H = 0; H<(int)hs; ++H)
		{
			sum += t[lx*hs+H];
		}
		U[i] = sum;
	}
}

//method4:

template<typename T> using Rav2 = cl::sycl::accessor<cl::sycl::vec<T, 2>, 1, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>;

template<typename T>
void method4(Rav2<T> M, Rav2<T> V, Wa<T> U, RWla<T> t, cl::sycl::nd_item<2> id, size_t n, size_t bs, size_t hs)
{
	using V2 = cl::sycl::vec<T, 2>;

	auto i = id.get_global().get(0);
	auto b = id.get_group().get(0);
	auto lx = id.get_local().get(0);
	auto ly = id.get_local().get(1);

	static const auto rat = 2;//bs/hs;
	static const auto hb = 8;
	static const auto rbs = hb*rat;
	size_t off = 0;
	T sum = (T)0;//, (T)0, (T)0};
	//auto sum2 = (T)0;
	/*auto sum3 = (T)0;
	auto sum4 = (T)0;*/
	auto Blim = (int)n/bs/hb;
	auto baseM0 = i*n/2;
	for(int B = 0; B<Blim; ++B)
	{
		auto idx = lx*hs+ly;
		if(idx < bs*hb/2)
		{
			auto v = V[(B*bs*hb)/2+idx];//V[B*bs*hb*2+lx];
			t[2*idx+0] = v.x();
			t[2*idx+1] = v.y();
		}
		id.barrier(cl::sycl::access::fence_space::local_space);
		auto baseV = (B*bs*hb + ly*rbs)/2;
		auto baseVt = ly*rbs;///2*rat;
		auto pbaseVt = t.get_pointer() + (long long)(ly*rbs);
		auto baseM = baseM0 + baseV;
		for(int k=0; k<hb*rat/2/*rbs/2*/; ++k)
		{
			//auto f1 = ;
			//auto f2 = t[hs*(k*2+1)+ly];
			//V2 f = V2{ t[hs*(k*2+0)+ly], t[hs*(k*2+1)+ly]};
			//auto f = t[hs*k+ly];
			//off = 2*i*n+B*bs+hs*(2*k)+ly;

			//V2 f = V2{ t[baseVt+(2*k+0)], t[baseVt+(2*k+1)] };
			V2 f = *(V2*)pbaseVt;
			//V2 f = V[baseV+k];
			//off = (i*n+base+k;//(2*k+0);

			//sum[0] += cl::sycl::dot(V2{ M[0*n+off+0], M[0*n+off+1] }, f);
			//sum[1] += cl::sycl::dot(V2{ M[1*n+off+0], M[1*n+off+1] }, f);
			//sum += cl::sycl::dot( M[baseM + k], V[baseV+k] );

			sum += cl::sycl::dot( M[baseM + k], f );

			pbaseVt += 2;
		}
		id.barrier(cl::sycl::access::fence_space::local_space);
	}

	//id.barrier(cl::sycl::access::fence_space::local_space);
	t[lx*hs+ly] = sum;
	id.barrier(cl::sycl::access::fence_space::local_space);

	if(ly == 0)
	{
		sum = (T)0;
		for(int H = 0; H<(int)hs; ++H)
		{
			sum += t[lx*hs+H];
		}
		U[i] = sum;
	}
}

template<typename T>
auto matvctmul_naive(cl::sycl::queue& queue, std::vector<T> const& m, std::vector<T> const& v)
{
	auto n = v.size();
	std::vector<T> res(n);

	Time t0, t1;
	{
		cl::sycl::buffer<T, 1> bM(m.data(), n*n);
		cl::sycl::buffer<T, 1> bV(v.data(), n);
		cl::sycl::buffer<T, 1> bU(res.data(), n);

		cl::sycl::range<1> r(n);

		t0 = std::chrono::high_resolution_clock::now();
		auto e0 = queue.submit([&](cl::sycl::handler& cgh)
		{
			auto M = bM.template get_access<cl::sycl::access::mode::read>(cgh);
			auto V = bV.template get_access<cl::sycl::access::mode::read>(cgh);
			auto U = bU.template get_access<cl::sycl::access::mode::write>(cgh);

			cgh.parallel_for<class MatVctMulNaive>(r, [=](cl::sycl::item<1> it)
			{
				method_0(M, V, U, it, n);
			});
		});
		queue.wait();
		t1 = std::chrono::high_resolution_clock::now();
	}

	return std::make_pair(res, ms(t0, t1));
}

template<int v> struct MatVctMul;

template<int version, typename T>
auto matvctmul(cl::sycl::queue& queue, std::vector<T> const& m, std::vector<T> const& v, size_t local_count)
{
	auto n = v.size();
	size_t local_mem_count = local_count;
	size_t range_div = 1;
	if( version == 1 ){ range_div = 8; }
	std::vector<T> res(n);

	Time t0, t1;
	{
		cl::sycl::buffer<T, 1> bM(m.data(), n*n);
		cl::sycl::buffer<T, 1> bV(v.data(), n);
		cl::sycl::buffer<T, 1> bU(res.data(), n);

		cl::sycl::nd_range<1> r(n / range_div, local_count);

		t0 = std::chrono::high_resolution_clock::now();
		auto e0 = queue.submit([&](cl::sycl::handler& cgh)
		{
			auto M = bM.template get_access<cl::sycl::access::mode::read>(cgh);
			auto V = bV.template get_access<cl::sycl::access::mode::read>(cgh);
			auto U = bU.template get_access<cl::sycl::access::mode::write>(cgh);

			cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write,
				cl::sycl::access::target::local> tmp(cl::sycl::range<1>(local_mem_count), cgh);

			cgh.parallel_for<MatVctMul<version>>(r, [=](cl::sycl::nd_item<1> it)
			{
				method(Int<version>(), M, V, U, tmp, it, n, local_count);
			});
		});
		queue.wait();
		t1 = std::chrono::high_resolution_clock::now();
	}

	return std::make_pair(res, ms(t0, t1));
}

template<typename T>
auto matvctmul_adv(cl::sycl::queue& queue, std::vector<T> const& m, std::vector<T> const& v, size_t local_count, size_t hs)
{
	auto n = v.size();
	size_t local_mem_count = local_count * hs;
	
	std::vector<T> res(n);

	Time t0, t1;
	{
		cl::sycl::buffer<T, 1> bM(m.data(), n*n);
		cl::sycl::buffer<T, 1> bV(v.data(), n);
		cl::sycl::buffer<T, 1> bU(res.data(), n);

		cl::sycl::nd_range<2> r( {n, hs}, {local_count, hs} );

		t0 = std::chrono::high_resolution_clock::now();
		auto e0 = queue.submit([&](cl::sycl::handler& cgh)
		{
			auto M = bM.template get_access<cl::sycl::access::mode::read>(cgh);
			auto V = bV.template get_access<cl::sycl::access::mode::read>(cgh);
			auto U = bU.template get_access<cl::sycl::access::mode::write>(cgh);

			cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> tmp(cl::sycl::range<1>(local_mem_count), cgh);

			cgh.parallel_for<class MatVctMulAdv>(r, [=](cl::sycl::nd_item<2> it)
			{
				method3(M, V, U, tmp, it, n, local_count, hs);
			});
		});
		queue.wait();
		t1 = std::chrono::high_resolution_clock::now();
	}

	return std::make_pair(res, ms(t0, t1));
}

template<typename T>
auto matvctmul_adv_v(cl::sycl::queue& queue, std::vector<T> const& m, std::vector<T> const& v, size_t local_count, size_t hs)
{
	auto n = v.size();
	size_t local_mem_count = local_count * hs;

	std::vector<T> res(n);

	Time t0, t1;
	{
		cl::sycl::buffer<cl::sycl::vec<T, 2>, 1> bM( (cl::sycl::vec<T, 2>*) m.data(), n*n/2);
		//cl::sycl::buffer<T, 1> bM( m.data(), n*n);
		cl::sycl::buffer<cl::sycl::vec<T, 2>, 1> bV( (cl::sycl::vec<T, 2>*) v.data(), n/2);
		cl::sycl::buffer<T, 1> bU(res.data(), n);

		cl::sycl::nd_range<2> r( {n, hs}, {local_count, hs} );

		t0 = std::chrono::high_resolution_clock::now();
		auto e0 = queue.submit([&](cl::sycl::handler& cgh)
		{
			auto M = bM.template get_access<cl::sycl::access::mode::read>(cgh);
			auto V = bV.template get_access<cl::sycl::access::mode::read>(cgh);
			auto U = bU.template get_access<cl::sycl::access::mode::write>(cgh);

			cl::sycl::accessor<T, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> tmp(cl::sycl::range<1>(local_mem_count), cgh);

			cgh.parallel_for<class MatVctMulAdvV>(r, [=](cl::sycl::nd_item<2> it)
			{
				method4(M, V, U, tmp, it, n, local_count, hs);
			});
		});
		queue.wait();
		t1 = std::chrono::high_resolution_clock::now();
	}

	return std::make_pair(res, ms(t0, t1));
}

template<typename T>
bool is_same( std::vector<T> const& u, std::vector<T> const& v )
{
	if( u.size() != v.size() ){ return false; }
	size_t i = 0;
	auto err = (T)0;
	for(; i<u.size(); ++i)
	{
		err = abs( (u[i] - v[i]) / (u[i]+v[i]));
		if( err > 5e-4 )
		{
			return false;
		}
	}
	return true;
}

int main()
{
	cl::sycl::queue queue{ cl::sycl::amd_selector() };

	using T = double;
	using R = std::pair<std::vector<T>, double>;

	// Size of vectors
	size_t n = 4096*4;

	// Host vectors
	std::vector<T> M(n*n);
	std::vector<T> V(n);

	// Initialize vectors on host
	for( auto& e : M ){ e = 0.0; }
	for(size_t i = 0; i < n; i++ )
	{
		for(size_t j = 0; j < n; j++ )
		{
			M[i*n+j] = (i+j+1) / (1.*n);
		}
	}

	for(size_t i = 0; i < n; i++ )
	{
		V[i] = 1/(i*i+1.) / (1.*n);
	}

	auto ref = matvctmul_naive(queue, M, V);

	auto summary = [&]( std::string const& title, std::vector<R> const& v )
	{
		std::cout << title << ": ";
		for(auto const& r : v){ std::cout << r.second << " ms (" << (is_same(r.first, ref.first) ? '+' : '-') << ") "; }
		std::cout << "\n";
	};

	summary("Naive", {ref, matvctmul_naive(queue, M, V)});

	summary("Version 1", {matvctmul<0>(queue, M, V, 1),
		                  matvctmul<0>(queue, M, V, 2),
		                  matvctmul<0>(queue, M, V, 4),
		                  matvctmul<0>(queue, M, V, 8),
		                  matvctmul<0>(queue, M, V, 16),
		                  matvctmul<0>(queue, M, V, 32),
		                  matvctmul<0>(queue, M, V, 64)	});
	
	
	/*summary("Version 2", {matvctmul<1>(queue, M, V, 1), matvctmul<1>(queue, M, V, 1)});
	
	summary("Version 3 (2)", {matvctmul_adv(queue, M, V, 2,  2),
		                      matvctmul_adv(queue, M, V, 4,  2),
		                      matvctmul_adv(queue, M, V, 8,  2),
		                      matvctmul_adv(queue, M, V, 16, 2),
		                      matvctmul_adv(queue, M, V, 32, 2)});

	summary("Version 3 (4)", {matvctmul_adv(queue, M, V, 4,  4),
		                      matvctmul_adv(queue, M, V, 8,  4),
		                      matvctmul_adv(queue, M, V, 16, 4),
		                      matvctmul_adv(queue, M, V, 32, 4)});*/

	summary("Version 3 (8)", {matvctmul_adv(queue, M, V, 8,  8),
		                      matvctmul_adv(queue, M, V, 16, 8),
		                      matvctmul_adv(queue, M, V, 32, 8)});

	/*summary("Version 3 (16)", {matvctmul_adv(queue, M, V, 16, 16),
		                       matvctmul_adv(queue, M, V, 16, 16)});

	summary("Version 4", {matvctmul_adv_v(queue, M, V, 8,  4),
		                  matvctmul_adv_v(queue, M, V, 16, 8)});*/


	//for(size_t i=0; i<U.size(); i++)
	{
		//std::cout << "result[" << i << "] = " << U[i] << "\n";
	}

	return 0;
}