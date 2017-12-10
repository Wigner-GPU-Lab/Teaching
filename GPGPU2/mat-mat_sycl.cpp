// SYCL include
#include <CL/sycl.hpp>
#include <random>

// Standard C++ includes
#include <iostream>
#include <vector>
#include <chrono>

template<int i> struct Int{};

using Time = decltype(std::chrono::high_resolution_clock::now());
auto ms(Time const& t0, Time const& t1){ return std::chrono::duration_cast<std::chrono::microseconds>( std::max(t1, t0) - std::min(t1, t0) ).count()/1000.0; }

template<typename T> using Ra   = cl::sycl::accessor<T, 2, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>;
template<typename T> using Wa   = cl::sycl::accessor<T, 2, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer>;
template<typename T> using RWla = cl::sycl::accessor<T, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;

//method0: textbook
template<typename T>
void method_0(Ra<T> A, Ra<T> B, Wa<T> C, cl::sycl::item<2> id, size_t n)
{
	auto ix = id.get(0);
	auto iy = id.get(1);

	auto tmp = (T)0;
	for(size_t k=0; k<n; ++k)
	{
		tmp += A[iy][k] * B[k][ix];
	}
	C[iy][ix] = tmp;
}

//method1 shared memory:
template<int ls, typename T>
void method_1(Ra<T> A, Ra<T> B, Wa<T> C, RWla<T> tA, RWla<T> tB, cl::sycl::nd_item<2> id, size_t n)
{
	auto ix = id.get_global().get(0);
	auto iy = id.get_global().get(1);
	auto lx = id.get_local().get(0);
	auto ly = id.get_local().get(1);

	auto sum = (T)0;
	auto blim = (int)n/ls;
	for(int b = 0; b<blim; ++b)
	{
		auto off1 = b*ls + lx;
		auto off2 = b*ls + ly;
		tA[ly][lx] = A[iy][off1];
		tB[lx][ly] = B[off2][ix];
		id.barrier(cl::sycl::access::fence_space::local_space);

		for(int k=0; k<ls; ++k)
		{
			sum += tA[ly][k] * tB[lx][k];
		}

		id.barrier(cl::sycl::access::fence_space::local_space);
	}
	C[iy][ix] = sum;
}

//method2 shared memory + vectorization:
template<typename T> using vRa   = cl::sycl::accessor<cl::sycl::vec<T, 2>, 2, cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>;
template<typename T> using vWa   = cl::sycl::accessor<cl::sycl::vec<T, 2>, 2, cl::sycl::access::mode::write, cl::sycl::access::target::global_buffer>;
template<typename T> using vRWla = cl::sycl::accessor<cl::sycl::vec<T, 2>, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;

template<int ls, int vs, typename T>
void method_2(vRa<T> A, vRa<T> B, vWa<T> C, vRWla<T> tA, vRWla<T> tB, cl::sycl::nd_item<2> id, size_t n)
{
	auto ix = id.get_global().get(0);
	auto iy = id.get_global().get(1);
	auto lx = id.get_local().get(0);
	auto ly = id.get_local().get(1);

	cl::sycl::vec<T, 2> sum1 = {(T)0, (T)0};
	cl::sycl::vec<T, 2> sum2 = {(T)0, (T)0};
	auto blim = (int)(n/vs)/ls;
	for(int b = 0; b<blim; ++b)
	{
		auto off1 = b*ls + lx;
		tA[2*ly+0][lx] = A[2*iy+0][off1];
		tA[2*ly+1][lx] = A[2*iy+1][off1];
		tB[2*ly+0][lx] = B[b*2*ls+2*ly+0][ix];
		tB[2*ly+1][lx] = B[b*2*ls+2*ly+1][ix];
		id.barrier(cl::sycl::access::fence_space::local_space);

		for(int k=0; k<ls; ++k)
		{
			auto va1 = tA[2*ly+0][k];
			auto va2 = tA[2*ly+1][k];
			auto vb1 = tB[2*k+0][lx];
			auto vb2 = tB[2*k+1][lx];
			sum1.x() += va1.x() * vb1.x() + va1.y() * vb2.x();
			sum1.y() += va1.x() * vb1.y() + va1.y() * vb2.y();
			sum2.x() += va2.x() * vb1.x() + va2.y() * vb2.x();
			sum2.y() += va2.x() * vb1.y() + va2.y() * vb2.y();
		}

		id.barrier(cl::sycl::access::fence_space::local_space);
	}
	C[2*iy+0][ix] = sum1;
	C[2*iy+1][ix] = sum2;
}

template<typename T>
auto matmatmul_naive(cl::sycl::queue& queue, std::vector<T> const& A, std::vector<T> const& B)
{
	auto n = (int)sqrt(A.size());
	std::vector<T> C(n*n);

	Time t0, t1;
	{
		cl::sycl::range<2> r(n, n);
		cl::sycl::buffer<T, 2> bA(A.data(), r);
		cl::sycl::buffer<T, 2> bB(B.data(), r);
		cl::sycl::buffer<T, 2> bC(C.data(), r);

		t0 = std::chrono::high_resolution_clock::now();
		auto e0 = queue.submit([&](cl::sycl::handler& cgh)
		{
			auto a = bA.template get_access<cl::sycl::access::mode::read>(cgh);
			auto b = bB.template get_access<cl::sycl::access::mode::read>(cgh);
			auto c = bC.template get_access<cl::sycl::access::mode::write>(cgh);

			cgh.parallel_for<class MatVctMulNaive>(r, [=](cl::sycl::item<2> it)
			{
				method_0(a, b, c, it, n);
			});
		});
		queue.wait();
		t1 = std::chrono::high_resolution_clock::now();
	}

	return std::make_pair(C, ms(t0, t1));
}

template<size_t ls, typename T>
auto matmatmul_shared(cl::sycl::queue& queue, std::vector<T> const& A, std::vector<T> const& B)
{
	auto n = (size_t)sqrt(A.size());
	std::vector<T> C(n*n);

	Time t0, t1;
	{
		cl::sycl::range<2> r(n, n);
		cl::sycl::buffer<T, 2> bA(A.data(), r);
		cl::sycl::buffer<T, 2> bB(B.data(), r);
		cl::sycl::buffer<T, 2> bC(C.data(), r);

		cl::sycl::nd_range<2> r2( {n, n}, {ls, ls} );

		t0 = std::chrono::high_resolution_clock::now();
		auto e0 = queue.submit([&](cl::sycl::handler& cgh)
		{
			auto a = bA.template get_access<cl::sycl::access::mode::read>(cgh);
			auto b = bB.template get_access<cl::sycl::access::mode::read>(cgh);
			auto c = bC.template get_access<cl::sycl::access::mode::write>(cgh);

			cl::sycl::accessor<T, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> tA(cl::sycl::range<2>(ls, ls), cgh);
			cl::sycl::accessor<T, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> tB(cl::sycl::range<2>(ls, ls), cgh);

			cgh.parallel_for<class MatVctMulShared>(r2, [=](cl::sycl::nd_item<2> it)
			{
				method_1<ls>(a, b, c, tA, tB, it, n);
			});
		});
		queue.wait();
		t1 = std::chrono::high_resolution_clock::now();
	}

	return std::make_pair(C, ms(t0, t1));
}

template<size_t ls, size_t vs, typename T>
auto matmatmul_vectorized(cl::sycl::queue& queue, std::vector<T> const& A, std::vector<T> const& B)
{
	auto n = (size_t)sqrt(A.size());
	std::vector<T> C(n*n);

	using V = cl::sycl::vec<T, vs>;

	Time t0, t1;
	{
		cl::sycl::range<2> r(n, n/vs);
		cl::sycl::buffer<V, 2> bA((V*)A.data(), r);
		cl::sycl::buffer<V, 2> bB((V*)B.data(), r);
		cl::sycl::buffer<V, 2> bC((V*)C.data(), r);

		cl::sycl::nd_range<2> r2( {n/vs, n/vs}, {ls, ls} );

		t0 = std::chrono::high_resolution_clock::now();
		auto e0 = queue.submit([&](cl::sycl::handler& cgh)
		{
			auto a = bA.template get_access<cl::sycl::access::mode::read>(cgh);
			auto b = bB.template get_access<cl::sycl::access::mode::read>(cgh);
			auto c = bC.template get_access<cl::sycl::access::mode::write>(cgh);

			cl::sycl::accessor<V, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> tA(cl::sycl::range<2>(2*ls, ls), cgh);
			cl::sycl::accessor<V, 2, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> tB(cl::sycl::range<2>(2*ls, ls), cgh);

			cgh.parallel_for<class MatVctMulSharedVct>(r2, [=](cl::sycl::nd_item<2> it)
			{
				method_2<ls, vs>(a, b, c, tA, tB, it, n);
			});
		});
		queue.wait();
		t1 = std::chrono::high_resolution_clock::now();
	}

	return std::make_pair(C, ms(t0, t1));
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
		if( err > 5e-6 )
		{
			std::cout << "mismatch " << i << "  " << err << "\n";
			return false;
		}
	}
	return true;
}

template<typename T>
auto cpu_naive(std::vector<T> const& A, std::vector<T> const& B)
{
	auto n = (int)sqrt(A.size());
	std::vector<T> C(n*n);
	auto t0 = std::chrono::high_resolution_clock::now();
	for(int i=0; i<n; ++i)
	{
		for(int j=0; j<n; ++j)
		{
			T sum = (T)0;
			for(int k=0; k<n; ++k)
			{
				sum += A[i*n+k] * B[k*n+j];
			}
			C[i*n+j] = sum;
		}
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	return std::make_pair(C, ms(t0, t1));;
}

template<int b, typename T>
auto cpu_blocked(std::vector<T> const& A, std::vector<T> const& B)
{
	auto n = (int)sqrt(A.size());
	std::vector<T> C(n*n);
	auto Bs = n / b;
	auto t0 = std::chrono::high_resolution_clock::now();
	for(int bi=0; bi<Bs; ++bi)
	{ //block index 1
		for(int bj=0; bj<Bs; ++bj)
		{ //block index 2
			for(int bk=0; bk<Bs; ++bk)
			{ //block index 3
				auto i0 = bi * b; auto j0 = bj * b; auto k0 = bk * b;
				for(int i=0; i<b; ++i )
				{
					auto ii = i0 + i;
					for(int j=0; j<b; ++j )
					{
						auto jj = j0 + j;
						double sum = 0.0;
						for(int k=0; k<b; ++k )
						{
							sum += A[ii*n+k0+k] * B[(k0+k)*n+jj];
						}
						C[ii*n+jj] += sum;
					}
				}
			}
		}
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	return std::make_pair(C, ms(t0, t1));;
}

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

int main()
{
	cl::sycl::queue queue{ cl::sycl::amd_selector() };

	using T = double;
	using R = std::pair<std::vector<T>, double>;

	// Size of vectors
	size_t n = 1024*4;

	// Host vectors
	std::vector<T> A(n*n);
	std::vector<T> B(n*n);

	long state = std::random_device()();

	// Initialize vectors on host
	for(size_t i = 0; i < n; i++ )
	{
		for(size_t j = 0; j < n; j++ )
		{
			A[i*n+j] = (T)((i+j+1) / (1.*n*n));//(T)uniform_rnd(state);//(i+j+1) / (1.*n*n);
			B[i*n+j] = (T)((i-j+2) / (1.*n*n));//(T)uniform_rnd(state);//(i-j+2) / (1.*n*n);
		}
	}

	auto ref = cpu_blocked<8>(A, B);//cpu_naive(A, B);

	auto summary = [&]( std::string const& title, std::vector<R> const& v )
	{
		std::cout << title << ": ";
		for(auto const& r : v){ std::cout << r.second << " ms (" << (is_same(r.first, ref.first) ? '+' : '-') << ") "; }
		std::cout << "\n";
	};

	//summary("CPU Blocked 4",  {ref, cpu_blocked<4>(A, B)});
	//summary("CPU Blocked 8",  {ref, cpu_blocked<8>(A, B)});
	//summary("CPU Blocked 16", {ref, cpu_blocked<16>(A, B)});
	//summary("CPU Blocked 32", {ref, cpu_blocked<32>(A, B)});
	summary("Naive",  {ref, matmatmul_naive(queue, A, B)});
	summary("Shared", {ref, matmatmul_shared<16>(queue, A, B)});
	summary("Vectorized", {ref, matmatmul_vectorized<16, 2>(queue, A, B)});

	return 0;
}