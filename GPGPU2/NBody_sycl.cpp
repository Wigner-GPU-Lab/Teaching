// SYCL include
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <numeric>
#include <vector>
#include <random>
#include <chrono>
#include <fstream>
#include "indexible.h"

template<typename T> auto sq  (T const& x){ return x*x; }
template<typename T> auto cube(T const& x){ return x*x*x; }

template<typename L, typename R> struct pair{ L left; R right; };
template<typename L, typename R> auto make_pair(L const& left, R const& right){ return pair<L, R>{left, right}; };

template<typename T> struct unpair;
template<typename A, typename B> struct unpair<pair<A, B>>{ using first = A; using second = B; };
template<typename A>             struct unpair<pair<A, A>>{ using common = A; };

template<typename Fi, typename T, int n, typename P = std::result_of_t<Fi(Vector<T, n>, Vector<T, n>)> >
auto NBodyForceCalc(cl::sycl::queue& queue, std::vector<Vector<T, n>> const& positions, Fi fi)
{
	size_t local_count = 256;

	size_t N = positions.size();

	using R = typename unpair<P>::common;

	std::vector<R> res( N );

	using Time = decltype(std::chrono::high_resolution_clock::now());
	Time t0, t1;
	queue.wait();
	{
		cl::sycl::buffer<Vector<T, n>, 1> b_src(positions.data(), N);
		cl::sycl::buffer<R,            1> b_res(res.data(), N);

		cl::sycl::nd_range<1> r(N, local_count);

		t0 = std::chrono::high_resolution_clock::now();
		queue.submit([&](cl::sycl::handler& cgh)
		{
			auto a_src = b_src.template get_access<cl::sycl::access::mode::read>(cgh);
			auto a_res = b_res.template get_access<cl::sycl::access::mode::write>(cgh);

			cgh.parallel_for<class NBodyKernel>(r, [=](cl::sycl::nd_item<1> id)
			{
				auto i = id.get_global().get(0);

				R sum{(T)0.0, (T)0.0, (T)0.0};
				auto self = a_src[i];
				for(int k=0; k<N; ++k)
				{
					//if(k != i)
					{
						sum = sum + fi(self, a_src[k]).left;
					}
				}
				a_res[i] = sum;
			});
		});
		//auto ev1 = e.get_complete().get();
		//queue.wait();
	}
	t1 = std::chrono::high_resolution_clock::now();

	return std::make_pair(res, std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count());
}

template<typename Fi, typename T, int n, typename P = std::result_of_t<Fi(Vector<T, n>, Vector<T, n>)> >
auto NBodyForceCalc1b(cl::sycl::queue& queue, std::vector<Vector<T, n>> const& positions, Fi fi)
{
	size_t local_count = 256;

	size_t N = positions.size();

	using R = cl::sycl::vec<T, 3>;//typename unpair<P>::common;

	std::vector<T> res0(N*3, (T)0.0);
	//cl::sycl::atomic<float> x;
	using Time = decltype(std::chrono::high_resolution_clock::now());
	Time t0, t1;
	queue.wait();
	{
		cl::sycl::buffer<Vector<T, n>, 1> b_src(positions.data(), N);
		cl::sycl::buffer<T, 1> b_res(res0.data(), 3*N);

		cl::sycl::nd_range<1> r(N, local_count);

		t0 = std::chrono::high_resolution_clock::now();
		queue.submit([&](cl::sycl::handler& cgh)
		{
			auto a_src = b_src.template get_access<cl::sycl::access::mode::read>(cgh);
			auto a_res = b_res.template get_access<cl::sycl::access::mode::atomic>(cgh);

			cgh.parallel_for<class NBodyKernel1b>(r, [=](cl::sycl::nd_item<1> id)
			{
				auto i = id.get_global().get(0);

				T sum[3]{ (T)0.0, (T)0.0, (T)0.0 };
				auto self = a_src[i];
				for (int k = 0; k<i; ++k)
				{
					auto res = fi(self, a_src[k]);
					sum[0] += res.left[0];
					sum[1] += res.left[1];
					sum[2] += res.left[2];
					a_res[k*3+0].fetch_add(res.right[0]);
					a_res[k*3+1].fetch_add(res.right[1]);
					a_res[k*3+2].fetch_add(res.right[2]);
				}
				a_res[3*i+0].store(sum[0]);
				a_res[3*i+1].store(sum[1]);
				a_res[3*i+2].store(sum[2]);
			});
		});
		//queue.wait();
	}
	t1 = std::chrono::high_resolution_clock::now();

	using U = typename unpair<P>::common;
	std::vector<U> res(N);
	for(int i=0; i<N; ++i)
	{
		res[i] = U{ res0[3*i+0], res0[3*i+0], res0[3*i+2] };
	}
	return std::make_pair(res, std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
}

template<typename Fi, typename T, int n, typename P = std::result_of_t<Fi(Vector<T, n>, Vector<T, n>)> >
auto NBodyForceCalc2(cl::sycl::queue& queue, std::vector<Vector<T, n>> const& positions, Fi fi)
{
	static const size_t local_count = 256;
	static const size_t L = local_count;
	size_t N = positions.size();
	auto nb = N / L;
	
	using R = typename unpair<P>::common;

	std::vector<R> res( N );
	//std::vector<R> trans( 2*nb*N );

	using Time = decltype(std::chrono::high_resolution_clock::now());
	Time t0, t1;
	queue.wait();
	{
		cl::sycl::buffer<Vector<T, n>, 1> b_src(positions.data(), N);
		cl::sycl::buffer<R, 1> b_res(res.data(), N);
		cl::sycl::buffer<R, 1> b_tmp(2*nb*N);

		cl::sycl::nd_range<1> r(nb*nb*L, L);

		t0 = std::chrono::high_resolution_clock::now();
		queue.submit([&](cl::sycl::handler& cgh)
		{
			auto a_src = b_src.template get_access<cl::sycl::access::mode::read>(cgh);
			auto a_res = b_tmp.template get_access<cl::sycl::access::mode::discard_write>(cgh);
			//cl::sycl::accessor<Vector<T, n>, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> locL(cl::sycl::range<1>(L), cgh);
			cl::sycl::accessor<Vector<T, n>, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> locR(cl::sycl::range<1>(L), cgh);
			cl::sycl::accessor<R,            1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> sumR(cl::sycl::range<1>(L), cgh);

			cgh.parallel_for<class NBodyKernel2a>(r, [=](cl::sycl::nd_item<1> id)
			{
				auto l = id.get_local().get(0);
				auto g0 = id.get_group().get(0);
				auto gx = g0 / nb;
				auto gy = g0 % nb;

				auto glL = gx * L + l;
				auto glR = gy * L + l;

				if(gx > gy)
				{ 
					a_res[(gy +  0)*N + glL] = R{(T)0.0, (T)0.0, (T)0.0};
					a_res[(gx + nb)*N + glR] = R{(T)0.0, (T)0.0, (T)0.0};
					return;
				}

				//locL[l] = a_src[glL];
				locR[l] = a_src[glR];

				sumR[l] = R{(T)0.0, (T)0.0, (T)0.0};
				R sumL  =  {(T)0.0, (T)0.0, (T)0.0};
				auto self = a_src[glL];//locL[l];
				auto o = l;
				for(int k=0; k<L; ++k)
				{
					id.barrier(cl::sycl::access::fence_space::local_space);
					auto f = fi(self, locR[o]);
					sumL    = sumL    + f.left;
					sumR[o] = sumR[o] + f.right;
					o += 1;
					if(o == L){ o = 0; }
				}
				auto c = (T)1.0;
				if(gy == gx){ c = (T)0.5; }
				id.barrier(cl::sycl::access::fence_space::local_space);
				a_res[(gx + nb)*N + glR] = sumR[l] * c;
				a_res[(gy +  0)*N + glL] = sumL    * c;
			});
		});

		cl::sycl::nd_range<1> r2(N, 256);
		queue.submit([&](cl::sycl::handler& cgh)
		{
			auto a_src = b_tmp.template get_access<cl::sycl::access::mode::read>(cgh);
			auto a_res = b_res.template get_access<cl::sycl::access::mode::discard_write>(cgh);
			
			cgh.parallel_for<class NBodyKernel2b>(r2, [=](cl::sycl::nd_item<1> id)
			{
				auto i = id.get_global().get(0);
				R sum = {(T)0.0, (T)0.0, (T)0.0};
				for(int k=0; k<2*nb; ++k)
				{
					sum = sum + a_src[k*N+i];
				}
				a_res[i] = sum;
			});
		});
		//queue.wait();
	}
	t1 = std::chrono::high_resolution_clock::now();

	return std::make_pair(res, std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count());
}

template<typename Fi, typename T, int n, typename P = std::result_of_t<Fi(Vector<T, n>, Vector<T, n>)> >
auto NBodyForceCalc(std::vector<Vector<T, n>> const& parts, Fi fi)
{
	using R = typename unpair<P>::common;
	size_t N = parts.size();
	std::vector<R> res( N );

	using Time = decltype(std::chrono::high_resolution_clock::now());
	Time t0 = std::chrono::high_resolution_clock::now(), t1;
	for(size_t i=0; i<N; ++i)
	{
		R sum{(T)0.0, (T)0.0, (T)0.0};
		auto self = parts[i];
		for(size_t k=0; k<N; ++k)
		{
			//if(k != i)
			{
				sum = sum + fi(self, parts[k]).left;
			}
		}
		res[i] = sum;
	}
	t1 = std::chrono::high_resolution_clock::now();

	return std::make_pair(res, std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count());
}

template<typename T, int N>
bool compare( std::vector<Vector<T, N>> const& u, std::vector<Vector<T, N>> const& v )
{
	if(u.size() != v.size()){ std::cout << "vector length mismatch in compare\n"; return false; }
	auto n = v.size();
	size_t i;
	T err = (T)0.0;
	for(i=0; i<n; ++i)
	{
		auto d = u[i] - v[i];
		auto scale = zip([](auto x, auto y){ return std::max(x, y); }, u[i], v[i]);
		auto divided = zip([](auto x, auto y){ return sq(x/y); }, d, scale);
		err = sqrt(reducel(add, divided)/N);
		if( err > 1e-6 ){ return false; }
	}
	return true;
}

int main()
{
	using T = double;
	T G = 1.0;

	cl::sycl::queue queue{ cl::sycl::amd_selector() };

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<T> distrP((T)-100.0, (T)100.0);
	std::uniform_real_distribution<T> distrM((T)0.1, (T)100.0);

	auto gen_rand_pos_mass = [&](size_t n)
	{
		std::vector<Vector<T, 4>> res(n);
		for(auto& x : res){ x = Vector<T, 4>{ distrP(gen), distrP(gen), distrP(gen), distrM(gen) }; }
		//for(size_t i=0; i<n; ++i){ res[i] = Vector<double, 4>{0.0*i, 0.0, 0.0, 0.1}; }
		return res;
	};

	auto interaction = [=](Vector<T, 4> const& p1, Vector<T, 4> const& p2)
	{
		auto e = std::numeric_limits<T>::epsilon();
		auto delta = Vector<T, 3>{ p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2] };
		auto f = -G*p1[3]*p2[3] / (e  + cube( cl::sycl::sqrt( lengthSq(delta) ) ));
		return make_pair(delta * (-f), delta * f);
	};

	//std::ofstream file("nbody_d.txt");
	size_t n = 1024;
	//for(size_t n=512; n<=4096*8; n *= 2)
	{
		auto particles1 = gen_rand_pos_mass(n);
		auto particles2 = gen_rand_pos_mass(n);

		auto resGPU1  = NBodyForceCalc(queue, particles1, interaction);
		auto resGPU1b = NBodyForceCalc1b(queue, particles1, interaction);
		auto resGPU2  = NBodyForceCalc2(queue, particles2, interaction);
		//resGPU1 = NBodyForceCalc(queue, particles1, interaction);
		//resGPU2 = NBodyForceCalc2(queue, particles2, interaction);
		auto resCPU1 = NBodyForceCalc(particles1, interaction);
		//auto resCPU2 = NBodyForceCalc(particles2, interaction);

		std::cout << (compare(resCPU1.first, resGPU1.first) ? std::string("Passed 1") : std::string("Failed 1")) << "\n";
		std::cout << (compare(resCPU1.first, resGPU1b.first) ? std::string("Passed 1b") : std::string("Failed 1b")) << "\n";
		std::cout << (compare(resCPU1.first, resGPU2.first) ? std::string("Passed 2") : std::string("Failed 2")) << "\n";
		//std::cout << "CPU1 time: " << resCPU1.second / 1000.0 / 1000.0 << " sec\n";
		std::cout << "GPU1  time: " << resGPU1.second / 1000.0 / 1000.0 << " sec\n";
		std::cout << "GPU1b time: " << resGPU1b.second / 1000.0 / 1000.0 << " sec\n";
		//std::cout << "CPU2 time: " << resCPU2.second / 1000.0 / 1000.0 << " sec\n";
		std::cout << "GPU2  time: " << resGPU2.second / 1000.0 / 1000.0 << " sec\n";

		//file << n << "   " << /*resCPU1.second / 1000.0 / 1000.0 << "   " <<*/ resGPU1.second / 1000.0 / 1000.0 << "   " << resGPU2.second / 1000.0 / 1000.0 << "\n";
	}

	

	return 0;
}