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

template<typename T> auto sq(T const& x){ return x*x; }
template<typename T> auto coth(T const& x){ return cosh(x) / sinh(x); }

double integrate_ising_energy( double beta, double J )
{
	double k = 1.0 / sq(sinh(2.0*beta*J));
	double sum = 0.0;
	double dt = 0.0001;
	for(double t = 0.0; t<pi/2; t+= dt)
	{
		sum += dt / sqrt(1.0 - 4.0*k*sq(sin(t)/(1.0 + k)));
	}
	return -J*coth(2.0*beta*J) * (1.0 + 2.0 / pi * (2.0*sq(tanh(2.0*beta*J))-1.0) * sum);
}

double uniform_rnd(long& state)
{
	const long A = 48271;    /* multiplier*/
	const long M = 2147483647; /* modulus */
	const long Q = M / A;      /* quotient */
	const long R = M % A;      /* remainder */
	long t = A * (state % Q) - R * (state / Q);
	if (t > 0){ state = t;     }
	else      { state = t + M; }
	return ((double) state / M);
}

template<typename T>
void randomize( std::vector<T>& grid )
{
	long state = std::random_device()();
	for(auto& x : grid)
	{
		x = uniform_rnd(state) < 0.5 ? (T)0 : (T)1;
	}
}

void make_periodic(size_t i, size_t N, size_t& ilo, size_t& ihi)
{
	ilo = i == 0   ? N-1 : i-1;
	ihi = i == N-1 ? 0   : i+1;
}

template<typename T>
void step( std::vector<T>& grid, size_t N, size_t nsteps, double J, double beta )
{
	size_t x0, x1, x2, y0, y1, y2;
	auto s = [&](size_t const& x, size_t const& y){ return grid[y*N+x] == (T)1 ? 1.0 : -1.0; };

	long state = std::random_device()();
	for(size_t n=0; n<nsteps; ++n)
	{
		x1 = (size_t)(uniform_rnd(state)*N);
		y1 = (size_t)(uniform_rnd(state)*N);

		make_periodic(x1, N, x0, x2);
		make_periodic(y1, N, y0, y2);

		auto c0 = grid[y1*N+x1];
		auto cell = c0 == (T)1 ? 1.0 : -1.0;
		auto sum = s(x1, y0) + s(x0, y1) + s(x2, y1) + s(x1, y2);
		auto Ecurrent = -cell * sum;
		auto Eflip    = +cell * sum;

		auto dE = J * (Eflip - Ecurrent);

		if( dE <= 0.0 || exp(-dE*beta) > uniform_rnd(state) )
		{
			grid[y1*N+x1] = (T)1 - c0;
		}
	}
}

template<typename T>
auto observables( std::vector<T>const& grid, size_t N, double J )
{
	double S = 0.0, E = 0.0;
	size_t x0, x1, x2, y0, y1, y2;
	auto s = [&](size_t const& x, size_t const& y){ return grid[y*N+x] == (T)1 ? 1.0 : -1.0; };
	for(size_t n=0; n<N; ++n)
	{
		y1 = n;
		for(size_t m=0; m<N; ++m)
		{
			x1 = m;
			make_periodic(x1, N, x0, x2);
			make_periodic(y1, N, y0, y2);

			auto sum = s(x1, y0) + s(x0, y1) + s(x2, y1) + s(x1, y2);
			auto cell = s(x1, y1);
			S += cell;
			E += -J * cell * sum;
		}
	}
	S /= (N*N);
	E /= (N*N);
	return std::make_pair(S, E/2.0);
}

template<typename T>
struct IsingGPU
{
	using R = double;
	long state;
	size_t gN, gL;
	cl::sycl::queue queue;
	cl::sycl::buffer<T, 2> grid;
	cl::sycl::buffer<long, 1> rng_states;
	IsingGPU( cl::sycl::queue q, size_t N_in, size_t L_in ):queue(q), gN(N_in), gL(L_in), grid(cl::sycl::range<2>{gN, gN}), rng_states(gN*gN/2){ state = std::random_device()(); }

	void randomize()
	{
		queue.submit([&](cl::sycl::handler& cgh)
		{
			{
				auto pr = rng_states.template get_access<cl::sycl::access::mode::write>();
				for(int i=0; i<gN*gN/2; ++i)
				{
					pr[i] = (long)(2147483647 * uniform_rnd(state));
				}
			}

			auto src = grid.template get_access<cl::sycl::access::mode::discard_write>(cgh);
			cl::sycl::range<2> range{ gN, gN };
			cgh.parallel_for<class Randomize>(range, [=](cl::sycl::item<2> id)
			{
				long q0 = (long)(15678348919 ^ (296547*id.get(0) * 45709*id.get(1)));
				long state = (long)(2147483647 * uniform_rnd(q0));
				src[id] = uniform_rnd(state) < 0.5 ? (T)1.0f :(T)-1.0f;
			});
		});
	}

	void step(int nsteps, int even_odd /*0 or 1*/, double J, double beta)
	{
		queue.submit([&, N=gN, L=gL](cl::sycl::handler& cgh)
		{
			auto src =       grid.template get_access<cl::sycl::access::mode::read_write>(cgh);
			auto rng = rng_states.template get_access<cl::sycl::access::mode::read_write>(cgh);
			//cl::sycl::accessor<I, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> loc(cl::sycl::range<1>(L*L), cgh);
			//cl::sycl::nd_range<2> step_range{ cl::sycl::range<2>{N, N/2}, cl::sycl::range<2>{L, L/2} };
			cl::sycl::range<2> step_range{ cl::sycl::range<2>{N, N/2} };

			cgh.parallel_for<class Step>(step_range, [=](cl::sycl::item<2> id)
			{
				size_t x   = id.get(0);
				size_t y0  = id.get(1);
				size_t xlo, xhi, ylo, yhi;

				long pstate = rng[y0*N+x];

				for(int eo0=0; eo0<1; ++eo0)
				{
					auto eo = (even_odd + eo0) % 2;
					//checkerboard on global level
					auto y = y0 * 2 + ((int)(x+eo) % 2);

					make_periodic(x, N, xlo, xhi);
					make_periodic(y, N, ylo, yhi);

					for(int k=0; k<nsteps; ++k)
					{
						//Neighbor sum:
						float sum = src[xlo][y] + src[xhi][y] + src[x][ylo] + src[x][yhi];

						//Get current cell:
						float cell = src[x][y];

						//update cell:
						float Ecurrent = -cell * sum;
						float Eflip    = +cell * sum;

						auto dE = J * (Eflip - Ecurrent);
				
						if( dE <= 0.0 || cl::sycl::exp(-dE*beta) > uniform_rnd(pstate) )
						{
							src[x][y] = -cell;
						}
					}
				}
				
				rng[y0*N+x] = pstate;
			});
			
			/*cgh.parallel_for<class Step>(step_range, [=](cl::sycl::nd_item<2> id)
			{
				size_t x   = id.get_global().get(0);
				size_t y0  = id.get_global().get(1);
				size_t lx  = id.get_local().get(0);
				size_t ly0 = id.get_local().get(1);
				size_t gx  = id.get_group().get(0);
				size_t gy0 = id.get_group().get(1);

				//checkerboard on global level
				auto gy = gy0 * 2 + even_odd + ((int)gx % 2)*((1-2*even_odd));

				//read data to shared:
				auto Gy = gy*L;
				auto Gx = gx*L;
				auto b = L/2;
				loc[ly0    *L+ lx] = src[Gx+lx][Gy+ly0  ];
				loc[(ly0+b)*L+ lx] = src[Gx+lx][Gy+ly0+b];

				long state = (long)(15678348919 ^ ((long)loc[ly0*L+ lx] + 2785632113*id.get_global_linear_id()));
				id.barrier(cl::sycl::access::fence_space::local_space);

				for(int eo=0; eo<2; ++eo)
				{
					//checkerboard on local level:
					auto ly = ly0 * 2 + eo + ((int)lx % 2)*((1-2*eo));
					
					//calculate private indices:
					auto xlo = (int)lx - 1;
					auto xhi = (int)lx + 1;

					auto ylo = (int)ly - 1;
					auto yhi = (int)ly + 1;

					float sum = -4.0f;
					
					//calculate neighbors sum
					//xlo, y same
					if( xlo <  0 )
					{
						if(Gx == 0){ sum += src[ N-1][Gy+ly]; }
						else       { sum += src[Gx-1][Gy+ly]; }
					}
					else           { sum += loc[ly * L + xlo]; }

					//xhi, y same
					if( xhi == L )
					{
						if(Gx + L >= N){ sum += src[0   ][Gy+ly]; }
						else           { sum += src[Gx+L][Gy+ly]; }
					}
					else               { sum += loc[ly * L + xhi];  }

					//x same, y lo
					if( ylo <  0 )
					{
						if(Gy == 0){ sum += src[Gx+lx][ N-1]; }
						else       { sum += src[Gx+lx][Gy-1]; }
					}
					else           { sum += loc[ylo * L + lx]; }
					
					//x same, y hi
					if( yhi == L )
					{
						if(Gy + L >= N){ sum += src[Gx+lx][0]; }
						else           { sum += src[Gx+lx][Gy+L]; }
					}
					else               { sum += loc[yhi * L + lx];  }

					//get current cell:
					auto cell = loc[ly*L+lx]-1.0f;

					//update cells:
					auto Ecurrent = -cell * sum;
					auto Eflip    = +cell * sum;

					auto dE = J * (Eflip - Ecurrent);
					id.barrier(cl::sycl::access::fence_space::local_space);
					if( dE <= 0.0 || cl::sycl::exp(-dE*beta) > uniform_rnd(state) )
					{
						loc[ly*L+lx] = (T)(1.0f - cell);
					}
					//state += (long)sum;
					id.barrier(cl::sycl::access::fence_space::local_space);
				}

				src[gx*L+lx][Gy+ly0  ] = loc[ly0    *L+ lx];
				src[gx*L+lx][Gy+ly0+b] = loc[(ly0+b)*L+ lx];

			});*/
		});
		//queue.wait();
	}

	std::pair<R, R> observables(double J, double beta)
	{
		std::vector<R> Es(gN);
		std::vector<R> Ss(gN);
		{
			cl::sycl::buffer<R, 1> S_tmp(Ss.data(), gN);
			cl::sycl::buffer<R, 1> E_tmp(Es.data(), gN);
			cl::sycl::range<1> step_range{ gN };

			queue.submit([&, N = gN](cl::sycl::handler& cgh)
			{
				auto src  = grid.template get_access<cl::sycl::access::mode::read>(cgh);
				auto resS = S_tmp.template get_access<cl::sycl::access::mode::discard_write>(cgh);
				auto resE = E_tmp.template get_access<cl::sycl::access::mode::discard_write>(cgh);
		
				cgh.parallel_for<class Obs>(step_range, [=](cl::sycl::item<1> id)
				{
					auto x = id.get(0);
					double S = 0.0, E = 0.0;

					size_t xlo = 0, xhi = 0;
					size_t ylo = 0, yhi = 0;
					make_periodic(x, N, xlo, xhi);
					for(size_t y=0; y<N; ++y)
					{
						make_periodic(y, N, ylo, yhi);
						float sum = src[xlo][y] + src[xhi][y] + src[x][ylo] + src[x][yhi];
						auto cell = src[x][y];
						S += cell;
						E += -J * cell * sum;
					}
			
					resS[x] = S;
					resE[x] = E;
				});
			});
			queue.wait();
		}

		return std::make_pair( std::accumulate(Ss.cbegin(), Ss.cend(), 0.0)/(gN*gN),
		                       std::accumulate(Es.cbegin(), Es.cend(), 0.0)/(gN*gN*2) );
	}
};

int main()
{
	double J = 2.0;
	double kTc = J * 2.0 / log(1.0 + sqrt(2.0));
	double kT = 0.95 * kTc;
	double beta = 1.0 / kT;
	double U = integrate_ising_energy(beta, J);
	double M = pow(1.0 - pow(sinh(2*beta*J), -4.0), 1./8.);

	size_t N = 1024;
	
	{
		cl::sycl::queue queue{ cl::sycl::amd_selector() };
		IsingGPU<float> Ising(queue, N, 8);
		Ising.randomize();
		auto res = Ising.observables(J, beta);
		std::cout << res.first << "   " << res.second << "\n";

		auto t0 = std::chrono::high_resolution_clock::now();

		int eo = 0;
		int kmax = 100'00;
		for(int k=0; k<kmax; ++k)
		{
			if(k % 1000 == 0)
			{
				/*std::cout << "----------------------------------------------------------------\n";
				auto pg = Ising.grid.template get_access<cl::sycl::access::mode::read>();
				for(int y=0; y<Ising.gN; ++y)
				{
					for(int x=0; x<Ising.gN; ++x)
					{
						std::cout << ((int)pg[x][y] == 1 ? 'X' : ' ');
					}
					std::cout << "\n";
				}
				std::cout << "----------------------------------------------------------------\n";*/
				//auto o = Ising.observables(J, beta);
				//std::cout << k << " avg spin = " << o.first << " exact: " << M << " avg energy = " << o.second << " exact: " << U << "\n";
			}
			Ising.step(4, 0, J, beta);
			Ising.step(4, 1, J, beta);
			queue.wait();
		}
		auto t1 = std::chrono::high_resolution_clock::now();
		auto gputime = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1.0;
	
		std::cout << "GPU performance: " <<  gputime / (kmax*4*2*2*(0.5*N*N)) << " ns / spinflip\n";
		auto o = Ising.observables(J, beta);
		std::cout << " avg spin = " << o.first << " exact: " << M << " avg energy = " << o.second << " exact: " << U << "\n";
	}
	return 0;
	std::vector<int> grid(N*N);

	auto t0 = std::chrono::high_resolution_clock::now();
	randomize(grid);
	auto kmax = N*N;
	for(size_t k=0; k<kmax; ++k)
	{
		step(grid, N, N*N*N, J, beta);
		auto o = observables(grid, N, J);
		//std::cout << "CPU performance: " << cputime / (N*N*kmax) << " ns / spinflip\n";
		std::cout << "avg spin = " << o.first << " exact: " << M << " avg energy = " << o.second << " exact: " << U << "\n";

		//std::cout << k << " / " << kmax << "\n";
	}
	auto t1 = std::chrono::high_resolution_clock::now();
	auto cputime = std::chrono::duration_cast<std::chrono::nanoseconds>(t1-t0).count() * 1.0;
	auto o = observables(grid, N, J);
	std::cout << "CPU performance: " << cputime / (N*N*kmax) << " ns / spinflip\n";
	std::cout << "avg spin = " << o.first << " exact: " << M << " avg energy = " << o.second << " exact: " << U << "\n";
	
	return 0;
}