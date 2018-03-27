#include <iostream>
#include <vector>
#include <fstream>

struct LVState
{
    double rabbits, wolves;
};

__host__ __device__ int size( LVState const& ){ return 2; }

template<typename F> __device__ LVState map(F f, LVState const& s){ return LVState{ f(s.rabbits), f(s.wolves) }; }
template<typename F> __device__ LVState zip(F f, LVState const& s, LVState const& s2){ return LVState{ f(s.rabbits, s2.rabbits), f(s.wolves, s2.wolves) }; }
template<typename F> __device__ auto reducel(F f, LVState const& s){ return f(s.rabbits, s.wolves); }

struct RungeKutta4Stepper
{
    double atol, rtol;

    template<typename TTime, typename TStep, typename TState, typename TRHS> __device__ 
    TState operator()( TTime time, TStep h, TState state, TRHS rhs, double* err) const
    {
        const auto scl_mul  = []__device__ (auto scl){ return [=]__device__ (auto x){ return scl*x; }; };
        const auto add      = []__device__ (auto const& x, auto const& y) { return x + y; };
        const auto rel_diff = [&]__device__ (auto const& x, auto const& y)
        {
           auto scale = atol + rtol * max(x, y);
           return (x-y)*(x-y)/scale/scale;
        };

        TState k1 = rhs(time, state);
        TState k2 = rhs(time + 0.5 * h, zip(add, state, map( scl_mul(h*0.5), k1 )));
        TState k3 = rhs(time + 0.5 * h, zip(add, state, map( scl_mul(h*0.5), k2 )));
	TState k4 = rhs(time + h,       zip(add, state, map( scl_mul(h),     k3 )));
	
        TState sum_state = zip(add, zip(add, k1, k4), map( scl_mul(2.0), zip(add, k2, k3)));
	TState res1 = zip(add, state, map( scl_mul(h/6.0), sum_state ) );//RK4 step
        TState res0 = zip(add, state, map( scl_mul(h), k1 ) );//Euler step

        *err = sqrt(reducel(add, zip(rel_diff, res1, res0)) / size(state));
        return res1;
    }
};

template<typename Stepper, typename T, typename H, typename RHS, typename S>
__global__ void step_impl(Stepper stepper, T t, int max_steps, H h0, RHS rhs, S* src, S* path, T* path0)
{
    auto i = blockIdx.x*blockDim.x+threadIdx.x;
    S s[2];
    int idx = 0;
    s[idx] = src[i];
    T time = t;
    H h = h0;

    double err = 1.0, lerr = 1e-4;

    int step = 0;
    while(step < max_steps)
    {
        do{
	    s[1-idx] = stepper(time, h, s[idx], rhs, &err);
            //if(i == 128 && err > 0.9){ printf("R %e %e %e\n", time, h, err); }
            
	    h = 0.95 * h * pow(err, -(1.0/4.0 - 0.75*0.4/4.0)) * pow(lerr, 0.4/4.0);         
            lerr = err;
        }while(err > 0.9);

        //if(i == 128){ printf("A %e %e %e\n", time, h, err); }	
        __syncthreads();
	path[step * blockDim.x * gridDim.x + i] = s[1-idx];
	path0[step * blockDim.x * gridDim.x + i] = time;
	step += 1;
        idx = 1 - idx;
        time = time + h;
    }
}

template<typename Stepper, typename T, typename H, typename S, typename RHS>
std::pair<std::vector<T>, std::vector<S>> step(Stepper stepper, T t, int max_steps, H h, std::vector<S>const& src, RHS rhs)
{
    size_t n = src.size();
    static const size_t blockSize = 256;
           const size_t gridSize  = (size_t)ceil((float)n/blockSize);
    std::vector<S> resS(n*max_steps);
    std::vector<T> resT(n*max_steps);

    S* d_src;
    S* d_resS;
    T* d_resT;

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_src,  n*sizeof(S));
    cudaMalloc(&d_resS, n*sizeof(S)*max_steps);
    cudaMalloc(&d_resT, n*sizeof(T)*max_steps);

    // Copy host vectors to device
    cudaMemcpy( d_src, src.data(), n*sizeof(S), cudaMemcpyHostToDevice);

    //Measure time:
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    step_impl<<<gridSize, blockSize>>>(stepper, t, max_steps, h, rhs, d_src, d_resS, d_resT);
    cudaEventRecord(stop);    
    
    cudaEventSynchronize(stop);    
    float cuda_time = 0.0f;//msec
    cudaEventElapsedTime(&cuda_time, start, stop);
    std::cout << "Elapsed time is: " << cuda_time << " msec\n";
    
    cudaMemcpy( resS.data(), d_resS, n*sizeof(S)*max_steps, cudaMemcpyDeviceToHost );
    cudaMemcpy( resT.data(), d_resT, n*sizeof(T)*max_steps, cudaMemcpyDeviceToHost );

    cudaFree(d_src);
    cudaFree(d_resS);
    cudaFree(d_resT);
    
    return std::make_pair(resT, resS);	
}

int main()
{
	using State = LVState;

	//Lotka-Volterra
	double a = 0.5, b = 0.1, c = 0.8, d = 0.3;

	RungeKutta4Stepper rk4{1e-3, 1e-3};

	// Size of vectors
	size_t n = 1024;

	// State vectors
	std::vector<State> initial_state(n);

	// Initialize vectors on host
	for(int i = 0; i < n; i++ )
	{
            initial_state[i].rabbits = i*0.05;
            initial_state[i].wolves  = i*0.01;
	}

	auto rhs = [=]__device__ (double t, State const& s)
        {
             return LVState{ a*s.rabbits - b*s.rabbits*s.wolves, d*s.rabbits*s.wolves - c*s.wolves };
        };

	auto res = step(rk4, 0.0, 2048*4, 1e-1, initial_state, rhs);

        {
          auto N = res.first.size() / n;
	  std::ofstream file("lv.txt");
	  for(decltype(N) i=0; i<N; i++)
 	  {
            file << res.first[i*n+128] << "   " << res.second[i*n+128].rabbits << "   " << res.second[i*n+128].wolves << "\n";
          }
		//std::cout << "result[" << i << "] = " << res[i].rabbits << ", " << res[i].wolves << "\n";
	}

	return 0;
}
