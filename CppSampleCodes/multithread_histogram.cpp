#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <future>

template <typename T>
struct atomic_wrapper
{
  std::atomic<T> data;

  atomic_wrapper():data(){}
  atomic_wrapper(std::atomic<T> const& a   ):data(a.load()){}
  atomic_wrapper(atomic_wrapper const& copy):data(copy.data.load()){}
  atomic_wrapper& operator=(atomic_wrapper const& copy){ data.store(copy.data.load()); return *this; }
};

int main()
{
	std::vector<double> vec(10'000'000);

	std::random_device random_source;
    std::normal_distribution<double> dist(5.0, 0.8);
	std::generate( vec.begin(), vec.end(), [&](){ return dist(random_source); } );

    static const int histo_size = 30;
	std::vector<atomic_wrapper<int>> atomic_histogram(histo_size);
	std::vector<int>                 non_atomic_histogram(histo_size);

	double low  = 0.0;
	double high = 10.0;

	auto atomic_averager = [&](auto it0, auto it1)
	{
		for( auto it=it0; it!=it1; ++it)
		{
			auto index = size_t((histo_size-1) * (((*it)-low)/(high-low)));
			if( index > 0 && index < histo_size-1 )
			{
				atomic_histogram[index].data += 1;
			}
		}
	};

    auto non_atomic_averager = [&](auto it0, auto it1)
	{
		for( auto it=it0; it!=it1; ++it)
		{
			auto index = size_t((histo_size-1) * (((*it)-low)/(high-low)));
			if( index > 0 && index < histo_size-1 )
			{
				non_atomic_histogram[index] += 1;
			}
		}
	};

	int max_num_of_threads = (int)std::thread::hardware_concurrency();
    std::cout << "Using " << max_num_of_threads << " threads.\n";
	std::vector<std::future<void>> atomic_futures(max_num_of_threads);
    std::vector<std::future<void>> non_atomic_futures(max_num_of_threads);

	auto time0 = std::chrono::high_resolution_clock::now();

	//start atomic threads:
	int step = 1 + (int)vec.size() / max_num_of_threads;
	for(int n=0; n<max_num_of_threads; ++n )
	{
		auto it0 = std::next(vec.begin(), std::max( n    * step, 0)              );
		auto it1 = std::next(vec.begin(), std::min((n+1) * step, (int)vec.size()));
		atomic_futures[n] = std::async( std::launch::async, atomic_averager, it0, it1 );
	}

	//wait on the futures and add results when they become available:
	std::for_each(atomic_futures.begin(), atomic_futures.end(), [](std::future<void>& f){ f.get(); } );

	int atomic_sum = 0;
	for(auto& h : atomic_histogram)
	{
		auto q = h.data.load();
		std::cout << q << "\n";
		atomic_sum += q;
	}

	auto time1 = std::chrono::high_resolution_clock::now();

    //start non-atomic threads:
	for(int n=0; n<max_num_of_threads; ++n )
	{
		auto it0 = std::next(vec.begin(), std::max( n    * step, 0)              );
		auto it1 = std::next(vec.begin(), std::min((n+1) * step, (int)vec.size()));
		non_atomic_futures[n] = std::async( std::launch::async, non_atomic_averager, it0, it1 );
	}

	//wait on the futures and add results when they become available:
	std::for_each(non_atomic_futures.begin(), non_atomic_futures.end(), [](std::future<void>& f){ f.get(); } );

	int non_atomic_sum = 0;
	for(auto& h : non_atomic_histogram)
	{
		std::cout << h << "\n";
		non_atomic_sum += h;
	}

    auto time2 = std::chrono::high_resolution_clock::now();

	std::cout << "Non-atomic sum is: " << non_atomic_sum << " \t ";
	std::cout << "Elapsed time is: " << std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count() << " msec\n";
    std::cout << "Atomic sum is:     " << atomic_sum << " \t ";
	std::cout << "Elapsed time is: " << std::chrono::duration_cast<std::chrono::milliseconds>(time1-time0).count() << " msec\n";

	//Sample output:
	//Non-atomic sum is: 6066376       Elapsed time is: 238 msec
    //Atomic sum is:     10000000      Elapsed time is: 1381 msec
}
