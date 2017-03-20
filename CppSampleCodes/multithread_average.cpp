#pragma once
#include <vector>
#include <random>
#include <numeric>
#include <future>
#include <chrono>
#include <iostream>

//Process a large vector with many threads via async:
int main()
{
	std::vector<double> vec(10'000'000);

	std::random_device random_source;
    std::normal_distribution<double> dist(5.0, 10.0); // normal distribution with average 5 and sigma = 10, generating doubles
	std::generate( vec.begin(), vec.end(), [&](){ return dist(random_source); } ); //fill the vector with random numbers

	auto averager = [](auto it0, auto it1)//lambda to average numbers between two iterators
	{
		auto difference = std::distance(it0, it1);
		auto sum = std::accumulate(it0, it1, 0.0, [](double x, double y){ return x+y; });
		return sum / difference;
	};

	int max_num_of_threads = (int)std::thread::hardware_concurrency(); //query number of threads
    std::cout << "Using " << max_num_of_threads << " threads.\n";
	std::vector<std::future<double>> futures(max_num_of_threads);
	
	auto time0 = std::chrono::high_resolution_clock::now();

	//start threads:
	auto step = 1 + (int)vec.size() / max_num_of_threads;
	for(int n=0; n<max_num_of_threads; ++n )
	{
		auto it0 = std::next(vec.begin(), std::max( n    * step, 0)              );
		auto it1 = std::next(vec.begin(), std::min((n+1) * step, (int)vec.size()));
		futures[n] = std::async( std::launch::async, averager, it0, it1 );
	}

	//wait on the futures and add results when they become available:
	auto parallel_result = std::accumulate(futures.begin(), futures.end(), 0.0, [](double acc, std::future<double>& f){ return acc + f.get(); } );

	auto time1 = std::chrono::high_resolution_clock::now();

    auto serial_result = std::accumulate(vec.begin(), vec.end(), 0.0);
    auto time2 = std::chrono::high_resolution_clock::now();

	std::cout << "Serial average is:   " << serial_result / (double)vec.size() << " \t Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(time2-time1).count() << " usec.\n";
    std::cout << "Parallel average is: " << parallel_result / (double)max_num_of_threads << " \t Elapsed time: " << std::chrono::duration_cast<std::chrono::microseconds>(time1-time0).count() << " usec\n";
}