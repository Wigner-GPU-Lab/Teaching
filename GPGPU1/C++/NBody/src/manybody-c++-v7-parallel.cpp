#include <array>
#include <fstream>
#include <vector>
#include <iterator>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <thread>
#include <future>

#define G 6.67384e-11

struct particle
{
	double mass;
	std::array<double, 3> pos;
	std::array<double, 3> v;
	std::array<double, 3> f;
};

std::array<double, 3> calculate_force(const particle& first, const particle& second)
{
    std::array<double, 3> dr
    {
        first.pos.at(0) - second.pos.at(0),
        first.pos.at(1) - second.pos.at(1),
        first.pos.at(2) - second.pos.at(2)
    };

    double dr_sq = std::accumulate(dr.cbegin(), dr.cend(), 0.0, [](const double& lhs, const double& rhs)
    {
        return std::pow(lhs, 2) + std::pow(rhs, 2);
    });
    double dr_sq_sqrt = std::sqrt(dr_sq);
    double f = -G * first.mass * second.mass / dr_sq_sqrt;

    return { f * dr.at(0), f * dr.at(1), f * dr.at(2) };
}

int main(int argc, char** argv)
{
    std::vector<particle> particles;

    // initialize
    {
        std::ifstream input(argv[1], std::ios::in);
        if (!input.is_open()) std::cerr << "Failed to open file " << argv[1] << std::endl;
        std::vector<double> data;

        std::cout << "Reading input file " << argv[1] << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        std::copy(
            std::istream_iterator<double>(input.ignore(std::numeric_limits<std::streamsize>::max(), '\n')),
            std::istream_iterator<double>(),
            std::back_inserter(data)); // Read data word-by-word from after first newline

        for (std::size_t i = 0; i < data.size(); i += 7)
        {
            particles.push_back(particle{
                data.at(i),
                { data.at(i + 1), data.at(i + 2), data.at(i + 3) },
                { data.at(i + 4), data.at(i + 5), data.at(i + 6) },
                { 0.0, 0.0, 0.0 } }
            );
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "I/O took " << std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch() - start.time_since_epoch()).count() << " milliseconds." << std::endl;

    } // initialize
    
    {
        std::cout << "Naive N^2" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        for (auto& part : particles)
            for (auto& running : particles)
            {
                auto force = calculate_force(part, running);

                part.f.at(0) += force.at(0);
                part.f.at(1) += force.at(1);
                part.f.at(2) += force.at(2);
            }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch() - start.time_since_epoch()).count() << " milliseconds." << std::endl;
    }

    {
        std::cout << "Cache optimized N^2" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        const std::size_t chunk_size = 2000;

        for (std::size_t I = 0; I < particles.size() / chunk_size; ++I)
            for (auto& part : particles)
                for (std::size_t i = I * chunk_size; i < (I + 1) * chunk_size; ++i)
                {
                    auto force = calculate_force(part, particles.at(i));

                    part.f.at(0) += force.at(0);
                    part.f.at(1) += force.at(1);
                    part.f.at(2) += force.at(2);
                }

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch() - start.time_since_epoch()).count() << " milliseconds." << std::endl;
    }

    {
        std::cout << "Parallel, naive N^2" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        const std::size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::future<void>> handles;

        for (std::size_t I = 0; I < num_threads; ++I) handles.push_back(std::async(std::launch::async, [&particles, I, num_threads]() {
            for (std::size_t i = I * particles.size() / num_threads; i < (I + 1) * particles.size() / num_threads; ++i)
                for (auto& running : particles)
                {
                    auto force = calculate_force(particles.at(i), running);

                    particles.at(i).f.at(0) += force.at(0);
                    particles.at(i).f.at(1) += force.at(1);
                    particles.at(i).f.at(2) += force.at(2);
                }
        }));
        for (auto& handle : handles)
            handle.wait();

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch() - start.time_since_epoch()).count() << " milliseconds." << std::endl;
    }

    {
        std::cout << "Parallel, cache optimized N^2" << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        const std::size_t chunk_size = 2000;
        const std::size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::future<void>> handles;

        for (std::size_t I = 0; I < num_threads; ++I) handles.push_back(std::async(std::launch::async, [&particles, I, num_threads, chunk_size]() {
            for (std::size_t J = 0; J < particles.size() / chunk_size; ++J)
                for (std::size_t i = I * particles.size() / num_threads; i < (I + 1) * particles.size() / num_threads; ++i)
                    for (std::size_t j = J * chunk_size; j < (J + 1) * chunk_size; ++j)
                    {
                        auto force = calculate_force(particles.at(i), particles.at(j));

                        particles.at(i).f.at(0) += force.at(0);
                        particles.at(i).f.at(1) += force.at(1);
                        particles.at(i).f.at(2) += force.at(2);
                    }
        }));
        for (auto& handle : handles)
            handle.wait();

        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch() - start.time_since_epoch()).count() << " milliseconds." << std::endl;
    }

	return 0;
}
