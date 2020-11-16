#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE
#include <thread>
#include <execution>
#include <numeric>

struct point
{
    double x, y;
};

int main()
{
    try
    {
        const std::size_t count = std::size_t(std::pow(2u, 24u)); // 4M, cast denotes floating-to-integral conversion,
                                                                  //     promises no data is lost, silences compiler warning
        std::vector<std::size_t> insides(count / std::thread::hardware_concurrency());

        // Create engines that align with serial version
        std::vector<std::default_random_engine> engines(1, std::default_random_engine{});
        std::generate_n(
            std::back_inserter(engines),
            std::thread::hardware_concurrency() - 1,
            [=, engine = std::default_random_engine{}]() mutable
        {
            engine.discard(count / std::thread::hardware_concurrency());
            return engine;
        });

        std::transform(
            std::execution::par_unseq,
            engines.begin(),
            engines.end(),
            insides.begin(),
            [=](std::default_random_engine engine)
        {
            std::vector<point> points;
            points.reserve(count / std::thread::hardware_concurrency());

            auto prpg = [&, distribution = std::uniform_real_distribution<double>{ -1.0, 1.0 }]() mutable
            {
                return point {
                    distribution(engine),
                    distribution(engine)
                };
            };

            std::generate_n(std::back_inserter(points), count / std::thread::hardware_concurrency(), prpg);

            return std::count_if(points.cbegin(), points.cend(), [](const point& p)
            {
                return p.x * p.x + p.y * p.y < 1;
            });
        });

        std::size_t inside = std::accumulate(insides.cbegin(), insides.cend(), size_t{0});

        std::cout.precision(10);

        constexpr double pi = 3.1415926535897932384626433832795028841971;
        std::cout << "pi up to machine precision: " << pi << std::endl;
        std::cout << "pi up to MC precision: " << 4 * static_cast<double>(inside) / (count) << std::endl;
    }
    catch (std::exception& error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
