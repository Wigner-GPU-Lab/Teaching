#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE

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
        std::vector<point> points;
        points.reserve(count);

        // Fill arrays with random values between -1 and 1
        auto prpg = [engine = std::default_random_engine{},
                     distribution = std::uniform_real_distribution<double>{ -1.0, 1.0 }]() mutable
        {
            return point {
                distribution(engine),
                distribution(engine)
            };
        };

        std::generate_n(std::back_inserter(points), count, prpg);

        std::size_t inside = std::count_if(points.cbegin(), points.cend(), [](const point& p)
        {
            return p.x * p.x + p.y * p.y < 1;
        });

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
