#include <vector>   //for std::vector
#include <chrono>   //for std::chrono::high_resolution_clock, std::chrono::duration_cast
#include <algorithm>//for std::for_each
#include <iostream> //for std::cout

int main()
{
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<double> data(100'000'000);
    
    auto t1 = std::chrono::high_resolution_clock::now();
    
    std::for_each(data.begin(), data.end(), [](double& elem){ elem = 0.0; });

    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Allocation took: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << " milliseconds." << std::endl;
    std::cout << "Zero out   took: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() << " milliseconds." << std::endl;
}