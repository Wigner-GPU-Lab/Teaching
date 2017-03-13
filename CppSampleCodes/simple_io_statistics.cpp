#include <vector>    //for std::vector
#include <random>    //for std::random_device, std::mt19937, std::normal_distribution
#include <fstream>   //for std::ifstream, std::ofstream
#include <iterator>  //for std::istream_iterator, std::ostream_iterator, std::back_inserter
#include <algorithm> //for std::generate, std::copy
#include <numeric>   //for std::accumulate
#include <iomanip>   //for std::setprecision
#include <iostream>  //for std::cout

int main()
{
    //create normal distributed random values and write to file
    {
        std::vector<double> data(1500);

        std::random_device rnd_device;
        std::mt19937 mersenne_engine(rnd_device());
        std::normal_distribution<double> dist(3.5, 1.37);

        std::generate(data.begin(), data.end(), [&]{ return dist(mersenne_engine); });

        std::ofstream output("simple_io_statistics.txt");
        if( output.is_open() )
        {
            std::copy( data.begin(), data.end(), std::ostream_iterator<double>(output, "\n") );
        }
        else
        {
            std::cout << "Could not create output file.\n";
            return -1;
        }
    }

    //read values and calculate statistics
    {
        std::vector<double> data;
        std::ifstream input("simple_io_statistics.txt");

        if( !input.is_open() )
        {
            std::cout << "Could not open input file.\n";
            return -1;
        }
        else
        {
            std::copy( std::istream_iterator<double>(input), std::istream_iterator<double>(), std::back_inserter(data) );
        }
 
        auto sum = std::accumulate(data.cbegin(), data.cend(), 0.0);
        auto average = sum / data.size();
        
        auto sqdev = std::accumulate(data.cbegin(), data.cend(), 0.0, [&](double acc, double x){ return acc + (x-average)*(x-average); });
        auto stdev = std::sqrt(sqdev / data.size());

        std::cout << "Sample average            = " << std::setprecision(16) << average << "\n";
        std::cout << "Sample standard deviation = " << std::setprecision(16) << stdev   << "\n";
    }
}