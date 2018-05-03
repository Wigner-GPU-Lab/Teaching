#include <chrono>
#include <random>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>

void Example22()
{
    std::random_device rnd_device;
    std::mt19937 mersenne_engine(rnd_device());
    std::uniform_int_distribution<int> dist(-10, 10);

    static const unsigned int N = 1024;
    static const unsigned int b = 16;
    static const unsigned int Bs = N / b;

    auto gen = std::bind(dist, mersenne_engine);

    std::vector<double> M1(N*N), M2(N*N), M3(N*N);
    std::generate(std::begin(M1), std::end(M1), gen);
    std::generate(std::begin(M2), std::end(M2), gen);
    std::generate(std::begin(M3), std::end(M3), [](){ return 0.0; } );

    auto t0 = std::chrono::high_resolution_clock::now();

    for( unsigned int i=0; i<N; ++i )
    {
        for( unsigned int j=0; j<N; ++j )
        {
            double sum = 0.0;
            for( unsigned int k=0; k<N; ++k )
            {
                sum += M1[i*N+k] * M2[k*N+j];
            }
            M3[i*N+j] = sum;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    for( unsigned int bi=0; bi<Bs; ++bi ) //block index 1
    {
        for( unsigned int bj=0; bj<Bs; ++bj ) //block index 2
        {
            for( unsigned int bk=0; bk<Bs; ++bk ) //block index 3
            {
                auto i0 = bi * b;
                auto j0 = bj * b;
                auto k0 = bk * b;
                for( unsigned int i=0; i<b; ++i )
                {
                    auto ii = i0 + i;
                    for( unsigned int j=0; j<b; ++j )
                    {
                        auto jj = j0 + j;
                        double sum = 0.0;
                        for( unsigned int k=0; k<b; ++k )
                        {
                            sum += M1[ii*N+k0+k] * M2[(k0+k)*N+jj];
                        }
                        M3[ii*N+jj] += sum;
                    }
                }
            }
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Naive implementation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count()*0.001 << " milliseconds." << std::endl;
    std::cout << "Block implementation took: " << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()*0.001 << " milliseconds." << std::endl;

}