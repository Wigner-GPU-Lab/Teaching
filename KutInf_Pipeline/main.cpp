#include <fstream>
#include <cmath>

int main()
{
    std::ofstream file("sin.dat");
    for(int i=0; i<1000; ++i)
    {
        double x = i*0.1; 
        file << x << "   " << std::sin(x) << "\n"; 
    }
    return 0;
}