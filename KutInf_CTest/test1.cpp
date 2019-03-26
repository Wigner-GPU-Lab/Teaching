#include "Integrate.h"

int main()
{
    //Known result:
    //https://www.wolframalpha.com/input/?i=Integrate%5BExp%5B-x*x%5D*Cos%5Bx%5D,+%7Bx,+0,+1%7D%5D
    const double Iref = 0.65617436273150682981668121630094195846148449707466;
    
    const double I = integrate([](double x){ return std::exp(-x*x)*std::cos(x); }, 1000, 0.0, 1.0);

    if(std::abs(I-Iref) < 1e-7)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}
