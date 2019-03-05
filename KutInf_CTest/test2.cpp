#include "Integrate.h"

int main()
{
    //Known result:
    //https://www.wolframalpha.com/input/?i=Integrate%5Bx*x-Exp%5B-x%5D,+%7Bx,+2,+3%7D%5D
    const double Iref = 6.247785118464584584418676;
    
    const double I = integrate([](double x){ return x*x-std::exp(-x); }, 5000, 2.0, 3.0);

    if(std::abs(I-Iref) < 1e-7)
    {
        return 0;
    }
    else
    {
        return -1;
    }
}