#include <iostream>
#include "Integrate.h"

int main()
{
    double x0 = 0.0;
    double x1 = 0.0;
    int n = 1;
    std::cout << "Numerical integrator for exp(-x*x)*cos(x) on the [x0, x1] interval with 'n' points.\n";
    std::cout << "Enter x0:  ";
    std::cin >> x0;

    std::cout << "\nEnter x1:  ";
    std::cin >> x1;
    
    std::cout << "\nEnter n:  ";
    std::cin >> n;

    if(n <= 0){ n = 100;            std::cout << "n was <= 0, was set to 100.\n"; }
    if(x1 < x0){ std::swap(x0, x1); std::cout << "x1 was < x0, they were swapped.\n";}

    double I = integrate([](double x){ return std::exp(-x*x)*std::cos(x); }, n, x0, x1);

    std::cout << "\nThe integral is: " << I << "\n";
    return 0;
}
