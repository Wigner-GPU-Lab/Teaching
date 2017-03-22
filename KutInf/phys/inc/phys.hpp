// Standard C++ includes
#include <functional>
#include <iostream>


struct discrete_function
{
	std::function<double(double)> f;
	double from, to, delta;
};

double sum(const discrete_function&);
double avarage(const discrete_function&);
std::function<double(double)> derivate(const discrete_function&);

std::ostream& operator<<(std::ostream&, const discrete_function&);
