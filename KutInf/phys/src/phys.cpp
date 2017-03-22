#include <phys.hpp>

double sum(const discrete_function& func)
{
	double result = 0.0;

	for (double x = func.from ; x < func.to ; x += func.delta)
		result += func.f(x);

	return result;
}


double avarage(const discrete_function& func)
{
	return sum(func) / (func.to - func.from);
}


std::function<double(double)> derivate(const discrete_function& func)
{
	return [=](double x){ return (func.f(x + func.delta) - func.f(x))  / func.delta; };
}


std::ostream& operator<<(std::ostream& lhs, const discrete_function& rhs)
{
	for (double x = rhs.from ; x < rhs.to ; x += rhs.delta)
		lhs << x << '\t' << rhs.f(x) << std::endl;

	return lhs;
}
