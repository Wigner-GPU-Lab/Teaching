#include <header.hpp>

int main()
{
	using signature = double(double);

	discrete_function f{static_cast<signature*>(std::sin), -M_PI, M_PI, M_PI / 100};

	std::ofstream file{ "sin.dat" };

	file << f;

	return 0;
}
