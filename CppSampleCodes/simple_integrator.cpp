#include <vector>   //for std::vector
#include <algorithm>//for std::generate
#include <numeric>  //for std::accumulate
#include <chrono>   //for std::chrono::high_resolution_clock, std::chrono::duration_cast
#include <iomanip>  //for std::setprecision
#include <iostream> //for std::cout
#include <cmath>

using count = long long;

template<typename F, typename T>
auto TrapezoidIntegrator1(count n, F const& f, T const& x0, T const& x1)
{
    double sum = 0.0;
	double dx = (x1 - x0) / (double)n;
	for (count i = 0; i<n; ++i)
	{
		sum += f(x0 + dx*(double)i);
	}
	return sum * dx;
};

template<typename F, typename T>
auto TrapezoidIntegrator2(count n, F const& f, T const& x0, T const& x1)
{
    double dx = (x1 - x0) / (double)n;
    count i = 0;
    
    std::vector<T> values(n);
    std::generate(values.begin(), values.end(), [=, &i]{ auto res = f(x0 + dx * (double)i); ++i; return res; });
    return dx * std::accumulate( values.begin(), values.end(), (T)0, [](T const& a, T const& b){ return a+b; } );
};

struct CountingIterator
{
    count i;
    CountingIterator():i{0}{}
    CountingIterator(count const& k):i{k}{}
    CountingIterator(CountingIterator const& cpy):i{cpy.i}{}

    count operator*()const{ return i; }
    auto& operator++(){ i+=1; return *this; }
    auto  operator++(int){ auto t = *(*this); i+=1; return t; }
    count* operator->(){ return &i; }
};
bool operator!=(CountingIterator const& i, CountingIterator const& j){ return i.i != j.i; }

template<typename F, typename T>
auto TrapezoidIntegrator3(count n, F const& f, T const& x0, T const& x1)
{
    double dx = (x1 - x0) / (double)n;
    return dx * std::accumulate( CountingIterator{0}, CountingIterator{n}, (T)0, [&](T const& a, auto const& b){ return a+f(x0 + dx * (double)b); } );
};


int main()
{
    double x0 = 1.;
    double x1 = 4.;

    auto integrand = [](double x) { return cos(x)*cos(x); };
	auto analytic = [](double a, double b) { return (-a + b - cos(a)*sin(a) + cos(b)*sin(b)) / 2.0; };

    double exact = analytic(x0, x1);
    auto t0 = std::chrono::high_resolution_clock::now();
    double approx1 = TrapezoidIntegrator1(500000, integrand, x0, x1 );
    auto t1 = std::chrono::high_resolution_clock::now();
    double approx2 = TrapezoidIntegrator2(500000, integrand, x0, x1 );
    auto t2 = std::chrono::high_resolution_clock::now();
    double approx3 = TrapezoidIntegrator3(500000, integrand, x0, x1 );
    auto t3 = std::chrono::high_resolution_clock::now();

    auto dt1 = std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
    auto dt2 = std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count();
    auto dt3 = std::chrono::duration_cast<std::chrono::microseconds>(t3-t2).count();

    std::cout << std::fixed << std::setprecision(16);
    std::cout << "Integrating cos(x)^2 from x0=" << x0 << " to x1=" << x1 << "." << std::endl;
    std::cout << "The exact value is:              " << exact << std::endl;
    std::cout << "The numerical approximation1 is: " << approx1 << " (" << dt1 << " us)" <<std::endl;
    std::cout << "The numerical approximation2 is: " << approx2 << " (" << dt2 << " us)" <<std::endl;
    std::cout << "The numerical approximation3 is: " << approx3 << " (" << dt3 << " us)" <<std::endl;
    std::cout << sizeof(int) << " " <<sizeof(long long);
}