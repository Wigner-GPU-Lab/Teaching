#include <cmath>

template<typename F, typename T>
T integrate(F f, int n, T x0, T x1)
{
  double dx = (x1-x0)/n, sum = 0.0;
  for(int i=1; i<=n-1; i++)
  {
    sum += f(x0+i*dx);
  }
  return dx/2.0*(f(x0) + 2.0*sum + f(x1));
}