#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>

template<typename I1, typename I2, typename F>
void zipassign2(I1 it1begin, I1 it1end, I2 it2begin, F&& f)
{
    auto it1 = it1begin;
    auto it2 = it2begin;
    while(it1!=it1end)
    {
        *it1 = f(*it1, *it2);
        ++it1;
        ++it2;
    }
}

template<typename T, int n>
struct Vector
{
    std::array<T, n> elems;
  
    T&       operator[](int i)     { return elems[i]; }
    T const& operator[](int i)const{ return elems[i]; }

    auto& operator+=( Vector<T, n> const& v )
    {
        zipassign2(elems.begin(), elems.end(), v.elems.begin(), std::plus<T>());
        return *this;
    }
  
    auto& operator-=( Vector<T, n> const& v )
    {
        zipassign2(elems.begin(), elems.end(), v.elems.begin(), std::minus<T>());
        return *this;
    }
  
    auto& operator*=( T const& scl )
    {
        std::for_each(elems.begin(), elems.end(), [&](T& x){ x*=scl; });
        return *this;
    }
  
    auto& operator/=( T const& scl )
    {
        std::for_each(elems.begin(), elems.end(), [&](T& x){ x/=scl; });
        return *this;
    }

    T length() const { return std::sqrt(sqlength()); }
    T sqlength() const { return dot(*this, *this); }
  
    auto begin(){ return elems.begin(); }
    auto cbegin() const { return elems.cbegin(); }
    auto end(){ return elems.end(); }
    auto cend() const { return elems.cend(); }
};

template<typename T, int n>
auto operator-(Vector<T, n>const& v)
{
  return ((T)-1)*v;
}

template<typename T, int n>
auto operator+(Vector<T, n>const& u, Vector<T, n>const& v)
{
  Vector<T, n> res;
  std::transform(u.cbegin(), u.cend(), v.cbegin(), res.begin(), std::plus<T>());
  return res;
}

template<typename T, int n>
auto operator-(Vector<T, n>const& u, Vector<T, n>const& v)
{
  Vector<T, n> res;
  std::transform(u.cbegin(), u.cend(), v.cbegin(), res.begin(), std::minus<T>());
  return res;
}

template<typename T, int n>
auto operator*(Vector<T, n>const& v, T const& c)
{
  Vector<T, n> res;
  std::transform(v.cbegin(), v.cend(), res.begin(), [&](T const& x){ return x*c; });
  return res;
}

template<typename T, int n>
auto operator*(T const& c, Vector<T, n>const& v )
{
  Vector<T, n> res;
  std::transform(v.cbegin(), v.cend(), res.begin(), [&](T const& x){ return c*x; });
  return res;
}

template<typename T, int n>
auto operator/(Vector<T, n>const& v, T const& c)
{
  Vector<T, n> res;
  std::transform(v.cbegin(), v.cend(), res.begin(), [&](T const& x){ return x/c; });
  return res;
}

template<typename T, int n>
auto dot( Vector<T, n>const& v, Vector<T, n>const& u )
{
  return std::inner_product(v.cbegin(), v.cend(), u.cbegin(), (T)0);
}

template<typename T>
auto cross( Vector<T, 3>const& v, Vector<T, 3>const& u )
{
  return Vector<T, 3>{{{v[1]*u[2]-v[2]*u[1],
                        v[2]*u[0]-v[0]*u[2],
                        v[0]*u[1]-v[1]*u[0]}}};
}

template<typename T, typename... Ts>
auto vec(T const& t, Ts const&... ts )
{
  return Vector<T, sizeof...(Ts)+1>{{{t, ts...}}};
}