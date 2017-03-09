#include <utility>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>

template<typename T, int n> struct Vector;

template<typename T, int n>
auto indices( Vector<T, n>const& v ){ return std::make_integer_sequence<int, n>(); }

template<typename T, typename... Ts>
auto vec(T const& t, Ts const&... ts )
{
  return Vector<T, sizeof...(Ts)+1>{{{t, ts...}}};
}

auto vcons = [](auto&&... xs){ return vec(xs...); };

auto map = [](auto&& g, auto&& f, auto&&... xs){ return g( f(xs)... ); };

auto zip = [](auto&& g, auto&& f, auto&&... xs)
{
      return [g, f, &xs...](auto&&... ys){ return g( f(xs, ys)... ); };
};

template<int... is, typename F, typename V>
auto imap(std::integer_sequence<int, is...> const&, F&& f, V&& v)
{
  return map( vcons, f, v[is]...);
}

template<int... is, typename F, typename U, typename V, int n>
auto izip(std::integer_sequence<int, is...> const&, F&& f, U&& u, Vector<V, n>const& v)
{
  return zip( vcons, f, u[is]...)( v[is]...);
}

template<typename F, typename Z>
auto foldl(F&& f, Z&& z){ return z; }

template<typename F, typename Z, typename T, typename... Ts>
auto foldl(F&& f, Z&& z, T const& t, Ts const&... ts){ return foldl(f, f(z, t), ts...); }

template<int... is, typename F, typename Z, typename T, int n>
auto ifoldl(std::integer_sequence<int, is...> const&, F&& f, Z&& z, Vector<T, n>const& v)
{
  return foldl(f, z, v[is]...); 
}

template<typename T, int n>
struct Vector
{
    std::array<T, n> elems;
  
    T&       operator[](int i)     { return elems[i]; }
    T const& operator[](int i)const{ return elems[i]; }

    auto& operator+=( Vector<T, n> const& v )
    {
        (void)izip(indices(*this), [](T& x, T const& y){ x+=y; return 0; }, *this, v);
        return *this;
    }
  
    auto& operator-=( Vector<T, n> const& v )
    {
        (void)izip(indices(*this), [](T& x, T const& y){ x-=y; return 0; }, *this, v);
        return *this;
    }
  
    auto& operator*=( T const& scl )
    {
      (void)imap(indices(*this), [&](T& x){ x*=scl; return 0; }, *this);
      return *this;
    }
  
    auto& operator/=( T const& scl )
    {
        (void)imap(indices(*this), [&](T& x){ x/=scl; return 0; }, *this);
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
auto operator-(Vector<T, n>const& v){ return ((T)-1)*v; }

template<typename U, typename V, int n>
auto operator+(Vector<U, n>const& u, Vector<V, n>const& v)
{
  return izip(indices(u), [](U const& x, V const& y){ return x+y; }, u, v);
}

template<typename U, typename V, int n>
auto operator-(Vector<U, n>const& u, Vector<V, n>const& v)
{
  return izip(indices(u), [](U const& x, V const& y){ return x-y; }, u, v);
}

template<typename X, typename T, int n>
auto operator*(Vector<X, n>const& v, T const& c)
{
  return imap(indices(v), [&](X const& x){ return x*c; }, v);
}

template<typename T, typename X, int n>
auto operator*(T const& c, Vector<X, n>const& v )
{
  return imap(indices(v), [&](X const& x){ return c*x; }, v);
}

template<typename X, typename T, int n>
auto operator/(Vector<X, n>const& v, T const& c)
{
  return imap(indices(v), [&](X const& x){ return x/c; }, v);
}

template<typename V, typename U, int n>
auto dot( Vector<V, n>const& v, Vector<U, n>const& u )
{
  using R = decltype(v[0]*u[0]);
  return ifoldl(indices(v), [](R const& x, R const& y){ return x+y; }, (R)0, izip(indices(v), [](V const& x, U const& y){ return x*y; }, v, u));
}

template<typename T>
auto cross( Vector<T, 3>const& v, Vector<T, 3>const& u )
{
  return Vector<T, 3>{{{v[1]*u[2]-v[2]*u[1],
                        v[2]*u[0]-v[0]*u[2],
                        v[0]*u[1]-v[1]*u[0]}}};
}

float f(float x, float y, float a, float b)
{
  auto v = vec(x, y);
  auto u = vec(a, b);
  return dot(v*5.0f, u);
}