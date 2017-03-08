#include <utility>
#include <array>
#include <algorithm>
#include <numeric>
#include <cmath>

template<typename T, int n> struct Vector;

template<typename T, typename... Ts>
auto vec(T const& t, Ts const&... ts )
{
  return Vector<T, sizeof...(Ts)+1>{{{t, ts...}}};
}

auto vcons = [](auto... xs){ return vec(xs...); };

auto map = [](auto g, auto f, auto... xs){ return g( f(xs)... ); };

auto zip = [](auto g, auto f, auto... xs)
{
      return [g, f, xs...](auto... ys){ return g( f(xs, ys)... ); };
};

template<int... is, typename F, typename T, int n>
auto imap(std::integer_sequence<int, is...> const&, F f, Vector<T, n>const& v)
{
  return map( vcons, f, v[is]...);
}

template<int... is, typename F, typename U, typename V, int n>
auto izip(std::integer_sequence<int, is...> const&, F f, Vector<U, n>const& u, Vector<V, n>const& v)
{
  return zip( vcons, f, u[is]...)( v[is]...);
}

template<typename F, typename Z>
auto foldl(F f, Z z){ return z; }

template<typename F, typename Z, typename T, typename... Ts>
auto foldl(F f, Z z, T const& t, Ts const&... ts){ return foldl(f, f(z, t), ts...); }

template<int... is, typename F, typename Z, typename T, int n>
auto ifoldl(std::integer_sequence<int, is...> const&, F f, Z z, Vector<T, n>const& v)
{
  return foldl(f, z, v[is]...); 
}

template<typename T, int n>
struct Vector
{
    using indices = std::make_integer_sequence<int, n>;
    std::array<T, n> elems;
  
    T&       operator[](int i)     { return elems[i]; }
    T const& operator[](int i)const{ return elems[i]; }

    auto& operator+=( Vector<T, n> const& v )
    {
        return (*this = *this + v);
    }
  
    auto& operator-=( Vector<T, n> const& v )
    {
        return  (*this = *this - v);
    }
  
    auto& operator*=( T const& scl )
    {
        return (*this = *this * scl);
    }
  
    auto& operator/=( T const& scl )
    {
        return (*this = *this / scl);
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

template<typename U, typename V, int n>
auto operator+(Vector<U, n>const& u, Vector<V, n>const& v)
{
  return izip(typename Vector<U, n>::indices(), [](U const& x, V const& y){ return x+y; }, u, v);
}

template<typename U, typename V, int n>
auto operator-(Vector<U, n>const& u, Vector<V, n>const& v)
{
  return izip(typename Vector<U, n>::indices(), [](U const& x, V const& y){ return x-y; }, u, v);
}

template<typename X, typename T, int n>
auto operator*(Vector<X, n>const& v, T const& c)
{
  return imap(typename Vector<X, n>::indices(), [&](X const& x){ return x*c; }, v);
}

template<typename T, typename X, int n>
auto operator*(T const& c, Vector<X, n>const& v )
{
  return imap(typename Vector<X, n>::indices(), [&](X const& x){ return c*x; }, v);
}

template<typename X, typename T, int n>
auto operator/(Vector<X, n>const& v, T const& c)
{
  return imap(typename Vector<X, n>::indices(), [&](X const& x){ return x/c; }, v);
}

template<typename V, typename U, int n>
auto dot( Vector<V, n>const& v, Vector<U, n>const& u )
{
  using R = decltype(v[0]*u[0]);
  return ifoldl(typename Vector<U, n>::indices(), [](R const& x, R const& y){ return x+y; }, (R)0, izip(typename Vector<U, n>::indices(), [](U const& x, V const& y){ return x*y; }, u, v));
}

template<typename T>
auto cross( Vector<T, 3>const& v, Vector<T, 3>const& u )
{
  return Vector<T, 3>{{{v[1]*u[2]-v[2]*u[1],
                        v[2]*u[0]-v[0]*u[2],
                        v[0]*u[1]-v[1]*u[0]}}};
}



int main()
{
  auto v = vec(2, 3);
  auto u = vec(3, 5);
  auto x = zip(vcons, std::plus<int>(), 2, 3)(3, 5);
  return dot(u, v);
}