#include <iostream>
#include <type_traits>

auto add = [](auto const& x, auto const& y) { return x + y; };
auto sub = [](auto const& x, auto const& y) { return x - y; };
auto mul = [](auto const& x, auto const& y) { return x * y; };

auto cmul_l = [](auto const& c) { return [=](auto const& x) { return c * x; }; };
auto cmul_r = [](auto const& c) { return [=](auto const& x) { return x * c; }; };
auto cdiv   = [](auto const& c) { return [=](auto const& x) { return x / c; }; };

auto sqsum = [](auto const& acc, auto const& x) { return acc + x*x; };

template<typename T, int n>
struct Vector
{
	T data[n];
	auto& operator[](int k) { return data[k]; }
	auto& operator[](int k)const { return data[k]; }
};

template<typename F, typename A, int n, typename B = std::result_of_t<F(A)>>
auto map(F&& f, Vector<A, n> const& v)
{
	Vector<B, n> res;
	for (int k = 0; k<n; ++k) { res[k] = f(v[k]); }
	return res;
}

template<typename F, typename A, typename B, int n, typename C = std::result_of_t<F(A, B)>>
auto zip(F&& f, Vector<A, n> const& v, Vector<B, n> const& u)
{
	Vector<C, n> res;
	for (int k = 0; k<n; ++k) { res[k] = f(v[k], u[k]); }
	return res;
}

template<typename F, typename Z, typename A, int n>
auto fold(F&& f, Z&& z, Vector<A, n> const& v)
{
	Z res = z;
	for (int k = 0; k<n; ++k) { res = f(res, v[k]); }
	return res;
}

template<typename T, typename V>
auto sum(T&& z, V&& v){ return fold(add, z, v); }

//implementations:
template<typename C, typename T, int n> auto operator*(C const& c, Vector<T, n> const& v) { return map(cmul_l(c), v); }
template<typename C, typename T, int n> auto operator*(Vector<T, n>const& v, C const& c) { return map(cmul_r(c), v); }
template<typename C, typename T, int n> auto operator/(Vector<T, n>const& v, C const& c) { return map(cdiv(c), v); }

template<typename T, int n> auto operator+(Vector<T, n> const& v, Vector<T, n> const& u) { return zip(add, v, u); }
template<typename T, int n> auto operator-(Vector<T, n> const& v, Vector<T, n> const& u) { return zip(sub, v, u); }

template<typename T, int n> auto dot(Vector<T, n> const& v, Vector<T, n> const& u) { return sum((T)0, zip(mul, v, u)); }

template<typename T, int n> auto sqlength(Vector<T, n> const& v) { return fold(sqsum, (T)0, v); }

template<typename T, int n> auto dyadic(Vector<T, n> const& v, Vector<T, n> const& u)
{
	return map([&](auto const& vi) { return map(cmul_r(vi), u); }, v);
}

template<typename T> auto cross0(Vector<T, 3> const& v, Vector<T, 3> const& u)
{
	return Vector<T, 3>{ {v[1] * u[2] - v[2] * u[1],
		                  v[2] * u[0] - v[0] * u[2],
		                  v[0] * u[1] - v[1] * u[0]}};
}

//Note this implementation is not optimally compiled by recent compilers...
template<typename T> auto cross(Vector<T, 3> const& a, Vector<T, 3> const& b)
{
	static const auto eps = 
	Vector<Vector<Vector<T, 3>, 3>, 3>{{ {{ {{ (T)0, (T)0, (T) 0 }}, {{ (T) 0, (T)0, (T)1 }}, {{ (T)0, (T)-1, (T)0 }} }},
	                                     {{ {{ (T)0, (T)0, (T)-1 }}, {{ (T) 0, (T)0, (T)0 }}, {{ (T)1, (T) 0, (T)0 }} }},
                                         {{ {{ (T)0, (T)1, (T) 0 }}, {{ (T)-1, (T)0, (T)0 }}, {{ (T)0, (T) 0, (T)0 }} }} }};
	return map([&](auto const& slice)
			{
				return dot( map([&](auto const& row){ return dot(row, b); }, slice), a);
			}, eps);
}

template<typename T, int sz>
std::ostream& operator<< (std::ostream& o, Vector<T, sz> const& v)
{
	o << "{ ";
	for (int i = 0; i<sz - 1; ++i) { o << v[i] << ", "; }
	o << v[sz - 1] << " }";
	return o;
}

template<typename T, int sz, int sz2>
std::ostream& operator<< (std::ostream& o, Vector<Vector<T, sz2>, sz> const& v)
{
	o << "\n{ ";
	for (int i = 0; i<sz - 1; ++i) { o << v[i] << "\n"; }
	o << v[sz - 1] << " }";
	return o;
}


int main()
{
	Vector<int, 2> v{ { 3, 5 } };
	Vector<int, 2> u{ { 7, 10 } };

	std::cout << "v = " << v << "\n";
	std::cout << "u = " << u << "\n";
	
	std::cout << "dot = " << dot(v, u) << "\n";

	std::cout << "dyadic = " << dyadic(v, u) << "\n";

	std::cout << "--------------------------------------\n";

	Vector<int, 3> a{ { 3, 5, 7 } };
	Vector<int, 3> b{ { -1, 4, 13 } };
	std::cout << "a = " << a << "\n";
	std::cout << "b = " << b << "\n";

	std::cout << cross0(a, b) << "\n";
	std::cout << cross(a, b) << "\n";

	
}
