#include <type_traits>
#include <utility>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

//Compile time Integer:
template<int i> struct Int
{
	static const int value = i;

	auto operator++()    const { return Int<(i+1)>(); }
	auto operator++(int) const { return Int<(i+1)>(); }
	auto operator--()    const { return Int<(i-1)>(); }
	auto operator--(int) const { return Int<(i-1)>(); }
};

template<typename T> auto operator*(Int<0> const&, T const& x) { return Int<0>(); }
template<typename T> auto operator*(T&& x, Int<0> const&) { return Int<0>(); }
template<typename T> auto operator+(Int<0> const&, T&& x) { return std::forward<T>(x); }
template<typename T> auto operator+(T && x, Int<0> const&) { return std::forward<T>(x); }
auto operator+(Int<0> const&, Int<0> const&) { return Int<0>(); }

template<typename T> auto operator*(Int<1> const&, T&& x) { return std::forward<T>(x); }
template<typename T> auto operator*(T const& x, Int<1> const&) { return std::forward<T>(x); }
template<typename T> auto operator*(Int<-1> const&, T const& x) { return -x; }
template<typename T> auto operator*(T const& x, Int<-1> const&) { return -x; }
auto operator*(Int<0> const&, Int<0> const&) { return Int<0>(); }
auto operator*(Int<1> const&, Int<1> const&) { return Int<1>(); }

const inline auto add = [](auto const& x, auto const& y) { return x + y; };
const inline auto sub = [](auto const& x, auto const& y) { return x - y; };
const inline auto mul = [](auto const& x, auto const& y) { return x * y; };
const inline auto sclmul_left  = [](auto const& c){ return [=](auto const& x) { return c * x; }; };
const inline auto sclmul_right = [](auto const& c){ return [=](auto const& x) { return x * c; }; };

template<typename... I>
struct Indices {};

template<typename Is> struct repack_indices;
template<int... is>
struct repack_indices<std::integer_sequence<int, is...>>
{
	using result = Indices<Int<is>...>;
};

template<typename I> struct Indices_impl;
template<int n> struct Indices_impl<Int<n>>
{
	using result = typename repack_indices< std::make_integer_sequence<int, n> >::result;
};


template<int n> HOST_DEVICE
auto make_indices(Int<n> const&)->typename Indices_impl<Int<n>>::result
{
	return typename Indices_impl<Int<n>>::result();
}

//Tuple implementation:
template<typename... Ts>
struct Tuple_;

template<> struct Tuple_<> {};

template<typename T> struct Tuple_<T> { T e; };

template<typename T, typename U, typename... Us> struct Tuple_<T, U, Us...> { T e; Tuple_<U, Us...> f; };

//Tuple indexers:
template<typename T> HOST_DEVICE
auto& idx_impl(T&& t, Int<0> const&)
{
	return t.e;
}

template<int i, typename T, typename I = Int<i - 1>> HOST_DEVICE
auto& idx_impl(T&& t, Int<i> const&)
{
	return idx_impl(t.f, I());
}

template<int i, typename T> HOST_DEVICE
auto& idx(T&& t, Int<i> const& id)
{
	return idx_impl(std::forward<T>(t), id);
}

//Tuple user front end struct:
template<typename... Ts>
struct Tuple
{
	Tuple_<Ts...> data;

	template<int n> decltype(auto) operator[] HOST_DEVICE (Int<n> const& N)const { static_assert(n<sizeof...(Ts), "Tuple overindexing"); return idx(data, N); }
	template<int n> decltype(auto) operator[] HOST_DEVICE (Int<n> const& N)      { static_assert(n<sizeof...(Ts), "Tuple overindexing"); return idx(data, N); }
};

template<typename... Ts> HOST_DEVICE
Tuple<Ts...> tuple(Ts... ts) { return Tuple<Ts...>{ {ts...}}; }

template<typename... As, typename... Bs> HOST_DEVICE
auto cons(Tuple<As...>const&, Bs... bs) { return tuple(std::forward<Bs>(bs)...); }

template<typename... Ts> HOST_DEVICE
auto size(Tuple<Ts...> const&)->Int<sizeof...(Ts)> { return Int<sizeof...(Ts)>(); }


template<typename T, typename... Is, typename V> HOST_DEVICE
auto append_impl(T t, Indices<Is...>const&, V v)
{
	return tuple( t[Is()]..., v );
}

template<typename T, typename V> HOST_DEVICE
auto append(T&& t, V&& val)
{
	return append_impl(std::forward<T>(t), make_indices(size(t)), std::forward<V>(val));
}


//Vector
template<typename T, int n>
struct Vector
{
	T data[n];

	template<int i> T const& operator[] HOST_DEVICE (Int<i> const&)const { static_assert(i<n, "Vector overindexing"); return data[i]; }
	template<int i> T&       operator[] HOST_DEVICE (Int<i> const&)      { static_assert(i<n, "Vector overindexing"); return data[i]; }

	T const& operator[] HOST_DEVICE (int const& i)const { return data[i]; }
	T&       operator[] HOST_DEVICE (int const& i) { return data[i]; }
};

template<typename T, typename... Ts> HOST_DEVICE
Vector<T, sizeof...(Ts)+1> vec(T const& t, Ts const&... ts) { return Vector<T, sizeof...(Ts)+1>{ {t, ts...}}; }

template<typename A, int m, typename... Bs> HOST_DEVICE
auto cons(Vector<A, m>const&, Bs const&... bs) { return vec(/*std::forward<Bs>*/(bs)...); }

template<typename T, int n> HOST_DEVICE
auto size(Vector<T, n> const&)->Int<n> { return Int<n>(); }

//map:
template<typename F, typename I, typename... Is> HOST_DEVICE
auto map_impl(F f, Indices<Is...>const&, I const& i)
{
	return cons(i, f(i[Is()])...);
}

template<typename F, typename I> HOST_DEVICE
auto map(F f, I const& i)
{
	return map_impl(/*std::forward<F>*/(f), make_indices(size(i)), /*std::forward<I>*/(i));
}

//zip:
template<typename F, typename I, typename J, typename... Is> HOST_DEVICE
auto zip_impl(F&& f, Indices<Is...>const&, I&& i, J&& j)
{
	return cons(i, f(i[Is()], j[Is()])...);
}

template<typename F, typename I, typename J> HOST_DEVICE
auto zip(F&& f, I&& i, J&& j)
{
	return zip_impl(std::forward<F>(f), make_indices(size(i)), std::forward<I>(i), std::forward<J>(j));
}

//foldl:
template<typename F, int n, typename Z, typename I> HOST_DEVICE
auto foldl_impl(F&& f, Int<n>const&, Int<n>const&, Z&& z, I&&)
{
	return std::forward<Z>(z);
}

template<typename F, int n, int c, typename Z, typename I> HOST_DEVICE
auto foldl_impl(F&& f, Int<n>const& N, Int<c>const& C, Z&& z, I && i)
{
	return f(foldl_impl(std::forward<F>(f), N, Int<c + 1>(), std::forward<Z>(z), i), i[C]);
}

template<typename F, typename Z, typename I> HOST_DEVICE
auto foldl(F&& f, Z&& z, I&& i)
{
	return foldl_impl(std::forward<F>(f), size(i), Int<0>(), std::forward<Z>(z), std::forward<I>(i));
}

//reduce:
template<typename F, int n, typename I> HOST_DEVICE
auto reduce_impl(F&& f, Int<n>const&, Int<n>const& C, I&& i)
{
	return i[--C];
}

template<typename F, int n, int c, typename I> HOST_DEVICE
auto reduce_impl(F&& f, Int<n>const& N, Int<c>const& C, I&& i)
{
	return f(reduce_impl(std::forward<F>(f), N, Int<c + 1>(), i), i[--C]);
}

template<typename F, typename I> HOST_DEVICE
auto reduce(F&& f, I&& i)
{
	return reduce_impl(std::forward<F>(f), size(i), Int<1>(), std::forward<I>(i));
}

template<typename T, typename U> auto dot(T const& a, U const& b)
{
	return foldl(add, Int<0>(), zip(mul, a, b));
}

template<typename T>
auto lengthSq(T const& a){ return dot(a, a); }

template<typename T, int n>
auto operator* (Vector<T, n> const& v, T const& c){ return map([=](auto x){ return x*c; }, v); }

template<typename T, int n>
auto operator+ (Vector<T, n> const& v, Vector<T, n> const& u){ return zip([](auto x, auto y){ return x+y; }, v, u); }

template<typename T, int n>
auto operator- (Vector<T, n> const& v, Vector<T, n> const& u){ return zip([](auto x, auto y){ return x-y; }, v, u); }

template<typename T> auto cross(Vector<T, 3> const& a, Vector<T, 3> const& b)
{
	using Z = Int<0>;
	using U = Int<1>;
	using M = Int<-1>;
	using eps =
		Tuple<
		Tuple< Tuple<Z, Z, Z>, Tuple<Z, Z, U>, Tuple<Z, M, Z> >,
		Tuple< Tuple<Z, Z, M>, Tuple<Z, Z, Z>, Tuple<U, Z, Z> >,
		Tuple< Tuple<Z, U, Z>, Tuple<M, Z, Z>, Tuple<Z, Z, Z> >
		>;

	return map([&](auto const& slice)
	{
		return dot(map([&](auto const& row) { return dot(row, b); }, slice), a);
	}, eps());
}

template<typename T> auto cross0(Vector<T, 3> const& v, Vector<T, 3> const& u)
{
	return Vector<T, 3>{ { v[1] * u[2] - v[2] * u[1],
		v[2] * u[0] - v[0] * u[2],
		v[0] * u[1] - v[1] * u[0] } };
}

