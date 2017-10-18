#include <type_traits>
#include <utility>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

//Compile time Integer:
template<int i> struct Int { static const int value = i; };

//Tuple implementation:
template<typename... Ts>
struct Tuple_;

template<> struct Tuple_<>{};

template<typename T> struct Tuple_<T>{ T e; };

template<typename T, typename U, typename... Us>
struct Tuple_<T, U, Us...>
{
	T e; 
	Tuple_<U, Us...> f;
};

//Tuple indexers:
template<typename T> HOST_DEVICE
auto idx_impl(T&& t, Int<0> const&)-> decltype(t.e)&
{
    return t.e;
}

template<int i, typename T, typename I=Int<i-1>> HOST_DEVICE
auto idx_impl(T&& t, Int<i> const&)->decltype(idx_impl(t.f, I()))&
{
    return idx_impl(t.f, I());
}

template<int i, typename T> HOST_DEVICE
auto idx(T&& t, Int<i> const& id)->decltype(idx_impl(std::forward<T>(t), id))
{
    return idx_impl(std::forward<T>(t), id);
}

//Tuple user front end struct:
template<typename... Ts>
struct Tuple
{
    Tuple_<Ts...> data;
    
    template<int n> decltype(auto) operator[] HOST_DEVICE (Int<n> const& N)const{ static_assert(n<sizeof...(Ts), "Tuple overindexing"); return idx(data, N); }
    template<int n> decltype(auto) operator[] HOST_DEVICE (Int<n> const& N)     { static_assert(n<sizeof...(Ts), "Tuple overindexing"); return idx(data, N); }
};

template<typename... Ts> HOST_DEVICE
Tuple<Ts...> tuple(Ts&&... ts){ return Tuple<Ts...>{{ts...}}; }

//map:
template<typename F, typename T1, typename T2, typename... Ts, typename... As> HOST_DEVICE
auto map_impl(F&& f, Tuple_<T1, T2, Ts...> const& t, As&&... as)->decltype(map_impl(std::forward<F>(f), t.f, std::forward<As>(as)..., f(t.e)))
{
    return map_impl(std::forward<F>(f), t.f, std::forward<As>(as)..., f(t.e));
}

template<typename F, typename T, typename... As, typename Q = typename std::result_of<F(T)>::type> HOST_DEVICE
Tuple<As..., Q> map_impl(F&& f, Tuple_<T> const& t, As&&... as)
{
    return tuple( std::forward<As>(as)..., f(t.e) );
}

template<typename F, typename... Ts> HOST_DEVICE
auto map(F&& f, Tuple<Ts...> const& t)->decltype(map_impl(f, t.data))
{
    return map_impl(f, t.data);
}

//zip:
template<typename F, typename T1, typename T2, typename... Ts, typename U1, typename U2, typename... Us, typename... As> HOST_DEVICE
auto zip_impl(F&& f, Tuple_<T1, T2, Ts...> const& t, Tuple_<U1, U2, Us...> const& u, As&&... as)->decltype(zip_impl(std::forward<F>(f), t.f, u.f, std::forward<As>(as)..., f(t.e, u.e)))
{
    return zip_impl(std::forward<F>(f), t.f, u.f, std::forward<As>(as)..., f(t.e, u.e));
}

template<typename F, typename T, typename U, typename... As, typename Q = typename std::result_of<F(T, U)>::type> HOST_DEVICE
Tuple<As..., Q> zip_impl(F&& f, Tuple_<T> const& t, Tuple_<U> const& u, As&&... as)
{
    return tuple( std::forward<As>(as)..., f(t.e, u.e) );
}

template<typename F, typename... Ts, typename... Us> HOST_DEVICE
auto zip(F&& f, Tuple<Ts...> const& t, Tuple<Us...> const& u)->decltype(zip_impl(f, t.data, u.data))
{
    return zip_impl(f, t.data, u.data);
}

//foldl:
template<typename F, typename Z, typename T1, typename T2, typename... Ts> HOST_DEVICE
auto foldl_impl(F&& f, Z&& z, Tuple_<T1, T2, Ts...> const& t)->decltype(foldl_impl(std::forward<F>(f), f(std::forward<Z>(z), t.e), t.f))
{
    return foldl_impl(std::forward<F>(f), f(std::forward<Z>(z), t.e), t.f);
}

template<typename F, typename Z, typename T, typename Q = typename std::result_of<F(Z, T)>::type> HOST_DEVICE
Q foldl_impl(F&& f, Z&& z, Tuple_<T> const& t)
{
    return f(std::forward<Z>(z), t.e);
}

template<typename F, typename Z, typename... Ts> HOST_DEVICE
auto foldl(F&& f, Z&& z, Tuple<Ts...> const& t)->decltype(foldl_impl(std::forward<F>(f), std::forward<Z>(z), t.data))
{
    return foldl_impl(std::forward<F>(f), std::forward<Z>(z), t.data);
}