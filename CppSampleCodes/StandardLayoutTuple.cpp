#include <type_traits>
#include <utility>

template<typename... Ts>
struct Tuple;

template<> struct Tuple<>{};

template<typename T> struct Tuple<T>{ T e; };

template<typename T, typename U, typename... Us> struct Tuple<T, U, Us...>{ T e; Tuple<U, Us...> f; };

template<typename... Ts>
Tuple<Ts...> tuple(Ts&&... ts){ return Tuple<Ts...>{ts...}; }

template<int i> struct Int{ static const int value = i; };

//idx:
template<typename T, typename... Ts>
auto idx_impl(Tuple<T, Ts...> const& t, Int<0> const&)->decltype(t.e)
{
    return t.e;
}

template<int i, typename... Ts, typename I=Int<i-1>>
auto idx_impl(Tuple<Ts...> const& t, Int<i> const&)->decltype(idx_impl(t.f, I()))
{
    return idx_impl(t.f, I());
}

template<int i, typename... Ts>
auto idx(Tuple<Ts...> const& t, Int<i> const& id)->decltype(idx_impl(t, id))
{
    return idx_impl(t, id);
}

//map:
template<typename F, typename T1, typename T2, typename... Ts, typename... As>
auto map_impl(F&& f, Tuple<T1, T2, Ts...> const& t, As&&... as)->decltype(map_impl(std::forward<F>(f), t.f, std::forward<As>(as)..., f(t.e)))
{
    return map_impl(std::forward<F>(f), t.f, std::forward<As>(as)..., f(t.e));
}

template<typename F, typename T, typename... As, typename Q = typename std::result_of<F(T)>::type>
Tuple<As..., Q> map_impl(F&& f, Tuple<T> const& t, As&&... as)
{
    //using Q = decltype(f(t.e));
    return Tuple<As..., Q>{ std::forward<As>(as)..., f(t.e) };
}

template<typename F, typename... Ts>
auto map(F&& f, Tuple<Ts...> const& t)->decltype(map_impl(f, t))
{
    return map_impl(f, t);
}

//foldl:
template<typename F, typename Z, typename T1, typename T2, typename... Ts>
auto foldl_impl(F&& f, Z&& z, Tuple<T1, T2, Ts...> const& t)->decltype(foldl_impl(std::forward<F>(f), f(std::forward<Z>(z), t.e), t.f))
{
    return foldl_impl(std::forward<F>(f), f(std::forward<Z>(z), t.e), t.f);
}

template<typename F, typename Z, typename T, typename Q = typename std::result_of<F(Z, T)>::type>
Q foldl_impl(F&& f, Z&& z, Tuple<T> const& t)
{
    return f(std::forward<Z>(z), t.e);
}

template<typename F, typename Z, typename... Ts>
auto foldl(F&& f, Z&& z, Tuple<Ts...> const& t)->decltype(foldl_impl(std::forward<F>(f), std::forward<Z>(z), t))
{
    return foldl_impl(std::forward<F>(f), std::forward<Z>(z), t);
}

int main()
{
    auto l = [](double const& x){ return x*x; };
    auto s = [](double const& x, double const& y){ return x+y; };

    auto u = tuple(2, 3.0, 5.0f);
    auto v = map(l, u);
    auto c = foldl(s, 100, u);
    double v1 = idx(v, Int<0>());
    double v2 = idx(v, Int<1>());
    double v3 = idx(v, Int<2>());
    /*Tuple<>                             t0{};
    Tuple<float>                        t1{3.3f};
    Tuple<float, float>                 t2{3.3f, 3.3};
    Tuple<float, float, float>          t3{3.3f, 3.3f, 3.3f};
    Tuple<float, float, float, float>   t4{3.3f, 3.3f, 3.3f, 4.4f};*/
    return std::is_standard_layout<decltype(v)>::value ? 10 : -10;
}
