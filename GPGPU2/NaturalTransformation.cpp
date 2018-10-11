#include <type_traits>
#include <iostream>
#include <vector>
#include <algorithm>
#include <optional>

template<typename A> using Maybe = std::optional<A>;

//helper base overload #clang bug
template<template<typename...> typename Functor, typename F, typename A, typename B = typename std::result_of<F(A)>::type>
Functor<B> fmap_impl(F f, Functor<A> in){ return fmap_impl(f, in); }

//fmap wrapper as function object
inline auto fmap = [](auto func, auto functor){ return fmap_impl(func, functor); };

//fmap implementation as a function template
template<typename F, typename A, typename B = typename std::result_of<F(A)>::type>
Maybe<B> fmap_impl(F f, Maybe<A> in)
{
    return in.has_value() ? Maybe<B>{ f(in.value()) } : Maybe<B>{ };
}

//Workaround fmap_impl predeclaration ambiguity in MSVC
template<typename A>
struct VectorImpl
{
	typedef std::vector<A> type;
};
template<typename A> using Vector = typename VectorImpl<A>::type;

//fmap implementation for std::vector
template<typename F, typename A, typename B = typename std::result_of<F(A)>::type>
Vector<B> fmap_impl(F f, Vector<A> in)
{
    Vector<B> res(in.size());
    std::transform(in.cbegin(), in.cend(), res.begin(), f);
    return res;
}

//Natural transformation, valid for all X!
template<typename X>
Vector<X> nt_Maybe_to_vector(Maybe<X> m)
{
    return m.has_value() ? Vector<X>{ m.value() } : Vector<X>{};
};

int main()
{
    auto is_even = [](int x){ return x % 2 == 0; };

    auto m = Maybe<int>{ 5 };
   
    //the following two ways get the same result:
    auto result1 = fmap(is_even, nt_Maybe_to_vector( m ) );
    auto result2 = nt_Maybe_to_vector( fmap(is_even, m ) );

    std::cout << (result1[0] ? "true" : "false") << "   " << (result2[0] ? "true" : "false") << "\n";
}
