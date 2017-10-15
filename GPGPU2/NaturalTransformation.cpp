
#include <type_traits>
#include <memory>
#include <vector>
#include <algorithm>

struct Just {};
struct Nothing {};

template<typename T> struct Maybe
{
	std::unique_ptr<T> p;

	Maybe(Nothing) :p{ nullptr } {}
	Maybe(Just, T value_in) :p{ new T{ value_in } } {}
	Maybe(Maybe const& cpy) : p{ cpy.is_valid() ? new T{ cpy.value() } : nullptr } {}

	T    value()    const { return *p; }
	bool is_valid() const { return p == nullptr ? false : true; }
};

//fmap implementation as a function template
template<typename F, typename A, typename B = typename std::result_of<F(A)>::type>
Maybe<B> fmap_impl(F f, Maybe<A> in)
{
	return in.is_valid() ? Maybe<B>{ Just(), f(in.value()) } : Maybe<B>{ Nothing() };
}

//fmap implementation for std::vector
template<typename F, typename A, typename B = typename std::result_of<F(A)>::type>
std::vector<B> fmap_impl(F f, std::vector<A> in)
{
    std::vector<B> res(in.size());
    std::transform(in.cbegin(), in.cend(), res.begin(), f);
    return res;
}

//fmap wrapper as function object
auto fmap = [](auto func, auto functor) { return fmap_impl(func, functor); };

//Natural transformation, valid for all X!
template<typename X>
std::vector<X> nt_Maybe_to_vector(Maybe<X> m)
{
    return m.is_valid() ? std::vector<X>{ m.value() } : std::vector<X>{};
};


int main()
{
    auto is_even = [](int x){ return x % 2 == 0; };

    auto m = Maybe<int>{ Just(), 5 };
   
    //the following two ways get the same result:
    auto result1 = fmap(is_even, nt_Maybe_to_vector( m ) );
    auto result2 = nt_Maybe_to_vector( fmap(is_even, m ) );

    printf("%s %s\n", result1[0] ? "true" : "false", result2[0] ? "true" : "false");
}