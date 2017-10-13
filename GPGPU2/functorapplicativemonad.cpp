#include <type_traits>
#include <iostream>
#include <cmath>
#include <typeinfo>
#include <memory>
#include <algorithm>

//identity function
//template<typename T> T id(T val){ return val; }

//identity as lambda
auto id = [](auto x) { return x; };

//composition as lambda
auto comp = [](auto f)
{
	return [&](auto g)
	{
		return [&](auto x)
		{
			return f(g(x));
		};
	};
};

//partial application as lambda
auto pa = [](auto f, auto x) { return [&](auto... ys) { return f(x, ys...); }; };

//some functions for fun
auto plus_one = [](auto x) { return x + 1; };
auto mul_two = [](auto x) { return x * 2; };
auto root = [](auto x) { return std::sqrt(x); };
auto cast_to_double = [](auto x) { return (double)x; };

//fmap wrapper as function object
auto fmap = [](auto func, auto functor) { return fmap_impl(func, functor); };

//Trivial Functor
template<typename T> struct Trivial
{
	T value;
};

//fmap implementation as a function template
template<typename F, typename A,
	typename B = typename std::result_of<F(A)>::type>
	Trivial<B> fmap_impl(F f, Trivial<A> in)
{
	return Trivial<B>{ f(in.value) };
}

void TestTrivialFunctor()
{
	//instantiate the functor:
	auto x = Trivial<int>{ 42 };

	//Check identity:
	auto test1a = fmap(id, x);
	auto test1b = id(x);
	std::cout << test1a.value << "   " << test1b.value << "\n";

	//Check composition:
	auto test2a = fmap(comp(plus_one)(mul_two), x);
	auto test2b = comp(pa(fmap, plus_one))(pa(fmap, mul_two))(x);
	std::cout << test2a.value << "   " << test2b.value << "\n";

	//Check type change:
	auto test3a = typeid(x).name();
	auto test3b = typeid(fmap(cast_to_double, x)).name();
	std::cout << test3a << "   " << test3b << "\n";
}

//----------------------------------
struct Just {};
struct Nothing {};

//Maybe functor:
//template<typename T> struct Maybe
//{
//	T value;
//	bool nothing;
//
//	Maybe(Nothing            ):value{ *((T*)this) }, nothing{true}{}
//	Maybe(Just,    T value_in):value{value_in},      nothing{false}{}
//  bool is_valid() const { return nothing ? false : true; }
//};

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

template<typename T>
std::ostream& operator<< (std::ostream& o, Maybe<T> const& m)
{
	if (m.is_valid()) { o << "Just " << (m.value()); }
	else { o << "Empty"; }
	return o;
}

void TestMaybeFunctor()
{
	//instantiate the functor:
	auto justx = Maybe<int>{ Just(), 42 };
	auto empty = Maybe<int>{ Nothing() };

	//auto q  =

	//Check identity:
	auto test1a = fmap(id, justx);
	auto test1b = id(justx);
	std::cout << test1a << "   " << test1b << "\n";

	auto test1c = fmap(id, empty);
	auto test1d = id(empty);
	std::cout << test1c << "   " << test1d << "\n";

	//Check composition:
	auto test2a = fmap(comp(plus_one)(mul_two), justx);
	auto test2b = comp(pa(fmap, plus_one))(pa(fmap, mul_two))(justx);
	std::cout << test2a << "   " << test2b << "\n";

	auto test2c = fmap(comp(plus_one)(mul_two), empty);
	auto test2d = comp(pa(fmap, plus_one))(pa(fmap, mul_two))(empty);
	std::cout << test2c << "   " << test2d << "\n";

	//Check type change:
	auto test3a = typeid(justx).name();
	auto test3b = typeid(fmap(cast_to_double, justx)).name();
	std::cout << test3a << "   " << test3b << "\n";

	auto test3c = typeid(empty).name();
	auto test3d = typeid(fmap(cast_to_double, empty)).name();
	std::cout << test3c << "   " << test3d << "\n";
}

//--------------------------------------

//Template holder struct:
template<template<typename> class Te> struct Template {};

//Applicative methods:
template<template<typename> class Functor, typename T>
auto pure(T value) { return pure_impl(Template<Functor>(), value); }

auto apply = [](auto ap_func, auto applicative) { return apply_impl(ap_func, applicative); };

//Make Maybe Applicative:
template<typename T>
auto pure_impl(Template<Maybe>, T value)
{
	return Maybe<T>{ Just(), value };
};

//apply implementation as a function template
template<typename F, typename A,
	typename B = typename std::result_of<F(A)>::type>
	Maybe<B> apply_impl(Maybe<F> f, Maybe<A> in)
{
	if (f.is_valid() && in.is_valid()) { return Maybe<B>{ Just(), (f.value())(in.value()) }; }
	else { return Maybe<B>{ Nothing() }; }
}

void TestMaybeApplicative()
{
	//instantiate the functor:
	auto justx = Maybe<int>{ Just(), 42 };
	auto empty = Maybe<int>{ Nothing() };

	//Check identity:
	auto test1a = apply(pure<Maybe>(id), justx);
	auto test1b = justx;
	std::cout << test1a << "   " << test1b << "\n";

	auto test1c = apply(pure<Maybe>(id), empty);
	auto test1d = empty;
	std::cout << test1c << "   " << test1d << "\n";

	//Check homomorphism:
	//(pure f) <*> (pure x) = pure (f x)
	auto test2a = apply(pure<Maybe>(plus_one), pure<Maybe>(137));
	auto test2b = pure<Maybe>(plus_one(137));
	std::cout << test2a << "   " << test2b << "\n";

	//Check associativity:
	//f <*> (g <*> x) = (pure (.)) <*> f <*> g <*> x
	auto f = pure<Maybe>(root);
	auto g = pure<Maybe>(mul_two);
	auto test3a = apply(f, apply(g, justx));
	auto test3b =
		apply(apply(apply(pure<Maybe>(comp), f), g), justx);
	std::cout << test3a << "   " << test3b << "\n";
}

//------------------------------------------------------

//Monad methods:
template<template<typename> class Applicative, typename T>
auto ret(T value) { return ret_impl(Template<Applicative>(), value); }

auto bind = [](auto m_func, auto monad) { return bind_impl(m_func, monad); };

//Make Maybe a Monad:
template<typename T>
auto ret_impl(Template<Maybe>, T value) { return Maybe<T>{ Just(), value }; };

//bind implementation as a function template
//Helper to extract template argument type:
template<typename T> struct GetTemplateArg;
template<template<typename> class Te, typename T>
struct GetTemplateArg<Te<T>>
{
	using type = T;
};

template<typename F, typename A,
	typename B =
	typename GetTemplateArg<typename std::result_of<F(A)>::type>::type>
	Maybe<B> bind_impl(Maybe<A> m, F f)
{
	if (m.is_valid()) { return f(m.value()); }
	else { return Maybe<B>{ Nothing() }; }
}

template<typename T, typename F>
auto operator>>(Maybe<T> m, F f) { return bind(m, f); }

void TestMaybeMonad()
{
	//instantiate the functor:
	auto x = 42;
	auto justx = Maybe<int>{ Just(), x };
	auto empty = Maybe<int>{ Nothing() };

	//halve if even
	auto f = [](int x) { return x % 2 == 0 ? Maybe<int>(Just(), x / 2) : Maybe<int>(Nothing()); };

	//root if non-negative
	auto g = [](auto x) { return x >= 0 ? Maybe<double>(Just(), root(x)) : Maybe<double>(Nothing()); };

	//Check left identity:
	auto test1a = ret<Maybe>(x) >> f;
	auto test1b = f(x);
	std::cout << test1a << "   " << test1b << "\n";
	auto test1c = ret<Maybe>(x + 1) >> f;
	auto test1d = f(x + 1);
	std::cout << test1c << "   " << test1d << "\n";

	//Check right identity:
	auto test2a = justx >> ret<Maybe, int>;
	auto test2b = justx;
	std::cout << test2a << "   " << test2b << "\n";

	//Check associativity:
	//(m >> f) >> g  =  m >> (\x -> f x >> g)
	auto test3a = (justx >> f) >> g;
	auto test3b = justx >> ([&](auto x) { return f(x) >> g; });
	std::cout << test3a << "   " << test3b << "\n";
}

#include <array>

//fmap implementation as a function template
template<typename F, typename A, size_t sz, typename B = typename std::result_of<F(A)>::type>
std::array<B, sz> fmap_impl(F f, std::array<A, sz> const& in)
{
	std::array<B, sz> result;
	std::transform(in.cbegin(), in.cend(), result.begin(), f);
	return result;
}

template<typename T, size_t sz>
std::ostream& operator<< (std::ostream& o, std::array<T, sz> const& arr)
{
	o << "{ ";
	for (size_t i = 0; i<sz - 1; ++i) { o << arr[i] << ", "; }
	o << arr[sz - 1] << " }";
	return o;
}

void TestArrayFunctor()
{
	//instantiate the functor:
	auto a = std::array<int, 3>{ {2, 42, 137}};

	//Check identity:
	auto test1a = fmap(id, a);
	auto test1b = id(a);
	std::cout << test1a << "   " << test1b << "\n";

	//Check composition:
	auto test2a = fmap_impl(comp(plus_one)(mul_two), a);
	auto test2b = comp(pa(fmap, plus_one))(pa(fmap, mul_two))(a);
	std::cout << test2a << "   " << test2b << "\n";

	//Check type change:
	auto test3a = typeid(a).name();
	auto test3b = typeid(fmap(cast_to_double, a)).name();
	std::cout << test3a << "   " << test3b << "\n";
}

int main()
{
	std::cout << "Trivial Functor:\n";
	TestTrivialFunctor();

	std::cout << "\n";
	std::cout << "Maybe Functor:\n";
	TestMaybeFunctor();

	std::cout << "\n";
	std::cout << "Maybe Applicative:\n";
	TestMaybeApplicative();

	std::cout << "\n";
	std::cout << "Maybe Monad:\n";
	TestMaybeMonad();

	std::cout << "\n";
	std::cout << "std::array Functor:\n";
	TestArrayFunctor();

	return 0;
}
