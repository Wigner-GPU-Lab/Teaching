//Type level integer value
template<int x>
struct Int
{
	static const int result = x;
};

//Type level list
template<typename... elems> struct List;

//Base template of type level function 'first', that return the first element of a list
template<typename L> struct Fst;

//Implementation of Fst (only if the type argument is a list):
template<typename A0, typename... As>
struct Fst<List<A0, As...>>{ using result = A0; };

//Example usage:
using FirstOfList = typename Fst<List<Int<1>, Int<2>, Int<3>>>::result;

//Base template of type level list indexer:
template<typename Icurrent, typename Igoal, typename L> struct Nth_impl;

//Implementation, induction step:
template<typename Icurrent, typename Igoal, typename A, typename... As>
struct Nth_impl<Icurrent, Igoal, List<A, As...>>
{
	using result = typename Nth_impl<Int<Icurrent::result+1>, Igoal, List<As...>>::result;
};

//Implementation, closing step:
template<typename I, typename A, typename... As> 
struct Nth_impl<I, I, List<A, As...>>  { using result = A; };

//Convenience helper alias:
template<typename L, typename I>
using Nth = typename Nth_impl<Int<0>, I, L>::result;

int main()
{
	return Nth<List<Int<12>, Int<24>, Int<36>>, Int<0>>::result;
}
