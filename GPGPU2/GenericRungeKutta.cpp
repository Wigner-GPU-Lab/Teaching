#include <ratio>
#include <iostream>
#include <thread>
#include "indexible.h"

template<int64_t n, int64_t d>
struct Rat
{
	static const int64_t nom = n;
	static const int64_t den = d;
	operator double() const { return (double)n / (double)d; }
};

template<int64_t n1, int64_t n2, int64_t d1, int64_t d2>
auto operator+( Rat<n1, d1>, Rat<n2, d2> )
{
	using res = std::ratio_add< std::ratio<n1, d1>, std::ratio<n2, d2> >;
	return Rat< res::num, res::den >();
}

template<int64_t n1, int64_t n2, int64_t d1, int64_t d2>
auto operator-( Rat<n1, d1>, Rat<n2, d2> )
{
	using res = std::ratio_subtract< std::ratio<n1, d1>, std::ratio<n2, d2> >;
	return Rat< res::num, res::den >();
}

template<int64_t n1, int64_t n2, int64_t d1, int64_t d2>
auto operator*( Rat<n1, d1>, Rat<n2, d2> )
{
	using res = std::ratio_multiply< std::ratio<n1, d1>, std::ratio<n2, d2> >;
	return Rat< res::num, res::den >();
}

template<int64_t n1, int64_t n2, int64_t d1, int64_t d2>
auto operator/( Rat<n1, d1>, Rat<n2, d2> )
{
	using res = std::ratio_divide< std::ratio<n1, d1>, std::ratio<n2, d2> >;
	return Rat< res::num, res::den >();
}

const inline auto state_adder  = [](auto s1,  auto s2){ return zip(add, s1, s2); };
const inline auto state_scaler = [](auto scl, auto s){ return map(sclmul_left(scl), s); };

template<typename As, typename Bs, typename Cs>
struct RungeKuttaStepper
{
	As as;
	Bs bs;
	Cs cs;

	template<typename TTime, typename TStep, typename TState, typename TRHS>
	auto eval_kn( Int<0>, TTime const& time, TStep const& h, TState const& state, TRHS&& rhs )
	{
		return tuple(rhs(time, state));
	}

	template<typename N, typename TTime, typename TStep, typename TState, typename TRHS>
	auto eval_kn( N n, TTime const& time, TStep const& h, TState const& state, TRHS&& rhs )
	{
		auto ks = eval_kn(--n, time, h, state, rhs );
		auto kc = rhs(	time + cs[n] * h,
						zip(add, state, map(sclmul_left(h), reduce(state_adder, zip(state_scaler, as[n], ks) ) ) ) );
		return append(ks, kc);
	}

	template<typename TTime, typename TStep, typename TState, typename TRHS>
	auto operator()( TTime const& time, TStep const& h, TState const& state, TRHS&& rhs )
	{
		auto ks = eval_kn( --size(bs), time, h, state, rhs );
		return zip( add, state, map( sclmul_left(h), reduce( state_adder, zip(state_scaler, bs, ks))));
	}
};

int main()
{
	//Ralston:
	/*using As = Tuple<	Tuple< Rat<0, 1> >,
						Tuple< Rat<2, 3> >	>;

	using Bs = Tuple<	Rat<1, 4>,
						Rat<3, 4>	>;

	using Cs = Tuple<	Rat<0, 1>,
						Rat<2, 3>	>;

	RungeKuttaStepper< As, Bs, Cs > ralston;*/

	//Standard RK4:
	using As = Tuple<	Tuple< Rat<0, 1> >,
		Tuple< Rat<1, 2> >,
		Tuple< Rat<0, 1>, Rat<1, 2> >,
		Tuple< Rat<0, 1>, Rat<0, 1>, Rat<1, 1> >
	>;

	using Bs = Tuple<	Rat<1, 6>,
		Rat<1, 3>,
		Rat<1, 3>,
		Rat<1, 6> >;

	using Cs = Tuple<	Rat<0, 1>,
		Rat<1, 2>,
		Rat<1, 2>,
		Rat<1, 1> >;

	RungeKuttaStepper< As, Bs, Cs > rk4;

	//Lotka-Volterra
	const double a = 0.5, b = 0.1, c = 0.8, d = 0.3;
	static const auto R = Int<0>();
	static const auto W = Int<1>();

	auto state = tuple(1.0, 2.0);
	auto rhs = [=](auto t, auto const& s)
	{
		return tuple(a*s[R]      - b*s[R]*s[W],
			         d*s[R]*s[W] - c*s[W]      );
	};

	double t = 0.0;
	const double dt = 0.01;
	while(t < 100.0)
	{
		state = rk4( t, dt, state, rhs );
		t += dt;
		std::cout << "t = " << t << "   R = " << state[R] << ",    W = " << state[W] << "\n";

		//std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}

	return 0;
}
