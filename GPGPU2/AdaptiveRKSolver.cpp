#include <utility>
#include <fstream>
#include <functional>
#include <iostream>
#include <ratio>
#include "indexible.h"

inline auto sq = [](auto x){ return x*x; };

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

//This is a type that represents a value and a flag to continue a calculation, or not
template<typename T>
struct escaper_t
{
	T value;
	bool should_continue;
	operator bool() const { return should_continue; }
};

template<typename T>
auto escaper(T&& value, bool should_continue){ return escaper_t<std::remove_reference_t<T>>{ std::forward<T>(value), should_continue }; }

//Helper for the Runge Kutta Stepper:
const inline auto state_adder  = [](auto s1,  auto s2){ return zip(add, s1, s2); };
const inline auto state_scaler = [](auto scl, auto s){ return map(sclmul_left(scl), s); };

template<typename As, typename Bs, typename Bps, typename Cs>
struct RungeKuttaStepper
{
	As  as;
	Bs  bs;
	Bps bps;
	Cs  cs;

	template<typename TTime, typename TStep, typename TState, typename TRHS>
	auto eval_kn( Int<0>, TTime const& time, TStep const& h, TState const& state, TRHS&& rhs ) const
	{
		return tuple(rhs(time, state));
	}

	template<typename N, typename TTime, typename TStep, typename TState, typename TRHS>
	auto eval_kn( N n, TTime const& time, TStep const& h, TState const& state, TRHS&& rhs ) const
	{
		auto ks = eval_kn(--n, time, h, state, rhs );
		auto kc = rhs( time + cs[n] * h,
			           zip(add, state, map(sclmul_left(h), reduce(state_adder, zip(state_scaler, as[n], ks) ) ) ) );
		return append(ks, kc);
	}


	//return value is a 3 element tuple: (last_state, new_state, state_error_est)
	template<typename TTime, typename TStep, typename TState, typename TRHS>
	auto operator()( TTime const& time, TStep const& h, TState const& state, TRHS&& rhs ) const
	{
		auto ks = eval_kn( --size(bs), time, h, state, rhs );
		return tuple(
					state,
			        zip( add, state, map( sclmul_left(h), reduce( state_adder, zip(state_scaler, bs, ks)))),
			        reduce( state_adder, zip(state_scaler, zip(sub, bs, bps), ks)) //State vector error estimate
		            );
	}
};

//Mutumorphism :: ((SH, S)->(SH, S)) -> ((Z, SH)->(Escaper (Z, SH))) -> Z -> SH -> S -> (Z, SH, S)
//UF: unfold function :: (SH, S)->(SH, S)
//FF:   fold function :: (Z, SH)-> Escaper_t (Z, SH)
//Z: zero (initial value) for the fold function and also the return type of it
//SH: the shared state type between the unfold and fold functions
//S: the seed (initial value) of the unfold function and also the return value of it
template<typename UF, typename FF, typename Z, typename SH, typename S>
auto mutu( UF&& uf, FF&& ff, Z&& zero, SH&& sh, S&& seed )
{
	auto ufs = uf( std::forward<SH>(sh), std::forward<S>(seed) );
	auto fs = ff( std::forward<Z>(zero), ufs.first );//fs is an escaper_t, implicitely convertible to bool
	while( fs )
	{
		ufs = uf( fs.value.second, ufs.second );
		fs = ff( fs.value.first, ufs.first );
	}
	return tuple(fs.value.first, fs.value.second, ufs.second);
}

//Tuple indices for easier readability
inline const auto iTime     = Int<0>();
inline const auto iStepsize = Int<1>();
inline const auto iState    = Int<2>();

inline const auto iLastState   = Int<0>();
inline const auto iNewState    = Int<1>();
inline const auto iStateErrEst = Int<2>();

//This is the Runge-Kutta stepper wrapper, that tries to step, and if the error is too large, tries again, with a smaller step size
template<typename TStepper, typename TTime, typename TState, typename TRHS>
auto RungeKuttaRepeater( TStepper&& stepper, TTime const& time, TTime const& h, TState const& state, TRHS&& rhs )
{
	struct FoldState
	{
		double atol, rtol, last_error;
		bool reject;
	};
	FoldState fs0{1e-7, 1e-7, 1e-4, false};

	struct UnfoldState{} ufs0;

	return mutu(
		[&]( auto&& ths, UnfoldState& ufs )
	    {
		   // return type: ( (time, h, (last_state, new_state, state_error_est)), UnfoldState )
		   return std::make_pair( tuple(ths[iTime], ths[iStepsize], stepper( ths[iTime], ths[iStepsize], ths[iState], rhs )), ufs );
	    },
		[]( FoldState& fs, auto&& thsds )
		{
			auto last_state = thsds[Int<2>()][iLastState];
			auto new_state  = thsds[Int<2>()][iNewState];
			auto& dstate    = thsds[Int<2>()][iStateErrEst];

			//Scale the error estimate and the new state by the tolerances and reduce to calculate a scalar, the final error estimate 
			auto err_pre = reduce( add, zip( [&](auto const& s, auto const& ds){ return sq(ds / (fs.atol + fabs(s)*fs.rtol)); }, new_state, dstate ) );
			auto err = sqrt(err_pre / (int)size(last_state).value);

			auto& t = thsds[iTime];
			auto& hc = thsds[iStepsize];

			auto hnext = 0.95 * fabs(hc) * pow( fabs(fs.last_error), 0.4/5.0) * pow(fabs(err), -0.7/5.0);
			
			fs.last_error = err;

			bool end = (err <= 1.0);
			
			//return type: Escaper_t ( pair(FoldState, (time, h, state)) )
			return escaper( std::make_pair(fs, tuple((end ? t+hc : t), hnext, (end ? new_state : last_state) )), !end );
		},
		fs0, tuple( time, h, state ), ufs0 );
}

//This driver keeps stepping the ODE by calling the Runge-Kutta repeater until some end condition is met
//and folds the produced (time, h, state) triplets into some user defined FoldState
template<typename TStepper, typename TTime, typename TState, typename TRHS, typename FFoldFunc, typename FoldState, typename FEndCondition>
auto ODEDriver(TStepper&& stepper, TTime&& t0, TState&& initial_state, TRHS&& rhs, FFoldFunc&& ff, FoldState&& foldstate, FEndCondition&& ec)
{
	FoldState foldstate0 = std::forward<FoldState>(foldstate);
	struct UnfoldState{} ufs0;

	return mutu(
		[&](auto && ths, auto&& ufs)
		{
			auto tmp = RungeKuttaRepeater(stepper, ths[iTime], ths[iStepsize], ths[iState], rhs)[Int<1>()];//this last index selects the (time, h, state) triplet from the return value
			return std::make_pair( tmp, ufs );
		},
		[&](FoldState& fs, auto && ths)
		{
			auto&& tmp = ff(fs, ths);
			return escaper( std::make_pair(fs, tmp), ec(tmp) );
		},
		foldstate0, tuple(t0, 1e-4, initial_state), ufs0);
}

struct LVState
{
	double r, w;
	LVState( double r0, double w0 ):r(r0), w(w0){}
	double const& operator[]( Int<0> )const{ return r; }
	double const& operator[]( Int<1> )const{ return w; }
};

auto size( LVState const& ){ return Int<2>(); }

template<typename... Bs>
auto cons(LVState const&, Bs const&... bs) { return LVState{bs...}; }

int main()
{
	//Dormand-Prince 5 stepper:
	//https://en.wikipedia.org/wiki/Dormand–Prince_method

	using As = Tuple<	Tuple< Rat<0, 1> >,
		Tuple< Rat<1, 5> >,
		Tuple< Rat<3, 40>, Rat<9, 40> >,
		Tuple< Rat<44, 45>, Rat<-56, 15>, Rat<32, 9> >,
		Tuple< Rat<19372, 6561>, Rat<-25360, 2187>, Rat<64448, 6561>, Rat<-212, 729> >,
		Tuple< Rat<9017, 3168>, Rat<-355, 33>, Rat<46732, 5247>, Rat<49, 176>, Rat<-5103, 18656> >,
		Tuple< Rat<35, 384>, Rat<0, 1>, Rat<500, 1113>, Rat<125, 192>, Rat<-2187, 6784>, Rat<11, 84> >
	>;

	using Bs = Tuple<	Rat<35, 384>,
		Rat<0, 1>,
		Rat<500, 1113>,
		Rat<125, 192>,
		Rat<-2187, 6784>,
		Rat<11, 84>,
		Rat<0, 1>
	>;
	using Bps = Tuple<	Rat<5179, 57600>,
		Rat<0, 1>,
		Rat<7571, 16695>,
		Rat<393, 640>,
		Rat<-92097, 339200>,
		Rat<187, 2100>,
		Rat<1, 40>
	>;

	using Cs = Tuple<	Rat<0, 1>,
		Rat<1, 5>,
		Rat<3, 10>,
		Rat<4, 5>,
		Rat<8, 9>,
		Rat<1, 1>,
		Rat<1, 1>
	>;

	RungeKuttaStepper< As, Bs, Bps, Cs > dps;

	//Lotka-Volterra
	double a = 0.5, b = 0.1, c = 0.8, d = 0.3;

	double t0 = 0.0;
	LVState s0{ 5.0, 5.0 };
	auto rhs = [=](double t, LVState const& s){ return LVState( a*s.r - b*s.r*s.w, d*s.r*s.w - c*s.w ); };
	auto fendcond = [](auto&& ths){ return ths[Int<0>()] < 120.0; };

	std::ofstream file("out.txt");

	struct Logger
	{
		std::ofstream* pfile;
	};

	auto foldfunc = []( Logger& log, auto&& ths )
	{
		auto& t = ths[ iTime ];
		auto& h = ths[ iStepsize ];
		auto& s = ths[ iState ];
		*log.pfile << t << "   " << h << "   " << s.r << "   " << s.w << "\n";
		return ths;
	};

	Logger logger{&file};

	auto res = ODEDriver(dps, t0, s0, rhs, foldfunc, logger, fendcond );
	//res is the (Logger, (final time, final stepsize, final state), final unfold state)
	std::cout << "Final time:     " << res[Int<1>()][iTime] << "\n";
	std::cout << "Final stepsize: " << res[Int<1>()][iStepsize] << "\n";
	std::cout << "Final state:    " << res[Int<1>()][iState].r << " rabbits, " << res[Int<1>()][iState].w << " wolves.\n";
	return 0;
}
