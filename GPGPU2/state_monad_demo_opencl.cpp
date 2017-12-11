#include <CL/cl.h>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

struct Unit{};

//State s is a Monad (v is the monadic parameter)
template<typename S, typename V> using State = std::function<std::pair<V, S>(S)>;

//Monad return :: a -> m a 
template<typename S, typename V> auto ret(V         val){ return State<S, V   >{ [=](S s ){ return std::make_pair(val,    s); } }; }
template<typename S>             auto get(             ){ return State<S, S   >{ [ ](S s ){ return std::make_pair(s,      s); } }; }
template<typename S>             auto put(S           s){ return State<S, Unit>{ [=](S s2){ return std::make_pair(Unit(), s); } }; }
template<typename S, typename V> auto run(State<S, V> s, S t){ return s(t); }

//Monad       bind :: m a -> (a -> m b) -> m b
//State monad bind :: State s v -> (v -> State s u) -> State s u
template<typename S, typename V, typename F> auto operator>>(State<S, V> s, F f)//f: V -> State<S, U>
{
	using StateSU = decltype( f(V()) );
	return StateSU( [=](S t)
	{
		/*std::pair<V, S>*/auto ps = run(s, t);
		auto s2 = f(ps.first);
		return run(s2, ps.second);
	} );
}

template<typename S> auto Get(   ){ return [ ](auto){ return get<S>(); }; }
template<typename S> auto Put(S s){ return [=](auto){ return put(s); }; }

template<typename V>
using clState = State<cl_int, V>;

auto mclGetNumberOfPlatforms()
{
	return [](Unit)
	{
		return clState<cl_uint>( [=](cl_int s)
		{
			if(s == CL_SUCCESS)
			{
				cl_uint nPlat = 0;
				auto res = clGetPlatformIDs(0, nullptr, &nPlat);
				return std::make_pair(nPlat, res);
			}
			else{ return std::make_pair((cl_uint)0, s); }
		});
	};
}

auto mclGetPlatforms()
{
	return [](cl_uint nPlat)
	{
		return clState<std::vector<cl_platform_id>>( [=](cl_int s)
		{
			if(s == CL_SUCCESS)
			{
				std::vector<cl_platform_id> v(nPlat);
				auto res = clGetPlatformIDs(nPlat, v.data(), nullptr);
				return std::make_pair(v, res);
			}
			else{ return std::make_pair(std::vector<cl_platform_id>{}, s); }
		});
	};
}

auto mclGetNumberOfDevices()
{
	return [](auto&& pls)
	{
		using V = std::vector<std::pair<cl_platform_id, cl_uint>>;
		return clState<V>( [=](cl_int s)
		{
			if(s == CL_SUCCESS)
			{
				cl_int res = CL_SUCCESS;
				V val(pls.size());
				for(size_t i=0; i<pls.size(); ++i)
				{
					val[i].first = pls[i];
					res = clGetDeviceIDs(pls[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &val[i].second);
					if( res != CL_SUCCESS){ break; }
				}
				return std::make_pair(val, res);
			}
			else{ return std::make_pair(V{}, s); }
		});
	};
}

auto mclGetDevices()
{
	return [](auto&& pls)
	{
		using V = std::vector<std::pair<cl_platform_id, std::vector<cl_device_id>>>;
		return clState<V>( [=](cl_int s)
		{
			if(s == CL_SUCCESS)
			{
				cl_int res = CL_SUCCESS;
				V val(pls.size());
				for(size_t i=0; i<pls.size(); ++i)
				{
					val[i].first = pls[i].first;
					val[i].second.resize(pls[i].second);
					res = clGetDeviceIDs(pls[i].first, CL_DEVICE_TYPE_ALL, pls[i].second, val[i].second.data(), nullptr);
					if( res != CL_SUCCESS){ break; }
				}
				return std::make_pair(val, res);
			}
			else{ return std::make_pair(V{}, s); }
		});
	};
}

auto mclGetDeviceNames()
{
	return [](auto&& pls)
	{
		using V = std::vector<std::pair<cl_platform_id, std::vector<std::string>>>;
		return clState<V>( [=](cl_int s)
		{
			if(s == CL_SUCCESS)
			{
				cl_int res = CL_SUCCESS;
				V val(pls.size());
				for(size_t i=0; i<pls.size(); ++i)
				{
					val[i].first = pls[i].first;
					auto n = pls[i].second.size();
					val[i].second.resize(n);
					for(size_t j=0; j<n; ++j)
					{
						size_t sz = 0;
						res = clGetDeviceInfo(pls[i].second[j], CL_DEVICE_NAME, 0, nullptr, &sz);
						if( res != CL_SUCCESS){ break; }
						val[i].second[j].resize(sz);
						res = clGetDeviceInfo(pls[i].second[j], CL_DEVICE_NAME, val[i].second[j].size(), (void*)val[i].second[j].data(), nullptr);
						if( res != CL_SUCCESS){ break; }
					}
				}
				return std::make_pair(val, res);
			}
			else{ return std::make_pair(V{}, s); }
		});
	};
}

int main()
{
	auto query_chain = ret<cl_int>(Unit())
		             >> mclGetNumberOfPlatforms()
		             >> mclGetPlatforms()
		             >> mclGetNumberOfDevices()
			         >> mclGetDevices()
		             >> mclGetDeviceNames();

	auto res = run(query_chain, CL_SUCCESS);

	for(auto const& p : res.first)
	{
		std::cout << "Platform:\n";
		for(auto const& d : p.second)
		{
			std::cout << "Device: " << d << "\n";
		}
	}
	return 0;
}