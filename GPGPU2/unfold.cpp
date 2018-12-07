#include <optional>
#include <vector>
#include <iostream>

template<typename UF, typename S>
auto unfoldr( UF uf, S seed)
{
	std::vector<decltype(uf(seed).value().first)> result;
	while( auto maybe_pair = uf(seed) )
	{
		result.push_back( maybe_pair.value().first );
		seed = maybe_pair.value().second;
	}
	return result;
}

int main()
{
	auto uf = [](auto p)
	{
		if(p.second == 0){ return std::optional<std::pair<int, std::pair<int, int>>>{}; }
		else             { return std::optional<std::pair<int, std::pair<int, int>>>{ std::make_pair( p.first, std::make_pair(p.first*2, p.second-1))}; }
	};

	auto res = unfoldr(uf, std::make_pair(2, 5));

	for(auto x : res)
	{
		std::cout << x << "\n";
	}
}
