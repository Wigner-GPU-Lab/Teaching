#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <string>
#include <sstream>
#include <vector>
#include <iomanip>

struct Continent
{
	std::string name;
	unsigned long long area;
	unsigned long long population;

	double density() const { return (double)population / (double)area; }
};

std::istream& operator>>( std::istream& s, Continent& c )
{
    std::getline(s, c.name, '\t');
    if(c.name.size() == 0){ return s; }

	s >> c.area;
	s >> c.population;

	std::string tmp;
    std::getline(s, tmp, '\n');
    return s;
}

std::ostream& operator<<( std::ostream& s, Continent const& c )
{
    s << std::left << std::setw(16) << c.name << std::right << std::setw(10) << c.area  << std::right << std::setw(12) << c.population;
    return s;
}

struct Continent2
{
	Continent2( Continent const& c ):name{c.name}, area{c.area}, population{c.population}, density{c.density()}{}
	std::string name;
	unsigned long long area;
	unsigned long long population;
	double density;
};

std::ostream& operator<<( std::ostream& s, Continent2 const& c )
{
    s << std::left << std::setw(16) << c.name << std::right << std::setw(10) << c.area  << std::right << std::setw(12) << c.population << std::right << std::setw(10) << c.density;
    return s;
}

template<typename Tout = Continent>
void PrintContinents(std::string const& header, std::vector<Continent> const& data)
{
	std::cout << header << "\n";
	std::copy( data.cbegin(), data.cend(), std::ostream_iterator<Tout>(std::cout, "\n") );
	std::cout << "\n\n\n";
}

int main()
{
	std::ifstream file("data.txt");
	std::vector<Continent> data;
	if( !file.is_open() )
    {
        std::cout << "Could not open input file.\n";
        return -1;
    }
    else
    {
		std::string tmp;
		std::getline(file, tmp, '\n');
        std::copy( std::istream_iterator<Continent>(file), std::istream_iterator<Continent>(), std::back_inserter(data) );
    }

	//Print data:
	PrintContinents("Original data:", data);

	//Sort by population:
	std::sort(data.begin(), data.end(), [](Continent const& c1, Continent const& c2){ return c1.population < c2.population;});

	//Print data:
	PrintContinents("Sorted by population:", data);

	//Sort by area:
	std::sort(data.begin(), data.end(), [](Continent const& c1, Continent const& c2){ return c1.area < c2.area;});

	//Print data:
	PrintContinents("Sorted by area:", data);

	//Sort by population density:
	std::sort(data.begin(), data.end(), [](Continent const& c1, Continent const& c2){ return c1.density() < c2.density(); });

	//Print data:
	PrintContinents<Continent2>("Sorted by population density:", data);

	//Copy based on predicate:
	std::vector<Continent> selection;
	std::copy_if( data.begin(), data.end(), std::back_inserter(selection), [](Continent const& c){ return c.density() < 50.0; });

	//Print data:
	PrintContinents<Continent2>("Selection by population density < 50.0:", selection);

	//Partition:
	auto partition_point = std::partition(data.begin(), data.end(), [](Continent const& c){ return c.area < 44579000 / 2; });

	std::vector<Continent> lower_part, higher_part;
	std::copy(data.begin(), partition_point, std::back_inserter(lower_part));
	std::copy(partition_point, data.end(),   std::back_inserter(higher_part));

	//Print data:
	PrintContinents<Continent2>("Area smaller than largest / 2:", lower_part);
	PrintContinents<Continent2>("Area larger than largest / 2:", higher_part);

	return 0;
}
