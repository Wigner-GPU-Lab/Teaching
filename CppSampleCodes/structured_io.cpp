#include <vector>    //for std::vector
#include <string>    //for std::string
#include <sstream>   //for std::stringstream
#include <fstream>   //for std::ifstream, std::ofstream
#include <iterator>  //for std::istream_iterator, std::ostream_iterator
#include <algorithm> //for std::copy, std::for_each
#include <iostream>  //for std::cout

struct Point3D
{
    double x, y, z;
};

std::istream& operator>>( std::istream& s, Point3D& p )
{
    s >> p.x >> p.y >> p.z;
    return s;
}

std::ostream& operator<<( std::ostream& s, Point3D const& p ){ s << p.x << " " << p.y << " " << p.z; return s; }

struct Particle
{
    std::string name;
    double energy;
    std::vector<Point3D> path;
};

std::istream& operator>>( std::istream& s, Particle& p )
{
    std::getline(s, p.name, ' ');
    if(p.name.size() == 0){ return s; }

    std::string tmp;
    std::getline(s, tmp, ' '); p.energy = std::stod(tmp);
    if(tmp.size() == 0){ return s; }

    std::getline(s, tmp, '\n');
    if(tmp.size() == 0){ return s; }
    
    std::stringstream ss(tmp);
    p.path.clear();
    std::copy(std::istream_iterator<Point3D>(ss), std::istream_iterator<Point3D>(), std::back_inserter(p.path) );
    return s;
}

std::ostream& operator<<( std::ostream& s, Particle const& p )
{
    s << "Particle: " << p.name << ", " << p.energy << "\n";
    for(auto const& pt : p.path){ s << "[" << pt << "]\n"; }
    return s;
}

int main()
{
    std::vector<Particle> data;

    {
        std::ifstream input("structured_io.txt");

        if( !input.is_open() )
        {
            std::cout << "Could not open input file.\n";
            return -1;
        }
        else
        {
            std::copy( std::istream_iterator<Particle>(input), std::istream_iterator<Particle>(), std::back_inserter(data) );
            std::for_each(data.begin(), data.end(), [&](Particle const& p){ std::cout << p << std::endl; });
        }
    }
}