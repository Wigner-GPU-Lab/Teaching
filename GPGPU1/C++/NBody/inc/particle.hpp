#ifndef PARTICLE_HPP
#define PARTICLE_HPP

// STL includes
#include <array>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <utility>

#define G 6.67384e-11

struct input_particle
{
    double mass;
    std::array<double, 3> pos;
    std::array<double, 3> v;
};

struct particle
{
    particle() = default;
    particle(const particle&) = default;
    particle(particle&&) = default;
    ~particle() = default;

    particle& operator=(const particle&) = default;
    particle& operator=(particle&&) = default;

    particle(const input_particle& in) : mass(in.mass), pos(in.pos), v(in.v), f{ 0, 0, 0 } {}

    double mass;
    std::array<double, 3> pos;
    std::array<double, 3> v;
    std::array<double, 3> f;
};

std::istream& operator >> (std::istream& input, input_particle& part)
{
    input >> part.mass;
    input >> part.pos[0];
    input >> part.pos[1];
    input >> part.pos[2];
    input >> part.v[0];
    input >> part.v[1];
    input >> part.v[2];

    return input;
}

std::istream& operator >> (std::istream& input, particle& part)
{
    input >> part.mass;
    input >> part.pos[0];
    input >> part.pos[1];
    input >> part.pos[2];
    input >> part.v[0];
    input >> part.v[1];
    input >> part.v[2];
    input >> part.f[0];
    input >> part.f[1];
    input >> part.f[2];

    return input;
}

std::vector<particle> read_particle_file(const std::string& filename)
{
    std::vector<particle> result;

    std::ifstream input(filename, std::ios::in);

	if (!input.is_open())
		std::cerr << "Could not open input file " << filename << std::endl;

    std::copy(std::istream_iterator<input_particle>(input),
              std::istream_iterator<input_particle>(),
              std::back_inserter(result));

    return result;
}


std::vector<particle> read_validation_file(const std::string& filename)
{
    std::vector<particle> result;

    std::ifstream input(filename, std::ios::in);

    if (!input.is_open())
        std::cerr << "Could not open validation file " << filename << std::endl;

    std::copy(
        std::istream_iterator<particle>(input),
        std::istream_iterator<particle>(),
        std::back_inserter(result));

    return result;
}


template <typename IteratorType>
void write_validation_file(IteratorType first, IteratorType last, const std::string& filename)
{
    std::stringstream stream;

	stream.precision( 20 );

    std::for_each(first, last, [&](const particle& part)
    {
        stream
            << part.mass << " "
            << part.pos.at(0) << " "
            << part.pos.at(1) << " "
            << part.pos.at(2) << " "
            << part.v.at(0) << " "
            << part.v.at(1) << " "
            << part.v.at(2) << " "
            << part.f.at(0) << " "
            << part.f.at(1) << " "
            << part.f.at(2) << std::endl;
    });

    std::ofstream output(filename);

	output.precision( 20 );

    output << stream.rdbuf();

    output.close();
}

template <typename IteratorType>
bool validate(IteratorType first, IteratorType last, const std::string& filename)
{
    std::vector<particle> particles(first, last);
    auto reference = read_validation_file(filename);
    double tolerance = 1e-3;

    if (particles.size() != reference.size()) return false;

    std::sort(particles.begin(), particles.end(), [](const particle& lhs, const particle& rhs) { return lhs.mass < rhs.mass; });
    std::sort(reference.begin(), reference.end(), [](const particle& lhs, const particle& rhs) { return lhs.mass < rhs.mass; });

    auto match = std::mismatch(particles.cbegin(), particles.cend(), reference.cbegin(), /*reference.cend(),*/ [&](const particle& part, const particle& ref)
    {
        return (std::abs((part.mass - ref.mass) / ref.mass) < tolerance) &&
            (std::abs((part.pos.at(0) - ref.pos.at(0)) / ref.pos.at(0)) < tolerance) &&
            (std::abs((part.pos.at(1) - ref.pos.at(1)) / ref.pos.at(1)) < tolerance) &&
            (std::abs((part.pos.at(2) - ref.pos.at(2)) / ref.pos.at(2)) < tolerance) &&
            (std::abs((part.v.at(0) - ref.v.at(0)) / ref.v.at(0)) < tolerance) &&
            (std::abs((part.v.at(1) - ref.v.at(1)) / ref.v.at(1)) < tolerance) &&
            (std::abs((part.v.at(2) - ref.v.at(2)) / ref.v.at(2)) < tolerance)/* &&
            (std::abs((part.f.at(0) - ref.f.at(0)) / ref.f.at(0)) < tolerance) &&
            (std::abs((part.f.at(1) - ref.f.at(1)) / ref.f.at(1)) < tolerance) &&
            (std::abs((part.f.at(2) - ref.f.at(2)) / ref.f.at(2)) < tolerance)*/;
    });

    if ((match.first != particles.cend()) || (match.second != reference.cend()))
    {
        std::cerr << "Mismatch found at " << std::distance(particles.cbegin(), match.first) << std::endl;

		std::cerr << "Output particle: " << std::endl;
		std::cerr << "\tmass = " << match.first->mass << std::endl;
		std::cerr << "\tposx = " << match.first->pos.at(0) << std::endl;
		std::cerr << "\tposy = " << match.first->pos.at(1) << std::endl;
		std::cerr << "\tposz = " << match.first->pos.at(2) << std::endl;
		std::cerr << "\tvelx = " << match.first->v.at(0) << std::endl;
		std::cerr << "\tvely = " << match.first->v.at(1) << std::endl;
		std::cerr << "\tvelz = " << match.first->v.at(2) << std::endl;
		std::cerr << "\tforx = " << match.first->f.at(0) << std::endl;
		std::cerr << "\tfory = " << match.first->f.at(1) << std::endl;
		std::cerr << "\tforz = " << match.first->f.at(2) << std::endl;

		std::cerr << "Reference particle: " << std::endl;
		std::cerr << "\tmass = " << match.second->mass << std::endl;
		std::cerr << "\tposx = " << match.second->pos.at(0) << std::endl;
		std::cerr << "\tposy = " << match.second->pos.at(1) << std::endl;
		std::cerr << "\tposz = " << match.second->pos.at(2) << std::endl;
		std::cerr << "\tvelx = " << match.second->v.at(0) << std::endl;
		std::cerr << "\tvely = " << match.second->v.at(1) << std::endl;
		std::cerr << "\tvelz = " << match.second->v.at(2) << std::endl;
		std::cerr << "\tforx = " << match.second->f.at(0) << std::endl;
		std::cerr << "\tfory = " << match.second->f.at(1) << std::endl;
		std::cerr << "\tforz = " << match.second->f.at(2) << std::endl;

		std::cerr << "Normalized deviation: " << std::endl;
		std::cerr << "\tmass = " << std::abs((match.first->mass - match.second->mass) / match.second->mass) << std::endl;
		std::cerr << "\tposx = " << std::abs((match.first->pos.at(0) - match.second->pos.at(0)) / match.second->pos.at(0)) << std::endl;
		std::cerr << "\tposy = " << std::abs((match.first->pos.at(1) - match.second->pos.at(1)) / match.second->pos.at(1)) << std::endl;
		std::cerr << "\tposz = " << std::abs((match.first->pos.at(2) - match.second->pos.at(2)) / match.second->pos.at(2)) << std::endl;
		std::cerr << "\tvelx = " << std::abs((match.first->v.at(0) - match.second->v.at(0)) / match.second->v.at(0)) << std::endl;
		std::cerr << "\tvely = " << std::abs((match.first->v.at(1) - match.second->v.at(1)) / match.second->v.at(1)) << std::endl;
		std::cerr << "\tvelz = " << std::abs((match.first->v.at(2) - match.second->v.at(2)) / match.second->v.at(2)) << std::endl;
		std::cerr << "\tforx = " << std::abs((match.first->f.at(0) - match.second->f.at(0)) / match.second->f.at(0)) << std::endl;
		std::cerr << "\tfory = " << std::abs((match.first->f.at(1) - match.second->f.at(1)) / match.second->f.at(1)) << std::endl;
		std::cerr << "\tforz = " << std::abs((match.first->f.at(2) - match.second->f.at(2)) / match.second->f.at(2)) << std::endl;

        return false;
    }

    return true;
}

double cube(const double& val)
{
    return val * val * val;
}

std::array<double, 3> burning_calculate_force(const particle& first, const particle& second)
{
    return
    {
        -G * first.mass * second.mass * (first.pos.at(0) - second.pos.at(0)) / cube(std::sqrt((first.pos.at(0) - second.pos.at(0))*(first.pos.at(0) - second.pos.at(0)) + (first.pos.at(1) - second.pos.at(1))*(first.pos.at(1) - second.pos.at(1)) + (first.pos.at(2) - second.pos.at(2))*(first.pos.at(2) - second.pos.at(2)))),
        -G * first.mass * second.mass * (first.pos.at(1) - second.pos.at(1)) / cube(std::sqrt((first.pos.at(0) - second.pos.at(0))*(first.pos.at(0) - second.pos.at(0)) + (first.pos.at(1) - second.pos.at(1))*(first.pos.at(1) - second.pos.at(1)) + (first.pos.at(2) - second.pos.at(2))*(first.pos.at(2) - second.pos.at(2)))),
        -G * first.mass * second.mass * (first.pos.at(2) - second.pos.at(2)) / cube(std::sqrt((first.pos.at(0) - second.pos.at(0))*(first.pos.at(0) - second.pos.at(0)) + (first.pos.at(1) - second.pos.at(1))*(first.pos.at(1) - second.pos.at(1)) + (first.pos.at(2) - second.pos.at(2))*(first.pos.at(2) - second.pos.at(2))))
    };
}
/*
std::array<double, 3> burning_calculate_force(const particle& first, const particle& second)
{
    return
    {
        -G * first.mass * second.mass * (first.pos.at(0) - second.pos.at(0)) / std::pow(std::sqrt((first.pos.at(0) - second.pos.at(0))*(first.pos.at(0) - second.pos.at(0)) + (first.pos.at(1) - second.pos.at(1))*(first.pos.at(1) - second.pos.at(1)) + (first.pos.at(2) - second.pos.at(2))*(first.pos.at(2) - second.pos.at(2))), 3),
        -G * first.mass * second.mass * (first.pos.at(1) - second.pos.at(1)) / std::pow(std::sqrt((first.pos.at(0) - second.pos.at(0))*(first.pos.at(0) - second.pos.at(0)) + (first.pos.at(1) - second.pos.at(1))*(first.pos.at(1) - second.pos.at(1)) + (first.pos.at(2) - second.pos.at(2))*(first.pos.at(2) - second.pos.at(2))), 3),
        -G * first.mass * second.mass * (first.pos.at(2) - second.pos.at(2)) / std::pow(std::sqrt((first.pos.at(0) - second.pos.at(0))*(first.pos.at(0) - second.pos.at(0)) + (first.pos.at(1) - second.pos.at(1))*(first.pos.at(1) - second.pos.at(1)) + (first.pos.at(2) - second.pos.at(2))*(first.pos.at(2) - second.pos.at(2))), 3)
    };
}
*/
std::array<double, 3> calculate_force(const particle& first, const particle& second)
{
    std::array<double, 3> dr
    {
        first.pos.at(0) - second.pos.at(0),
        first.pos.at(1) - second.pos.at(1),
        first.pos.at(2) - second.pos.at(2)
    };

	double dr_sq = dr.at( 0 )*dr.at( 0 ) + dr.at( 1 )*dr.at( 1 ) + dr.at( 2 )*dr.at( 2 );
    double dr_sq_sqrt = std::sqrt(dr_sq);
	double cube_dr_sq_sqrt = dr_sq_sqrt * dr_sq_sqrt * dr_sq_sqrt;
    double f = -G * first.mass * second.mass / cube_dr_sq_sqrt;

    return { f * dr.at(0), f * dr.at(1), f * dr.at(2) };
}

std::array<double, 3> cutoff_calculate_force(const particle& first, const particle& second, double cutoff_sq)
{
	std::array<double, 3> dr
	{
		first.pos.at(0) - second.pos.at(0),
		first.pos.at(1) - second.pos.at(1),
		first.pos.at(2) - second.pos.at(2)
	};

	double dr_sq = dr.at(0)*dr.at(0) + dr.at(1)*dr.at(1) + dr.at(2)*dr.at(2);

	if (dr_sq > cutoff_sq) return{ 0.0, 0.0, 0.0 };

	double dr_sq_sqrt = std::sqrt(dr_sq);
	double cube_dr_sq_sqrt = dr_sq_sqrt * dr_sq_sqrt * dr_sq_sqrt;
	double f = -G * first.mass * second.mass / cube_dr_sq_sqrt;

	return{ f * dr.at(0), f * dr.at(1), f * dr.at(2) };
}

void forward_euler(particle& part, const double& dt)
{
    auto dt_per_m = dt / part.mass;
    auto dt_sq = dt*dt;
    auto half_dt_sq_per_m = dt_sq / part.mass;

    part.pos.at(0) += part.v.at(0) * dt + part.f.at(0) * half_dt_sq_per_m;
    part.pos.at(1) += part.v.at(1) * dt + part.f.at(1) * half_dt_sq_per_m;
    part.pos.at(2) += part.v.at(2) * dt + part.f.at(2) * half_dt_sq_per_m;

    part.v.at(0) += part.f.at(0) * dt_per_m;
    part.v.at(1) += part.f.at(1) * dt_per_m;
    part.v.at(2) += part.f.at(2) * dt_per_m;

	part.f = { 0.0, 0.0, 0.0 };
}

template <typename RndIt>
std::pair<RndIt, RndIt> mask_range( RndIt first1, RndIt last1, RndIt first2, RndIt last2 )
{
	std::pair<RndIt, RndIt> result;

	// mask_range assumes that the input ranges are well-formed, meaning
	// they are not reversed, and they originate from the same container.

	// If the input and the mask ranges don't overlap, return null-range
	if ( ((last2 < first1) || (last1 < first2)) )
	{
		result = std::make_pair( first1, first1 );
	}
	// otherwise return intersection
	else
	{
		// NOTE: intersection may be null-range if the ranges only overlap on "begin" and "end"
		result = std::make_pair( std::max( first1, first2 ), std::min( last1, last2 ) );
	}

	return result;
}

#endif // PARTICLE_HPP
