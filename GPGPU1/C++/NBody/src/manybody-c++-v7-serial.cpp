// STL includes
#include <chrono>
#include <numeric>
#include <iostream>

// TCLAP includes
#include <tclap/CmdLine.h>

// Manybody includes
#include <particle.hpp>
#include <octree.hpp>

struct particle_key_accessor
{
	const std::array<double, 3>& operator()( const particle& in )
	{
		return in.pos;
	}
};

int main( int argc, char** argv )
{
	TCLAP::CmdLine cli( "Manybody v4: Octree" );

	TCLAP::ValueArg<std::string> input_arg( "i", "input", "Path to input file", true, "./", "path" );
	TCLAP::ValueArg<std::string> output_arg( "o", "output", "Path to output file", false, "", "path" );
	TCLAP::ValueArg<std::string> validate_arg( "v", "validate", "Path to validation file", false, "", "path" );
	TCLAP::ValueArg<std::size_t> step_arg( "n", "", "Number of steps to take", false, 1, "positive integral" );
	TCLAP::ValueArg<std::size_t> threshold_arg( "t", "threshold", "Threshold of node size", false, 100, "positive integral" );
	TCLAP::SwitchArg quiet_arg( "q", "quiet", "Suppress standard output", false );

	cli.add( input_arg );
	cli.add( output_arg );
	cli.add( validate_arg );
	cli.add( step_arg );
	cli.add( threshold_arg );
	cli.add( quiet_arg );

	try
	{
		cli.parse( argc, argv );
	}
	catch ( TCLAP::ArgException &e )
	{
		std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
	}

	std::vector<particle> particles( read_particle_file( input_arg.getValue() ) );
	octree<particle, particle_key_accessor> tree( { 0.0, 0.0, 0.0 }, 1e6, threshold_arg.getValue() );

	{
		auto start = std::chrono::high_resolution_clock::now();

		for ( const auto& particle : particles )
			tree.insert( particle );

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Building octree of size " << tree.size() << " with node size of " << threshold_arg.getValue() << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch() - start.time_since_epoch()).count() << " milliseconds." << std::endl;

		std::cout << "Depth of tree " << tree.depth() << std::endl;
		std::cout << "Leaf node count of tree " << tree.nodes() << std::endl;
	}

	std::cout << (tree.begin() != tree.end()) << std::endl;

	{
		std::cout << "Octree" << std::endl;
		auto start = std::chrono::high_resolution_clock::now();

		for ( std::size_t n = 0; n < step_arg.getValue(); ++n )
		{
			// Intra-node N^2
			{
				for ( auto& node : tree )
					for ( auto IT = node.begin(); IT != node.end(); ++IT )
					{
						IT->f = { 0.0, 0.0, 0.0 };

						for ( auto it = node.cbegin(); it != node.cend(); ++it )
						{
							if ( IT != it )
							{
								auto force = calculate_force( *IT, *it );

								IT->f.at( 0 ) += force.at( 0 );
								IT->f.at( 1 ) += force.at( 1 );
								IT->f.at( 2 ) += force.at( 2 );
							}
						}
					}
			}

			// Inter-node N*n^2
			{
				typedef std::pair<double, octree<particle, particle_key_accessor>::pos_type> center_of_mass;

				std::vector<center_of_mass> centers_of_mass( tree.nodes() );


				// Calculate node aggregates (V2)
				std::transform( tree.cbegin(), tree.cend(), centers_of_mass.begin(), centers_of_mass.begin(), []( const octree<particle, particle_key_accessor>::node_type& node, center_of_mass& com )
				{
					if ( node.empty() )
						return center_of_mass( 0.0, { 0.0, 0.0, 0.0 } );
					else
					{
						com.first = node.cbegin()->mass;
						com.second = node.cbegin()->pos;

						com.second.at( 0 ) *= com.first;
						com.second.at( 1 ) *= com.first;
						com.second.at( 2 ) *= com.first;

						std::for_each( std::next( node.cbegin() ), node.cend(), [&node, &com]( const particle& part )
						{
							com.second.at( 0 ) += part.pos.at( 0 ) * part.mass;
							com.second.at( 1 ) += part.pos.at( 1 ) * part.mass;
							com.second.at( 2 ) += part.pos.at( 2 ) * part.mass;

							com.first += part.mass;
						} );

						com.second.at( 0 ) /= com.first;
						com.second.at( 1 ) /= com.first;
						com.second.at( 2 ) /= com.first;

						return com;
					}
				} );

				/*
				// Calculate node aggregates (V1)
				for (auto it = tree.begin(); it != tree.end(); ++it)
				{
				auto i = std::distance(tree.begin(), it);

				for (auto& part : *it)
				centers_of_mass.at(i).first += part.mass;

				centers_of_mass.at(i).second = it->center();

				std::for_each(it->cbegin(), it->cend(), [&](const particle& part)
				{
				centers_of_mass.at(i).second.at(0) += (it->center().at(0) - part.pos.at(0)) * part.mass / centers_of_mass.at(i).first;
				centers_of_mass.at(i).second.at(1) += (it->center().at(1) - part.pos.at(1)) * part.mass / centers_of_mass.at(i).first;
				centers_of_mass.at(i).second.at(2) += (it->center().at(2) - part.pos.at(2)) * part.mass / centers_of_mass.at(i).first;
				});

				//std::cout << "center = {" << it->center().at(0) << " " << it->center().at(1) << " " << it->center().at(2) << "}" << std::endl;
				//std::cout << "mass   = " << centers_of_mass.at(i).first << std::endl;
				//std::cout << "CoM    = {" << centers_of_mass.at(i).second.at(0) << " " << centers_of_mass.at(i).second.at(1) << " " << centers_of_mass.at(i).second.at(2) << "}" << std::endl;
				//std::cout << std::endl;

				}
				*/

				/*
				// Particle interaction with aggregates (omits if() inside loops)
				for (auto IT = tree.begin(); IT != tree.end(); ++IT)
				{
				std::vector<center_of_mass>::size_type i = 0;

				for (auto it = tree.begin(); it != IT; ++it)
				{
				//auto i = std::distance( it, IT );

				for (auto& part : *IT)
				{
				auto force = calculate_force(part, { centers_of_mass.at(i).first, centers_of_mass.at(i).second, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0} });

				part.f.at(0) += force.at(0);
				part.f.at(1) += force.at(1);
				part.f.at(2) += force.at(2);
				}

				++i;
				}

				++i;

				for (auto it = std::next(IT); it != tree.end(); ++it)
				{
				//auto i = std::distance( IT, it );

				for (auto& part : *IT)
				{
				auto force = calculate_force(part, { centers_of_mass.at(i).first, centers_of_mass.at(i).second,{ 0.0, 0.0, 0.0 },{ 0.0, 0.0, 0.0 } });

				part.f.at(0) += force.at(0);
				part.f.at(1) += force.at(1);
				part.f.at(2) += force.at(2);
				}

				++i;
				}
				}
				*/

				// Particle interaction with aggregates
				for ( auto IT = tree.begin(); IT != tree.end(); ++IT )
				{
					std::vector<center_of_mass>::size_type i = 0;

					for ( auto it = tree.cbegin(); it != tree.cend(); ++it )
					{
						if ( IT != it )
						{
							for ( auto& part : *IT )
							{
								auto force = calculate_force( part, { centers_of_mass.at( i ).first, centers_of_mass.at( i ).second,{ 0.0, 0.0, 0.0 },{ 0.0, 0.0, 0.0 } } );

								part.f.at( 0 ) += force.at( 0 );
								part.f.at( 1 ) += force.at( 1 );
								part.f.at( 2 ) += force.at( 2 );
							}
						}

						++i;
					}
				}

			}

			// Forward-Euler
			for ( auto& node : tree )
				for ( auto& part : node )
					forward_euler( part, 0.001 );

		}

		auto end = std::chrono::high_resolution_clock::now();
		std::cout << "Computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch() - start.time_since_epoch()).count() << " milliseconds." << std::endl;
	}

	{
		if ( output_arg.getValue() != "" )
		{
			if ( !quiet_arg.getValue() ) std::cout << "Exporting results into " << output_arg.getValue() << std::endl;

			std::vector<particle> temp;

			for ( auto& node : tree )
				std::copy( node.cbegin(), node.cend(), std::back_inserter( temp ) );

			write_validation_file( temp.cbegin(), temp.cend(), output_arg.getValue() );
		}
	}

	{
		if ( validate_arg.getValue() != "" )
		{
			if ( !quiet_arg.getValue() ) std::cout << "Validating results against " << validate_arg.getValue() << std::endl;

			std::vector<particle> temp;

			for ( auto& node : tree )
				std::copy( node.cbegin(), node.cend(), std::back_inserter( temp ) );

			if ( !validate( temp.cbegin(), temp.cend(), validate_arg.getValue() ) ) exit( EXIT_FAILURE );
		}
	}

	return 0;
}