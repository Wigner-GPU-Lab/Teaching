// STL includes
#include <chrono>
#include <numeric>
#include <iostream>

// TCLAP includes
#include <tclap/CmdLine.h>

// Manybody includes
#include <particle.hpp>

int main( int argc, char** argv )
{
	std::string banner = "Manybody v5 serial: (N^2)/2, cache aware, unaliased, minimal calculating";
	TCLAP::CmdLine cli( banner );

	TCLAP::ValueArg<std::string> input_arg( "i", "input", "Path to input file", true, "./", "path" );
	TCLAP::ValueArg<std::string> output_arg( "o", "output", "Path to output file", false, "", "path" );
	TCLAP::ValueArg<std::string> validate_arg( "v", "validate", "Path to validation file", false, "", "path" );
	TCLAP::ValueArg<std::size_t> iterate_arg( "n", "", "Number of iterations to take", false, 1, "positive integral" );
	TCLAP::SwitchArg quiet_arg( "q", "quiet", "Suppress standard output", false );

	cli.add( input_arg );
	cli.add( output_arg );
	cli.add( validate_arg );
	cli.add( iterate_arg );
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
	const size_t cache_size = 1000;

	if ( particles.size() % cache_size != 0 )
	{
		std::cerr << "cache size is not a divisor of particle count" << std::endl;
		exit( EXIT_FAILURE );
	}

	{
		if ( !quiet_arg.getValue() ) std::cout << banner << std::endl;
		auto start = std::chrono::high_resolution_clock::now();

		for ( std::size_t n = 0; n < iterate_arg.getValue(); ++n )
		{
			for ( auto cache = particles.begin(); cache != particles.end(); cache += cache_size )
			{
				for ( auto IT = particles.begin(); IT != particles.end(); ++IT )
				{
					auto masked_before_IT = mask_range( particles.begin(), IT, cache, cache + cache_size );

					for ( auto it = masked_before_IT.first; it != masked_before_IT.second; ++it )
					{
						auto force = calculate_force( *IT, *it );

						IT->f.at( 0 ) += force.at( 0 );
						IT->f.at( 1 ) += force.at( 1 );
						IT->f.at( 2 ) += force.at( 2 );

						it->f.at( 0 ) -= force.at( 0 );
						it->f.at( 1 ) -= force.at( 1 );
						it->f.at( 2 ) -= force.at( 2 );
					}
				}
			}

			for ( auto& part : particles )
				forward_euler( part, 0.001 );
		}

		auto end = std::chrono::high_resolution_clock::now();
		if ( !quiet_arg.getValue() ) std::cout << "Computation took " << std::chrono::duration_cast<std::chrono::milliseconds>(end.time_since_epoch() - start.time_since_epoch()).count() << " milliseconds." << std::endl;
	}

	{
		if ( output_arg.getValue() != "" )
		{
			if ( !quiet_arg.getValue() ) std::cout << "Exporting results into " << output_arg.getValue() << std::endl;

			write_validation_file( particles.cbegin(), particles.cend(), output_arg.getValue() );
		}
	}

	{
		if ( validate_arg.getValue() != "" )
		{
			if ( !quiet_arg.getValue() ) std::cout << "Validating results against " << validate_arg.getValue() << std::endl;

			if ( !validate( particles.cbegin(), particles.cend(), validate_arg.getValue() ) ) exit( EXIT_FAILURE );
		}
	}

	return 0;
}