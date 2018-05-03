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
	std::string banner = "Manybody v3 serial: N^2, cache unaware, unaliased, minimal calculating";
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

	{
		if ( !quiet_arg.getValue() ) std::cout << banner << std::endl;
		auto start = std::chrono::high_resolution_clock::now();

		for ( std::size_t n = 0; n < iterate_arg.getValue(); ++n )
		{
			for ( auto IT = particles.begin(); IT != particles.end(); ++IT )
			{
				decltype(particle::f) force{ 0.0, 0.0, 0.0 };

				for ( auto it = particles.cbegin(); it != IT; ++it )
				{
					auto f = calculate_force( *IT, *it );

					force.at( 0 ) += f.at( 0 );
					force.at( 1 ) += f.at( 1 );
					force.at( 2 ) += f.at( 2 );
				}

				for ( auto it = std::next(IT); it != particles.cend(); ++it )
				{
					auto f = calculate_force( *IT, *it );

					force.at( 0 ) += f.at( 0 );
					force.at( 1 ) += f.at( 1 );
					force.at( 2 ) += f.at( 2 );
				}

				IT->f = force;
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