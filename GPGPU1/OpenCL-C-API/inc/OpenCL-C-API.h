#ifndef HEADER_HPP
#define HEADER_HPP

// C standard includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

// OpenCL includes
#include <CL/cl.h>

// OpenCL error handling function
void checkErr( cl_int err, const char * name );

// Probe for available OpenCL platforms
void probe_platforms();

// Select platform with most DP capable devices
cl_platform_id select_platform();

// Test device for DP capability
cl_bool is_device_DP_capable( cl_device_id device );

// Count devices with DP capability
unsigned int count_dp_capable_devices(cl_platform_id platform);

// Select devices with DP capability
cl_device_id* select_devices( cl_platform_id platform );

// Create standard context
cl_context create_standard_context( cl_device_id* devices );

// Load kernel file
char* load_program_file( const char* filename );

// Build program file for all devices in a context
cl_program build_program_source( cl_context context, const char* source );

// Obtain kernel from program
cl_kernel obtain_kernel( cl_program program, const char* name );


#endif // HEADER_HPP