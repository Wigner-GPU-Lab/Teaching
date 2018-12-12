// SYCL include
#include <CL/sycl.hpp>
#include <CL/cl.h>
#include <CL/cl_gl.h>

// GLEW include
#include <GL/glew.h>

// GLM includes
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <SFML/Window.hpp>
#include <SFML/OpenGL.hpp>
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Shape.hpp>
#include <SFML/Graphics/RectangleShape.hpp>

// Standard C++ includes
#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <memory>

#include <random>

struct Point
{
	GLfloat x, y, z, m;
};

struct Force
{
	GLfloat x, y, z, rho;
};

struct Velocity
{
	GLfloat vx, vy, vz, rho;
};

struct Color
{
	GLfloat r, g, b, a;
};

const float Pi = 3.1415926535f;

inline const auto sq = [](auto x){ return x*x; };
inline const auto cube = [](auto x){ return x*x*x; };
inline const auto hyp = [](auto x, auto y, auto z){ return cl::sycl::sqrt(x*x+y*y+z*z); };
inline const auto dist = [](auto x1, auto y1, auto z1, auto x2, auto y2, auto z2){ return hyp(x1-x2, y1-y2, z1-z2); };
inline const auto dot = [](auto x1, auto y1, auto z1, auto x2, auto y2, auto z2){ return x2*x1+y2*y1+z2*z1; };

class SYCL_Force_Kernel;
class SYCL_Step_Kernel;

constexpr cl::sycl::access::mode sycl_read = cl::sycl::access::mode::read;
constexpr cl::sycl::access::mode sycl_write = cl::sycl::access::mode::write;
constexpr cl::sycl::access::mode sycl_discard_write = cl::sycl::access::mode::discard_write;
constexpr cl::sycl::access::mode sycl_read_write = cl::sycl::access::mode::read_write;
using V4 = cl::sycl::vec<float, 4>;

auto event_time( cl::sycl::event const& e )//milliseconds
{
	return (e.get_profiling_info<cl::sycl::info::event_profiling::command_end>() - e.get_profiling_info<cl::sycl::info::event_profiling::command_start>())/1'000'000.0;
}

inline auto clErr(cl_int e, std::string const& success_text = std::string{})
{
	if(e != CL_SUCCESS)
	{
		std::cout << "OpenCL error: " << e << "\n";
		int i; std::cin >> i;
	}
	else if(success_text.size() > 0){ std::cout << success_text << "\n"; }
};

inline bool glErr(std::string const& step_name)
{
	int Error;
	if((Error = glGetError()) != GL_NO_ERROR)
	{
		std::string ErrorString;
		switch(Error)
		{
		case GL_INVALID_ENUM:
			ErrorString = "GL_INVALID_ENUM";
			break;
		case GL_INVALID_VALUE:
			ErrorString = "GL_INVALID_VALUE";
			break;
		case GL_INVALID_OPERATION:
			ErrorString = "GL_INVALID_OPERATION";
			break;
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			ErrorString = "GL_INVALID_FRAMEBUFFER_OPERATION";
			break;
		case GL_OUT_OF_MEMORY:
			ErrorString = "GL_OUT_OF_MEMORY";
			break;
		default:
			ErrorString = "UNKNOWN";
			break;
		}
		std::cout << step_name << " encountered an OpenGL error: " << ErrorString << "\n";
		int i; std::cin >> i;
	}
	else{ std::cout << step_name << " success\n"; }
	return Error == GL_NO_ERROR;
}

const char* vertex_shader_str = R"raw(
#version 330

// VS locations
#define POSITION	0
#define COLOR		1

// FS locations
#define FRAG_COLOR	0

layout(location = POSITION) in vec4 Pin;
layout(location = COLOR) in vec4 in_Color;

out block
{
	vec4 Position;
	vec4 Color;
} VS_Out;

uniform mat4 matWorld;
uniform mat4 matView;
uniform mat4 matProj;

void main()
{
	gl_Position = matProj * matView * matWorld * vec4(Pin.xyz, 1.0f);
	VS_Out.Position = gl_Position;
	VS_Out.Color = in_Color;
}
)raw";

const char* fragment_shader_str = R"raw(
#version 330

// VS locations
#define POSITION	0
#define COLOR		1

// FS locations
#define FRAG_COLOR	0

in block
{
	vec4 Position;
	vec4 Color;
} FS_In;

// FS index = 0 kimenete a gl_FragColor, de igazabol foloseges igy allitani
//layout(location = FRAG_COLOR, index = 0) out vec4 Color;

void main()
{
	gl_FragColor = FS_In.Color;
//	Color = FS_In.Color;
}
)raw";

struct PhysParams
{
	//Liquid
	static const double viscosity() { return 20.0; }
	static const double rho0()      { return 0.20; }

	//Gas
	//static const double viscosity() { return 100.01; }
	//static const double rho0()      { return 0.002; }

	static const double h(){ return 0.5; }
	static const double G(){ return 0.8; }

	static const double dt(){ return 0.00004; }
};

//Gas equation of state
/*inline const auto EoS = [](auto rho)->float
{ 
	auto p = 1000.0*(rho - PhysParams().rho0());
	return (p < 0.0f) ? 0.0f : (float)p;
};*/

//Liquid equation of state
inline const auto EoS = [](auto rho)->float
{
	auto u = rho/PhysParams().rho0();
	auto p = 5e-16*PhysParams().rho0() / 7.0 * (cl::sycl::pow(u, 7.0) - 1.0)+0.0;
	return (p < 0.0f) ? 0.0f : (float)p;
};

int main()
{
	static const int nParticles = (1 << 11) + 0*(1 << 11);// (1 << 14);//1 << 16; //8192;//4*16384;
	std::cout << "Program Started with " << nParticles << " particles.\n";

	std::vector<Point>	        points(nParticles);
	std::vector<Velocity>		velocities(nParticles);
	std::vector<Force>		    forces(nParticles);
	std::vector<Color>			colors(nParticles);
	{
		std::mt19937 eng;
		eng.seed(std::random_device()());

		std::normal_distribution<float>       dnormal(0.0f, 2.0f);
		std::uniform_real_distribution<float> duniform_r(0.0f, 1.0f);
		std::uniform_real_distribution<float> duniform_ang(0.0f, 360.0f);

		std::uniform_real_distribution<float> duniform2(1.0f, 50.0f);
		std::exponential_distribution<float>  dexp(1.2f);
		std::uniform_real_distribution<float> duniform3(-0.5f, 0.5f);

		for(size_t n = 0; n<nParticles; ++n)
		{
			auto& p = points[n];
			p.m = 2048.0f/nParticles;

			auto ch = (dnormal(eng) < 0.0f) ? 1.0f : -1.0f;

			auto r0     = duniform_r(eng)*0.8f;
			auto angle = duniform_ang(eng)/180.0f * Pi;

			p.x = duniform3(eng);
			p.y = 40.0f*duniform_r(eng);
			p.z = duniform3(eng);

			auto r = hyp(p.x, p.y, p.z);


			velocities[n].vx = -0*ch*1.40f;
			velocities[n].vy = 0.0f;
			velocities[n].vz = 0*ch * 0.2f;

			colors[n] = Color{1.0f, 0.5f, 0.2f, 1.0f};

			velocities[n].rho = 1.0f;
			forces[n].rho = 1.0f;
		}
	}

	sf::RenderWindow window(sf::VideoMode(1024, 768),
		"SPH with " + std::to_string(nParticles) + " particles",
		sf::Style::Default,
		sf::ContextSettings(24, 8, 1, 3, 3, sf::ContextSettings::Attribute::Core));

	if( sf::Uint32(window.getSettings().majorVersion * 10 + window.getSettings().majorVersion) < 33 )
	{
		std::cerr << "Highest OpenGL version is " << window.getSettings().majorVersion << "." << window.getSettings().minorVersion << " Exiting..." << std::endl;
		std::exit(EXIT_FAILURE);
	}

	if( glewInit() != GLEW_OK ){ std::cout << "Cannot initialize glew.\n"; return -1; }
	else{ std::cout << "glew initialized.\n"; }

	const int iPlatform = 0;
	const int iDevice = 0;

	GLint  gl_status;			// Hiba allapot
	GLuint vertexShaderObj;		// Vertex arnyalo objektum
	GLuint fragmentShaderObj;	// Fragment arnyalo objektum
	GLuint glProgram;			// OpenGL program
	GLuint glbPoint[2];			// Point Buffer Object
	GLuint glbColor;			// Color Buffer Object
	GLuint glvao;				// Vertex Array Object

	glm::mat4 mWorld;		// Vilag matrix
	glm::mat4 mView;		// Nezeti matrix
	glm::mat4 mProj;		// Vetitesi matrix

	glm::vec3 vEye;
	float eye_dist = 10.0f;
	float cam_angle_x = 00.0f / 180.0 * Pi;
	float cam_angle_y = 00.0f / 180.0 * Pi;

	GLint worldMatrixLocation;		// Vilag matrix helye az arnyalo kodjaban
	GLint viewMatrixLocation;		// Nezeti matrix helye az arnyalo kodjaban
	GLint projectionMatrixLocation;	// Vetitesi matrix helye az arnyalo kodjaban

	cl_platform_id		platform;
	cl_device_id		device;
	cl_context			cl_ctx;
	cl_command_queue	cl_queue;
	cl_mem				clbPoint[2];
	cl_mem				clbColor;

	cl::sycl::context	sycl_ctx;
	cl::sycl::queue		sycl_queue;

	struct SYCL_Buffers
	{
		std::unique_ptr<cl::sycl::buffer<Point,        1>> points[2];
		std::unique_ptr<cl::sycl::buffer<Velocity,     1>> velocities[2];
		std::unique_ptr<cl::sycl::buffer<Force,        1>> forces[2];
		std::unique_ptr<cl::sycl::buffer<Color,        1>> colors;
	};
	SYCL_Buffers sycl_buffers;

	{
		std::vector<cl_platform_id> platforms;
		std::vector<cl_device_id>   devices;
		std::string text;	text.resize(256);
		cl_uint n = 0;

		clErr(clGetPlatformIDs(0, nullptr, &n));
		std::cout << "Number of platforms: " << n << "\n";
		
		platforms.resize(n);

		clErr(clGetPlatformIDs((cl_uint)platforms.size(), platforms.data(), nullptr));

		platform = platforms[iPlatform];
		
		clErr(clGetPlatformInfo(platform, CL_PLATFORM_NAME, text.size(), (void*)text.data(), 0));
		std::cout << "Selected platform index: " << iPlatform << " name: " << text.data() << "\n";

		clErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &n));
		std::cout << "Number of GPU devices: " << n << "\n";

		devices.resize(n);

		clErr(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, (cl_uint)devices.size(), devices.data(), &n));

		device = devices[iDevice];

		clErr(clGetDeviceInfo(device, CL_DEVICE_NAME, text.size(), (void*)text.data(), 0));
		std::cout << "Selected GPU device index: " << iDevice << " name: " << text.data() << "\n";
	}

	auto load_and_compile_shader = [](auto shader_type, std::string const& path)->GLuint
	{
		std::basic_string<GLchar> string;

		if(path.size() != 0)
		{
			std::basic_ifstream<GLchar> file(path);
			if(!file.is_open()){ std::cout << "Cannot open shader file: " << path << "\n"; return 0; }
			string = std::basic_string<GLchar>( std::istreambuf_iterator<GLchar>(file), (std::istreambuf_iterator<GLchar>()));
		}
		else
		{
			string = std::basic_string<GLchar>{ shader_type == GL_VERTEX_SHADER ? vertex_shader_str : fragment_shader_str };
		}
		std::vector<const GLchar*> c_strings{ string.c_str() };

		auto shaderObj = glCreateShader(shader_type);
		if(!glErr("glCreateShader")){ return 0; }
		
		GLint gl_status = 0;
		glShaderSource(shaderObj, (GLsizei)c_strings.size(), c_strings.data(), NULL);
		glCompileShader(shaderObj);
		glGetShaderiv(shaderObj, GL_COMPILE_STATUS, &gl_status);

		if (!gl_status)
		{
			GLint log_size;
			glGetShaderiv(shaderObj, GL_INFO_LOG_LENGTH, &log_size);
			std::basic_string<GLchar> log(log_size, ' ');
			glGetShaderInfoLog(shaderObj, log_size, NULL, &(*log.begin()));
			std::cout << "Failed to compile shader: " << std::endl << log << std::endl;
		}
		else
		{
			std::cout << "Shader " << path << " compiled successfully\n";
		}

		return shaderObj;
	};

	vertexShaderObj   = load_and_compile_shader(GL_VERTEX_SHADER,   "");
	fragmentShaderObj = load_and_compile_shader(GL_FRAGMENT_SHADER, "");

	if(!vertexShaderObj || !fragmentShaderObj){ return -1; }

	glProgram = glCreateProgram();

	glAttachShader(glProgram, vertexShaderObj);
	glAttachShader(glProgram, fragmentShaderObj);

	glLinkProgram(glProgram);

	gl_status = 0;
	glGetProgramiv(glProgram, GL_LINK_STATUS, &gl_status);

	if(!gl_status)
	{
		char temp[256];
		glGetProgramInfoLog(glProgram, 256, 0, temp);
		std::cout << "Failed to link program: " << temp << std::endl;
		glDeleteProgram(glProgram);
	}
	else{ std::cout << "Shaders linked successfully\n"; }

	glUseProgram(glProgram);
	glErr("glUseProgram");

	// Generate buffers
	auto glSetupBuffers = [&](GLuint& buffer, auto const& datavec)
	{
		glGenBuffers(1, &buffer); glErr("glGenBuffers(buffer)");

		// Buffer objektum hasznalatba vetele
		glBindBuffer(GL_ARRAY_BUFFER, buffer); glErr("glBindBuffer(buffer)");

		// Memoria lefoglalasa, de meg ne masoljunk bele.
		glBufferData(GL_ARRAY_BUFFER, datavec.size() * sizeof(datavec[0]), NULL, GL_STATIC_DRAW); glErr("glBufferData()");

		// Toltsuk fel a buffer (jelen esetben egeszet) a geometriaval
		glBufferSubData(GL_ARRAY_BUFFER, 0, datavec.size() * sizeof(datavec[0]), datavec.data()); glErr("glBufferSubData()");

		// Buffer feltoltese utan hasznalat vege
		glBindBuffer(GL_ARRAY_BUFFER, 0); glErr("glBindBuffer(0)");
	};

	glSetupBuffers(glbPoint[0], points);
	glSetupBuffers(glbPoint[1], points);
	glSetupBuffers(glbColor, colors);

	// Osszefogo VAO letrehozasa es hasznalatba vetele
	glGenVertexArrays(1, &glvao); glErr("glGenVertexArrays(vao)");
	glBindVertexArray(glvao);	  glErr("glBindVertexArrays(vao)");

	// Belso buffer aktivalasa
	glBindBuffer(GL_ARRAY_BUFFER, glbPoint[0]); glErr("glBindBuffer(glbPoint)");
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Point), (GLvoid *)0);	glErr("glVertexAttribPointer(position)");
	glBindBuffer(GL_ARRAY_BUFFER, glbColor); glErr("glBindBuffer(glbColor)");
	glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE,	sizeof(Color), (GLvoid *)0);	glErr("glVertexAttribPointer(color)");
	
	// Vertex attributum indexek aktivalasa
	glEnableVertexAttribArray(0); glErr("glEnableVertexAttribArray(0)");
	glEnableVertexAttribArray(1); glErr("glEnableVertexAttribArray(1)");

	glEnable(GL_BLEND);
	glDepthMask(GL_TRUE);
	glPointSize(3.0f);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	{
		vEye = glm::vec3(0.f, 0.f, eye_dist);

		// Matrixok beallitasa
		mWorld = glm::mat4(1.0f); // Modell es vilag koordinatak megegyeznek

		mWorld = glm::rotate(mWorld, cam_angle_x, glm::vec3(0, 1, 0));
		mWorld = glm::rotate(mWorld, cam_angle_y, glm::vec3(1, 0, 0));

		mView = glm::lookAt(vEye, glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));

		mProj = glm::perspective(45.0f,                                            // 90 fokos nyilasszog
			((float)window.getSize().x) / window.getSize().y, // ablakmereteknek megfelelo nezeti arany
			0.01f,                                            // Kozeli vagosik
			1000.0f);                                         // Tavoli vagosik

															  // Arnyalo uniformis valtozoinak gazda oldali leiroinak letrehozasa
		worldMatrixLocation      = glGetUniformLocation(glProgram, "matWorld"); glErr("glGetUniformLocation(matWorld)");
		viewMatrixLocation       = glGetUniformLocation(glProgram, "matView");  glErr("glGetUniformLocation(matView)");
		projectionMatrixLocation = glGetUniformLocation(glProgram, "matProj");  glErr("glGetUniformLocation(matProj)");

		// Matrixok beallitasa az arnyalokban
		glUniformMatrix4fv(worldMatrixLocation,      1, GL_FALSE, &mWorld[0][0]); glErr("glUniformMatrix4fv(worldMatrix)");
		glUniformMatrix4fv(viewMatrixLocation,       1, GL_FALSE, &mView [0][0]); glErr("glUniformMatrix4fv(viewMatrix)");
		glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &mProj [0][0]); glErr("glUniformMatrix4fv(projectionMatrix)");

	}

	auto CreateClStuff = [&]
	{
		auto glContext = wglGetCurrentContext();
		auto gldc      = wglGetCurrentDC();

		cl_context_properties cps[] =
		{
			CL_CONTEXT_PLATFORM, (cl_context_properties) platform,
			CL_GL_CONTEXT_KHR,   (cl_context_properties) glContext,
			CL_WGL_HDC_KHR,      (cl_context_properties) gldc,
			  0
		};

		{
			cl_int status = 0;
			cl_ctx = clCreateContext(cps, 1, &device, 0, 0, &status);
			clErr(status, "OpenCL Context created successfully");

			cl_queue_properties qprop[] = {(cl_queue_properties)CL_QUEUE_PROPERTIES, (cl_queue_properties)CL_QUEUE_PROFILING_ENABLE, (cl_queue_properties)0};

			cl_queue = clCreateCommandQueueWithProperties(cl_ctx, device, qprop, &status);
			clErr(status, "OpenCL Command Queue created successfully");

			clbPoint[0] = clCreateFromGLBuffer( cl_ctx, CL_MEM_READ_WRITE, glbPoint[0], &status );
			clErr(status);

			clbPoint[1] = clCreateFromGLBuffer( cl_ctx, CL_MEM_READ_WRITE, glbPoint[1], &status );
			clErr(status);

			clbColor = clCreateFromGLBuffer( cl_ctx, CL_MEM_READ_WRITE, glbColor, &status );
			clErr(status);
		}

		try
		{
			sycl_ctx = cl::sycl::context(cl_ctx);
			sycl_queue = cl::sycl::queue(cl_queue, sycl_ctx);
			std::cout << "SYCL query: selected platform: " << sycl_queue.get_context().get_platform().get_info<cl::sycl::info::platform::name>() << "\n";
			std::cout << "SYCL query: selected device:   " << sycl_queue.get_device().get_info<cl::sycl::info::device::name>() << "\n";

			auto extensions = sycl_ctx.get_devices()[0].get_info<cl::sycl::info::device::extensions>();
			auto cl_khr_gl_event_supported = std::find(extensions.cbegin(), extensions.cend(), "cl_khr_gl_sharing") != extensions.cend();
			std::cout << "Interop is " << std::string{cl_khr_gl_event_supported ? "supported" : "NOT SUPPORTED"} << "\n";
	
			sycl_buffers.points[0] = std::make_unique<cl::sycl::buffer<Point,    1>>(clbPoint[0], sycl_queue);
			sycl_buffers.points[1] = std::make_unique<cl::sycl::buffer<Point,    1>>(clbPoint[1], sycl_queue);
			sycl_buffers.colors    = std::make_unique<cl::sycl::buffer<Color,    1>>(clbColor, sycl_queue);

			sycl_buffers.velocities[0] = std::make_unique<cl::sycl::buffer<Velocity, 1>>(velocities.data(), nParticles);
			sycl_buffers.velocities[1] = std::make_unique<cl::sycl::buffer<Velocity, 1>>(velocities.data(), nParticles);
			sycl_buffers.forces[0]     = std::make_unique<cl::sycl::buffer<Force,    1>>(forces.data(), nParticles);
			sycl_buffers.forces[1]     = std::make_unique<cl::sycl::buffer<Force,    1>>(forces.data(), nParticles);
		}
		catch (cl::sycl::exception e){ std::cout << "Exception encountered in SYCL: " << e.what() << "\n"; return -1; }

		return 0;
	};

	auto sycl_step = [&](PhysParams pp, int iBuffer, float t)
	{
		size_t N = (size_t)nParticles;
		cl::sycl::range<2> R2{N, N};
		cl::sycl::range<1> R{N};
		const static size_t Ls = 16;
		cl::sycl::nd_range<1> nR(N, Ls);
		
		const float h = (float)pp.h();
		const float rech = (float)pow(h, -1.0);
		const float h2 = (float)pow(h, 2.0);
		const float rech3 = (float)pow(h, -3.0);
		const float rech4 = (float)pow(h, -4.0);
		const float rech8 = (float)pow(h, -8.0);

		const float coeff_visc  = (float)(pp.viscosity() * 45.0/Pi*pow((double)h, -5.0));
		const float coeff_press = (float)(1.0  / 2.0 *     45.0/Pi*pow((double)h, -4.0));

		auto boundary = [](Point& p, Velocity& v, float r)
		{
			const float d = 3.5f;
			const float q = 1.0f;
			const float q2 = 0.00001f;
			const float a = 0.1f;
			if(p.y < -1.5f){ p.y = -1.5f+q2; v.vy *= -q; }
			if(p.x > +d)   { p.x = +d-q2; v.vx *= -q; }
			if(p.x < -d)   { p.x = -d+q2; v.vx *= -q; }
			if(p.z > +d*a)   { p.z = +d*a-q2*a; v.vz *= -q; }
			if(p.z < -d*a)   { p.z = -d*a+q2*a; v.vz *= -q; }
			/*if(r > 80.0f)
			{
				p.x *= 0.0f;
				p.y *= 0.0f;
				p.z *= 0.0f;
				vel.vx *= 0.5f;
				vel.vy *= 0.5f;
				vel.vz *= 0.5f;
			}*/
		};

		auto interaction = [&](int buffer)
		{
			return sycl_queue.submit([&](cl::sycl::handler &cgh)
			{
				auto P = sycl_buffers.points[buffer]    ->template get_access<sycl_read>(cgh);
				auto V = sycl_buffers.velocities[buffer]->template get_access<sycl_read>(cgh);
				auto C = sycl_buffers.colors            ->template get_access<sycl_discard_write>(cgh);
				auto F = sycl_buffers.forces[buffer]    ->template get_access<sycl_discard_write>(cgh);

				cl::sycl::accessor<Point,    1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> Lp(cl::sycl::range<1>(Ls), cgh);
				cl::sycl::accessor<Velocity, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> Lv(cl::sycl::range<1>(Ls), cgh);

				cgh.parallel_for<SYCL_Force_Kernel>(nR, [=](cl::sycl::nd_item<1> item)
				{
					auto idx = item.get_global_id(0);
					auto l   = item.get_local_id(0);

					const auto eps = std::numeric_limits<float>::epsilon() * 1000.0f;
					auto fgix = 0.0f;
					auto fgiy = -(float)pp.G()*100000.0f;
					auto fgiz = 0.0f;

					auto fvix = 0.0f;
					auto fviy = 0.0f;
					auto fviz = 0.0f;

					auto fpix = 0.0f;
					auto fpiy = 0.0f;
					auto fpiz = 0.0f;

					auto pi = P[idx];
					auto m  = pi.m;
					auto vi = V[idx];
					auto Pressurei = EoS(vi.rho);
					auto Gmi = (float)(pp.G()*m);
					auto rhoi = 0.0f;

					for(int i=0; i<N/Ls; ++i)
					{
						Lp[l] = P[i*Ls+l];
						Lv[l] = V[i*Ls+l];
						item.barrier(cl::sycl::access::fence_space::local_space);
						auto o = l;
						for(int u=0; u<Ls; ++u)
						{
							const auto pj = Lp[o];
							const auto vj = Lv[o];
							float mj   = pj.m;
							float rhoj = vj.rho;

							o += 1;
							if(o >= Ls){ o = 0; }
							
							auto rij = dist(pi.x, pi.y, pi.z, pj.x, pj.y, pj.z);
							/*auto f = -Gmi * mj / (eps  + cube( rij ));
							fgix = 0.0f;//fgix + (pi.x - pj.x) * f;
							fgiy = -pp.G()*1000.0f*mj;//fgiy + (pi.y - pj.y) * f;
							fgiz = 0.0f;//fgiz + (pi.z - pj.z) * f;*/

							float mask = (rij >= h) ? 0.0f : 1.0f;
							auto qij = rij * rech;
							rhoi += mask * mj * cube(1.0f - sq(qij));
							if(rij >= h){ continue; }

							auto invqij = (1.0f - qij)*mask;
							float Avisc  = -coeff_visc  * mj/rhoj*invqij;
							float Ap     = coeff_press * mj * (Pressurei + EoS(rhoj)) / (eps+rhoj) *   sq(invqij) / (eps+rij);

							fpix += Ap * (pi.x - pj.x);
							fpiy += Ap * (pi.y - pj.y);
							fpiz += Ap * (pi.z - pj.z);

							fvix += (Avisc * (vi.vx - vj.vx));
							fviy += (Avisc * (vi.vy - vj.vy));
							fviz += (Avisc * (vi.vz - vj.vz));
						}
						item.barrier(cl::sycl::access::fence_space::local_space);
					}

					rhoi = (float)(315.0f/64.0f*rhoi/Pi)*rech3;

					float fg = hyp(fgix, fgiy, fgiz);
					float fv = hyp(fvix, fviy, fviz);
					float fp = hyp(fpix, fpiy, fpiz);
					Color col;
					col.r = fp / (fv+fp+fg);
					col.g = fg / (fv+fp+fg);
					col.b = fv / (fv+fp+fg);
					col.a = rhoi/((float)PhysParams().rho0()*4.0f);
					C[idx] = col;
					F[idx] = Force{fgix+fpix+fvix, fgiy+fpiy+fviy, fgiz+fpiz+fviz, rhoi};
				});
			});
		};

		auto step = [&](int buffer)
		{
			sycl_queue.submit([&](cl::sycl::handler &cgh)
			{
				auto Pr = sycl_buffers.points[buffer]      ->template get_access<sycl_read>(cgh);
				auto Vr = sycl_buffers.velocities[buffer]  ->template get_access<sycl_read>(cgh);
				auto Pw = sycl_buffers.points[1-buffer]    ->template get_access<sycl_discard_write>(cgh);
				auto Vw = sycl_buffers.velocities[1-buffer]->template get_access<sycl_read_write>(cgh);
				auto F = sycl_buffers.forces[buffer]       ->template get_access<sycl_read>(cgh);

				cgh.parallel_for<SYCL_Step_Kernel>(R, [=](cl::sycl::item<1> item)
				{
					auto i = item.get_linear_id();

					auto p = Pr[i];
					auto f = F[i];
					auto v = Vr[i];
					auto dthalf = (float)PhysParams().dt() * 0.5f;
					auto recmdt = dthalf / p.m * (2.0f*buffer);

					auto vold = Vw[i];
					auto add_vold = (buffer == 1) ? 1.0f : 0.0f;

					p.x += (v.vx + add_vold*vold.vx) * dthalf;
					p.y += (v.vy + add_vold*vold.vy) * dthalf;
					p.z += (v.vz + add_vold*vold.vz) * dthalf;

					v.vx = v.vx + recmdt * f.x;
					v.vy = v.vy + recmdt * f.y;
					v.vz = v.vz + recmdt * f.z;
					v.rho = f.rho;

					auto r = hyp(p.x, p.y, p.z);
					boundary(p, v, r);

					Pw[i] = p;
					Vw[i] = v;
				});
			});
		};
		
		interaction(iBuffer);
		step(iBuffer);
		sycl_queue.wait();
	};

	
	CreateClStuff();

	float t = 0.0f;

	int iLeapFrogBuffer = 0;

	size_t nFrame = 0;
	bool CL_Is_Initialized = false;

	double forceCalcTime = 0.0;
	auto tlast = std::chrono::high_resolution_clock::now();

	bool lMouse = false;
	//mouse click coords
	int mcx = 0;
	int mcy = 0;
	while (window.isOpen())
	{
		sf::Event event;
		bool quit = false;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				quit = true;

			if ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::Escape))
				quit = true;

			// Resize event: adjust the viewport
			if (event.type == sf::Event::Resized)
			{
				window.setView(sf::View(sf::FloatRect(0, 0, (float)event.size.width, (float)event.size.height)));
				glViewport(0, 0, event.size.width, event.size.height);
				mProj = glm::perspective( 45.0f, ((float)event.size.width)/event.size.height, 0.01f, 1000.0f);
				glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &mProj[0][0]);
			}

			if(event.type == sf::Event::MouseButtonPressed  && event.mouseButton.button == sf::Mouse::Button::Left){ lMouse = true; std::cout << "L\n"; mcx = event.mouseButton.x; mcy = event.mouseButton.y; }
			if(event.type == sf::Event::MouseButtonReleased && event.mouseButton.button == sf::Mouse::Button::Left){ lMouse = false; std::cout << "R\n";  }
			if(event.type == sf::Event::MouseWheelScrolled)
			{
				eye_dist *= (1.0f + event.mouseWheelScroll.delta * 0.05f);
				std::cout << eye_dist << "\n";
				vEye = glm::vec3(0.f, 0.f, eye_dist);
				mView = glm::lookAt(vEye, glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 1.f, 0.f));
				glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &mView [0][0]); glErr("glUniformMatrix4fv(viewMatrix)");
			}

			if(event.type == sf::Event::MouseMoved && lMouse)
			{
				cam_angle_x += 5.0f * (event.mouseMove.x - mcx) * Pi / 180.0f / (1.0f * window.getSize().x);
				cam_angle_y += 5.0f * (event.mouseMove.y - mcy) * Pi / 180.0f / (1.0f * window.getSize().y);

				mWorld = glm::mat4(1.0f);
				mWorld = glm::rotate(mWorld, cam_angle_x, glm::vec3(0, 1, 0));
				mWorld = glm::rotate(mWorld, cam_angle_y, glm::vec3(1, 0, 0));
				glUniformMatrix4fv(worldMatrixLocation, 1, GL_FALSE, &mWorld[0][0]); glErr("glUniformMatrix4fv(worldMatrix)");
			}
		}

		glFinish();

		{
			auto t0 = std::chrono::high_resolution_clock::now();
			cl_mem bs[2] = {clbPoint[1-iLeapFrogBuffer], clbColor};
			clErr(clEnqueueAcquireGLObjects(cl_queue, 2, bs, 0, nullptr, nullptr));
			sycl_step(PhysParams(), iLeapFrogBuffer, t);
			clErr(clEnqueueReleaseGLObjects(cl_queue, 2, bs, 0, nullptr, nullptr));
			clFinish(cl_queue);
			forceCalcTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - t0).count()*1.0;
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glBindVertexArray(glvao);	  //glErr("glBindVertexArrays(vao)");
		glBindBuffer(GL_ARRAY_BUFFER, glbPoint[1-iLeapFrogBuffer]); //glErr("glBindBuffer(glbPoint)");
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Point), (GLvoid *)0);	//glErr("glVertexAttribPointer(position)");
		glDrawArrays(GL_POINTS, 0, (GLsizei)nParticles);
		glFinish();

		window.display();

		nFrame += 1;
		t += (float)PhysParams().dt();
		iLeapFrogBuffer = 1 - iLeapFrogBuffer;

		auto tcurr = std::chrono::high_resolution_clock::now();
		auto tFrame = std::chrono::duration_cast<std::chrono::milliseconds>(tcurr - tlast);
		std::cout << "Frame time: " << tFrame.count() << " mss, Force calculation took: " << (int)forceCalcTime << " mss (" << (int)(forceCalcTime / tFrame.count() * 100) << "%)\n";
		tlast = tcurr;

		if(quit) window.close();
	}

	sycl_buffers.points[0].reset(nullptr);
	sycl_buffers.points[1].reset(nullptr);
	sycl_buffers.forces[0].reset(nullptr);
	sycl_buffers.forces[1].reset(nullptr);
	sycl_buffers.velocities[0].reset(nullptr);
	sycl_buffers.velocities[1].reset(nullptr);
	sycl_buffers.colors.reset(nullptr);

	glDeleteBuffers( 1, &glbPoint[0] );
	glDeleteBuffers( 1, &glbPoint[1] );
	glDeleteBuffers( 1, &glbColor );

	return 0;
}