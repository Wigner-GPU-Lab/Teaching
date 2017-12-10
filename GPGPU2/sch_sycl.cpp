#include <CL/sycl.hpp>
#include <CL/cl_gl.h>
#include <CL/cl_gl_ext.h>

// GLEW include
#include <GL/glew.h>

// SFML includes
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>

// GLM includes
#include <glm/glm.hpp>
#include <glm/ext.hpp>

// C++ Standard includes
#include <iostream>
#include <fstream>
#include <complex>
#include <string>
#include <iterator>
#include <memory>
#include <thread>

struct Float4
{
    float x, y, r, b;
};

template<typename T>
auto sq(T const& x){ return x*x; };

struct CL
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

	cl::sycl::queue sycl_queue;
	cl::sycl::context sycl_context;

    CL():platform{nullptr}, device{nullptr}, context{nullptr}, queue{nullptr}, program{nullptr}, kernel{nullptr}
    {
		cl_platform_id platforms[3];
        auto status = clGetPlatformIDs(3, platforms, NULL);
		platform = platforms[2];
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        auto glContext = wglGetCurrentContext();
        auto gldc = wglGetCurrentDC();
        cl_context_properties cps[] =
        {   CL_GL_CONTEXT_KHR,   (cl_context_properties) glContext,
            CL_WGL_HDC_KHR,      (cl_context_properties) gldc,
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,  0 };
        context = clCreateContext(cps, 1, &device, 0, 0, &status);
		sycl_context = cl::sycl::context(context);
        queue = clCreateCommandQueueWithProperties(context, device, nullptr, &status);
		sycl_queue = cl::sycl::queue(queue, sycl_context);
    }

    ~CL()
    {
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        clReleaseDevice(device);
    }

    void set_kernel(std::string const& fn, std::string const& kernelname)
    {
        cl_int status;
        std::ifstream file(fn);
        std::string source(std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
        size_t      sourceSize = source.size();
        const char* sourcePtr = source.c_str();
        program = clCreateProgramWithSource(context, 1, &sourcePtr, &sourceSize, &status);

        status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
        if (status != CL_SUCCESS)
        {
            size_t len = 0;
            status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
            std::unique_ptr<char[]> log = std::make_unique<char[]>(len);
            status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log.get(), nullptr);
            std::cout << log.get() << "\n";
            MessageBoxA(0, log.get(), nullptr, 0);
        }
        kernel = clCreateKernel(program, kernelname.c_str(), &status);
        if(!kernel){ std::cout << "Kernel FAIL\n"; }
    }

    template<typename T>
    cl::sycl::buffer<T, 1> create_from_gl_buffer( std::vector<T>& data, cl_mem_flags const& flags, GLuint glid )
    {
        cl_int status;
        auto clid = clCreateFromGLBuffer( context, CL_MEM_READ_WRITE, glid, &status );
		return cl::sycl::buffer<T, 1>(clid, sycl_queue, cl::sycl::event());
    }

};

int main()
{
    sf::RenderWindow window(sf::VideoMode(1000, 600), "Hello Schroedinger");
    CL cl;
  
    std::vector<Float4> lines, linesu;

    auto psi0 = [](auto x)
    {
        using namespace std;
        return exp(complex<double>(0.0, 1.0)*14.*(x+3))*exp(-sq(x+3));
    };

    auto u = [](auto x)
    {
        using namespace std;
        return 2.0*exp(-sq(x/0.4)) + 10. * (exp(-sq((x-12.)/2.0)) + exp(-sq((x+12.)/2.0)));
    };

    int n = 600;
    double x0 = -12.0;
    double x1 = +12.0;
    std::vector<double> PsiR;
    std::vector<double> PsiI;
    std::vector<double> U;
    lines.resize(n);
    linesu.resize(n);
    PsiR.resize(2*n);
    PsiI.resize(2*n);
    
    U.resize(n);
    auto dx = (x1-x0)/(n-1);
    for(int i=0; i<n; ++i)
    {
        auto x = x0 + i * dx;
        auto psi = psi0(x);
        PsiR[0*n+i] = psi.real();
        PsiI[0*n+i] = psi.imag();
        U[i] = u(x);

        lines[i].x = (float)x / 12.f;
        lines[i].y = -0.75f;
        lines[i].r = 1.0f;
        lines[i].b = 0.0f;

        linesu[i].x = (float)x / 12.f;
        linesu[i].y = -0.75f + 0.25f*(float)U[i];
        linesu[i].r = 0.0f;
        linesu[i].b = 1.0f;
    }

    // GLEW inicializalas
    if (glewInit() != GLEW_OK) std::exit(EXIT_FAILURE);

	GLuint lines_vbo = 0;
	glGenBuffers(1, &lines_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, lines_vbo);
	glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(Float4), lines.data(), GL_STATIC_DRAW);

    GLuint linesu_vbo = 0;
	glGenBuffers(1, &linesu_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, linesu_vbo);
	glBufferData(GL_ARRAY_BUFFER, linesu.size() * sizeof(Float4), linesu.data(), GL_STATIC_DRAW);

	GLuint vao = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, lines_vbo);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

    GLuint vaou = 0;
	glGenVertexArrays(1, &vaou);
	glBindVertexArray(vaou);
	glBindBuffer(GL_ARRAY_BUFFER, linesu_vbo);
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(0);

	std::ifstream vs_file("C:\\Users\\u235a\\Desktop\\Schrodinger\\shader\\Schrodinger-vertex.glsl");
    std::string vs_code{ std::istreambuf_iterator<char>(vs_file), std::istreambuf_iterator<char>() };
	
	std::ifstream fs_file("C:\\Users\\u235a\\Desktop\\Schrodinger\\shader\\Schrodinger-fragment.glsl");
    std::string fs_code{ std::istreambuf_iterator<char>(fs_file), std::istreambuf_iterator<char>() };

	GLint result = GL_FALSE;
    int logLength;

	auto pvs_code = vs_code.c_str();
	auto pfs_code = fs_code.c_str();

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &pvs_code, NULL);
	glCompileShader(vs);

	glGetShaderiv(vs, GL_COMPILE_STATUS, &result);
    glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &logLength);
    {
	    std::vector<char> ShaderError((logLength > 1) ? logLength : 1);
        glGetShaderInfoLog(vs, logLength, NULL, &ShaderError[0]);
        std::cerr << std::string(ShaderError.cbegin(), ShaderError.cend()) << std::endl;
    }

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &pfs_code, NULL);
	glCompileShader(fs);

	glGetShaderiv(fs, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &logLength);
    {
	    std::vector<char> ShaderError((logLength > 1) ? logLength : 1);
        glGetShaderInfoLog(fs, logLength, NULL, &ShaderError[0]);
        std::cerr << std::string(ShaderError.cbegin(), ShaderError.cend()) << std::endl;
    }

	GLuint shader = glCreateProgram();
	glAttachShader(shader, vs);
	glAttachShader(shader, fs);

	glBindAttribLocation(shader, 0, "vertex_position");
	glLinkProgram(shader);

	cl::sycl::buffer<double, 1> bPsiR(PsiR.data(), PsiR.size());
	cl::sycl::buffer<double, 1> bPsiI(PsiI.data(), PsiI.size());
	cl::sycl::buffer<double, 1> bU(U.data(), U.size());
	
	cl::sycl::buffer<Float4, 1> blines = cl.create_from_gl_buffer(lines, CL_MEM_READ_WRITE, lines_vbo);

	double h = 0.001;//0.0008;
	double m = 30.0;//60.0;
    
    int idx = 0;
    auto step = [&]
    {
		cl.sycl_queue.submit([&](cl::sycl::handler& cgh)
		{
			auto aPsiR  = bPsiR.template get_access<cl::sycl::access::mode::read_write>(cgh);
			auto aPsiI  = bPsiI.template get_access<cl::sycl::access::mode::read_write>(cgh);
			auto aU     = bU.   template get_access<cl::sycl::access::mode::read_write>(cgh);
			auto aLines = blines.template get_access<cl::sycl::access::mode::write>(cgh);
			cl::sycl::range<1> r{(size_t)n};
			cgh.parallel_for<class SchrodingerKernel>(r, [=](cl::sycl::item<1> id)
			{
				int i = (int)id.get(0);
				double dr, di;
				double dxdx = sq(dx);
				int read = idx*n;
				int write = (1-idx)*n;
				if(i==0)
				{
					dr = (aPsiR[read+0] - 2.0*aPsiR[read+1] + aPsiR[read+2])/dxdx;
					di = (aPsiI[read+0] - 2.0*aPsiI[read+1] + aPsiI[read+2])/dxdx;
				}
				else if(i==n-1)
				{
					dr = (aPsiR[read+n-3] - 2.0*aPsiR[read+n-2] + aPsiR[read+n-1])/dxdx;
					di = (aPsiI[read+n-3] - 2.0*aPsiI[read+n-2] + aPsiI[read+n-1])/dxdx;
				}
				else
				{
					dr = (aPsiR[read+i-1] - 2.0*aPsiR[read+i] + aPsiR[read+i+1])/dxdx;
					di = (aPsiI[read+i-1] - 2.0*aPsiI[read+i] + aPsiI[read+i+1])/dxdx;
				}

				aPsiR[write+i] = aPsiR[read+i] + h * (aU[i] * aPsiI[read+i] - di / (2.0*m));
				aPsiI[write+i] = aPsiI[read+i] + h * (dr / (2.0*m) - aU[i] * aPsiR[read+i]);

				float sz = (float)n;
				Float4 l;
				l.x = (i-sz/2.0f)/sz;
				l.y = -0.75f + 0.25f*(float)cl::sycl::sqrt( sq(aPsiR[write+i]) + sq(aPsiI[write+i]));
				aLines[i] = l;
			});
		});
		//cl.sycl_queue.wait();
        idx = 1 - idx;
    };
		
    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
			if     (event.type == sf::Event::Closed  ){ window.close(); }
			else if(event.type == sf::Event::Resized )
			{
				window.setView(sf::View(sf::FloatRect(0.0f, 0.0f, event.size.width*1.0f, event.size.height*1.0f)));
				glViewport(0, 0, event.size.width, event.size.height);
			}
        }

        step();
        window.clear();

        glUseProgram(shader);
		glBindVertexArray(vaou);
        glLineWidth(2.0f);
		glDrawArrays(GL_LINE_STRIP, 0, n);

		glUseProgram(shader);
		glBindVertexArray(vao);
        glLineWidth(2.0f);
		glDrawArrays(GL_LINE_STRIP, 0, n);

        window.display();
    }

    return 0;
}