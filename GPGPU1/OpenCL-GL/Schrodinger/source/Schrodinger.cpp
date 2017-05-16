#include <Schrodinger.hpp>

struct Float4
{
    float x, y, r, b;
};

auto sq = [](auto const& x){ return x*x; };

struct CL
{
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    struct Buffer
    {
        void* ptr;
        size_t size;
        cl_mem id;
        bool is_gl_object;
    };

    std::vector<Buffer> buffers;
    CL():platform{nullptr}, device{nullptr}, context{nullptr}, queue{nullptr}, program{nullptr}, kernel{nullptr}
    {
        auto status = clGetPlatformIDs(1, &platform, NULL);
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

        auto glContext = wglGetCurrentContext();
        auto gldc = wglGetCurrentDC();
        cl_context_properties cps[] =
        {   CL_GL_CONTEXT_KHR,   (cl_context_properties) glContext,
            CL_WGL_HDC_KHR,      (cl_context_properties) gldc,
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform,  0 };
        context = clCreateContext(cps, 1, &device, 0, 0, &status);
        queue = clCreateCommandQueueWithProperties(context, device, nullptr, &status);
    }

    ~CL()
    {
        for(auto const& b : buffers){ clReleaseMemObject(b.id); }
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

    void set_num_buffers(int n){ buffers.resize(n); }
    template<typename T>
    void create_buffer( std::vector<T>& data, cl_mem_flags const& flags )
    {
        cl_int status;
        buffers.push_back({data.data(), data.size() * sizeof(T), clCreateBuffer(context, flags, data.size() * sizeof(T), data.data(), &status), false});
    }

    template<typename T>
    void create_from_gl_buffer( std::vector<T>& data, cl_mem_flags const& flags, GLuint glid )
    {
        cl_int status;
        auto clid = clCreateFromGLBuffer( context, CL_MEM_READ_WRITE, glid, &status );
        buffers.push_back({data.data(), data.size() * sizeof(T), clid, true});
    }

    template<typename T>
    void set_buffer_at_index(int i, std::vector<T>const& data)
    {
        cl_mem id = nullptr;
        bool is_gl = false;
        for(auto const& b : buffers){ if(data.data() == b.ptr){ id = b.id; is_gl = b.is_gl_object; break; } }
        if(!id){ return; }

        cl_int status;
        if(is_gl)
        {
            status = clEnqueueAcquireGLObjects(queue, 1, &id, 0, nullptr, nullptr );
        }
        status = clSetKernelArg(kernel, i, sizeof(id), &id);
    }

    template<typename T>
    void set_value_at_index(int i, T const& data)
    {
        auto status = clSetKernelArg(kernel, i, sizeof(T), &data);
    }

    void start_kernel( int n )
    {
        size_t thread_count = (size_t)n;
        auto status = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &thread_count, nullptr, 0, nullptr, nullptr);
    }

    template<typename T>
    void read_buffer( std::vector<T>& data )
    {
        cl_mem id = nullptr;
        for(auto const& b : buffers){ if(data.data() == b.ptr){ id = b.id; break; } }
        if(!id){ return; }
        auto status = clEnqueueReadBuffer(queue, id, false, 0, data.size() * sizeof(T), data.data(), 0, nullptr, nullptr);
    }

    void finish()
    {
        auto status = clFinish(queue);
        for(auto const& b : buffers)
        {
            if(b.is_gl_object){ clEnqueueReleaseGLObjects( queue, 1, &b.id, 0, nullptr, nullptr ); }
        }
    }
};

//OpenGL Functions Signatures:
typedef void (*F_GenBuffers) (GLsizei, GLuint*);
typedef void (*F_BindBuffer) (GLenum target, GLuint buffer);
typedef void (*F_BufferData) (GLenum target, GLsizeiptr size, const GLvoid * data, GLenum usage);
typedef void (*F_GenVertexArrays) (GLsizei n, GLuint *arrays);
typedef void (*F_BindVertexArray) (GLuint array);
typedef void (*F_VertexAttribPointer) (	GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const GLvoid * pointer);
typedef void (*F_EnableVertexAttribArray) (GLuint index);
typedef GLuint (*F_CreateShader) (GLenum shaderType);

typedef void (*F_ShaderSource) (GLuint shader, GLsizei count, const GLchar **string, const GLint *length);
typedef void (*F_CompileShader) (GLuint shader);
typedef GLuint (*F_CreateProgram) (void);
typedef void (*F_AttachShader) (GLuint program, GLuint shader);
typedef void (*F_LinkProgram) (GLuint program);
typedef GLuint (*F_UseProgram) (GLuint program);

typedef void (*F_BindAttribLocation) (GLuint program, GLuint index, const GLchar *name);
typedef void (*F_GetShaderiv) (GLuint shader, GLenum pname, GLint *params);
typedef void (*F_GetShaderInfoLog) (GLuint shader, GLsizei maxLength, GLsizei* length, GLchar* infoLog);

int main()
{
    //sf::ContextSettings context(24, 8, 2, 3, 3);
    //sf::RenderWindow window(sf::VideoMode(1024, 768), "OpenGL 2D Function", sf::Style::Default, context);
    sf::RenderWindow window(sf::VideoMode(1000, 600), "Hello Schrödinger");
    CL cl;
    cl.set_kernel( "sss.cl", "sssh" );

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

    //Initialize openGL objects:
    //OpenGL Functions Pointers:
    auto glGenBuffers      = (F_GenBuffers)     wglGetProcAddress("glGenBuffers");
    auto glBindBuffer      = (F_BindBuffer)     wglGetProcAddress("glBindBuffer");
    auto glBufferData      = (F_BufferData)     wglGetProcAddress("glBufferData");
    auto glGenVertexArrays = (F_GenVertexArrays)wglGetProcAddress("glGenVertexArrays");
    auto glBindVertexArray = (F_BindVertexArray)wglGetProcAddress("glBindVertexArray");
    auto glVertexAttribPointer = (F_VertexAttribPointer)wglGetProcAddress("glVertexAttribPointer");
    auto glEnableVertexAttribArray = (F_EnableVertexAttribArray)wglGetProcAddress("glEnableVertexAttribArray");
    auto glCreateShader = (F_CreateShader)wglGetProcAddress("glCreateShader");
    auto glShaderSource = (F_ShaderSource)wglGetProcAddress("glShaderSource");
    auto glCompileShader = (F_CompileShader)wglGetProcAddress("glCompileShader");
    auto glCreateProgram = (F_CreateProgram)wglGetProcAddress("glCreateProgram");
    auto glAttachShader = (F_AttachShader)wglGetProcAddress("glAttachShader");
    auto glLinkProgram = (F_LinkProgram)wglGetProcAddress("glLinkProgram");
    auto glUseProgram = (F_UseProgram)wglGetProcAddress("glUseProgram");
    auto glBindAttribLocation = (F_BindAttribLocation)wglGetProcAddress("glBindAttribLocation");
    auto glGetShaderiv = (F_GetShaderiv)wglGetProcAddress("glGetShaderiv");
    auto glGetShaderInfoLog = (F_GetShaderInfoLog)wglGetProcAddress("glGetShaderInfoLog");

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

	std::ifstream vs_file("schvs.glsl");
	std::string vs_code{std::istreambuf_iterator<char>(vs_file), std::istreambuf_iterator<char>()};
	
	std::ifstream fs_file("schfs.glsl");
	std::string fs_code{std::istreambuf_iterator<char>(fs_file), std::istreambuf_iterator<char>()};

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
        //MessageBoxA(0, &ShaderError[0], "", 0);
    }

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &pfs_code, NULL);
	glCompileShader(fs);

	glGetShaderiv(fs, GL_COMPILE_STATUS, &result);
    glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &logLength);
    {
	    std::vector<char> ShaderError((logLength > 1) ? logLength : 1);
        glGetShaderInfoLog(fs, logLength, NULL, &ShaderError[0]);
        //MessageBoxA(0, &ShaderError[0], "", 0);
    }

	GLuint shader = glCreateProgram();
	glAttachShader(shader, vs);
	glAttachShader(shader, fs);

	glBindAttribLocation(shader, 0, "vertex_position");
	glLinkProgram(shader);

    cl.create_buffer( PsiR,      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR );
    cl.create_buffer( PsiI,      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR );
    cl.create_buffer( U,         CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR );
    cl.create_from_gl_buffer( lines,     CL_MEM_READ_WRITE, lines_vbo );

    cl.set_buffer_at_index(0, PsiR);
    cl.set_buffer_at_index(1, PsiI);
    cl.set_buffer_at_index(2, U);
    
    cl.set_value_at_index(4, dx);
    cl.set_value_at_index(5, n);
    
    int idx = 0;
    auto step = [&]
    {
        cl.set_buffer_at_index(3, lines);
        cl.set_value_at_index(6, idx);
        cl.start_kernel(n);
        cl.finish();
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