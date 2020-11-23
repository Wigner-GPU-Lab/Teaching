#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN 1
	#include <windows.h>
#else
#endif

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#ifdef _WIN32
#else
	#include <GL/glx.h>
	#include <GL/glext.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>

#include <string>
#include <algorithm>
#include <vector>
#include <random>
#include <iostream>

__host__ __device__ float sq(float x){ return x*x; }
__host__ __device__ float cube(float x){ return x*x*x; }

__global__ void gpu_nbody_opt(float3* V, float4* P, float G, float dt) 
{
    unsigned int N = blockDim.x*gridDim.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float3 sum = {0.0f, 0.0f, 0.0f};
    float4 Pi = P[i];
    float x = Pi.x;
    float y = Pi.y;
	float z = Pi.z;
	
	for(int j=0; j<N; ++j)
    {
        float4 Pj = P[j];
        float dx = Pj.x - x;
        float dy = Pj.y - y;
        float dz = Pj.z - z;
        float rsqrt = i != j ? rsqrtf( sq(dx) + sq(dy) + sq(dz) ) : 0.0f;
        float rec = Pj.w * cube(rsqrt);
        sum.x += dx * rec;
        sum.y += dy * rec;
        sum.z += dz * rec;
    }
    
    //a = F/m = -G * sum
    //pos = pos + vel * dt + acc / 2 *dt^2
    float3 Vi = V[i];
    float vx = Vi.x + G*sum.x * dt;
    float vy = Vi.y + G*sum.y * dt;
    float vz = Vi.z + G*sum.z * dt;

    x = x + (Vi.x + G/2*sum.x * dt)*dt;
    y = y + (Vi.y + G/2*sum.y * dt)*dt;
    z = z + (Vi.z + G/2*sum.z * dt)*dt;

    P[i] = float4{x, y, z, Pi.w};
    V[i] = float3{vx, vy, vz};

}

struct Velocity{ float x, y, z; };
struct Particle{ float x, y, z, m; };

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	template<typename fptr_type>
	fptr_type load_extension_pointer(const char* name){ return reinterpret_cast<fptr_type>(wglGetProcAddress(name)); }
#else
	template<typename fptr_type>
	fptr_type load_extension_pointer(const char* name){ return reinterpret_cast<fptr_type>(glXGetProcAddressARB((const GLubyte*)name)); }
#endif

static void error_callback(int error, const char* description)
{
    std::cout << "Error: " << description << "\n";
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
	{
        glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}

static inline const char* glErrorToString(GLenum err)
{
#define CASE_RETURN_MACRO(arg) case arg: return #arg
    switch(err)
    {
        CASE_RETURN_MACRO(GL_NO_ERROR);
        CASE_RETURN_MACRO(GL_INVALID_ENUM);
        CASE_RETURN_MACRO(GL_INVALID_VALUE);
        CASE_RETURN_MACRO(GL_INVALID_OPERATION);
        CASE_RETURN_MACRO(GL_OUT_OF_MEMORY);
        CASE_RETURN_MACRO(GL_STACK_UNDERFLOW);
        CASE_RETURN_MACRO(GL_STACK_OVERFLOW);
#ifdef GL_INVALID_FRAMEBUFFER_OPERATION
        CASE_RETURN_MACRO(GL_INVALID_FRAMEBUFFER_OPERATION);
#endif
        default: break;
    }
#undef CASE_RETURN_MACRO
    return "*UNKNOWN*";
}

bool checkGLError(const char* msg = "")
{
	GLenum gl_error = glGetError();
    if (gl_error != GL_NO_ERROR)
    {
		std::cout << "GL error: " << glErrorToString(gl_error) << " msg: " << msg << "\n";
		return false;
	}
	return true;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

int main(void)
{
    //Simulation parameters:
    const int N = 8192*4;
    const int block_sz = 1024;
    const int n_blocks = N / block_sz;
    const float G = 2e-4f;
    const float dt = 5e-3f;

    //Create window:
	int width = 640;
	int height = 480;

    glfwSetErrorCallback(error_callback);
	if (!glfwInit()){ return -1; }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "Simple example", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
	}
    glfwSetKeyCallback(window, key_callback);
    glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	glGetError();
	if (err != GLEW_OK)
	{
		std::cout << "glewInit failed: " << glewGetErrorString(err);
		return -1;
	}

    glViewport(0, 0, width, height);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	//verify the openGL version we got:
	{
		int p = glfwGetWindowAttrib(window, GLFW_OPENGL_PROFILE);
		std::string version = glfwGetVersionString();
		std::string opengl_profile = "";
		if      (p == GLFW_OPENGL_COMPAT_PROFILE){ opengl_profile = "OpenGL Compatibility Profile"; }
		else if (p == GLFW_OPENGL_CORE_PROFILE  ){ opengl_profile = "OpenGL Core Profile"; }
		std::cout << "GLFW version: " << version << "\n";
		std::cout << "GLFW OpenGL profile: " << opengl_profile << "\n";

		std::cout << "OpenGL: GL version: " <<  glGetString(GL_VERSION) << "\n";
		std::cout << "OpenGL: GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << "\n";
		std::cout << "OpenGL: Vendor: " << glGetString(GL_VENDOR) << "\n";

		std::cout << "GLEW: Glew version: " << glewGetString(GLEW_VERSION) << "\n";
	}
    
    //Generate data for Nbody simulation:
    std::vector<Particle> particles(N);
    std::vector<Velocity> velocities(N);

    std::mt19937 mersenne_engine{43};  // Generates random integers
    std::uniform_real_distribution<float> dist1{-8.5f, 8.5f};
    std::uniform_real_distribution<float> dist2{0.5f, 15.0f};
    std::uniform_real_distribution<float> dist3{-0.5f, 0.5f};
    auto gen = [&]()
    { 
        float disp = dist3(mersenne_engine) > 0.0f ? -10.0f : +10.0f;
        return Particle{dist1(mersenne_engine) + disp, dist1(mersenne_engine) - disp/2, dist1(mersenne_engine), dist2(mersenne_engine)};
    };
    std::generate(particles.begin(), particles.end(), gen);
    auto genv = [&]()
    {
        return Velocity{dist1(mersenne_engine)*0.11f, dist1(mersenne_engine)*0.01f, dist1(mersenne_engine)*0.001f};
    };
    std::generate(velocities.begin(), velocities.end(), genv);
    /*particles[0].m = 20000.0f;
    particles[1].m = 32000.0f;
    particles[2].m = 15000.0f;
    particles[3].m = 18000.0f;*/

	//Compile shaders:
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
			//string = std::basic_string<GLchar>{ shader_type == GL_VERTEX_SHADER ? vertex_shader_str : fragment_shader_str };
			return 0;
		}
		const GLchar* tmp = string.c_str();

		auto shaderObj = glCreateShader(shader_type);
		if(!checkGLError()){ return 0; }
		
		GLint gl_status = 0;
		glShaderSource(shaderObj, (GLsizei)1, &tmp, NULL);
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

	GLuint vertexShaderObj   = load_and_compile_shader(GL_VERTEX_SHADER,   "vertex.glsl");
	GLuint fragmentShaderObj = load_and_compile_shader(GL_FRAGMENT_SHADER, "fragment.glsl");
	if(!vertexShaderObj && !fragmentShaderObj){ std::cout << "Failed to load and compile shaders\n"; return -1; }

	GLuint glProgram = glCreateProgram();
	{
		glAttachShader(glProgram, vertexShaderObj);
		glAttachShader(glProgram, fragmentShaderObj);
		glLinkProgram(glProgram);

		GLint gl_status = 0;
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
		if(!checkGLError()){ return -1; }
	}

    //Create buffers:
    GLuint glbuffer;
    cudaGraphicsResource* cuda_glbuffer = nullptr;
	GLuint glvao;
	
	//Create buffer ID
	glGenBuffers(1, &glbuffer);
	if(!checkGLError("glGenBuffers")){ return -1; }

	// Select the GL Buffer as the active one: 
	glBindBuffer(GL_ARRAY_BUFFER, glbuffer);
	if(!checkGLError("glBindBuffer")){ return -1; }

	// Allocate memory for the buffer:
	glBufferData(GL_ARRAY_BUFFER, particles.size() * sizeof(Particle), NULL, GL_STATIC_DRAW);
	if(!checkGLError("glBufferData")){ return -1; }

	// Upload data to the buffer:
	glBufferSubData(GL_ARRAY_BUFFER, 0, particles.size() * sizeof(Particle), particles.data());
	if(!checkGLError("glBufferSubData")){ return -1; }

	// Create and activate the Vertex Array Object
	// these API functions are missing on windows from the CUDA SDK glew, so we load them manually:
	typedef void (*Fnt_GenVertexArrays) (GLsizei n, GLuint *arrays);
	typedef void (*Fnt_BindVertexArray) (GLuint array);
	Fnt_GenVertexArrays pglGenVertexArrays = load_extension_pointer<Fnt_GenVertexArrays>("glGenVertexArrays");
	Fnt_BindVertexArray pglBindVertexArray = load_extension_pointer<Fnt_BindVertexArray>("glBindVertexArray");

	pglGenVertexArrays(1, &glvao); if(!checkGLError("glGenVertexArrays")){ return -1; }
	pglBindVertexArray(glvao);	   if(!checkGLError("glBindVertexArray")){ return -1; }

	// Register buffers into the VAO
	glBindBuffer(GL_ARRAY_BUFFER, glbuffer);
	if(!checkGLError("glBindBuffer(geo)")){ return -1; }
	glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Particle), (GLvoid *)0);
	if(!checkGLError("glVertexAttribPointer(geo)")){ return -1; }
	
	// register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_glbuffer, glbuffer, cudaGraphicsMapFlagsWriteDiscard));
	
	glEnableVertexAttribArray(0);
	if(!checkGLError("glEnableVertexAttribArray(0)")){ return -1; }
	
	//Cuda source arrays:
    float4* cuda_particles = nullptr;
    float3* cuda_velocities = nullptr;
	auto cuerr = cudaMalloc( (void**)&cuda_particles, particles.size()*sizeof(Particle) );
    if( cuerr != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(cuerr) << "\n"; return -1; }
    
    cuerr = cudaMalloc( (void**)&cuda_velocities, velocities.size()*sizeof(Velocity) );
	if( cuerr != cudaSuccess){ std::cout << "Error allocating CUDA memory: " << cudaGetErrorString(cuerr) << "\n"; return -1; }
	
	cuerr = cudaMemcpy( cuda_particles, particles.data(), particles.size()*sizeof(Particle), cudaMemcpyHostToDevice );
    if( cuerr != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(cuerr) << "\n"; return -1; }

    cuerr = cudaMemcpy( cuda_velocities, velocities.data(), velocities.size()*sizeof(Velocity), cudaMemcpyHostToDevice );
    if( cuerr != cudaSuccess){ std::cout << "Error copying memory to device: " << cudaGetErrorString(cuerr) << "\n"; return -1; }


	std::cout << "Entering render loop\n";
    while (!glfwWindowShouldClose(window))
    {
		// map OpenGL buffer object for writing from CUDA
		float4* ptr;
		checkCudaErrors(cudaGraphicsMapResources(1, &cuda_glbuffer, 0));
		size_t num_bytes;
		checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&ptr, &num_bytes, cuda_glbuffer));

		// execute the kernel
		dim3 dimGrid( n_blocks );
        dim3 dimBlock( block_sz );
		gpu_nbody_opt<<<dimGrid, dimBlock>>>(cuda_velocities, ptr, G, dt);

		// unmap buffer object
		checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_glbuffer, 0));

		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT);
        
        pglBindVertexArray(glvao);
		glDrawArrays(GL_POINTS, 0, N);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}