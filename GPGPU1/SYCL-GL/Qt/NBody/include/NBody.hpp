#pragma once

// NBody configure
#include <NBody-Config.hpp>

// NBody includes
#include <NBodyStep.hpp>

// C++ behavioral defines
#define _USE_MATH_DEFINES

// Base class include
#include <InteropWindow.hpp>

// SYCL include
#include <CL/sycl.hpp>

// Graphics utility includes
#include <QMatrix4x4>
#include <QVector>

// C++ includes
#include <array>
#include <fstream>
#include <memory>
#include <future>
#include <random>
#include <memory>
#include <sstream>
#include <algorithm>

using real = cl_float;
using real4 = cl::sycl::float4;


class NBody : public InteropWindow
{
    Q_OBJECT

public:

    explicit NBody(QWindow *parent = 0);
	~NBody() = default;

    virtual void initializeGL() override;
    virtual void initializeCL() override;
    virtual void updateScene() override;
    virtual void render() override;
    virtual void render(QPainter* painter) override;
    virtual void resizeGL(QResizeEvent* event_in) override;
    virtual bool event(QEvent *event_in) override;

private:

	enum Buffer
	{
		Front = 0,
		Back = 1
	};

	// Simulation related variables
	std::size_t particle_count;
	real x_abs_range, y_abs_range, z_abs_range,
		mass_min, mass_max;

	std::size_t dev_id;

	// Host-side containers
	std::vector<real4> pos_mass;
	std::vector<real4> velocity;

	// OpenCL related variables
    cl::sycl::context context;              // Context
    cl::sycl::device device;                // Device
    cl::sycl::queue compute_queue;          // CommandQueue
    bool cl_khr_gl_event_supported;

	std::array<cl::sycl::buffer<real4>, 2> posBuffs;   // Simulation data buffers
	std::array<cl::sycl::buffer<real4>, 2> velBuffs;   // Simulation data buffers
	std::array<cl::Event, 2> acquire_release;

	std::vector<cl::Memory> interop_resources;  // Bloat
	std::vector<cl::Event> acquire_wait_list,   // Bloat
	                       release_wait_list;   // Bloat

	// OpenGL related variables
	std::unique_ptr<QOpenGLShader> vs, fs;
	std::unique_ptr<QOpenGLShaderProgram> sp;
	std::array<std::unique_ptr<QOpenGLBuffer>, 2> vbos;
	std::array<std::unique_ptr<QOpenGLVertexArrayObject>, 2> vaos;

	bool rightMouseButtonPressed;           // Variables to enable dragging
	QPoint mousePos;                        // Variables to enable dragging
    float dist, phi, theta;                 // Mouse polar coordinates
    bool imageDrawn;                        // Whether image has been drawn since last iteration
	bool needMatrixReset;                   // Whether matrices need to be reset in shaders

	void mouseDrag(QMouseEvent* event_in);  // Handle mouse dragging
	void mouseWheel(QWheelEvent* event_in); // Handle mouse wheel movement

	void setMatrices();                     // Update shader matrices
    /*
	QVector3D vecEye;                     // Camera position
	QVector3D vecTarget;                  // Viewed point position
    QVector3D vecUp;                      // Up vector

	QMatrix4x4 matWorld;                  // World matrix
	QMatrix4x4 matView;                   // Viewing matrix
	QMatrix4x4 matProj;                   // Perspective projection matrix
    */
};
