#pragma once

// NBody configure
#include <SYCL-Conway-Config.hpp>

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
#include <array>        // std::array
#include <fstream>
#include <memory>
#include <future>
#include <random>
#include <memory>
#include <sstream>
#include <algorithm>
#include <memory>       // std::unique_ptr


namespace kernels { struct ConwayStep; }


class Conway : public InteropWindow
{
    Q_OBJECT

public:

    explicit Conway(std::size_t plat_id,
                    cl_bitfield dev_type,
                    QWindow *parent = 0);
    ~Conway() = default;

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

    std::size_t dev_id;

    // OpenGL related variables
    std::unique_ptr<QOpenGLShader> vs, fs;
    std::unique_ptr<QOpenGLShaderProgram> sp;
    std::unique_ptr<QOpenGLBuffer> vbo;
    std::unique_ptr<QOpenGLVertexArrayObject> vao;
    std::array<std::unique_ptr<QOpenGLTexture>, 2> texs;
    GLuint TEX;

    // OpenCL related variables
    std::array<cl::ImageGL, 2> CL_latticeImages;
    std::vector<cl::Memory> interop_resources;  // Bloat
    bool cl_khr_gl_event_supported;

    // SYCL related variables
    cl::sycl::context context;              // Context
    cl::sycl::device device;                // Device
    cl::sycl::queue compute_queue;          // CommandQueue

    std::array<std::unique_ptr<cl::sycl::image<2>>, 2> latticeImages;   // Simulation data images

    bool imageDrawn;                        // Whether image has been drawn since last iteration
    bool needMatrixReset;                   // Whether matrices need to be reset in shaders

    void setMatrices();                     // Update shader matrices
};
