#pragma once

// Base class include
#include <InteropWindow.hpp>

// Graphics utility includes
#include <QMatrix4x4>
#include <QVector>

// C++ includes
#include <fstream>
#include <memory>
#include <future>

#define REAL cl_float

struct KGParams
{
    cl_uint r_max;          // Number of radial coordinates
    cl_uint L_max;          // Number of largest L multipole component
    REAL dr;                // Radial step size
    REAL dt;                // Temporal step size
    REAL t_max;             // Length of temporal integration
    REAL lambda;            // Self-interacting coefficient
    REAL sigma;             // Dissipation coefficient
    cl_uint cutoff_r;       // Number of radial coordinates from free end to center cutoff around
    cl_uint cutoff_w;       // 99.99% ("four nine") cutoff distance from center
    REAL a;                 // Amplitude of initial Gausses
    REAL b;                 // Distance of Gausses from sides in radial coordinates
    REAL c;                 // Sigma of Gausses in radial coordinates
};

class QGripper : public InteropWindow
{
    Q_OBJECT

public:
    explicit QGripper(QWindow *parent = 0);
    ~QGripper();

    virtual void initializeGL() override;
    virtual void initializeCL() override;
    virtual void updateScene() override;
    virtual void render() override;
    virtual void render(QPainter* painter) override;
    virtual void resizeGL(QResizeEvent* event_in) override;
    virtual bool event(QEvent *event_in) override;
    /*
    typedef cl::make_kernel<
        cl_int,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::LocalSpaceArg&,
        cl::LocalSpaceArg&,
        cl::LocalSpaceArg&,
        cl::Buffer&> RK4Functor;

    typedef cl::make_kernel<
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&,
        cl::LocalSpaceArg&,
        cl::LocalSpaceArg&,
        cl_int> GauntMulFunctor;

    typedef cl::make_kernel<
        cl_int,
        cl::Buffer&,
        cl::BufferGL&> UpdateVertexFunctor;

    typedef cl::make_kernel<
        cl::Buffer&,
        cl::Buffer&,
        cl::Buffer&> InterpolateFunctor;
        */
private:
    // Simulation related variables
    KGParams params;                        // Simulation parameters
    REAL t;                                 // Actual time
    unsigned int sysWidth;                  // System width
	unsigned int sysHeight;                 // System height
	int stepNum;                            // Number of actual step
	int graphLength;                        // Number of sampling points
    bool imageDrawn;                        // Whether image has been drawn since last iteration
    float scale;                            // Z scaling factor of raw view

    StdLogger* myLog;                       // Logger used by Gaunt
    Gaunt<REAL>* gaunt;                     // Gaunt coefficient calculator

    // Host-side containers
	std::array<std::vector<REAL>,4> data;   // Simulation data
	std::vector<cl_float4> mesh;            // Mesh data
    std::vector<GLuint> indx;               // Index Buffer array
	std::vector<cl_float4> meshGraph;       // Mesh data for graph

    // OpenCL related variables
	std::array<cl::Buffer,3> dataBuffs;     // Simulation data buffers
	cl::Buffer gauntBuff;                   // Gaunt coefficient buffer
	cl::Buffer gauntIndexBuff;              // Gaunt index buffer
	cl::Buffer gauntMarkBuff;               // Gaunt accelerating buffer
    cl::Buffer gauntMulStageBuff;           // Gaunt multiplication stage buffer
	cl::Buffer gauntMulResultBuff;          // Gaunt multiplication result buffer
	std::array<std::array<cl::Buffer,3>,3> auxBuffs;    // RK4 temp buffers
	std::vector<cl::Memory> vertexBuffs;    // Display vertex buffers

	cl::Kernel rk4;                         // Device kernel for RK4
    cl::Kernel interpolate;                 // Device kernel to interpolate first radial co-ordinate
    cl::Kernel gaunt_mul;                   // Device kernel for Gaunt-matrix multiplication
	cl::Kernel upd_vtx;                     // Device kernel to update vertices

	cl::NDRange rk4_global, rk4_local;          // RK4 work-sizes
    cl::NDRange int_global, int_local;          // Interpolate work-sizes
    cl::NDRange mul_global, mul_local;          // Gaunt matrix multiplication work-sizes
	cl::NDRange upd_vtx_global, upd_vtx_local;  // Vertex update work-sizes
	cl::NDRange upd_dyn_global, upd_dyn_local;  // Dynamical variable update work-sizes

    // OpenGL related variables
    QOpenGLShader* m_vs;
	QOpenGLShader* m_fs;
    QOpenGLShaderProgram* m_sp;
	QOpenGLBuffer* m_vbo;
    QOpenGLBuffer* m_ibo;
    QOpenGLVertexArrayObject* m_vao;

    bool validateParams(const KGParams& params_in); // Validates parameters given for computation

    bool rightMouseButtonPressed;           // Variables to enable dragging
	QPoint mousePos;                        // Variables to enable dragging
    float dist, phi, theta;                 // Mouse polar coordinates
    bool needMatrixReset;                   // Whether matrices need to be reset in shaders

    void mouseDrag(QMouseEvent* event_in);  // Handle mouse dragging
    void mouseWheel(QWheelEvent* event_in); // Handle mouse wheel movement

    void setMatrices();                     // Update shader matrices

    QVector3D m_vecEye;                     // Camera position
    QVector3D m_vecTarget;                  // Viewed point position

    QMatrix4x4 m_matWorld;                  // World matrix
    QMatrix4x4 m_matView;                   // Viewing matrix
    QMatrix4x4 m_matProj;                   // Perspective projection matrix

    // Helper functions
    void setDynamicKernelParams(int);       // Nomen est omen
    int gauntIndexer(int l, int m);         // Used for combining l,m indices
	cl_int2 reverseGauntIndexer(int i);     // Used for reverting l,m combination
};
