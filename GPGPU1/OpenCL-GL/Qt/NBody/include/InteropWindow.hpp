#pragma once

// OpenCL behavioral defines
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

// OpenCL include
#include <CL/cl2.hpp>

// Qt5 includes
#include <QtGui>
#include <QOpenGLFunctions_3_3_Core>

// Logging
#include <QMessageLogger>
#include <QDebug>

// C++ includes
#include <climits>
#include <chrono>
#include <algorithm>



#define QTIMER
//#define STDTIMER


class InteropWindow : public QWindow
{
    Q_OBJECT

public:

    explicit InteropWindow(std::size_t plat_id = 0,
                           cl_bitfield dev_type = CL_DEVICE_TYPE_DEFAULT,
                           QWindow *parent = 0);
    ~InteropWindow();

public slots:
    virtual void renderLater() final;   // Function that notifies application to refresh window contents (non-blocking)
    virtual void renderNow() final;     // Function that forces instant refresh (blocking)

    void setAnimating(bool animating);  // Turn on/off self-animation
    void setMaxIPS(int IPS);            // Set the maximum Iterations Per Second
    void setMaxFPS(int FPS);            // Set the maximum Frames Per Second
                                        // (NOTE: When both IPS and FPS are limited the application will burn the object's thread!)

    void setDeviceType(cl_bitfield);    // Set device type to be used
    void setPlatformId(std::size_t);    // Set implementation vendor to be used

protected:

    cl_int CL_err;                      // Can be used to store OpenCL errors
    GLint GL_err;                       // Can be used to store OpenGL errors

    // Core functionality to be overriden
    virtual void initializeGL() = 0;                    // Function that initializes all OpenGL assets needed to draw a scene
    virtual void initializeCL() = 0;                    // Function that initializes all OpenCL assets needed to draw a scene
    virtual void updateScene() = 0;                     // Function that holds scene update guaranteed not to conflict with drawing
    virtual void render() = 0;                          // Function that does the native rendering
    virtual void render(QPainter* painter) = 0;         // Function the overlays content on the native scene
    virtual void resizeGL(QResizeEvent* event_in) = 0;  // Function that handles render area resize
    virtual bool event(QEvent* event_in) override;      // Override of QWindow event handler function

    cl::Platform& CLplatform();                         // Get associated variable
    std::vector<cl::Device>& CLdevices();               // Get associated variable
    cl::Context& CLcontext();                           // Get associated variable
    std::vector<cl::CommandQueue>& CLcommandqueues();   // Get associated variable
  
    QOpenGLFunctions_3_3_Core* glFuncs; // OpenGL functions Qt5.1

    const int getActIPS();
    const int getActFPS();
    const int getMaxIPS();
    const int getMaxFPS();
    const bool getAnimating();

    void checkCLerror();
    void checkGLerror();

private:
    bool m_gl_context_initialized;      // Flag indicating OpenGL context state
    bool m_cl_context_initialized;      // Flag indicating OpenCL context state
    bool m_assets_initialized;          // Flag indicating assets state
    bool m_render_pending;              // Flag indicating whether a refresh is enqueued on the application event loop
    bool m_animating;                   // Flag indicating whether the window is self-animating

    int m_max_IPS, m_max_FPS;           // Nomen est omen
    int m_act_IPS, m_act_FPS;           // Nomen est omen

    cl_bitfield m_device_type;          // Device type to be used
    std::size_t m_platform_id;          // Implementation vendor to be used
#ifdef QTIMER
    QElapsedTimer m_IPS_limiter, m_FPS_limiter; // Limiters
#endif
#ifdef STDTIMER
    std::chrono::time_point<std::chrono::high_resolution_clock> m_IPS_limiter, m_FPS_limiter; // Limiters
#endif

    // Resource handles
    QPlatformNativeInterface* plat_int;   // Platform native interface to obtain OS-spcific handles

    QOpenGLContext* m_gl_context;                               // Context used by the window
    QOpenGLContext* m_painter_context;                          // Painter context
    QOpenGLPaintDevice* m_gl_paintdevice;                       // Paint Engine used by the window
    QPainter* m_painter;                                        // Painter used to draw managed content
    cl::Platform m_cl_platform;                                 // OpenCL platform used
    std::vector<cl::Device> m_cl_devices;                       // OpenCL device used
    cl::Context m_cl_context;                                   // OpenCL context used
    std::vector<cl::CommandQueue> m_cl_commandqueues;           // OpenCL commandqueue used

    virtual void exposeEvent(QExposeEvent* event_in) override;  // Override of QWindow expose event
    virtual void resizeEvent(QResizeEvent* event_in) override;  // Override of QWindow on resize event

    void createGLcontext_helper();                              // Nomen est omen
    void createCLcontext_helper();                              // Nomen est omen
    void render_helper();                                       // Nomen est omen
    void updateScene_helper();                                  // Nomen est omen

    bool lookForDeviceType(cl_bitfield devtype);                // Helper function to enumerates interop capable devices of given type
    bool lookForDeviceType(cl::Platform&, cl_bitfield);         // Helper function to enumerates interop capable devices of given type on a certain platform

    QVector<cl_context_properties> interopCLcontextProps(const cl::Platform& plat);          // Context properties of interop context

    bool detectFormatMismatch(QSurfaceFormat left, QSurfaceFormat right);
    void printFormatMismatch();
    const char* convertCLerrorToString(cl_int error);
    const char* convertGLerrorToString(GLint error);
};
