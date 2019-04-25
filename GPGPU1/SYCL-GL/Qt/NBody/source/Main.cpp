// Qt5 includes
#include <QGuiApplication>
#include <QMessageLogger>
#include <QCommandLineParser>

// Custom made includes
#include <NBody.hpp>

// SYCL include
#ifdef _MSC_VER 
#pragma warning( push )
#pragma warning( disable : 4310 ) /* Prevents warning about cast truncates constant value */
#pragma warning( disable : 4100 ) /* Prevents warning about unreferenced formal parameter */
#endif
#include <CL/sycl.hpp>
#ifdef _MSC_VER 
#pragma warning( pop )
#endif

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    QCoreApplication::setApplicationName("SYCL-GL NBody sample");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("Sample application demonstrating OpenCL-OpenGL interop");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addOptions({
        {{"p", "platform"}, "The index of the platform to use", "unsigned integral", "0"},
        {{"d", "device"}, "The index of the device to use", "unsigned integral", "0"},
        {{"t", "type"}, "Device type to use", "[cpu|gpu|acc]", "def"},
        {{"x", "particles"}, "Number of particles", "unsigned integral", "8192"}
    });

    parser.process(app);

    cl_bitfield dev_type = CL_DEVICE_TYPE_DEFAULT;
    std::size_t plat_id = 0u, dev_id = 0u, count = 8192u;

    if (!parser.value("platform").isEmpty()) plat_id = parser.value("platform").toULong();
    if (!parser.value("device").isEmpty()) dev_id = parser.value("device").toULong();
    if(!parser.value("type").isEmpty())
    {
        if(parser.value("type") == "cpu")
            dev_type = CL_DEVICE_TYPE_CPU;
        else if(parser.value("type") == "gpu")
            dev_type = CL_DEVICE_TYPE_GPU;
        else if(parser.value("type") == "acc")
            dev_type = CL_DEVICE_TYPE_ACCELERATOR;
        else
        {
            qFatal("NBody: Invalid device type: valid values are [cpu|gpu|acc]. Using CL_DEVICE_TYPE_DEFAULT instead.");
        }
    }
    if(!parser.value("particles").isEmpty()) count = parser.value("particles").toULong();

    NBody nbody(plat_id, dev_id, dev_type, count);
    nbody.setVisibility(QWindow::Maximized);
    nbody.setAnimating(true);

    // Qt5 constructs
    QSurfaceFormat my_surfaceformat;

    // Setup desired format
    my_surfaceformat.setRenderableType(QSurfaceFormat::RenderableType::OpenGL);
    my_surfaceformat.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
    my_surfaceformat.setSwapBehavior(QSurfaceFormat::SwapBehavior::DoubleBuffer);
    my_surfaceformat.setOption(QSurfaceFormat::DebugContext);
    my_surfaceformat.setMajorVersion(3);
    my_surfaceformat.setMinorVersion(3);
    my_surfaceformat.setRedBufferSize(8);
    my_surfaceformat.setGreenBufferSize(8);
    my_surfaceformat.setBlueBufferSize(8);
    my_surfaceformat.setAlphaBufferSize(8);
    my_surfaceformat.setDepthBufferSize(24);
    my_surfaceformat.setStencilBufferSize(8);
    my_surfaceformat.setStereo(false);

    nbody.setFormat(my_surfaceformat);

    return app.exec();
}
