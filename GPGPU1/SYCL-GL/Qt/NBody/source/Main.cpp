// Qt5 includes
#include <QGuiApplication>
#include <QMessageLogger>
#include <QCommandLineParser>

// Custom made includes
#include <NBody.hpp>


int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    QCoreApplication::setApplicationName("OpenCL-GL NBody sample");
    QCoreApplication::setApplicationVersion("1.0");

    QCommandLineParser parser;
    parser.setApplicationDescription("Sample application demonstrating OpenCL-OpenGL interop");
    parser.addHelpOption();
    parser.addVersionOption();
    parser.addOptions({
        {{"d", "device"}, "Device type to use", "[cpu|gpu|acc]", "cpu"},
        {{"p", "platformId"}, "The index of the platform to use", "unsigned integral", "0"},
        {{"x", "particles"}, "Number of particles", "unsigned integral", "8192"}
    });

    parser.process(app);

    cl_bitfield dev_type;
    std::size_t plat_id, count;
	if(!parser.value("device").isEmpty())
    {
        if(parser.value("device") == "cpu")
            dev_type = CL_DEVICE_TYPE_CPU;
        else if(parser.value("device") == "gpu")
            dev_type = CL_DEVICE_TYPE_GPU;
        else if(parser.value("device") == "acc")
            dev_type = CL_DEVICE_TYPE_ACCELERATOR;
        else
        {
            qFatal("NBody: Invalid device type: valid values are [cpu|gpu|acc]. Using CL_DEVICE_TYPE_DEFAULT instead.");
        }
    }
    else dev_type = CL_DEVICE_TYPE_DEFAULT;
    if(!parser.value("platformId").isEmpty())
    {
        plat_id = parser.value("platformId").toULong();
    }
    else plat_id = 0;
    if(!parser.value("particles").isEmpty())
    {
        count = parser.value("particles").toULong();
    }
    else count = 8192u;

    NBody nbody(plat_id, dev_type, count);
    nbody.setVisibility(QWindow::AutomaticVisibility);
    nbody.setWidth(1280);
    nbody.setHeight(720);
    nbody.setMaxFPS(60);
    //nbody.setMaxIPS(60);
	nbody.setAnimating(true);

    // Qt5 constructs
    QSurfaceFormat my_surfaceformat;
    
    // Setup desired format
    my_surfaceformat.setRenderableType(QSurfaceFormat::RenderableType::OpenGL);
    my_surfaceformat.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
    my_surfaceformat.setSwapBehavior(QSurfaceFormat::SwapBehavior::DoubleBuffer);
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