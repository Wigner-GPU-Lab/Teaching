// Qt5 includes
#include <QGuiApplication>
#include <QMessageLogger>

// Custom made includes
#include <NBody.hpp>


int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    // Qt5 constructs
    QSurfaceFormat my_surfaceformat;
    
    // Setup desired format
    my_surfaceformat.setRenderableType(QSurfaceFormat::RenderableType::OpenGL);
    my_surfaceformat.setProfile(QSurfaceFormat::OpenGLContextProfile::CoreProfile);
    my_surfaceformat.setSwapBehavior(QSurfaceFormat::SwapBehavior::DoubleBuffer);
    my_surfaceformat.setMajorVersion(4);
    my_surfaceformat.setMinorVersion(2);
    my_surfaceformat.setRedBufferSize(8);
    my_surfaceformat.setGreenBufferSize(8);
    my_surfaceformat.setBlueBufferSize(8);
    my_surfaceformat.setAlphaBufferSize(8);
    my_surfaceformat.setDepthBufferSize(24);
    my_surfaceformat.setStencilBufferSize(8);
    my_surfaceformat.setStereo(false);

	NBody nbody;
	nbody.setDeviceType(CL_DEVICE_TYPE_GPU);
	nbody.setFormat(my_surfaceformat);
	nbody.setVisibility(QWindow::Maximized);
    nbody.setMaxFPS(60);
    //nbody.setMaxIPS(10);
	nbody.setAnimating(true);

    return app.exec();
}