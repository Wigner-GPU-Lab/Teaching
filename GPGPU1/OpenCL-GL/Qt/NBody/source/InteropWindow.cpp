#include <InteropWindow.hpp>

InteropWindow::InteropWindow(QWindow *parent)
    : QWindow(parent)
    , CL_err(0)
    , GL_err(0)
    , m_gl_context_initialized(false)
    , m_cl_context_initialized(false)
    , m_assets_initialized(false)
    , m_render_pending(false)
    , m_animating(false)
    , m_max_IPS(INT_MAX)
    , m_max_FPS(INT_MAX)
    , m_act_IPS(0)
    , m_act_FPS(0)
    , m_device_type(CL_DEVICE_TYPE_DEFAULT)
    , m_platform_vendor()
    , m_gl_context(0)
    , m_gl_paintdevice(0)
{
    qDebug("InteropWindow: Entering constructor");
    setSurfaceType(QWindow::OpenGLSurface);
    plat_int = QGuiApplication::platformNativeInterface();
#ifdef QTIMER
    m_IPS_limiter.start();
    m_FPS_limiter.start();
#endif
#ifdef STDTIMER
    m_IPS_limiter = std::chrono::high_resolution_clock::now();
    m_FPS_limiter = m_IPS_limiter;
#endif
    qDebug("InteropWindow: Leaving constructor");
}


InteropWindow::~InteropWindow()
{
    if(m_gl_context!=0) delete m_gl_context;
    if(m_gl_paintdevice!=0) delete m_gl_paintdevice;
}


void InteropWindow::createGLcontext_helper()
{
    qDebug("InteropWindow: Entering createGLcontext_helper");
    m_gl_context = new QOpenGLContext(this);
    m_gl_context->setFormat(requestedFormat());

    m_painter_context = new QOpenGLContext(this);
    QSurfaceFormat painter_format = requestedFormat();
    painter_format.setMajorVersion(2);
    painter_format.setMinorVersion(0);
    m_painter_context->setFormat(painter_format);

    if(!m_painter_context->create()) qWarning("InteropWindow: Failed to create painter context");
    else
    {
        m_painter_context->makeCurrent(this);
        {
            // Create QPainter-compatible OpenGL drawing device
            m_gl_paintdevice = new QOpenGLPaintDevice(size());
            m_painter = new QPainter(m_gl_paintdevice);
            m_painter->end();
        }
        m_painter_context->doneCurrent();
    }

    if(!m_gl_context->create()) qWarning("InteropWindow: Failed to create OpenGL context");

    if(detectFormatMismatch(m_gl_context->format(), requestedFormat()))
    {
        qWarning("InteropWindow: The requested window format and the acquired context format mismatch");
        printFormatMismatch();
    }
    qDebug("InteropWindow: Leaving createGLcontext_helper");
}


void InteropWindow::createCLcontext_helper()
{
    qDebug("InteropWindow: Entering createCLcontext_helper");
    m_gl_device = nativeGLdevice();
    
    if(!lookForDeviceType(m_device_type)) qFatal("InteropWindow: No interoperable device could be found!");
    else
    {
        QString platform_name(m_cl_platform.getInfo<CL_PLATFORM_NAME>(&CL_err).c_str()); checkCLerror();
        qDebug() << "InteropWindow: Selected plaform: " << platform_name;
        QVector<QString> device_names;
        for (auto& device : m_cl_devices)
        {
            device_names.push_back(device.getInfo<CL_DEVICE_NAME>(&CL_err).c_str()); checkCLerror();
        }
        qDebug() << "InteropWindow: Selected devices: " << device_names;
    }
    qDebug("InteropWindow: Leaving createCLcontext_helper");
}


bool InteropWindow::lookForDeviceType(cl_bitfield devtype)
{
    bool dev_found = false;

    std::vector<cl::Platform> plats;

    if(m_platform_vendor.isNull())
    {
        qDebug("InteropWindow: No platform preference. Choosing based on device preference");
        CL_err = cl::Platform::get(&plats); checkCLerror();
        for(auto& platform : plats)
        {
            dev_found = lookForDeviceType(platform, devtype);
            if(dev_found == true) break;
        }
    }
    else
    {
        qDebug("InteropWindow: Looking for platform: %s", m_platform_vendor);
        CL_err = cl::Platform::get(&plats); checkCLerror();
        auto it = std::find_if(plats.begin(), plats.end(), [&](cl::Platform& elem) -> bool
            {return elem.getInfo<CL_PLATFORM_VENDOR>(&CL_err) == m_platform_vendor.toStdString();}
        );
        if(it != plats.end())
        {
            dev_found = lookForDeviceType(*it, devtype);
        }
        else
        {
            qDebug("InteropWindow: %s not found", m_platform_vendor);
            qDebug("InteropWindow: Possible platforms are:");
            for(auto& platform : plats) {qDebug("InteropWindow:\t%s", platform.getInfo<CL_PLATFORM_VENDOR>(&CL_err).c_str()); checkCLerror();}
        }
    }

    return dev_found;
}


bool InteropWindow::lookForDeviceType(cl::Platform& plat_in, cl_bitfield devtype_in)
{
    bool found = false;

    QVector<cl_context_properties> properties = interopCLcontextProps(plat_in);

    // Method 1 //
    /*
    // Create a context holding all devices capable of interoperating with the OpenGL context
    auto possible_context = cl::Context(devtype_in, properties.data(), nullptr, nullptr, &CL_err); checkCLerror();
    std::vector<cl::Device> possible_devices(possible_context.getInfo<CL_CONTEXT_DEVICES>(&CL_err)); checkCLerror();
    */
    // Method 2 //
    
    // Attempt to create an interop context for all the platform devices one by one, and remove those that fail context creation
    std::vector<cl::Device> possible_devices;

    CL_err = plat_in.getDevices(devtype_in, &possible_devices); checkCLerror();
    std::remove_if(possible_devices.begin(), possible_devices.end(), [&properties](const cl::Device& device)
    {
        cl_int err = CL_SUCCESS;
        try
        {
            cl::Context(device, properties.data(), nullptr, nullptr);
        }
        catch (cl::Error e)
        {
            err = e.err();
        }
        
        return err != CL_SUCCESS;
    });
    //cl::Context possible_context(possible_devices, properties.data(), nullptr, nullptr, &CL_err); checkCLerror();
    
    if (!possible_devices.empty())
    {
        qDebug("InteropWindow: Interop capable device(s) found");
        m_cl_platform = plat_in;

        // Multi-device init
        //m_cl_devices = possible_devices;
        //m_cl_context = possible_context;

        // Single-device init
        m_cl_devices.push_back(possible_devices.at(0));
        m_cl_context = cl::Context(possible_devices.at(0), properties.data(), nullptr, nullptr, &CL_err); checkCLerror();
        for (auto& dev : m_cl_devices)
        {
            m_cl_commandqueues.push_back(cl::CommandQueue(m_cl_context, dev, 0, &CL_err));
            checkCLerror();
        }

        found = true;
    }

    return found;
}


InteropWindow::gl_device InteropWindow::nativeGLdevice()
{
#ifdef _WIN32
    // Native way
    QPair<HDC, HGLRC> m_gl_device_native;
    m_gl_device_native.first = wglGetCurrentDC();
    m_gl_device_native.second = wglGetCurrentContext();

    qDebug() << "InteropWindow: Window on screen " << this->screen()->name() << " has native HDC = " << m_gl_device_native.first << " and HGLRC = " << m_gl_device_native.second;

    return m_gl_device_native;
#endif
#ifdef __linux__
    // Native way
    QPair<Display*, GLXContext> m_gl_device_native;
    m_gl_device_native.first = glXGetCurrentDisplay();
    m_gl_device_native.second = glXGetCurrentContext();

    qDebug() << "InteropWindow: Window on screen " << this->screen()->name() << " has native Screen = " << m_gl_device_native.first << " and GLXContext = " << m_gl_device_native.second;

    return m_gl_device_native;
#endif
}


QVector<cl_context_properties> InteropWindow::interopCLcontextProps(const cl::Platform& plat)
{
    QVector<cl_context_properties> result;

    result.append(CL_CONTEXT_PLATFORM);
    result.append(reinterpret_cast<cl_context_properties>(plat()));
#ifdef _WIN32
    result.append(CL_WGL_HDC_KHR);
    result.append(reinterpret_cast<cl_context_properties>(m_gl_device.first));
    result.append(CL_GL_CONTEXT_KHR);
    result.append(reinterpret_cast<cl_context_properties>(m_gl_device.second));
#endif
#ifdef __linux__
    result.append(CL_GLX_DISPLAY_KHR);
    result.append(reinterpret_cast<cl_context_properties>(m_gl_device.first));
    result.append(CL_GL_CONTEXT_KHR);
    result.append(reinterpret_cast<cl_context_properties>(m_gl_device.second));
#endif
    result.append(0);

    return result;
}


void InteropWindow::render_helper()
{
    // Limit FPS by checking time elapsed since last draw
#ifdef QTIMER
    if(m_FPS_limiter.hasExpired(1000./m_max_FPS))
#endif
#ifdef STDTIMER
    auto now = std::chrono::high_resolution_clock::now();
    if(std::chrono::duration_cast<std::chrono::milliseconds>(now - m_FPS_limiter).count() > 1000/m_max_FPS)
#endif
    {
        m_gl_context->makeCurrent(this);
        {   
            // Call unimplemented native OpenGL drawing function
            render();
        }
        m_gl_context->swapBuffers(this);
        
        m_painter_context->makeCurrent(this);
        {
            // Call unimplemented QOpenGLPainting drawing function
            m_painter->begin(m_gl_paintdevice);
            render(m_painter);
            m_painter->end();
        }
        m_painter_context->doneCurrent();
        
        // Restart the limiter once the drawing is done
#ifdef QTIMER
        m_act_FPS = 1000/m_FPS_limiter.restart();
#endif
#ifdef STDTIMER
        now = std::chrono::high_resolution_clock::now();
        m_act_FPS = 1000/std::chrono::duration_cast<std::chrono::milliseconds>(now - m_FPS_limiter).count();
        m_FPS_limiter = now;
#endif
    }
}


void InteropWindow::updateScene_helper()
{
    // Limit IPS by checking time elapsed since last iteration
#ifdef QTIMER
    if(m_IPS_limiter.hasExpired(1000./m_max_IPS))
#endif
#ifdef STDTIMER
    auto now = std::chrono::high_resolution_clock::now();
    if(std::chrono::duration_cast<std::chrono::milliseconds>(now - m_IPS_limiter).count() > 1000/m_max_IPS)
#endif
    {
        // Context needs to be made current because clEnqueueAcquireGLObejects needs it
        m_gl_context->makeCurrent(this);
        {
            // Call unimplemented scene update function
            updateScene();
        }
        m_gl_context->doneCurrent();

        // Restart the limiter only once the update is done
#ifdef QTIMER
        m_act_IPS = 1000/m_IPS_limiter.restart();
#endif
#ifdef STDTIMER
        now = std::chrono::high_resolution_clock::now();
        m_act_IPS = 1000/std::chrono::duration_cast<std::chrono::milliseconds>(now - m_IPS_limiter).count();
        m_IPS_limiter = now;
#endif   
    }

    // If self-animation is turned on, draw the new frame after which the scene will be updated again.
    if(m_animating) renderLater();
}


void InteropWindow::renderLater()
{
    if (!m_render_pending)
    {
        // Set pending helper
        m_render_pending = true;

        // Send message to application event loop to update contents of window
        QCoreApplication::postEvent(this, new QEvent(QEvent::UpdateRequest));
    }
}


bool InteropWindow::event(QEvent *event_in)
{
    // Process messages arriving from application
    switch (event_in->type())
    {
    case QEvent::UpdateRequest:
        // In case a content update is requested, we redraw
        renderNow();
        return true;

    default:
        // In case InteropWindow does not implement handling of the even, we pass it on to the base class
        return QWindow::event(event_in);
    }
}


void InteropWindow::exposeEvent(QExposeEvent *event_in)
{
    // Macro does void cast to silence "unused variable" error
    Q_UNUSED(event_in);

    if (isExposed()) renderNow();
}


void InteropWindow::resizeEvent(QResizeEvent *event_in)
{
    if (isExposed())
    {
        m_gl_context->makeCurrent(this);
        {
            // Call unimplemented resize handler
            resizeGL(event_in);
        }
        m_gl_context->doneCurrent();
        /*
        m_painter_context->makeCurrent(this);
        {
            m_gl_paintdevice->setSize(size());
            m_painter->begin(m_gl_paintdevice);
            m_painter->setViewport(0, 0, event_in->size().width(), event_in->size().height());
            m_painter->end();
        }
        m_painter_context->doneCurrent();
        */
        renderNow();
    }
}


void InteropWindow::renderNow()
{
    // If the window is not present on the desktop, omit drawing
    if (!isExposed()) return;

    // Reset pending helper
    m_render_pending = false;

    // Initialize context if it isn't yet
    if(!m_gl_context_initialized)
    {
        createGLcontext_helper();
        m_gl_context_initialized = true;

        // Activate context on the window's surface. Clause is to indicate active context
        bool valid = m_gl_context->makeCurrent(this);
        {
            // Initialize OpenCL context, OpenGL functions and assets once we have a valid OpenGL context.
            if(valid && m_gl_context_initialized && !m_cl_context_initialized)
            {
                // First initalize the OpenGL functions
                glFuncs = m_gl_context->versionFunctions<QOpenGLFunctions_3_3_Core>();
                glFuncs->initializeOpenGLFunctions();

                // Create the OpenCL context
                createCLcontext_helper();

                // Initialize OpenGL assets
                initializeGL();
                m_assets_initialized = true;

                // Initialize OpenCL assets
                initializeCL();
                m_cl_context_initialized = true;
            }
        }
        m_gl_context->doneCurrent();
    }

    // Do the actual drawing
    render_helper();

    // If self-animation is turned on, update scene which in turn will draw again.
    if(m_animating) updateScene_helper();
}


void InteropWindow::setAnimating(bool animating)
{
    m_animating = animating;

    if(animating) renderLater();
}


void InteropWindow::setMaxIPS(int IPS) {m_max_IPS = IPS;}


void InteropWindow::setMaxFPS(int FPS) {m_max_FPS = FPS;}


void InteropWindow::setDeviceType(cl_bitfield in) {m_device_type = in;}


void InteropWindow::setPlatformVendor(QString in) {m_platform_vendor = in;}


cl::Platform& InteropWindow::CLplatform() {return m_cl_platform;}


std::vector<cl::Device>& InteropWindow::CLdevices() {return m_cl_devices;}


cl::Context& InteropWindow::CLcontext() {return m_cl_context;}


std::vector<cl::CommandQueue>& InteropWindow::CLcommandqueues() {return m_cl_commandqueues;}


const int InteropWindow::getActIPS() {return m_act_IPS;}


const int InteropWindow::getActFPS() {return m_act_FPS;}


const int InteropWindow::getMaxIPS() {return m_max_IPS;}


const int InteropWindow::getMaxFPS() {return m_max_FPS;}


const bool InteropWindow::getAnimating() {return m_animating;}


void InteropWindow::checkCLerror()
{
    if(CL_err != CL_SUCCESS)
    {
        qWarning("OpenCL Error: %s", convertCLerrorToString(CL_err));
        CL_err = 0;
    }
}


void InteropWindow::checkGLerror()
{
    GL_err = glGetError();
    if(GL_err != GL_NO_ERROR)
    {
        qWarning("OpenGL Error: %s", convertGLerrorToString(GL_err));
        GL_err = 0;
    }
}


bool InteropWindow::detectFormatMismatch(QSurfaceFormat& left, QSurfaceFormat& right)
{
    if(left.renderableType() != right.renderableType()) return true;
    if(left.majorVersion() != right.majorVersion()) return true;
    if(left.minorVersion() != right.minorVersion()) return true;
    if(left.swapBehavior() != right.swapBehavior()) return true;
    if(left.profile() != right.profile()) return true;
    if(left.redBufferSize() != right.redBufferSize()) return true;
    if(left.greenBufferSize() != right.greenBufferSize()) return true;
    if(left.blueBufferSize() != right.blueBufferSize()) return true;
    if(left.alphaBufferSize() != right.alphaBufferSize()) return true;
    return false;
}


void InteropWindow::printFormatMismatch()
{
    const char *req_rendertype, *acq_rendertype;
    switch(requestedFormat().renderableType())
    {
    case QSurfaceFormat::RenderableType::DefaultRenderableType :
        req_rendertype = "Default  ";
        break;
    case QSurfaceFormat::RenderableType::OpenGL :
        req_rendertype = "OpenGL   ";
        break;
    case QSurfaceFormat::RenderableType::OpenGLES :
        req_rendertype = "OpenGLES ";
        break;
    default:
        req_rendertype = "Unkown   ";
        break;
    }
    switch(m_gl_context->format().renderableType())
    {
    case QSurfaceFormat::RenderableType::DefaultRenderableType :
        acq_rendertype = "Default  ";
        break;
    case QSurfaceFormat::RenderableType::OpenGL :
        acq_rendertype = "OpenGL   ";
        break;
    case QSurfaceFormat::RenderableType::OpenGLES :
        acq_rendertype = "OpenGLES ";
        break;
    default:
        acq_rendertype = "Unkown   ";
        break;
    }

    int req_MV, req_mv, acq_MV, acq_mv;
    req_MV = requestedFormat().majorVersion();
    req_mv = requestedFormat().minorVersion();
    acq_MV = m_gl_context->format().majorVersion();
    acq_mv = m_gl_context->format().minorVersion();

    const char *req_swapbehavior, *acq_swapbehavior;
    switch(requestedFormat().swapBehavior())
    {
    case QSurfaceFormat::SwapBehavior::DefaultSwapBehavior :
        req_swapbehavior = "Default  ";
        break;
    case QSurfaceFormat::SwapBehavior::SingleBuffer :
        req_swapbehavior = "Single   ";
        break;
    case QSurfaceFormat::SwapBehavior::DoubleBuffer :
        req_swapbehavior = "Double   ";
        break;
    case QSurfaceFormat::SwapBehavior::TripleBuffer :
        req_swapbehavior = "Triple   ";
        break;
    default:
        req_swapbehavior = "Unkown   ";
        break;
    }
    switch(m_gl_context->format().swapBehavior())
    {
    case QSurfaceFormat::SwapBehavior::DefaultSwapBehavior :
        acq_swapbehavior = "Default  ";
        break;
    case QSurfaceFormat::SwapBehavior::SingleBuffer :
        acq_swapbehavior = "Single   ";
        break;
    case QSurfaceFormat::SwapBehavior::DoubleBuffer :
        acq_swapbehavior = "Double   ";
        break;
    case QSurfaceFormat::SwapBehavior::TripleBuffer :
        acq_swapbehavior = "Triple   ";
        break;
    default:
        acq_swapbehavior = "Unkown   ";
        break;
    }

    const char *req_ctxprofile, *acq_ctxprofile;
    switch(requestedFormat().profile())
    {
    case QSurfaceFormat::OpenGLContextProfile::CompatibilityProfile :
        req_ctxprofile = "Compat   ";
        break;
    case QSurfaceFormat::OpenGLContextProfile::CoreProfile :
        req_ctxprofile = "Core     ";
        break;
    case QSurfaceFormat::OpenGLContextProfile::NoProfile :
        req_ctxprofile = "NoProfile";
        break;
    default:
        req_ctxprofile = "Unkown   ";
        break;
    }
    switch(m_gl_context->format().profile())
    {
    case QSurfaceFormat::OpenGLContextProfile::CompatibilityProfile :
        acq_ctxprofile = "Compat   ";
        break;
    case QSurfaceFormat::OpenGLContextProfile::CoreProfile :
        acq_ctxprofile = "Core     ";
        break;
    case QSurfaceFormat::OpenGLContextProfile::NoProfile :
        acq_ctxprofile = "NoProfile";
        break;
    default:
        acq_ctxprofile = "Unkown   ";
        break;
    }

    int req_r, req_g, req_b, req_a, acq_r, acq_g, acq_b, acq_a;
    req_r = requestedFormat().redBufferSize();
    req_g = requestedFormat().greenBufferSize();
    req_b = requestedFormat().blueBufferSize();
    req_a = requestedFormat().alphaBufferSize();
    acq_r = m_gl_context->format().redBufferSize();
    acq_g = m_gl_context->format().greenBufferSize();
    acq_b = m_gl_context->format().blueBufferSize();
    acq_a = m_gl_context->format().alphaBufferSize();

    qWarning("InteropWindow: Property name  | Requested | Acquired");
    qWarning("InteropWindow: RenderableType | %s | %s", req_rendertype, acq_rendertype);
    qWarning("InteropWindow: ContextProfile | %s | %s", req_ctxprofile, acq_ctxprofile);
    qWarning("InteropWindow: SwapBehavior   | %s | %s", req_swapbehavior, acq_swapbehavior);
    qWarning("InteropWindow: OpenGL version | %d.%d       | %d.%d", req_MV, req_mv, acq_MV, acq_mv);
    qWarning("InteropWindow: R.G.B.A bits   | %d.%d.%d.%d   | %d.%d.%d.%d", req_r, req_g, req_b, req_a, acq_r, acq_g, acq_b, acq_a);
    qWarning("InteropWindow: Depth bits     | %d        | %d", requestedFormat().depthBufferSize(), m_gl_context->format().depthBufferSize());
    qWarning("InteropWindow: Stencil bits   | %d         | %d", requestedFormat().stencilBufferSize(), m_gl_context->format().stencilBufferSize());
    qWarning("InteropWindow: Stereo         | %d         | %d", requestedFormat().stereo(), m_gl_context->format().stereo());
}


const char* InteropWindow::convertCLerrorToString(cl_int error)
{
    switch(error)
    {
         case 0: return "CL_SUCCESS"; break;
         case -1: return "CL_DEVICE_NOT_FOUND"; break;
         case -2: return "CL_DEVICE_NOT_AVAILABLE"; break;
         case -3: return "CL_COMPILER_NOT_AVAILABLE"; break;
         case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE"; break;
         case -5: return "CL_OUT_OF_RESOURCES"; break;
         case -6: return "CL_OUT_OF_HOST_MEMORY"; break;
         case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE"; break;
         case -8: return "CL_MEM_COPY_OVERLAP"; break;
         case -9: return "CL_IMAGE_FORMAT_MISMATCH"; break;
         case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED"; break;
         case -11: return "CL_BUILD_PROGRAM_FAILURE"; break;
         case -12: return "CL_MAP_FAILURE"; break;
         case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET"; break;
         case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST"; break;
         case -15: return "CL_COMPILE_PROGRAM_FAILURE"; break;
         case -16: return "CL_LINKER_NOT_AVAILABLE"; break;
         case -17: return "CL_LINK_PROGRAM_FAILURE"; break;
         case -18: return "CL_DEVICE_PARTITION_FAILED "; break;
         case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE"; break;

         case -30: return "CL_INVALID_VALUE"; break;
         case -31: return "CL_INVALID_DEVICE_TYPE"; break;
         case -32: return "CL_INVALID_PLATFORM"; break;
         case -33: return "CL_INVALID_DEVICE"; break;
         case -34: return "CL_INVALID_CONTEXT"; break;
         case -35: return "CL_INVALID_QUEUE_PROPERTIES"; break;
         case -36: return "CL_INVALID_COMMAND_QUEUE"; break;
         case -37: return "CL_INVALID_HOST_PTR"; break;
         case -38: return "CL_INVALID_MEM_OBJECT"; break;
         case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR"; break;
         case -40: return "CL_INVALID_IMAGE_SIZE"; break;
         case -41: return "CL_INVALID_SAMPLER"; break;
         case -42: return "CL_INVALID_BINARY"; break;
         case -43: return "CL_INVALID_BUILD_OPTIONS"; break;
         case -44: return "CL_INVALID_PROGRAM"; break;
         case -45: return "CL_INVALID_PROGRAM_EXECUTABLE"; break;
         case -46: return "CL_INVALID_KERNEL_NAME"; break;
         case -47: return "CL_INVALID_KERNEL_DEFINITION"; break;
         case -48: return "CL_INVALID_KERNEL"; break;
         case -49: return "CL_INVALID_ARG_INDEX"; break;
         case -50: return "CL_INVALID_ARG_VALUE"; break;
         case -51: return "CL_INVALID_ARG_SIZE"; break;
         case -52: return "CL_INVALID_KERNEL_ARGS"; break;
         case -53: return "CL_INVALID_WORK_DIMENSION"; break;
         case -54: return "CL_INVALID_WORK_GROUP_SIZE"; break;
         case -55: return "CL_INVALID_WORK_ITEM_SIZE"; break;
         case -56: return "CL_INVALID_GLOBAL_OFFSET"; break;
         case -57: return "CL_INVALID_EVENT_WAIT_LIST"; break;
         case -58: return "CL_INVALID_EVENT"; break;
         case -59: return "CL_INVALID_OPERATION"; break;
         case -60: return "CL_INVALID_GL_OBJECT"; break;
         case -61: return "CL_INVALID_BUFFER_SIZE"; break;
         case -62: return "CL_INVALID_MIP_LEVEL"; break;
         case -63: return "CL_INVALID_GLOBAL_WORK_SIZE"; break;
         case -64: return "CL_INVALID_PROPERTY"; break;
         case -65: return "CL_INVALID_IMAGE_DESCRIPTOR"; break;
         case -66: return "CL_INVALID_COMPILER_OPTIONS"; break;
         case -67: return "CL_INVALID_LINKER_OPTIONS"; break;
         case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT"; break;
         default: return "Unknown OpenCL error"; break;
     }
}


const char* InteropWindow::convertGLerrorToString(GLint error)
{
	switch(error)
	{
	    case GL_INVALID_ENUM: return "GL_INVALID_ENUM"; break;
	    case GL_INVALID_VALUE: return "GL_INVALID_VALUE"; break;
	    case GL_INVALID_OPERATION: return "GL_INVALID_OPERATION"; break;
	    case GL_INVALID_FRAMEBUFFER_OPERATION: return "GL_INVALID_FRAMEBUFFER_OPERATION"; break;
	    case GL_OUT_OF_MEMORY: return "GL_OUT_OF_MEMORY"; break;
	    default: return "Unknown OpenGL error"; break;
	}
}
