#include <SYCL-Conway.hpp>


Conway::Conway(std::size_t plat,
               std::size_t dev,
               cl_bitfield type,
               QWindow *parent)
    : InteropWindow(plat, type, parent)
    , dev_id(dev)
    , cl_khr_gl_event_supported(false)
    , imageDrawn(false)
    , needMatrixReset(true)
{
}

// Override unimplemented InteropWindow function
void Conway::initializeGL()
{
    qDebug("Conway: Entering initializeGL");
    std::unique_ptr<QOpenGLDebugLogger> log(new QOpenGLDebugLogger(this));
    if (!log->initialize()) qWarning("Conway: QDebugLogger failed to initialize");

    // Initialize OpenGL resources
    vs = std::make_unique<QOpenGLShader>(QOpenGLShader::Vertex, this);
    fs = std::make_unique<QOpenGLShader>(QOpenGLShader::Fragment, this);
    sp = std::make_unique<QOpenGLShaderProgram>(this);
    vbo = std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::VertexBuffer);
    vao = std::make_unique<QOpenGLVertexArrayObject>(this);
    texs = { std::make_unique<QOpenGLTexture>(QOpenGLTexture::Target::Target2D),
             std::make_unique<QOpenGLTexture>(QOpenGLTexture::Target::Target2D) };

    // Initialize frame buffer
    glFuncs->glViewport(0, 0, width(), height());
    glFuncs->glClearColor(0.0, 0.0, 0.0, 1.0);
    glFuncs->glDisable(GL_DEPTH_TEST);
    glFuncs->glDisable(GL_CULL_FACE);

    // Create shaders
    qDebug("Conway: Building shaders...");
    if (!vs->compileSourceFile( (shader_location + "/Vertex.glsl").c_str())) qWarning("%s", vs->log().data());
    if (!fs->compileSourceFile( (shader_location + "/Fragment.glsl").c_str())) qWarning("%s", fs->log().data());
    qDebug("Conway: Done building shaders");

    // Create and link shaderprogram
    qDebug("Conway: Linking shaders...");
    if (!sp->addShader(vs.get())) qWarning("Conway: Could not add vertex shader to shader program");
    if (!sp->addShader(fs.get())) qWarning("Conway: Could not add fragment shader to shader program");
    if (!sp->link()) qWarning("%s", sp->log().data());
    qDebug("Conway: Done linking shaders");

    // Init device memory
    qDebug("Conway: Initializing OpenGL buffers...");

    std::vector<float> quad =
        //  vertices  , tex coords
        //  x  ,   y  ,  u  ,   v
        { -1.0f, -1.0f, 0.0f, 0.0f,
          -1.0f,  1.0f, 0.0f, 1.0f,
           1.0f, -1.0f, 1.0f, 0.0f,
           1.0f,  1.0f, 1.0f, 1.0f };

    if (!vbo->create()) qWarning("Conway: Could not create VBO");
    if (!vbo->bind()) qWarning("Conway: Could not bind VBO");
    vbo->setUsagePattern(QOpenGLBuffer::StaticDraw);
    vbo->allocate(quad.data(), (int)quad.size() * sizeof(float));
    vbo->release();

    qDebug("Conway: Done initializing OpenGL buffers");

    // Setup VAO for the VBO
    if (!vao->create()) qWarning("Conway: Could not create VAO");

    vao->bind();
    {
        if (!vbo->bind()) qWarning("Conway: Could not bind VBO");

        // Setup shader attributes (can only be done when a VBO is bound, VAO does not store shader state
        if (!sp->bind()) qWarning("Conway: Failed to bind shaderprogram");
        sp->enableAttributeArray(0);
        sp->enableAttributeArray(1);
        sp->setAttributeArray(0, GL_FLOAT, (GLvoid *)(NULL), 2, sizeof(cl::sycl::float4));
        sp->setAttributeArray(1, GL_FLOAT, (GLvoid *)(NULL + 2 * sizeof(float)), 2, sizeof(cl::sycl::float4));
        sp->release();
    }
    vao->release();

    std::vector<std::array<float, 4>> texels;
    std::generate_n(std::back_inserter(texels),
                    width() * height(),
        [prng = std::ranlux48{ std::random_device{}() },
         dist = std::uniform_int_distribution<std::uint32_t>{ 0, 1 }]() mutable
    {
        auto rand = dist(prng);
        return std::array<float, 4>{ (float)rand, (float)rand, (float)rand, 0.f };
    });

    // Quote from the QOpenGLTexture documentation of Qt 5.12
    //
    // The typical usage pattern for QOpenGLTexture is:
    //  -  Instantiate the object specifying the texture target type
    //  -  Set properties that affect the storage requirements e.g.storage format, dimensions
    //  -  Allocate the server - side storage
    //  -  Optionally upload pixel data
    //  -  Optionally set any additional properties e.g.filtering and border options
    //  -  Render with texture or render to texture

    for (auto& tex : texs)
    {
        tex->setSize(width(), height());
        tex->setFormat(QOpenGLTexture::TextureFormat::RGBA32F);
        tex->allocateStorage(QOpenGLTexture::PixelFormat::RGBA, QOpenGLTexture::PixelType::Float32);
        tex->setData(QOpenGLTexture::PixelFormat::RGBA, QOpenGLTexture::PixelType::Float32, texels.data());
        tex->generateMipMaps();
    }

    for (const QOpenGLDebugMessage& message : log->loggedMessages()) qDebug() << message << "\n";

     qDebug("Conway: Leaving initializeGL");
}

// Override unimplemented InteropWindow function
void Conway::initializeCL()
{
    qDebug("Conway: Entering initializeCL");

    // Translate OpenGL handles to OpenCL
    std::transform(texs.cbegin(), texs.cend(), CL_latticeImages.begin(), [this](const std::unique_ptr<QOpenGLTexture>& tex)
    {
        return cl::ImageGL{ CLcontext(), CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, tex->textureId() };
    });

    // Translate OpenCL handles to SYCL
    auto async_error_handler = [](cl::sycl::exception_list errors)
    {
        for (auto error : errors)
        {
            try { std::rethrow_exception(error); }
            catch (cl::sycl::exception e)
            {
                qDebug() << e.what();
                std::exit(e.get_cl_code());
            }
            catch (std::exception e)
            {
                qDebug() << e.what();
                std::exit(EXIT_FAILURE);
            }
        }
    };

    try
    {
        context = cl::sycl::context{ CLcontext()(), async_error_handler };
        device = cl::sycl::device{ CLdevices().at(dev_id)() };
        compute_queue = cl::sycl::queue{ CLcommandqueues().at(dev_id)(), context };

        std::transform(CL_latticeImages.cbegin(), CL_latticeImages.cend(), latticeImages.begin(), [this](const cl::ImageGL & image)
        {
            return std::make_unique<cl::sycl::image<2>>(image(), context);
        });
    }
    catch(cl::sycl::exception e)
    {
        qDebug() << e.what();
        std::exit(e.get_cl_code());
    }

    qDebug("Conway: Querying device capabilities");
    auto extensions = device.get_info<cl::sycl::info::device::extensions>();
    cl_khr_gl_event_supported = std::find(extensions.cbegin(), extensions.cend(), "cl_khr_gl_event") != extensions.cend();

    // Init bloat vars
    std::copy(CL_latticeImages.cbegin(), CL_latticeImages.cend(), std::back_inserter(interop_resources));

    qDebug("Conway: Leaving initializeCL");
}


// Override unimplemented InteropWindow function
void Conway::updateScene()
{
    // NOTE 1: When cl_khr_gl_event is NOT supported, then clFinish() is the only portable
    //         sync method and hence that will be called.
    //
    // NOTE 2.1: When cl_khr_gl_event IS supported AND the possibly conflicting OpenGL
    //           context is current to the thread, then it is sufficient to wait for events
    //           of clEnqueueAcquireGLObjects, as the spec guarantees that all OpenGL
    //           operations involving the acquired memory objects have finished. It also
    //           guarantees that any OpenGL commands issued after clEnqueueReleaseGLObjects
    //           will not execute until the release is complete.
    //         
    //           See: opencl-1.2-extensions.pdf (Rev. 15. Chapter 9.8.5)

    cl::Event acquire, release;

    CLcommandqueues().at(dev_id).enqueueAcquireGLObjects(&interop_resources, nullptr, &acquire);

    try
    {
        compute_queue.submit([&](cl::sycl::handler& cgh)
        {
            using namespace cl::sycl;

            auto old_lattice = latticeImages[Buffer::Front]->get_access<float4, access::mode::read>(cgh);
            auto new_lattice = latticeImages[Buffer::Back]->get_access<float4, access::mode::write>(cgh);

            sampler periodic{ coordinate_normalization_mode::normalized,
                              addressing_mode::repeat,
                              filtering_mode::nearest };

            float2 d = float2{ 1, 1 } / float2{ old_lattice.get_range()[0], old_lattice.get_range()[1] };

            cgh.parallel_for<kernels::ConwayStep>(range<2>{ old_lattice.get_range() },
                                                  [=](const item<2> i)
            {
                // Convert unnormalized floating coords offsetted by self to normalized uv
                auto uv = [=, s = float2{ i.get_id()[0], i.get_id()[1] }, d2 = d * 0.5f](float2 in) { return (s + in) * d + d2; };

                auto old = [=](float2 in) { return old_lattice.read(uv(in), periodic).r() > 0.5f; };
                auto next = [=](bool v) { new_lattice.write((int2)i.get_id(), float4{ v, v, v, 1.f }); };

                std::array<bool, 8> neighbours = {
                      old(float2{-1,+1}), old(float2{0,+1}), old(float2{+1,+1}),
                      old(float2{-1,0}),                     old(float2{+1,0}),
                      old(float2{-1,-1}), old(float2{0,-1}), old(float2{+1,-1}) };

                bool self = old(float2{0,0});

                auto count = std::count(neighbours.cbegin(), neighbours.cend(), true);

                next(self ? (count < 2 || count > 3 ? 0.f : 1.f) : (count == 3 ? 1.f : 0.f));
            });
        });
    }
    catch (cl::sycl::compile_program_error e)
    {
        qDebug() << e.what();
        std::exit(e.get_cl_code());
    }
    catch (cl::sycl::exception e)
    {
        qDebug() << e.what();
        std::exit(e.get_cl_code());
    }
    catch (std::exception e)
    {
        qDebug() << e.what();
        std::exit(EXIT_FAILURE);
    }

    CLcommandqueues().at(dev_id).enqueueReleaseGLObjects(&interop_resources, nullptr, &release);

    // Wait for all OpenCL commands to finish
    if (!cl_khr_gl_event_supported) cl::finish();
    else release.wait();
  
    // Swap front and back buffer handles
    std::swap(CL_latticeImages[Front], CL_latticeImages[Back]);
    std::swap(latticeImages[Front], latticeImages[Back]);
    std::swap(texs[Front], texs[Back]);
    
    imageDrawn = false;
}

// Override unimplemented InteropWindow function
void Conway::render()
{
    std::unique_ptr<QOpenGLDebugLogger> log(new QOpenGLDebugLogger(this));
    if (!log->initialize()) qWarning("Conway: QDebugLogger failed to initialize");

    // Update matrices as needed
    if(needMatrixReset) setMatrices();

    // Clear Frame Buffer and Z-Buffer
    glFuncs->glClear(GL_COLOR_BUFFER_BIT);

    // Draw
    if(!sp->bind()) qWarning("QGripper: Failed to bind shaderprogram");
    vao->bind();

    texs[Buffer::Front]->bind();

    glFuncs->glDrawArrays(GL_TRIANGLE_STRIP, 0, static_cast<GLsizei>(4));

    texs[Buffer::Front]->release();
    vao->release();
    sp->release();

    // Wait for all drawing commands to finish
    if (!cl_khr_gl_event_supported) glFuncs->glFinish();
    else glFuncs->glFlush();

    imageDrawn = true;

    for (const QOpenGLDebugMessage& message : log->loggedMessages()) qDebug() << message << "\n";
}

// Override unimplemented InteropWindow function
void Conway::render(QPainter* painter)
{
    QString text("SYCL-Conway: ");
    text.append("IPS = ");
    text.append(QString::number(getActIPS()));
    text.append(" | FPS = ");
    text.append(QString::number(getActFPS()));
    
    this->setTitle(text);
}

// Override InteropWindow function
void Conway::resizeGL(QResizeEvent* event_in)
{
    glFuncs->glViewport(0, 0, event_in->size().width(), event_in->size().height());
    checkGLerror();

    needMatrixReset = true; // projection matrix need to be recalculated
}

// Override InteropWindow function
bool Conway::event(QEvent *event_in)
{
    QKeyEvent* keyboard_event;

    // Process messages arriving from application
    switch (event_in->type())
    {
    case QEvent::KeyPress:
        keyboard_event = static_cast<QKeyEvent*>(event_in);

        if(keyboard_event->key() == Qt::Key::Key_Space) setAnimating(!getAnimating());
        return true;

    default:
        // In case InteropWindow does not implement handling of the even, we pass it on to the base class
        return InteropWindow::event(event_in);
    }
}

// Helper function
void Conway::setMatrices()
{
    // Set camera to view the origo from the z-axis with up along the y-axis
    // and distance so the entire sim space is visible with given field-of-view
    QVector3D vecTarget{ 0, 0, 0 };
    QVector3D vecUp{ 0, 1, 0 };
    QVector3D vecEye = vecTarget + QVector3D{ 0, 0, 1 };

    QMatrix4x4 matWorld; // Identity

    QMatrix4x4 matView; // Identity
    matView.lookAt(vecEye, vecTarget, vecUp);

    QMatrix4x4 matProj; // Identity
    matProj.ortho((float)width(), (float)width(),
                  (float)height(), (float)height(),
                  std::numeric_limits<float>::epsilon(),
                  std::numeric_limits<float>::max());

    sp->bind();
    sp->setUniformValue("mat_MVP", matProj * matView * matWorld);
    sp->release();

    needMatrixReset = false;
}
