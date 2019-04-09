#include <SYCL-Conway.hpp>


Conway::Conway(std::size_t plat_id,
               cl_bitfield dev_type,
               QWindow *parent)
    : InteropWindow(plat_id, dev_type, parent)
    , dev_id(0)
    , imageDrawn(false)
    , needMatrixReset(true)
{
}


// Override unimplemented InteropWindow function
void Conway::initializeGL()
{
    qDebug("Conway: Entering initializeGL");
    // Initialize OpenGL resources
    vs = std::make_unique<QOpenGLShader>(QOpenGLShader::Vertex, this);
    fs = std::make_unique<QOpenGLShader>(QOpenGLShader::Fragment, this);
    sp = std::make_unique<QOpenGLShaderProgram>(this);
    vbo = std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::VertexBuffer);
    vao = std::make_unique<QOpenGLVertexArrayObject>(this);
    texs = { std::make_unique<QOpenGLTexture>(QOpenGLTexture::Target::Target2D),
             std::make_unique<QOpenGLTexture>(QOpenGLTexture::Target::Target2D) };

    // Initialize frame buffer
    glFuncs->glViewport(0, 0, width(), height());   checkGLerror();
    glFuncs->glClearColor(0.0, 0.0, 0.0, 1.0);      checkGLerror();
    glFuncs->glDisable(GL_DEPTH_TEST);              checkGLerror();
    glFuncs->glDisable(GL_CULL_FACE);               checkGLerror();

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
        //  x  ,   y  ,  x  ,   y
        { -1.0f, -1.0f, 0.0f, 0.0f,
          -1.0f,  1.0f, 0.0f, 1.0f,
           1.0f,  1.0f, 1.0f, 1.0f,
           1.0f, -1.0f, 1.0f, 0.0f,
          -1.0f, -1.0f, 0.0f, 0.0f };

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
        sp->enableAttributeArray(0);  checkGLerror();
        sp->enableAttributeArray(1);  checkGLerror();
        sp->setAttributeArray(0, GL_FLOAT, (GLvoid *)(NULL), 2, sizeof(cl::sycl::float4));                      checkGLerror();
        sp->setAttributeArray(1, GL_FLOAT, (GLvoid *)(NULL + 2 * sizeof(float)), 2, sizeof(cl::sycl::float4));  checkGLerror();
        sp->release(); checkGLerror();
    }
    vao->release();;

    std::vector<cl::sycl::uint4> texels;
    std::generate_n(std::back_inserter(texels),
                    width() * height(),
                    [prng = std::default_random_engine{},
                     dist = std::uniform_int_distribution<cl::sycl::uint4::element_type>{ 0, 1 }]() mutable
    {
        return cl::sycl::uint4{ 1, 1, 1, dist(prng) };
    });

    for (auto& tex : texs)
    {
        if (!tex->create()) qWarning("Failed to create texture");
        tex->bind();
        tex->setFormat(QOpenGLTexture::TextureFormat::RGBA8U);
        tex->setMinificationFilter(QOpenGLTexture::Filter::Nearest);
        tex->setMagnificationFilter(QOpenGLTexture::Filter::Nearest);
        tex->setWrapMode(QOpenGLTexture::WrapMode::Repeat);
        tex->setMipMaxLevel(0);
        tex->setSize(width(), height());
        tex->setData(QOpenGLTexture::PixelFormat::RGBA, QOpenGLTexture::PixelType::UInt32, texels.data());
    }

    qDebug("Conway: Leaving initializeGL");
}

// Override unimplemented InteropWindow function
void Conway::initializeCL()
{
    qDebug("Conway: Entering initializeCL");

    // Translate OpenGL handles to OpenCL
    std::transform(texs.cbegin(), texs.cend(), CL_latticeImages.begin(), [this](const std::unique_ptr<QOpenGLTexture>& tex)
    {
        return cl::ImageGL{ CLcontext(), CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, vbo->bufferId() };
    });

    // Translate OpenCL handles to SYCL
    context = cl::sycl::context{ CLcontext()() };
    device = cl::sycl::device{ CLdevices().at(dev_id)() };
    compute_queue = cl::sycl::queue{ CLcommandqueues().at(dev_id)(), context };

    std::transform(CL_latticeImages.cbegin(), CL_latticeImages.cend(), latticeImages.begin(), [this](const cl::ImageGL& image)
    {
        return std::make_unique<cl::sycl::image<2>>(image(), context);
    });

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
    acquire.wait(); cl::finish();

    compute_queue.submit([&](cl::sycl::handler& cgh)
    {
        auto old_lattice = latticeImages[Buffer::Front]->get_access<cl::sycl::uint4, cl::sycl::access::mode::read>(cgh);
        auto new_lattice = latticeImages[Buffer::Back]->get_access<cl::sycl::uint4, cl::sycl::access::mode::write>(cgh);
        
        cgh.parallel_for<kernels::ConwayStep>(cl::sycl::range<2>{ old_lattice.get_range() },
                                              [=](const cl::sycl::item<2> item)
        {
            using namespace cl::sycl;
            using elem_type = cl::sycl::uint4::element_type;

            sampler sampler(coordinate_normalization_mode::unnormalized,
                            addressing_mode::repeat,
                            filtering_mode::nearest);

            auto old = [=](cl::sycl::id<2> id) { return old_lattice.read(id.operator cl::sycl::int2(), sampler).a(); };

            auto id = item.get_id();

            std::array<elem_type, 8> neighbours =
                { old(id + id<2>(-1,+1)), old(id + id<2>(0,+1)), old(id + id<2>(+1,+1)),
                  old(id + id<2>(-1,0)),                         old(id + id<2>(+1,0)),
                  old(id + id<2>(-1,-1)), old(id + id<2>(0,-1)), old(id + id<2>(+1,-1))
                };
            elem_type self = old(id);

            auto count = std::count(neighbours.cbegin(), neighbours.cend(), 1);

            auto val = self ?
                (count < 2 || count > 3 ? 0 : 1) :
                (count == 3 ? 1 : 0);

            new_lattice.write(id.operator cl::sycl::int2(), cl::sycl::uint4{ 1, 1, 1, val });
        });
    });

    CLcommandqueues().at(dev_id).enqueueReleaseGLObjects(&interop_resources, nullptr, &release);

    // Wait for all OpenCL commands to finish
    if (!cl_khr_gl_event_supported)
        cl::finish();
    else
        release.wait();
    
    // Swap front and back buffer handles
    std::swap(CL_latticeImages[Front], CL_latticeImages[Back]);
    std::swap(latticeImages[Front], latticeImages[Back]);
    
    imageDrawn = false;
}

// Override unimplemented InteropWindow function
void Conway::render()
{
    // Update matrices as needed
    if(needMatrixReset) setMatrices();

    // Clear Frame Buffer and Z-Buffer
    glFuncs->glClear(GL_COLOR_BUFFER_BIT); checkGLerror();

    // Draw
    if(!sp->bind()) qWarning("QGripper: Failed to bind shaderprogram");
    vao->bind(); checkGLerror();

    glFuncs->glDrawArrays(GL_TRIANGLE_FAN, 0, static_cast<GLsizei>(5)); checkGLerror();

    vao->release(); checkGLerror();
    sp->release(); checkGLerror();

    // Wait for all drawing commands to finish
    if (!cl_khr_gl_event_supported)
    {
        glFuncs->glFinish(); checkGLerror();
    }
    else
    {
        glFuncs->glFlush(); checkGLerror();
    }
    imageDrawn = true;
}

// Override unimplemented InteropWindow function
void Conway::render(QPainter* painter)
{
    QString text("QGripper: ");
    text.append("IPS = ");
    text.append(QString::number(getActIPS()));
    text.append(" | FPS = ");
    text.append(QString::number(getActFPS()));
     
    //painter->setBackgroundMode(Qt::TransparentMode);
    //painter->setPen(Qt::white);
    //painter->setFont(QFont("Arial", 30));
    //painter->drawText(QRect(0, 0, 280, 50), Qt::AlignLeft, text);
    
    this->setTitle(text);

    //Q_UNUSED(painter);
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
    matProj.ortho(-0.5f * width(),
                  +0.5f * width(),
                  -0.5f * height(),
                  +0.5f * height(),
                  std::numeric_limits<float>::epsilon(),
                  std::numeric_limits<float>::max());

    sp->bind();
    sp->setUniformValue("mat_MVP", matProj * matView * matWorld);
    sp->release();
}
