#include <NBody.hpp>


NBody::NBody(std::size_t plat_id,
	         cl_bitfield dev_type,
	         std::size_t particle_count,
	         QWindow *parent)
    : InteropWindow(plat_id, dev_type, parent)
	, particle_count(particle_count)
	, x_abs_range(192.f)
	, y_abs_range(128.f)
	, z_abs_range(32.f)
	, mass_min(100.f)
	, mass_max(500.f)
	, dev_id(0)
    //, CL_velBuffs{ { cl::Buffer{ CLcontext(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, particle_count * sizeof(real4) },
    //                 cl::Buffer{ CLcontext(), CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, particle_count * sizeof(real4) } } }
    , posBuffs{ { cl::sycl::buffer<real4>{ cl::sycl::range<1>{1} },
                  cl::sycl::buffer<real4>{ cl::sycl::range<1>{1} } } }
    , velBuffs{ { cl::sycl::buffer<real4>{ cl::sycl::range<1>{1} },
                  cl::sycl::buffer<real4>{ cl::sycl::range<1>{1} } } }
    , dist(3 * std::max({ x_abs_range,
                          y_abs_range ,
                          z_abs_range }))
    , phi(0)
    , theta(0)
    , imageDrawn(false)
    , needMatrixReset(true)
{
}


// Override unimplemented InteropWindow function
void NBody::initializeGL()
{
	qDebug("NBody: Entering initializeGL");
	// Initialize OpenGL resources
	vs = std::make_unique<QOpenGLShader>(QOpenGLShader::Vertex, this);
	fs = std::make_unique<QOpenGLShader>(QOpenGLShader::Fragment, this);
	sp = std::make_unique<QOpenGLShaderProgram>(this);
	vbos = { std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::VertexBuffer),
		     std::make_unique<QOpenGLBuffer>(QOpenGLBuffer::VertexBuffer) };
	vaos = { std::make_unique<QOpenGLVertexArrayObject>(this),
		     std::make_unique<QOpenGLVertexArrayObject>(this) };

	// Initialize frame buffer
	glFuncs->glViewport(0, 0, width(), height());   checkGLerror();
	glFuncs->glClearColor(0.0, 0.0, 0.0, 1.0);      checkGLerror();
	glFuncs->glDisable(GL_DEPTH_TEST);              checkGLerror();
	glFuncs->glDisable(GL_CULL_FACE);               checkGLerror();

	// Initialize simulation data
	qDebug("NBody: Allocating host-side memory");
	pos_mass.reserve(particle_count);

	qDebug("NBody: Setting initial states");
	using uni = std::uniform_real_distribution<real>;

	std::generate_n(std::back_inserter(pos_mass),
		            particle_count,
		            [prng = std::default_random_engine(),
		             x_dist = uni(-x_abs_range, x_abs_range),
		             y_dist = uni(-y_abs_range, y_abs_range),
		             z_dist = uni(-z_abs_range, z_abs_range),
		             m_dist = uni(mass_min, mass_max)]() mutable
	{
		return real4{ x_dist(prng),
			          y_dist(prng),
			          z_dist(prng),
			          m_dist(prng) };
	});

	// Create shaders
	qDebug("NBody: Building shaders...");
	if (!vs->compileSourceFile( (shader_location + "/Vertex.glsl").c_str())) qWarning("%s", vs->log().data());
	if (!fs->compileSourceFile( (shader_location + "/Fragment.glsl").c_str())) qWarning("%s", fs->log().data());
	qDebug("NBody: Done building shaders");

	// Create and link shaderprogram
	qDebug("NBody: Linking shaders...");
	if (!sp->addShader(vs.get())) qWarning("NBody: Could not add vertex shader to shader program");
	if (!sp->addShader(fs.get())) qWarning("NBody: Could not add fragment shader to shader program");
	if (!sp->link()) qWarning("%s", sp->log().data());
	qDebug("NBody: Done linking shaders");

	// Init device memory
	qDebug("NBody: Initializing OpenGL buffers...");

	for (auto& vbo : vbos)
	{
		if (!vbo->create()) qWarning("QGripper: Could not create VBO");
		if (!vbo->bind()) qWarning("QGripper: Could not bind VBO");
		vbo->setUsagePattern(QOpenGLBuffer::StaticDraw);
		vbo->allocate((int)pos_mass.size() * sizeof(real4));
		vbo->write(0, pos_mass.data(), (int)pos_mass.size() * sizeof(real4));
		vbo->release();
	}
    pos_mass.clear();

	qDebug("NBody: Done initializing OpenGL buffers");

	// Setup VAOs for the VBOs
	std::transform(vbos.begin(), vbos.end(), vaos.begin(), [this](std::unique_ptr<QOpenGLBuffer>& vbo)
	{
		auto vao = std::make_unique<QOpenGLVertexArrayObject>(this);
        if (!vao->create()) qWarning("QGripper: Could not create VAO");

		vao->bind();
        {
            if (!vbo->bind()) qWarning("QGripper: Could not bind VBO");

            // Setup shader attributes (can only be done when a VBO is bound, VAO does not store shader state
            if (!sp->bind()) qWarning("QGripper: Failed to bind shaderprogram");
            sp->enableAttributeArray(0);  checkGLerror();
            sp->enableAttributeArray(1);  checkGLerror();
            sp->setAttributeArray(0, GL_FLOAT, (GLvoid *)(NULL), 3, sizeof(real4));                     checkGLerror();
            sp->setAttributeArray(1, GL_FLOAT, (GLvoid *)(NULL + 3 * sizeof(real)), 1, sizeof(real4));  checkGLerror();
            sp->release(); checkGLerror();
        }
		vao->release();

		return std::move(vao);
	});

	qDebug("NBody: Leaving initializeGL");
}

// Override unimplemented InteropWindow function
void NBody::initializeCL()
{
	qDebug("NBody: Entering initializeCL");

    context = cl::sycl::context{ CLcontext()() };
    device = cl::sycl::device{ CLdevices().at(dev_id)() };
    compute_queue = cl::sycl::queue{ CLcommandqueues().at(dev_id)(), context };

	qDebug("NBody: Querying device capabilities");
    auto extensions = device.get_info<cl::sycl::info::device::extensions>();
	cl_khr_gl_event_supported = std::find(extensions.cbegin(), extensions.cend(), "cl_khr_gl_event") != extensions.cend();

	// Create Buffers
	qDebug("NBody: Creating SYCL buffer objects");
    for (auto& CL_velBuff : CL_velBuffs)
    {
        CL_velBuffs = { cl::Buffer{ CLcontext(), CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE, particle_count * sizeof(real4) },
                        cl::Buffer{ CLcontext(), CL_MEM_HOST_WRITE_ONLY | CL_MEM_READ_WRITE, particle_count * sizeof(real4) } };
    }

    std::transform(CL_velBuffs.cbegin(), CL_velBuffs.cend(),
                   velBuffs.begin(),
                   [&](const cl::Buffer& CL_velBuff)
    {
        cl::sycl::buffer<real4> velBuff{ CL_velBuff(), context };

        auto access = velBuff.get_access<cl::sycl::access::mode::discard_write>();

        std::fill_n(access.get_pointer(), access.get_count(), real4{ 0.f, 0.f, 0.f, 0.f });

        return velBuff;
    });

	std::transform(vbos.cbegin(), vbos.cend(), posBuffs.begin(), [this](const std::unique_ptr<QOpenGLBuffer>& vbo)
	{
        return cl::sycl::buffer<real4>{ cl::BufferGL{ CLcontext(), CL_MEM_READ_WRITE, vbo->bufferId() }(), context };
	});

	// Init bloat vars
	std::transform(vbos.cbegin(), vbos.cend(),
                   std::back_inserter(interop_resources),
                   [this](const std::unique_ptr<QOpenGLBuffer>& vbo)
	{
		return cl::BufferGL{ CLcontext(), CL_MEM_READ_WRITE, vbo->bufferId() };
	});

	qDebug("NBody: Leaving initializeCL");
}


template <typename T, std::size_t J>
struct unroll_step
{
    unroll_step(std::size_t i, T&& t)
    {
        unroll_step<T, J - 1>(i, std::forward<T>(t));
        t(i + J);
    }
};

template <typename T>
struct unroll_step<T, (std::size_t)0>
{
    unroll_step(std::size_t i, T&& t) { t(i); }
};

template <std::size_t N, typename T>
void unroll_loop(std::size_t first, std::size_t last, T&& t)
{
    std::size_t i = first;

    for (; i + N < last; i += N)
        unroll_step<T, N - 1>(i, std::forward<T>(t));

    for (; i < last; ++i)
        t(i);
}


// Override unimplemented InteropWindow function
void NBody::updateScene()
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
    acquire.wait();

    compute_queue.submit([&](cl::sycl::handler& cgh)
    {
        auto pos = posBuffs[DoubleBuffer::Front].get_access<cl::sycl::access::mode::read_write>();
        auto new_pos = posBuffs[DoubleBuffer::Back].get_access<cl::sycl::access::mode::read_write>();
        auto vel = velBuffs[DoubleBuffer::Front].get_access<cl::sycl::access::mode::read_write>();
        auto new_vel = velBuffs[DoubleBuffer::Back].get_access<cl::sycl::access::mode::read_write>();

        cgh.parallel_for<kernels::NBodyStep>(cl::sycl::range<1>{ particle_count }, [=](const cl::sycl::item<1> item)
        {
            /*
            real4 myPos = pos[item];
            real4 acc = { 0.f, 0.f, 0.f, 0.f };
            real epsSqr = std::numeric_limits<real>::epsilon();
            real deltaTime = real(0.005);

            auto xyz = [](real4& vec) { return vec.swizzle<0, 1, 2>(); };
            auto interact = [&](const size_t i)
            {
                real4 p = pos[i];
                real4 r;
                xyz(r) = xyz(p) - xyz(myPos);
                real distSqr = r.x() * r.x() + r.y() * r.y() + r.z() * r.z();

                real invDist = 1.0f / sqrt(distSqr + epsSqr);
                real invDistCube = invDist * invDist * invDist;
                real s = p.w() * invDistCube;

                // accumulate effect of all particles
                xyz(acc) += s * xyz(r);
            };

            // NOTE 1:
            //
            // This loop construct unrolls the outer loop in chunks of UNROLL_FACTOR
            // up to a point that still fits into numBodies. After the unrolled part
            // the remainder os particles are accounted for. (NOTE 1.1: the loop variable)
            // 'j' is not used anywhere in the body. It's only used as a trivial
            // unroll construct. NOTE 1.2: 'i' is left intact after the unrolled loops.
            // The finishing loop picks up where the other left off.
            //
            // NOTE 2:
            //
            // epsSqr is used to omit self interaction checks alltogether by introducing
            // a minimal skew in the deistance calculation. Thus, the almost infinity
            // in invDistCube is suppressed by the ideally 0 distance calulated by
            // r.xyz = p.xyz - myPos.xyz where the right-hand side hold identical values.
            //
            constexpr size_t factor = 8; // loop unroll depth
            unroll_loop<factor>(0, particle_count, interact);

            real4 oldVel = vel[item];

            // updated position and velocity
            real4 newPos;
            xyz(newPos) = xyz(myPos) + xyz(oldVel) * deltaTime + xyz(acc) * real(0.5) * deltaTime * deltaTime;
            newPos.w() = myPos.w();

            real4 newVel;
            xyz(newVel) = xyz(oldVel) + xyz(acc) * deltaTime;
            newVel.w() = oldVel.w();

            // write to global memory
            new_pos[item] = newPos;
            new_vel[item] = newVel;
            */
            /*
            new_pos[item] = pos[item];
            new_vel[item] = vel[item];
            */
            /*
            real4 myPos = pos[item];
            real4 acc = { 0.f, 0.f, 0.f, 0.f };
            real epsSqr = std::numeric_limits<real>::epsilon();
            real deltaTime = real(0.005);

            auto xyz = [](real4& vec) { return vec.swizzle<0, 1, 2>(); };
            for (int i = 0; i < particle_count; ++i)
            {
                real4 p = pos[i];
                real4 r;
                xyz(r) = xyz(p) - xyz(myPos);
                real distSqr = r.x() * r.x() + r.y() * r.y() + r.z() * r.z();

                real invDist = 1.0f / sqrt(distSqr + epsSqr);
                real invDistCube = invDist * invDist * invDist;
                real s = p.w() * invDistCube;

                // accumulate effect of all particles
                auto eff = s * xyz(r);
                acc += real4{ eff.x(), eff.y(), eff.z(), 0.0f };
            }

            real4 oldVel = vel[item];

            // updated position and velocity
            real4 newPos;
            xyz(newPos) = xyz(myPos) + xyz(oldVel) * deltaTime + xyz(acc) * real(0.5) * deltaTime * deltaTime;
            newPos.w() = myPos.w();

            real4 newVel;
            xyz(newVel) = xyz(oldVel) + xyz(acc) * deltaTime;
            newVel.w() = oldVel.w();
            */
            new_pos[item] = real4{ 0, 0, 0, 0 };
            new_vel[item] = real4{ 0, 0, 0, 0 };
        });
    });

    CLcommandqueues().at(dev_id).enqueueReleaseGLObjects(&interop_resources, nullptr, &release);

    // Wait for all OpenCL commands to finish
    if (!cl_khr_gl_event_supported)
        cl::finish();
    else
        release.wait();
    
    // Swap front and back buffer handles
    std::swap(vaos[Front], vaos[Back]);
    std::swap(posBuffs[Front], posBuffs[Back]);
    std::swap(velBuffs[Front], velBuffs[Back]);
    
    imageDrawn = false;
}

// Override unimplemented InteropWindow function
void NBody::render()
{
    // Update matrices as needed
    if(needMatrixReset) setMatrices();

    // Clear Frame Buffer and Z-Buffer
    glFuncs->glClear(GL_COLOR_BUFFER_BIT); checkGLerror();

    // Draw
    if(!sp->bind()) qWarning("QGripper: Failed to bind shaderprogram");
    vaos[Back]->bind(); checkGLerror();

    glFuncs->glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particle_count)); checkGLerror();

    vaos[Back]->release(); checkGLerror();
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
void NBody::render(QPainter* painter)
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
void NBody::resizeGL(QResizeEvent* event_in)
{
    glFuncs->glViewport(0, 0, event_in->size().width(), event_in->size().height());
    checkGLerror();

    needMatrixReset = true; // projection matrix need to be recalculated
}

// Override InteropWindow function
bool NBody::event(QEvent *event_in)
{
    QMouseEvent* mouse_event;
    QWheelEvent* wheel_event;
    QKeyEvent* keyboard_event;

    // Process messages arriving from application
    switch (event_in->type())
    {
    case QEvent::MouseMove:
        mouse_event = static_cast<QMouseEvent*>(event_in);

        if((mouse_event->buttons() & Qt::MouseButton::RightButton) && // If RMB is pressed AND
           (mousePos != mouse_event->pos()))                          // Mouse has moved 
            mouseDrag(mouse_event);

        mousePos = mouse_event->pos();
        return true;

    case QEvent::Wheel:
        wheel_event = static_cast<QWheelEvent*>(event_in);

        mouseWheel(wheel_event);
        return true;

    case QEvent::KeyPress:
        keyboard_event = static_cast<QKeyEvent*>(event_in);

        if(keyboard_event->key() == Qt::Key::Key_Space) setAnimating(!getAnimating());
        return true;

    default:
        // In case InteropWindow does not implement handling of the even, we pass it on to the base class
        return InteropWindow::event(event_in);
    }
}

// Input handler function
void NBody::mouseDrag(QMouseEvent* event_in)
{
    phi += (event_in->x() - mousePos.x());
	theta += (event_in->y() - mousePos.y());
    
    needMatrixReset = true;
    
    if(!getAnimating()) renderNow();
}

// Input handler function
void NBody::mouseWheel(QWheelEvent* event_in)
{
    QPoint numPixels = event_in->pixelDelta();
    QPoint numDegrees = event_in->angleDelta() / 4;

    if (!numPixels.isNull())
    {
        dist += (float)sqrt(pow((double)x_abs_range,2)+pow((double)y_abs_range,2)) * 1.1f * numPixels.y() * (-0.02f);
        dist = abs(dist);

        needMatrixReset = true;
    }
    else if (!numDegrees.isNull())
    {
        QPoint numSteps = numDegrees / 15;
        dist += (float)sqrt(pow((double)x_abs_range,2)+pow((double)y_abs_range,2)) * 1.1f * numSteps.y() * (-0.02f);
        dist = abs(dist);

        needMatrixReset = true;
    }

    if(!getAnimating()) renderNow();
}

// Helper function
void NBody::setMatrices()
{
    // Set shader variables
    const float fov = 45.f;
    const float max_range = std::max({ x_abs_range,
                                       y_abs_range ,
                                       z_abs_range });

    // Set camera to view the origo from the z-axis with up along the y-axis
    // and distance so the entire sim space is visible with given field-of-view
    QVector3D vecTarget{ 0, 0, 0 };
    QVector3D vecUp{ 0, 1, 0 };
    QVector3D vecEye = vecTarget + QVector3D{ 0, 0, dist };

    QMatrix4x4 matWorld; // Identity
    matWorld.rotate(theta, { 2, 0, 0 }); // theta rotates around z-axis
    matWorld.rotate(phi,   { 0, 0, 2 }); // theta rotates around x-axis

    QMatrix4x4 matView; // Identity
    matView.lookAt(vecEye, vecTarget, vecUp);

    QMatrix4x4 matProj; // Identity
    matProj.perspective(fov,
                        static_cast<float>(this->width()) / this->height(),
                        std::numeric_limits<float>::epsilon(),
                        std::numeric_limits<float>::max());

    sp->bind();
    sp->setUniformValue("mat_MVP", matProj * matView * matWorld);
    sp->setUniformValue("mat_M", matWorld);
    sp->release();
}
