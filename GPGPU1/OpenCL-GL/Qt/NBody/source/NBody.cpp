#include <NBody.hpp>


NBody::NBody(QWindow* parent)
    : InteropWindow(parent)
	, particle_count(4096u)
	, x_abs_range(192.f)
	, y_abs_range(128.f)
	, z_abs_range(32.f)
	, mass_min(100.f)
	, mass_max(1000.f)
	, dev_id(0)
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
	glFuncs->glEnable(GL_DEPTH_TEST);               checkGLerror();
	glFuncs->glDisable(GL_CULL_FACE);               checkGLerror();

	// Initialize simulation data
	qDebug("NBody: Allocating host-side memory");
	pos_mass.reserve(particle_count);
	forces.reserve(particle_count);

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

	std::fill_n(std::back_inserter(velocity), particle_count, real4{ 0, 0, 0, 0 });

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

	qDebug("NBody: Done initializing OpenGL buffers");

	// Setup VAOs for the VBOs
	std::transform(vbos.begin(), vbos.end(), vaos.begin(), [this](std::unique_ptr<QOpenGLBuffer>& vbo)
	{
		auto vao = std::make_unique<QOpenGLVertexArrayObject>(this);

		vao->bind();
		vbo->bind();
		sp->enableAttributeArray(0);  checkGLerror();
		sp->enableAttributeArray(1);  checkGLerror();
		sp->setAttributeArray(0, GL_FLOAT, (GLvoid *)(NULL), 3, sizeof(real4));   checkGLerror();
		sp->setAttributeArray(1, GL_FLOAT, (GLvoid *)(NULL + 3 * sizeof(real)), 1, sizeof(real4));   checkGLerror();
		vao->release();

		return std::move(vao);
	});


	qDebug("NBody: Leaving initializeGL");
}

// Override unimplemented InteropWindow function
void NBody::initializeCL()
{
	qDebug("NBody: Entering initializeCL");

	// Load, compile and initialize
	qDebug("NBody: Loading kernel code");
	std::ifstream kernel_stream{ kernel_location + "/Kernels.cl" };
	if (!kernel_stream.is_open()) qWarning("NBody: Cannot open kernel/Kernels.cl");

	std::string kernel_string{ std::istreambuf_iterator<char>{kernel_stream}, std::istreambuf_iterator<char>{} };

	qDebug("NBody: Optimizing kernel");
	// Define compile-time constants, compute domains and types.
	std::stringstream compile_options;

	qDebug("NBody: Querying device capabilities");
	if (sizeof(real) == 8)
	{
		bool fp64 = CLdevices().at(dev_id).getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_fp64", 0) != std::string::npos;

		if (!fp64) qWarning("NBody: Selected device does not support double precision");

        compile_options << "-D USE_FP64 ";

		compile_options << "-D real=" << "double ";
		compile_options << "-D real4=" << "double4 ";
	}
	else
	{
		compile_options << "-cl-single-precision-constant" << " ";

		compile_options << "-D real=" << "float ";
		compile_options << "-D real4=" << "float4 ";
	}

	qDebug("NBody: Building kernel");
	qDebug((std::string("NBody: Build options are:") + compile_options.str()).c_str());
	cl::Program prog{ CLcontext(), kernel_string };

	try
	{
		prog.build({ CLdevices().at(dev_id) }, compile_options.str().c_str());
	}
	catch (cl::BuildError err)
	{
		qWarning((std::string{ "NBody: Build exception: " } + err.what()).c_str());

		for (const auto& log : err.getBuildLog())
		{
			qWarning((std::string{ "NBody: Device " } + log.first.getInfo<CL_DEVICE_NAME>() + " has build log:\n").c_str());
			qWarning(log.second.c_str());
		}
	}

	step_kernel = cl::Kernel{ prog, "nbody_sim" };

	gws = cl::NDRange{ particle_count };
	lws = cl::NullRange;

	// Create Buffers
	qDebug("NBody: Creating OpenCL buffer objects");
	for (auto& velBuff : velBuffs)
		velBuff = cl::Buffer{ CLcontext(), velocity.begin(), velocity.end(), false };

	std::transform(vbos.cbegin(), vbos.cend(), posBuffs.begin(), [this](const std::unique_ptr<QOpenGLBuffer>& vbo)
	{
		return cl::BufferGL{ CLcontext(), CL_MEM_READ_WRITE, vbo->bufferId() };
	});

	compute_queue = cl::CommandQueue{ CLcontext(), CLdevices().at(dev_id) };

	// Init bloat vars
	std::transform(posBuffs.cbegin(), posBuffs.cend(), std::back_inserter(interop_resources), [](const cl::BufferGL& buf)
	{
		return cl::Memory{ buf };
	});

	qDebug("NBody: Leaving initializeCL");
}

// Override unimplemented InteropWindow function
void NBody::updateScene()
{
	compute_queue.enqueueAcquireGLObjects(&interop_resources, nullptr, &acquire_release[0]);

	auto compute_event = kernel_functor{ step_kernel }(cl::EnqueueArgs{ compute_queue, gws },
		                                               posBuffs[Front],
		                                               posBuffs[Back],
		                                               velBuffs[Front],
		                                               velBuffs[Back],
		                                               (cl_uint)particle_count,
		                                               (real)0.005,
		                                               (real)1.0);

	compute_queue.enqueueReleaseGLObjects(&interop_resources, nullptr, &acquire_release[1]);
    
    // Wait for all OpenCL commands to finish
    compute_queue.finish();
    //t += params.dt; stepNum++;
    imageDrawn = false;

    // Swap fron and back handles
    std::swap(vaos[Front], vaos[Back]);
    std::swap(posBuffs[Front], posBuffs[Back]);
    std::swap(velBuffs[Front], velBuffs[Back]);
}

// Override unimplemented InteropWindow function
void NBody::render()
{
    // Update matrices as needed
    if(needMatrixReset) setMatrices();

    // Clear Frame Buffer and Z-Buffer
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); checkGLerror();

    // Draw
    if(!sp->bind()) qWarning("QGripper: Failed to bind shaderprogram");
    vaos[Front]->bind(); checkGLerror();

    glFuncs->glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(particle_count)); checkGLerror();

    vaos[Front]->release(); checkGLerror();
    sp->release(); checkGLerror();

    // Wait for all drawing commands to finish
    glFuncs->glFinish();
    checkGLerror();

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
     
    painter->setBackgroundMode(Qt::TransparentMode);
    painter->setPen(Qt::white);
    painter->setFont(QFont("Arial", 30));
    painter->drawText(QRect(0, 0, 280, 50), Qt::AlignLeft, text);
    
    this->setTitle(text);

    Q_UNUSED(painter);
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

        if((mouse_event->buttons() & Qt::MouseButton::RightButton) && (mousePos != mouse_event->pos())) mouseDrag(mouse_event);
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
    phi += (event_in->x() - mousePos.x()) * 0.005f;
	theta += (event_in->y() - mousePos.y()) * -0.005f;
    
    needMatrixReset = true;
    
    if(!getAnimating()) renderNow();
}

// Input handler function
void NBody::mouseWheel(QWheelEvent* event_in)
{
    QPoint numPixels = event_in->pixelDelta();
    QPoint numDegrees = event_in->angleDelta() / 8;

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
    dist = max_range / std::tan(fov); // tan(alfa) = opposite / adjacent

    // Set camera to view the origo from the z-axis with up along the y-axis
    // and distance so the entire sim space is visible with given field-of-view
    QVector3D vecTarget{ 0, 0, 0 };
    QVector3D vecUp{ 0, 1, 0 };
    QVector3D vecEye = vecTarget + QVector3D{ 0, 0, max_range + dist };

    QMatrix4x4 matWorld; // Identity
    matWorld.rotate(theta, { 0, 0, 1 }); // theta rotates around z-axis
    matWorld.rotate(phi,   { 1, 0, 0 }); // theta rotates around x-axis

    QMatrix4x4 matView; // Identity
    matView.lookAt(vecEye, vecTarget, vecUp);

    QMatrix4x4 matProj; // Identity
    matProj.perspective(fov,
                        static_cast<float>(this->width()) / this->height(),
                        dist + z_abs_range,
                        dist + 2 * z_abs_range);

    sp->bind();
    sp->setUniformValue("mat_MVP", matProj * matView * matWorld);
    sp->setUniformValue("mat_M", matWorld);
    sp->release();
}
