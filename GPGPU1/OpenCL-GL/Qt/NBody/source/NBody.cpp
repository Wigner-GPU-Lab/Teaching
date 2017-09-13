#include <QGripper.hpp>


QGripper::QGripper(QWindow* parent)
    : InteropWindow(parent)
{
    // User defined params
    params.r_max = 512;
    params.L_max = 1;

    params.dr = 0.1;
    params.dt = 0.01;
    params.t_max = 800.0;

    params.lambda = 0.00;

    params.sigma = 0.1;
    params.cutoff_r = params.r_max - 40;
    params.cutoff_w = 10;

    params.a = 5.;
    params.b = params.r_max/3;
    params.c = params.r_max/0.8;

    // Internal variable init
    t = 0.0f;
    stepNum = 0;
    graphLength = (int)ceil(params.t_max/params.dt);
    scale = 100.0 / params.a;

    sysWidth = params.r_max;
    sysHeight = gauntIndexer(params.L_max,params.L_max)+1;

    rightMouseButtonPressed = false;
	imageDrawn = false;

    myLog = new StdLogger(LogLevel::DEBUG);
	gaunt = new Gaunt<REAL>(myLog);
}


QGripper::~QGripper()
{
    if(!m_vbo) delete m_vbo;
    if(!m_ibo) delete m_ibo;
    if(!gaunt) delete gaunt;
}

// Override unimplemented InteropWindow function
void QGripper::initializeGL()
{
    qDebug("QGripper: Entering initializeGL");
    glFuncs->glViewport(0, 0, width(), height());   checkGLerror();
    glFuncs->glClearColor(0.0, 0.0, 0.0, 1.0);      checkGLerror();
    glFuncs->glEnable(GL_DEPTH_TEST);               checkGLerror();
    glFuncs->glEnable(GL_CULL_FACE);                checkGLerror();
    glFuncs->glEnable(GL_PRIMITIVE_RESTART);        checkGLerror();
    glFuncs->glPrimitiveRestartIndex(UINT_MAX);     checkGLerror();

    // Initialize simulation data
    qDebug("QGripper: Allocating host-side memory");
	mesh = std::vector<cl_float4>(sysWidth * sysHeight);
	meshGraph = std::vector<cl_float4>(graphLength);
    for(auto& vec : data) vec = std::vector<REAL>(sysWidth * sysHeight);

	qDebug("QGripper: Setting initial states");
    // GAUSS GORBET L=2 M=0 MODUSBAN
    for(unsigned int var = 0 ; var < 4 ; ++var)
    {
	    for(unsigned int y = 0 ; y < sysHeight ; ++y)
	    {
            for(unsigned int x = 0 ; x < sysWidth ; ++x)
            {
                if(var == 0)
                {
		            double gaussL = params.a*exp(-1.0*pow(x - params.b, 2) / (2*params.c));
                    double gaussR = params.a*exp(-1.0*pow(x - (params.r_max - params.b), 2) / (2*params.c));
                    cl_float4 mesh_value = {x, y, gaussR + gaussL, 1.0};
                    cl_float4 mesh_zero  = {x, y, FLT_EPSILON, 1.0};

		            if((x != 0) && (x != sysWidth-1) && (/*y == gauntIndexer(2,0)*/true))
                    {
                        data.at(var).at(y*sysWidth + x) = mesh_value.s[2];
                        mesh.at(y*sysWidth + x) = mesh_value;
                    }
		            else
                    {
                        data.at(var).at(y*sysWidth + x) = mesh_zero.s[2];
                        mesh.at(y*sysWidth + x) = mesh_zero;
                    }
                }
                else data.at(var).at(y*sysWidth + x) = REAL(0);
            }
	    }
    }

    for(unsigned int y = 0 ; y < sysHeight ; ++y)
    {
        for(unsigned int x = 0 ; x < sysWidth ; ++x)
        {
            if(x == 0) indx.push_back(UINT_MAX);
            indx.push_back(static_cast<GLuint>(y*sysWidth+x));
        }
    }

    qDebug("QGripper: Calculating Gaunt-coefficients");
    gaunt->compute(params.L_max);
    gaunt->loadBalanceMarkers();
    
    // Create shaders
    m_vs = new QOpenGLShader(QOpenGLShader::Vertex, this);
    m_fs = new QOpenGLShader(QOpenGLShader::Fragment, this);
    qDebug("QGripper: Building shaders...");
    if(!m_vs->compileSourceFile("Resources/QGripper_Shaders_vertex.glsl")) qWarning("%s",m_vs->log().data());
    if(!m_fs->compileSourceFile("Resources/QGripper_Shaders_fragment.glsl")) qWarning("%s",m_fs->log().data());
    qDebug("QGripper: Done building shaders");

    // Create and link shaderprogram
    m_sp = new QOpenGLShaderProgram(this);
    qDebug("QGripper: Linking shaders...");
    if(!m_sp->addShader(m_vs)) qWarning("QGripper: Could not add vertex shader to shader program");
    if(!m_sp->addShader(m_fs)) qWarning("QGripper: Could not add fragment shader to shader program");
    if(!m_sp->link()) qWarning("%s",m_sp->log().data());
    qDebug("QGripper: Done linking shaders");

    // Set shader variables
    dist = (float)sqrt(pow((double)sysWidth,2)+pow((double)sysHeight,2)) * 1.1f;
    phi = 3.f*(float)M_PI / 2.f;
    theta = 2.f*(float)M_PI / 4.f;

    m_vecTarget = QVector3D((float)sysWidth / 2, (float)sysHeight / 2, 0.0f);
    m_vecEye = m_vecTarget + QVector3D(dist*cos(phi)*sin(theta),dist*sin(phi)*sin(theta),dist*cos(theta));

    m_matWorld.setToIdentity();
    m_matWorld.scale(1.0, 1.0, scale);

    m_matView.setToIdentity();
    m_matView.lookAt(m_vecEye, m_vecTarget, QVector3D(0.f, 0.f, 1.f));

    m_matProj.setToIdentity();
    m_matProj.perspective( 45.0f, static_cast<float>(this->width()) / this->height(), 0.01f, 10000.0f);

    m_sp->bind();
    m_sp->setUniformValue("mat_MVP", m_matProj*m_matView*m_matWorld);
    m_sp->setUniformValue("mat_M", m_matWorld);
    m_sp->release();

    // Init device memory
    qDebug("QGripper: Initializing OpenGL buffers...");

    m_vbo = new QOpenGLBuffer(QOpenGLBuffer::VertexBuffer);
	if(!m_vbo->create()) qWarning("QGripper: Could not create VBO");
	if(!m_vbo->bind()) qWarning("QGripper: Could not bind VBO");
	m_vbo->setUsagePattern(QOpenGLBuffer::StaticDraw);
	m_vbo->allocate(mesh.size()*sizeof(cl_float4));
    m_vbo->write(0, mesh.data(), mesh.size()*sizeof(cl_float4));
	m_vbo->release();

    m_ibo = new QOpenGLBuffer(QOpenGLBuffer::IndexBuffer);
	if(!m_ibo->create()) qWarning("QGripper: Could not create IBO");
	if(!m_ibo->bind()) qWarning("QGripper: Could not bind IBO");
	m_ibo->setUsagePattern(QOpenGLBuffer::StaticDraw);
	m_ibo->allocate(indx.size()*sizeof(GLuint));
	m_ibo->write(0, indx.data(), indx.size()*sizeof(GLuint));
	m_ibo->release();
    qDebug("QGripper: Done initializing OpenGL buffers");

    // Setup VAO
    m_vao = new QOpenGLVertexArrayObject(this);
    m_vao->bind();
    m_vbo->bind();
    m_ibo->bind();
    m_sp->enableAttributeArray(0);  checkGLerror();
    m_sp->enableAttributeArray(1);  checkGLerror();
    m_sp->setAttributeArray(0, GL_FLOAT, (GLvoid *)NULL, 3, sizeof(cl_float4));                         checkGLerror();
    m_sp->setAttributeArray(1, GL_FLOAT, (GLvoid *)(NULL + 3*sizeof(cl_float)), 1, sizeof(cl_float4));  checkGLerror();
    m_vao->release();
    
    qDebug("QGripper: Leaving initializeGL");
}

// Override unimplemented InteropWindow function
void QGripper::initializeCL()
{
    qDebug("QGripper: Entering initializeCL");
    // Load, compile and initialize
    qDebug("QGripper: Loading kernel files");
    std::ifstream kernel_file_rk4("Resources/QGripper_Kernels_RK4.cl"); if(!kernel_file_rk4.is_open()) qWarning("QGripper: Cannot open QGripper_Kernels_RK4.cl");
	std::string prog_rk4( std::istreambuf_iterator<char>(kernel_file_rk4), (std::istreambuf_iterator<char>()));
    std::ifstream kernel_file_mul("Resources/QGripper_Kernels_GauntMul.cl"); if(!kernel_file_mul.is_open()) qWarning("QGripper: Cannot open QGripper_Kernels_GauntMul.cl");
	std::string prog_mul( std::istreambuf_iterator<char>(kernel_file_mul), (std::istreambuf_iterator<char>()));
    std::ifstream kernel_file("Resources/QGripper_Kernels.cl"); if(!kernel_file.is_open()) qWarning("QGripper: Cannot open QGripper_Kernels.cl");
	std::string prog( std::istreambuf_iterator<char>(kernel_file), (std::istreambuf_iterator<char>()));

    qDebug("QGripper: Querying device parameters");
    cl_uint dev_sharedMemory = CLdevices().at(0).getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&CL_err);
	checkCLerror();
    cl_uint dev_max_wgs = CLdevices().at(0).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>(&CL_err);
	checkCLerror();
    cl_uint dev_pref_real_vector_width = CLdevices().at(0).getInfo<sizeof(REAL) == 8 ? CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE : CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>(&CL_err);
	checkCLerror();
    std::string dev_vendor = CLdevices().at(0).getInfo<CL_DEVICE_VENDOR>(&CL_err);
    checkCLerror();

    // The matrix multiplication preferred vector width depends on whether the real16*sysHeight fits into shared memory or not.
    cl_uint mul_pref_real_vector_width = std::pow(2.0,std::floor(std::log(std::min(std::floor(static_cast<double>(dev_sharedMemory)/(sizeof(REAL)*sysHeight*2)),16.0))/std::log(2.0)));
    // TODO: Change log(#/log(2.0)) ==> log2(#) once VS supports it.

    // Define compile-time constants, compute domains and types.
    std::stringstream compile_options;

    // Global defines
    std::string real_string = sizeof(REAL) == 8 ? "double" : "float";
    if(MY_DEBUG) compile_options << "-D MY_DEBUG=1" << " ";
    if(dev_vendor == std::string("Advanced Micro Devices, Inc.")) compile_options << "-D AMD" << " ";
    if(dev_vendor == std::string("Intel(R) Corporation")) compile_options << "-D INTEL" << " ";
    if(dev_vendor == std::string("Nvidia Corporation")) compile_options << "-D NVIDIA" << " ";
    if(sizeof(REAL) == 8) compile_options << "-D USE_FP64" << " ";
    else compile_options << "-cl-single-precision-constant" << " ";
    compile_options << "-cl-opt-disable" << " ";
    compile_options << "-I Resources/" << " ";
    compile_options << "-D XSIZE=" << sysWidth << " ";
    compile_options << "-D YSIZE=" << sysHeight << " ";
    compile_options << "-D REAL=" << real_string << " ";
    compile_options << "-D REAL4=" << real_string << 4 << " ";

    // RK4 type defines
    compile_options << "-D REALVEC_WIDTH=" << dev_pref_real_vector_width << " ";
    if(dev_pref_real_vector_width != 1)
    {
        compile_options << "-D INTVEC=" << "int" << dev_pref_real_vector_width << " ";
        compile_options << "-D REALVEC=" << real_string << dev_pref_real_vector_width << " ";
        compile_options << "-D CONVERT_REALVEC(x)=convert_" << real_string << dev_pref_real_vector_width << "(x)" << " ";
        compile_options << "-D CONVERT_INTVEC(x)=convert_int" << dev_pref_real_vector_width << "(x)" << " ";
        if(sizeof(REAL) == 8) compile_options << "-D CONVERT_INTEGRALVEC(x)=convert_long" << dev_pref_real_vector_width << "(x)" << " ";
        else compile_options << "-D CONVERT_INTEGRALVEC(x)=convert_int" << dev_pref_real_vector_width << "(x)" << " ";
    }
    else
    {
        compile_options << "-D INTVEC=" << "int" << " ";
        compile_options << "-D REALVEC=" << real_string << " ";
        compile_options << "-D CONVERT_REALVEC(x)=convert_" << real_string << "(x)" << " ";
        compile_options << "-D CONVERT_INTVEC(x)=convert_int" << "(x)" << " ";
        if(sizeof(REAL) == 8) compile_options << "-D CONVERT_INTEGRALVEC(x)=convert_long" << "(x)" << " ";
        else compile_options << "-D CONVERT_INTEGRALVEC(x)=convert_int" << "(x)" << " ";
    }

    // Gaunt mul type defines
    compile_options << "-D MULSTRIDES=" << static_cast<int>(std::ceil(static_cast<double>(sysHeight)/dev_max_wgs)) << " ";
    compile_options << "-D MULREALVEC=" << real_string << mul_pref_real_vector_width << " ";
    compile_options << "-D MULREALVEC_WIDTH=" << mul_pref_real_vector_width << " ";

    qDebug("QGripper: Building OpenCL kernel code");
    qDebug((std::string("QGripper: Build options are:") + compile_options.str()).c_str());
    std::vector<cl::Program> programs;
    programs.push_back(cl::Program(CLcontext(), prog_rk4, CL_FALSE, &CL_err));  checkCLerror();
    programs.push_back(cl::Program(CLcontext(), prog_mul, CL_FALSE, &CL_err));  checkCLerror();
    programs.push_back(cl::Program(CLcontext(), prog    , CL_FALSE, &CL_err));  checkCLerror();

    for(auto& program : programs)
    {
        qDebug("QGripper: Building programs...");
        CL_err = program.build(CLdevices(), compile_options.str().c_str());
        if(CL_err != CL_SUCCESS)
        {
            for (auto& device : CLdevices())
            {
                cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(device, &CL_err); checkCLerror();
                if (status == CL_BUILD_ERROR)
                {
                    qWarning((std::string("QGripper: Build failed on device: ") + device.getInfo<CL_DEVICE_NAME>(&CL_err)).c_str()); checkCLerror();
                    qWarning(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device, &CL_err).c_str()); checkCLerror();
                }
            }
        }
    }
    qDebug("QGripper: Done building kernel code");

    rk4         = cl::Kernel(programs.at(0), "kgRK4",         &CL_err); checkCLerror();
    gaunt_mul   = cl::Kernel(programs.at(1), "kgGauntMul",    &CL_err); checkCLerror();
    interpolate = cl::Kernel(programs.at(2), "kgInterpolate", &CL_err); checkCLerror();
    upd_vtx     = cl::Kernel(programs.at(2), "kgUpdVtx",      &CL_err); checkCLerror();

    // Set workgroup dimensions and sizes
    qDebug("QGripper: Querying kernel preferences");
    cl_uint rk4_pref_wgs = rk4.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLdevices().at(0), &CL_err);
	checkCLerror();
    cl_uint mul_pref_wgs = gaunt_mul.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLdevices().at(0), &CL_err);
	checkCLerror();
    cl_uint vtx_pref_wgs = upd_vtx.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(CLdevices().at(0), &CL_err);
	checkCLerror();

    // Set RK4 work sizes
    cl_uint rk4_max_wgs_width;
    cl_uint rk4_max_wgs_height;
    cl_uint rk4_shared_memory;
    /*
    if(sysHeight % dev_pref_real_vector_width == 0)
    {
        rk4_max_wgs_width = greatestCommonFactor(rk4_pref_wgs, sysWidth);
        rk4_max_wgs_height = rk4_pref_wgs / rk4_max_wgs_width;
        rk4_shared_memory = (rk4_max_wgs_width+6)*rk4_max_wgs_height*dev_pref_real_vector_width*sizeof(REAL)*3;
    }
    else rk4_max_wgs_height = 0;

    while(rk4_shared_memory > dev_sharedMemory)
    {
        rk4_max_wgs_width /= 2;
        rk4_shared_memory = (rk4_max_wgs_width+6)*rk4_max_wgs_height*dev_pref_real_vector_width*sizeof(REAL)*3;
    }
    */
    rk4_shared_memory = 0;

    // Set Gaunt mul work sizes
    cl_uint mul_max_wgs_width = std::min(std::pow(2.0, std::floor(std::log(dev_sharedMemory/(mul_pref_real_vector_width*sysHeight*sizeof(REAL)*2))/std::log(2.0))), static_cast<double>(sysWidth)/mul_pref_real_vector_width);
    // TODO: Change log(#/log(2.0)) ==> log2(#) once VS supports it.
    cl_uint mul_max_wgs_height = std::min(dev_max_wgs, sysHeight);

    cl_uint mul_shared_memory = mul_max_wgs_width*mul_pref_real_vector_width*sysHeight*sizeof(REAL)*2;

    // Set upd_vtx work sizes
    cl_uint max_vtx_wgs_width = vtx_pref_wgs > sysWidth ? sysWidth : nearestInferiorPowerOf2(vtx_pref_wgs);
    /*
    rk4_global = cl::NDRange(sysWidth, sysHeight/dev_pref_real_vector_width);
    rk4_local = cl::NDRange(rk4_max_wgs_width, rk4_max_wgs_height);

    int_global = cl::NDRange(1, sysHeight/dev_pref_real_vector_width);
    int_local = cl::NullRange;
    */
    rk4_global = cl::NDRange(sysWidth, sysHeight);
    rk4_local = cl::NullRange;

    int_global = cl::NDRange(1, sysHeight);
    int_local = cl::NullRange;

    mul_global = cl::NDRange(sysWidth/mul_pref_real_vector_width, mul_max_wgs_height);
    mul_local = cl::NDRange(mul_max_wgs_width, mul_max_wgs_height);

    upd_vtx_global = cl::NDRange(sysWidth, sysHeight);
    upd_vtx_local = cl::NullRange;

	std::stringstream wgs_report;
    // Compute domains
    wgs_report << "\t" << "Compute domain:" << std::endl;
    wgs_report << "\t\t" << "Radial extent: " << sysWidth << std::endl;
    wgs_report << "\t\t" << "Multipole extent: " << sysHeight << std::endl;
    // Device capabilities
    wgs_report << "\t" << "Device capabilities:" << std::endl;
    wgs_report << "\t\t" << "Local memory available: " << dev_sharedMemory << " Bytes" << std::endl;
    wgs_report << "\t\t" << "Maximum work-group size: " << dev_max_wgs << std::endl;
    // RK4 report
    wgs_report << "\tRK4:" << std::endl;
    wgs_report << "\t\tVector width: " << dev_pref_real_vector_width << std::endl;
    wgs_report << "\t\tGlobal work dimensions: {" << rk4_global[0] << "," << rk4_global[1] << "}" << std::endl;
    wgs_report << "\t\tPreferred work-group dimensions: {" << rk4_local[0] << "," << rk4_local[1] << "}" << std::endl;
    wgs_report << "\t\tPreferred work-group size: " << rk4_local[0] * rk4_local[1] << std::endl;
    wgs_report << "\t\tLocal memory used: " << rk4_shared_memory << " Bytes" << std::endl;
    // Gaunt report
    wgs_report << "\tGaunt multiplication:" << std::endl;
    wgs_report << "\t\tVector width: " << mul_pref_real_vector_width << std::endl;
    wgs_report << "\t\tGlobal work dimensions: {" << mul_global[0] << "," << mul_global[1] << "}" << std::endl;
    wgs_report << "\t\tPreferred work-group dimensions: {" << mul_local[0] << "," << mul_local[1] << "}" << std::endl;
    wgs_report << "\t\tPreferred work-group size: " << mul_local[0] * mul_local[1] << std::endl;
    wgs_report << "\t\tLocal memory used: " << mul_shared_memory << " Bytes" << std::endl;
    // Interpolation report
    wgs_report << "\tInterpolate:" << std::endl;
    wgs_report << "\t\tVector width: " << dev_pref_real_vector_width << std::endl;
    wgs_report << "\t\tGlobal work dimensions: {" << int_global[0] << "," << int_global[1] << "}" << std::endl;
    wgs_report << "\t\tPreferred work-group dimensions: {" << int_local[0] << "," << int_local[1] << "}" << std::endl;
    wgs_report << "\t\tPreferred work-group size: " << int_local[0] * int_local[1] << std::endl;
    wgs_report << "\t\tLocal memory used: " << 0 << " Bytes" << std::endl;
    // Update vertex report
    wgs_report << "\tUpdate vertex:" << std::endl;
    wgs_report << "\t\tVector width: " << 1 << std::endl;
    wgs_report << "\t\tGlobal work dimensions: {" << upd_vtx_global[0] << "," << upd_vtx_global[1] << "}" << std::endl;
    wgs_report << "\t\tPreferred work-group dimensions: {" << upd_vtx_local[0] << "," << upd_vtx_local[1] << "}" << std::endl;
    wgs_report << "\t\tPreferred work-group size: " << upd_vtx_local[0] * upd_vtx_local[1] << std::endl;
    wgs_report << "\t\tLocal memory used: " << 0 << " Bytes" << std::endl;

	qDebug() << wgs_report.str().c_str();

    // Create Buffers
    qDebug("QGripper: Creating OpenCL buffer objects");
    size_t bufferSize = data[0].size()*sizeof(REAL);
    for(unsigned int var = 0 ; var < dataBuffs.size() ; ++var)
    {
        dataBuffs.at(var) = cl::Buffer(CLcontext(), CL_MEM_READ_WRITE, bufferSize, NULL, &CL_err);
        checkCLerror();
        for(unsigned int aux = 0 ; aux < auxBuffs.at(var).size() ; ++aux)
        {
	        auxBuffs.at(var).at(aux) = cl::Buffer(CLcontext(), CL_MEM_READ_WRITE, bufferSize, NULL, &CL_err);
	        checkCLerror();
        }
    }
	checkCLerror();
    gauntBuff = cl::Buffer(CLcontext(), CL_MEM_READ_ONLY, gaunt->getGauntCoefficients().size() * sizeof(REAL), NULL, &CL_err);
    checkCLerror();
    gauntIndexBuff = cl::Buffer(CLcontext(), CL_MEM_READ_ONLY, gaunt->getGauntIndices().size() * sizeof(cl_ushort2), NULL, &CL_err);
    checkCLerror();
	gauntMarkBuff = cl::Buffer(CLcontext(), CL_MEM_READ_ONLY, gaunt->getGauntBalancedMarkers().size() * sizeof(cl_uint3), NULL, &CL_err);
    checkCLerror();
    gauntMulResultBuff = cl::Buffer(CLcontext(), CL_MEM_READ_WRITE, bufferSize, NULL, &CL_err);
    checkCLerror();
    vertexBuffs.push_back(cl::BufferGL(CLcontext(), CL_MEM_WRITE_ONLY, m_vbo->bufferId(), &CL_err));
	checkCLerror();

    // Setting kernel arguments
    qDebug("QGripper: Setting kernel arguments");
    // rk4_step must be set inside loop
    /*
    CL_err = rk4.setArg(1, cl::Local(rk4_shared_memory/3));     checkCLerror();
    CL_err = rk4.setArg(2, cl::Local(rk4_shared_memory/3));     checkCLerror();
    CL_err = rk4.setArg(3, cl::Local(rk4_shared_memory/3));     checkCLerror();
    CL_err = rk4.setArg(4, dataBuffs.at(0));                    checkCLerror();
    CL_err = rk4.setArg(5, dataBuffs.at(1));                    checkCLerror();
    CL_err = rk4.setArg(6, dataBuffs.at(2));                    checkCLerror();
    CL_err = rk4.setArg(7, auxBuffs.at(0).at(0));               checkCLerror();
    CL_err = rk4.setArg(8, auxBuffs.at(0).at(1));               checkCLerror();
    CL_err = rk4.setArg(9, auxBuffs.at(0).at(2));               checkCLerror();
    CL_err = rk4.setArg(10, gauntMulResultBuff);                checkCLerror();
    */
    CL_err = rk4.setArg(1, dataBuffs.at(0));                    checkCLerror();
    CL_err = rk4.setArg(2, dataBuffs.at(1));                    checkCLerror();
    CL_err = rk4.setArg(3, dataBuffs.at(2));                    checkCLerror();
    CL_err = rk4.setArg(4, auxBuffs.at(0).at(0));               checkCLerror();
    CL_err = rk4.setArg(5, auxBuffs.at(0).at(1));               checkCLerror();
    CL_err = rk4.setArg(6, auxBuffs.at(0).at(2));               checkCLerror();
    CL_err = rk4.setArg(7, gauntMulResultBuff);                 checkCLerror();
    CL_err = rk4.setArg(8, auxBuffs.at(1).at(0));               checkCLerror();
    CL_err = rk4.setArg(9, auxBuffs.at(1).at(1));               checkCLerror();
    CL_err = rk4.setArg(10, auxBuffs.at(1).at(2));               checkCLerror();
    CL_err = rk4.setArg(11, auxBuffs.at(2).at(0));               checkCLerror();
    CL_err = rk4.setArg(12, auxBuffs.at(2).at(1));               checkCLerror();
    CL_err = rk4.setArg(13, auxBuffs.at(2).at(2));               checkCLerror();
    // source and target must be set inside loop

    // source must be set inside the loop
    CL_err = gaunt_mul.setArg(1, gauntMulResultBuff);           checkCLerror();
    CL_err = gaunt_mul.setArg(2, gauntBuff);                    checkCLerror();
	CL_err = gaunt_mul.setArg(3, gauntIndexBuff);               checkCLerror();
	CL_err = gaunt_mul.setArg(4, gauntMarkBuff);                checkCLerror();
    CL_err = gaunt_mul.setArg(5, cl::Local(mul_shared_memory/2));  checkCLerror();
    CL_err = gaunt_mul.setArg(6, cl::Local(mul_shared_memory/2));  checkCLerror();
    CL_err = gaunt_mul.setArg(7, 2);                            checkCLerror();

	CL_err = upd_vtx.setArg(0, graphLength);		            checkCLerror();
	CL_err = upd_vtx.setArg(1, dataBuffs.at(0));                checkCLerror();
	CL_err = upd_vtx.setArg(2, vertexBuffs.at(0));              checkCLerror();

    CL_err = interpolate.setArg(0, dataBuffs.at(0));            checkCLerror();
    CL_err = interpolate.setArg(1, dataBuffs.at(1));            checkCLerror();
    CL_err = interpolate.setArg(2, dataBuffs.at(2));            checkCLerror();

	qDebug("QGripper: Dispatching inital data to device");
    for(unsigned int var = 0 ; var < dataBuffs.size() ; var++)
    {
        CL_err = CLcommandqueues().at(0).enqueueWriteBuffer(dataBuffs.at(var), CL_TRUE, 0, bufferSize, data.at(var).data(), NULL, NULL);
	    checkCLerror();
    }
    CL_err = CLcommandqueues().at(0).enqueueWriteBuffer(gauntBuff, CL_TRUE, 0, gaunt->getGauntCoefficients().size() * sizeof(REAL), gaunt->getGauntCoefficients().data(), NULL, NULL);
	checkCLerror();
    CL_err = CLcommandqueues().at(0).enqueueWriteBuffer(gauntIndexBuff, CL_TRUE, 0, gaunt->getGauntIndices().size() * sizeof(cl_ushort2), gaunt->getGauntIndices().data(), NULL, NULL);
	checkCLerror();
    CL_err = CLcommandqueues().at(0).enqueueWriteBuffer(gauntMarkBuff, CL_TRUE, 0, gaunt->getGauntBalancedMarkers().size() * sizeof(cl_uint3), gaunt->getGauntBalancedMarkers().data(), NULL, NULL);
	checkCLerror();

    qDebug("QGripper: Leaving initializeCL");
}

// Override unimplemented InteropWindow function
void QGripper::updateScene()
{
    std::vector<cl::Event> mul_events(4);
    std::vector<cl::Event> rk4_events(4);
    cl::Event int_event;
    cl::Event vtx_event;

    for(cl_int i = 0 ; i < 4 ; ++i)
	{
        setDynamicKernelParams(i);

        //CL_err = CLcommandqueue().enqueueNDRangeKernel(gaunt_mul, cl::NullRange, mul_global, mul_local, nullptr, &mul_events.at(i));
        checkCLerror();

        CL_err = CLcommandqueues().at(0).finish(); checkCLerror();

        CL_err = CLcommandqueues().at(0).enqueueNDRangeKernel(rk4, cl::NullRange, rk4_global, rk4_local, NULL, NULL/*&rk4_events.at(i)*/);
	    checkCLerror();

        CL_err = CLcommandqueues().at(0).finish(); checkCLerror();

        //CL_err = CLcommandqueue().enqueueNDRangeKernel(interpolate, cl::NDRange(1,0), int_global, int_local, NULL, NULL/*&int_event*/);
        checkCLerror();

        CL_err = CLcommandqueues().at(0).finish(); checkCLerror();
    }
    
    if(imageDrawn)
	{
        CL_err = CLcommandqueues().at(0).enqueueAcquireGLObjects(&vertexBuffs, NULL, NULL);
		checkCLerror();
        CL_err = CLcommandqueues().at(0).enqueueNDRangeKernel(upd_vtx, cl::NullRange, upd_vtx_global, upd_vtx_local, NULL, NULL/*&vtx_event*/);
		checkCLerror();
        CL_err = CLcommandqueues().at(0).enqueueReleaseGLObjects(&vertexBuffs, NULL, NULL);
		checkCLerror();

		imageDrawn = false;
	}
    
    // Wait for all OpenCL commands to finish
    CL_err = CLcommandqueues().at(0).finish();
    checkCLerror();
    t += params.dt; stepNum++;
}

// Override unimplemented InteropWindow function
void QGripper::render()
{
    // Update matrices as needed
    if(needMatrixReset) setMatrices();

    // Clear Frame Buffer and Z-Buffer
    glFuncs->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); checkGLerror();
    glFuncs->glEnable(GL_PRIMITIVE_RESTART); checkGLerror();

    // Draw
    if(!m_sp->bind()) qWarning("QGripper: Failed to bind shaderprogram");
    m_vao->bind(); checkGLerror();

    glFuncs->glDrawArrays(GL_POINTS, 0, sysWidth * sysHeight);
    //glFuncs->glDrawElements(GL_LINE_STRIP, indx.size(), GL_UNSIGNED_INT, (GLvoid*)NULL); checkGLerror();

    m_vao->release(); checkGLerror();
    m_sp->release(); checkGLerror();

    // Wait for all drawing commands to finish
    glFuncs->glFinish();
    checkGLerror();

    imageDrawn = true;
}

// Override unimplemented InteropWindow function
void QGripper::render(QPainter* painter)
{
    QString text("QGripper: ");
    text.append("IPS = ");
    text.append(QString::number(getActIPS()));
    text.append(" | FPS = ");
    text.append(QString::number(getActFPS()));
    /*  
    painter->setBackgroundMode(Qt::TransparentMode);
    painter->setPen(Qt::white);
    painter->setFont(QFont("Arial", 30));
    painter->drawText(QRect(0, 0, 280, 50), Qt::AlignLeft, text);
    */
    this->setTitle(text);

    Q_UNUSED(painter);
}

// Override InteropWindow function
void QGripper::resizeGL(QResizeEvent* event_in)
{
    glFuncs->glViewport(0, 0, event_in->size().width(), event_in->size().height());
    checkGLerror();

    m_matProj.setToIdentity();
    m_matProj.perspective( 45.0f, static_cast<float>(event_in->size().width()) / event_in->size().height(), 0.01f, 10000.0f);

    needMatrixReset = true;
}

// Override InteropWindow function
bool QGripper::event(QEvent *event_in)
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
void QGripper::mouseDrag(QMouseEvent* event_in)
{
    phi += (event_in->x() - mousePos.x()) * 0.005f;
	theta += (event_in->y() - mousePos.y()) * -0.005f;
    
    needMatrixReset = true;
    
    if(!getAnimating()) renderNow();
}

// Input handler function
void QGripper::mouseWheel(QWheelEvent* event_in)
{
    QPoint numPixels = event_in->pixelDelta();
    QPoint numDegrees = event_in->angleDelta() / 8;

    if (!numPixels.isNull())
    {
        dist += (float)sqrt(pow((double)sysWidth,2)+pow((double)sysHeight,2)) * 1.1f * numPixels.y() * (-0.02f);
        dist = abs(dist);

        needMatrixReset = true;
    }
    else if (!numDegrees.isNull())
    {
        QPoint numSteps = numDegrees / 15;
        dist += (float)sqrt(pow((double)sysWidth,2)+pow((double)sysHeight,2)) * 1.1f * numSteps.y() * (-0.02f);
        dist = abs(dist);

        needMatrixReset = true;
    }

    if(!getAnimating()) renderNow();
}

// Helper function
void QGripper::setMatrices()
{
    m_vecEye = m_vecTarget + QVector3D(dist*cos(phi)*sin(theta),dist*sin(phi)*sin(theta),dist*cos(theta));
	m_matView.setToIdentity();
    m_matView.lookAt(m_vecEye, m_vecTarget, QVector3D(0.f, 0.f, 1.f));
    m_sp->bind();
    m_sp->setUniformValue("mat_MVP", m_matProj*m_matView*m_matWorld);
    m_sp->release();
}

// Nomen est omen
void QGripper::setDynamicKernelParams(int rk4step)
{
    switch(rk4step)
    {
    case 0:

        CL_err = gaunt_mul.setArg(0, dataBuffs.at(1)); checkCLerror();
        CL_err = rk4.setArg(0, rk4step); checkCLerror();
        /*CL_err = rk4.setArg(11, dataBuffs.at(0));      checkCLerror();
        CL_err = rk4.setArg(12, dataBuffs.at(1));      checkCLerror();
        CL_err = rk4.setArg(13, dataBuffs.at(2));      checkCLerror();
        CL_err = rk4.setArg(14, auxBuffs.at(1).at(0)); checkCLerror();
        CL_err = rk4.setArg(15, auxBuffs.at(1).at(1)); checkCLerror();
        CL_err = rk4.setArg(16, auxBuffs.at(1).at(2)); checkCLerror();
        CL_err = interpolate.setArg(0, auxBuffs.at(1).at(0)); checkCLerror();
        CL_err = interpolate.setArg(1, auxBuffs.at(1).at(1)); checkCLerror();
        CL_err = interpolate.setArg(2, auxBuffs.at(1).at(2)); checkCLerror();*/

        break;
    case 1:

        CL_err = gaunt_mul.setArg(0, auxBuffs.at(1).at(1)); checkCLerror();
        /*CL_err = rk4.setArg(0, rk4step); checkCLerror();
        CL_err = rk4.setArg(14, auxBuffs.at(1).at(0)); checkCLerror();
        CL_err = rk4.setArg(15, auxBuffs.at(1).at(1)); checkCLerror();
        CL_err = rk4.setArg(16, auxBuffs.at(1).at(2)); checkCLerror();
        CL_err = rk4.setArg(14, auxBuffs.at(2).at(0)); checkCLerror();
        CL_err = rk4.setArg(15, auxBuffs.at(2).at(1)); checkCLerror();
        CL_err = rk4.setArg(16, auxBuffs.at(2).at(2)); checkCLerror();
        CL_err = interpolate.setArg(0, auxBuffs.at(2).at(0)); checkCLerror();
        CL_err = interpolate.setArg(1, auxBuffs.at(2).at(1)); checkCLerror();
        CL_err = interpolate.setArg(2, auxBuffs.at(2).at(2)); checkCLerror();*/

        break;
    case 2:

        CL_err = gaunt_mul.setArg(0, auxBuffs.at(2).at(1)); checkCLerror();
        /*CL_err = rk4.setArg(0, rk4step); checkCLerror();
        CL_err = rk4.setArg(14, auxBuffs.at(2).at(0)); checkCLerror();
        CL_err = rk4.setArg(15, auxBuffs.at(2).at(1)); checkCLerror();
        CL_err = rk4.setArg(16, auxBuffs.at(2).at(2)); checkCLerror();
        CL_err = rk4.setArg(14, auxBuffs.at(1).at(0)); checkCLerror();
        CL_err = rk4.setArg(15, auxBuffs.at(1).at(1)); checkCLerror();
        CL_err = rk4.setArg(16, auxBuffs.at(1).at(2)); checkCLerror();
        CL_err = interpolate.setArg(0, auxBuffs.at(1).at(0)); checkCLerror();
        CL_err = interpolate.setArg(1, auxBuffs.at(1).at(1)); checkCLerror();
        CL_err = interpolate.setArg(2, auxBuffs.at(1).at(2)); checkCLerror();*/

        break;
    case 3:

        CL_err = gaunt_mul.setArg(0, auxBuffs.at(1).at(1)); checkCLerror();
        CL_err = rk4.setArg(0, rk4step); checkCLerror();
        /*CL_err = rk4.setArg(11, auxBuffs.at(1).at(0)); checkCLerror();
        CL_err = rk4.setArg(12, auxBuffs.at(1).at(1)); checkCLerror();
        CL_err = rk4.setArg(13, auxBuffs.at(1).at(2)); checkCLerror();
        CL_err = rk4.setArg(14, dataBuffs.at(0));      checkCLerror();
        CL_err = rk4.setArg(15, dataBuffs.at(1));      checkCLerror();
        CL_err = rk4.setArg(16, dataBuffs.at(2));      checkCLerror();
        CL_err = interpolate.setArg(0, dataBuffs.at(0)); checkCLerror();
        CL_err = interpolate.setArg(1, dataBuffs.at(1)); checkCLerror();
        CL_err = interpolate.setArg(2, dataBuffs.at(2)); checkCLerror();*/

        break;
    default:
        qDebug("setDynamicKernelParams(int) called with invalid rk4 step param");
    }
}

// Input handler function
int QGripper::gauntIndexer(int l, int m)
{
	// Comes from the simplified form of the sum of arithmetic progression
	return l*(l+1)+m;
}

// Input handler function
cl_int2 QGripper::reverseGauntIndexer(int i)
{
	// Speed is not critical here, but should look for a faster reverse indexer
	// since this will be evaluated inside the kernels.
	cl_int2 result;
	result.s[0] = round((-1.+sqrt(1.+4*i))/2.);
	result.s[1] = i-result.s[0]*(result.s[0]+1);
	return result;
}