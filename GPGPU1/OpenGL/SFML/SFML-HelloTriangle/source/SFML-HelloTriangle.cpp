#include <SFML-HelloTriangle.hpp>


////////////////////////////////////////////////////
/////////////////////// MAIN ///////////////////////
////////////////////////////////////////////////////


int main()
{
    // Fo ablak letrehozasa (ContextSettings-ben lehet OpenGL verziot (deafult a legnagyobb), anti-aliasingot es stencil buffer melyseget is allitani)
    sf::RenderWindow window(sf::VideoMode(1024, 768), "SFML / OpenGL test", sf::Style::Default, sf::ContextSettings(32));

	// OpenGL kontext verzio ellenorzese
	if( sf::Uint32(window.getSettings().majorVersion * 10 + window.getSettings().majorVersion) < 33 )
	{
		std::cerr << "Highest OpenGL version is " << window.getSettings().majorVersion << "." << window.getSettings().minorVersion << " Exiting..." << std::endl;
		std::exit(EXIT_FAILURE);
	}
	
	// GLEW inicializalas
	if(glewInit() != GLEW_OK) std::exit(EXIT_FAILURE);
	
	// Arnyalok betoltese fajlbol, forditasa es linkelese
	loadShaders();

	// Program hasznalata
	glUseProgram(glProgram); checkError("glUseProgram");

	// geometria kialakitasa
	Vertex geom[] =
	{
		//			 p.x	p.y	  p.z			   c.R	 c.G   c.B
		{glm::vec3( cos(0.f   * PI/180), sin(0.f   * PI/180), 0.0f), glm::vec3(1.0f, 0.0f, 0.0f)},
		{glm::vec3( cos(120.f * PI/180), sin(120.f * PI/180), 0.0f), glm::vec3(0.0f, 1.0f, 0.0f)},
		{glm::vec3( cos(240.f * PI/180), sin(240.f * PI/180), 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)},
	};
	
	// Generaljunk egy buffer objektumot
	glGenBuffers(1, &m_vbo); checkError("glGenBuffers(m_vbo)");

	// Buffer objektum hasznalatba vetele
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo); checkError("glBindBuffer(m_vbo)");

	// Memoria lefoglalasa, de meg ne masoljunk bele.
    glBufferData(GL_ARRAY_BUFFER, sizeof(geom), NULL, GL_STATIC_DRAW); checkError("glBufferData(m_vbo)");

	// Toltsuk fel a buffer (jelen esetben egeszet) a geometriaval
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(geom), geom); checkError("glBufferSubData(m_vbo)");

	// Buffer feltoltese utan hasznalat vege
	glBindBuffer(GL_ARRAY_BUFFER, 0); checkError("glBindBuffer(0)");

	// Osszefogo VAO letrehozasa es hasznalatba vetele
	glGenVertexArrays(1, &m_vao); checkError("glGenVertexArrays(m_vao)");
    glBindVertexArray(m_vao); checkError("glBindVertexArrays(m_vao)");

		// Belso buffer aktivalasa
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo); checkError("glBindBuffer(m_vbo)");
		glVertexAttribPointer(	0,						// A modositando tulajdonsaghoz tartozo index
								3,						// Hany darab adatot olvassunk
								GL_FLOAT,				// Adat amit olvasunk
								GL_FALSE,				// Arnyalo olvasaskor automatikus normalizalas
								sizeof(Vertex),			// Mekkora kozokkel szerepelnek az adatok
								(GLvoid *)0);			// Elso elem pointere (nem gazda oldali pointer!)
		checkError("glVertexAttribPointer(position)");
		glVertexAttribPointer(	1,						// A modositando tulajdonsaghoz tartozo index
								3,						// Hany darab adatot olvassunk
								GL_FLOAT,				// Adat amit olvasunk
								GL_FALSE,				// Arnyalo olvasaskor automatikus normalizalas
								sizeof(Vertex),			// Mekkora kozokkel szerepelnek az adatok
								(GLvoid *)(0 + sizeof(glm::vec3)));	// Elso elem pointere (nem gazda oldali pointer!)
		checkError("glVertexAttribPointer(color)");
		// Vertex Buffer Object hasznalat vege
		//glBindBuffer(GL_ARRAY_BUFFER, 0); checkError("glBindBuffer(0)");

		// Vertex attributum indexek aktivalasa
		glEnableVertexAttribArray(0); checkError("glEnableVertexAttribArray(0)");
		glEnableVertexAttribArray(1); checkError("glEnableVertexAttribArray(3)");

	// Osszefoglalo VAO bealliatasanak vege
	glBindVertexArray(0); checkError("glBindVertexArray(0)");

	m_vecEye = glm::vec3(0.f, 0.f, 3.f);

	// Matrixok beallitasa
	m_matWorld = glm::mat4(1.0f);										// Modell es vilag koordinatak megegyeznek

	m_matView = glm::lookAt(m_vecEye,
							glm::vec3(0.f, 0.f, 0.f),
							glm::vec3(0.f, 1.f, 0.f));

    m_matProj = glm::perspective(45.0f,                                            // 90 fokos nyilasszog
                                 ((float)window.getSize().x) / window.getSize().y, // ablakmereteknek megfelelo nezeti arany
                                 0.01f,                                            // Kozeli vagosik
                                 100.0f);                                          // Tavoli vagosik

	// Arnyalo uniformis valtozoinak gazda oldali leiroinak letrehozasa
	worldMatrixLocation = glGetUniformLocation(glProgram, "matWorld"); checkError("glGetUniformLocation(matWorld)");
	viewMatrixLocation = glGetUniformLocation(glProgram, "matView"); checkError("glGetUniformLocation(matView)");
	projectionMatrixLocation = glGetUniformLocation(glProgram, "matProj"); checkError("glGetUniformLocation(matProj)");

	// Matrixok beallitasa az arnyalokban
	glUniformMatrix4fv(worldMatrixLocation, 1, GL_FALSE, &m_matWorld[0][0]); checkError("glUniformMatrix4fv(worldMatrix)");
	glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &m_matView[0][0]); checkError("glUniformMatrix4fv(viewMatrix)");
	glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &m_matProj[0][0]); checkError("glUniformMatrix4fv(projectionMatrix)");

	// Ora peldanyositas
	sf::Clock clock;

	// FPS szamlalohoz valtozok
    sf::String string("0");
    sf::Font font;
    sf::Text FPSdrawable(string, font);         // Renderelheto elem letrehozasa
	unsigned int FPS = 0, refFrameCount = 0;

    // Program ciklus
    while (window.isOpen())
    {
        // Esemenyek kezelese
        sf::Event event;
		bool quit = false;
        while (window.pollEvent(event))
        {
            // Ablak bezarasa
            if (event.type == sf::Event::Closed)
				quit = true;

            // Escape gomb lenyomasa
            if ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::Escape))
				quit = true;

			// Fel nyil gomb lenyomasa
			if ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::Up))
			{
				if(m_vecEye.z > 0.15) m_vecEye += glm::vec3(0.f, 0.f, -0.1f);
				m_matView = glm::lookAt(m_vecEye, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
				glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &m_matView[0][0]); checkError("glUniformMatrix4fv(viewMatrix)");
			}

			// Le nyil gomb lenyomasa
			if ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::Down))
			{
				m_vecEye += glm::vec3(0.f, 0.f, 0.1f);
				m_matView = glm::lookAt(m_vecEye, glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
				glUniformMatrix4fv(viewMatrixLocation, 1, GL_FALSE, &m_matView[0][0]); checkError("glUniformMatrix4fv(viewMatrix)");
			}

            // Nezeti matrix beallitasa ha az ablak atmeretezodott
            if (event.type == sf::Event::Resized)
			{
				window.setView(sf::View(sf::FloatRect(0, 0, (float)event.size.width, (float)event.size.height)));
				glViewport(0, 0, event.size.width, event.size.height);
				m_matProj = glm::perspective( 45.0f, ((float)event.size.width)/event.size.height, 0.01f, 100.0f);
				glUniformMatrix4fv(projectionMatrixLocation, 1, GL_FALSE, &m_matProj[0][0]);
			}
        }

        // A fo ablak legyen az aktiv kontext
        window.setActive();

		// Rajzolo rutin

		// Kapcsoljuk be a hatrafele nezo lapok eldobasat
		//glEnable(GL_CULL_FACE);
		
		//m_matWorld = glm::rotate(0.01f, glm::vec3(0,0,1)) * glm::rotate(0.02f, glm::vec3(0,1,0)) * m_matWorld;
		glm::rotate(m_matWorld, 0.01f, glm::vec3(0, 0, 1));
		glm::rotate(m_matWorld, 0.02f, glm::vec3(0, 1, 0));
		glUniformMatrix4fv(worldMatrixLocation, 1, GL_FALSE, &m_matWorld[0][0]);

        // Torlesi szin beallitasa
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		
		// Toroljuk a framebuffert (GL_COLOR_BUFFER_BIT) es a melysegi Z-buffert (GL_DEPTH_BUFFER_BIT)
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); checkError("glClear");

		// Vertex Array hasznalatba vetel
		glBindVertexArray(m_vao); checkError("glBindVertexArray(m_vao)");

		// Tenyleges rajzolas
		glDrawArrays(	GL_TRIANGLES,	// Primitiv tipus
						0,				// A vertex bufferek hanyadik indexu elemetol kezdve rajzolunk ki
						3);				// Hany csucspontot hasznalunk a kirajzolashoz
		checkError("glDrawArrays");
		// Vertex Array hasznalat vege
		glBindVertexArray(0); checkError("glBindVertexArrays(0)");

		// Kapcsoljuk ki a hatrafele nezo lapok eldobasat, mert az sf::Drawable cuccok kulonben nem latszodnak
		glDisable(GL_CULL_FACE);
		
		// FPS szoveg frissitese
		if( clock.getElapsedTime().asSeconds() > 1.f )
		{
			sf::Time elapsedTime = clock.getElapsedTime();	// Get elapsed time since last update
			std::stringstream temp; temp << (int)(refFrameCount/elapsedTime.asSeconds());
			FPSdrawable.setString(sf::String(temp.str()));
			clock.restart();
			refFrameCount = 0;
		}
		
		// Minden befejezetlen OpenGL parancs bevarasa
		glFinish(); checkError("glFinish");
		
		// FPS szoveg kirajzolasa
        window.pushGLStates();									// Minden OpenGL allapot elmentese
        //FPSdrawable.setColor(sf::Color(255, 255, 255, 170));	// Szin beallitasa
        FPSdrawable.setPosition(10.f, 10.f);					// Pozicio beallitasa
        window.draw(FPSdrawable);								// Szoveg render
        window.popGLStates();									// OpenGL allapotok visszatoltese
		
		// Program hasznalata nem allitodik vissza
		glUseProgram(glProgram); checkError("glUseProgram");
		
        // Uzenet az ablakkezelonek, hogy frissitse az ablak tartalmat
        window.display(); refFrameCount++;

		// Ha kilepesi esemenyt kaptunk, zarjuk be az ablakot
		if(quit) window.close();
    }

	// OpenGL takaritas
	glDeleteBuffers( 1, &m_vbo );

    return EXIT_SUCCESS;
}


////////////////////////////////////////////////////////////////////
/////////////////////// Fuggveny kifejetesek ///////////////////////
////////////////////////////////////////////////////////////////////


// Hibakezelo fuggveny
inline void checkErr(int err, const char * name)
{
	if (err != EXIT_SUCCESS)
	{
		std::cerr << "ERROR: " << name << " (" << err << ")" << std::endl;
		exit(EXIT_FAILURE);
	}
}

// OpenGL hibakezelo fuggveny
inline bool checkError(const char* Title)
{
	int Error;
	if((Error = glGetError()) != GL_NO_ERROR)
	{
		std::string ErrorString;
		switch(Error)
		{
		case GL_INVALID_ENUM:
			ErrorString = "GL_INVALID_ENUM";
			break;
		case GL_INVALID_VALUE:
			ErrorString = "GL_INVALID_VALUE";
			break;
		case GL_INVALID_OPERATION:
			ErrorString = "GL_INVALID_OPERATION";
			break;
		case GL_INVALID_FRAMEBUFFER_OPERATION:
			ErrorString = "GL_INVALID_FRAMEBUFFER_OPERATION";
			break;
		case GL_OUT_OF_MEMORY:
			ErrorString = "GL_OUT_OF_MEMORY";
			break;
		default:
			ErrorString = "UNKNOWN";
			break;
		}
		fprintf(stdout, "OpenGL Error(%s): %s\n", ErrorString.c_str(), Title);
	}
	return Error == GL_NO_ERROR;
}

// Arnyalok betoltese
void loadShaders()
{
	std::basic_ifstream<GLchar> vs_file(VERTEX_SHADER_PATH);
	checkErr(vs_file.is_open() ? EXIT_SUCCESS : -1, "ifstream() cannot access file");

	std::basic_string<GLchar> vs_string( std::istreambuf_iterator<GLchar>(vs_file), (std::istreambuf_iterator<GLchar>()));
    std::vector<const GLchar*> vs_c_strings{ vs_string.c_str() };

    vertexShaderObj = glCreateShader(GL_VERTEX_SHADER);
    checkErr(!vertexShaderObj, "Failed to create vertex shader handle");

    glShaderSource(vertexShaderObj, (GLsizei)vs_c_strings.size(), vs_c_strings.data(), NULL);
    glCompileShader(vertexShaderObj);
    glGetShaderiv(vertexShaderObj, GL_COMPILE_STATUS, &GL_err);

    if (!GL_err)
    {
        GLint log_size;
        glGetShaderiv(vertexShaderObj, GL_INFO_LOG_LENGTH, &log_size);
        std::basic_string<GLchar> log(log_size, ' ');
        glGetShaderInfoLog(vertexShaderObj, log_size, NULL, &(*log.begin()));
        std::cout << "Failed to compile shader: " << std::endl << log << std::endl;
        std::exit(EXIT_FAILURE);
    }

	std::ifstream fs_file(FRAGMENT_SHADER_PATH);
	checkErr(fs_file.is_open() ? EXIT_SUCCESS : -1, "ifstream() cannot access file");

    std::basic_string<GLchar> fs_string( std::istreambuf_iterator<GLchar>(fs_file), (std::istreambuf_iterator<GLchar>()));
    const GLchar* fs_c_string = fs_string.c_str();

	fragmentShaderObj = glCreateShader(GL_FRAGMENT_SHADER);
	checkErr(!fragmentShaderObj, "Failed to create fragment shader handle");

    glShaderSource(fragmentShaderObj, 1, &fs_c_string, NULL);
	glCompileShader(fragmentShaderObj);
	glGetShaderiv(fragmentShaderObj, GL_COMPILE_STATUS, &GL_err);

    if (!GL_err)
    {
        GLint log_size;
        glGetShaderiv(fragmentShaderObj, GL_INFO_LOG_LENGTH, &log_size);
        std::basic_string<GLchar> log(log_size, ' ');
        glGetShaderInfoLog(fragmentShaderObj, log_size, NULL, &(*log.begin()));
        std::cout << "Failed to compile shader: " << std::endl << log << std::endl;
        std::exit(EXIT_FAILURE);
    }

	glProgram = glCreateProgram();

	glAttachShader(glProgram, vertexShaderObj);
	glAttachShader(glProgram, fragmentShaderObj);

	glLinkProgram(glProgram);

	// check if program linked (disabled for now)
	GL_err = 0;
	glGetProgramiv(glProgram, GL_LINK_STATUS, &GL_err);

	if(!GL_err)
	{
		char temp[256];
		glGetProgramInfoLog(glProgram, 256, 0, temp);
		std::cout << "Failed to link program: " << temp << std::endl;
		glDeleteProgram(glProgram);
		exit(EXIT_FAILURE);
	}
}