////////////////////////////////////////////////////////////
////////////////////// Header fajlok ///////////////////////
////////////////////////////////////////////////////////////

// Arnyalok helyei
#include <SFML-HelloTriangle-config.hpp>

// Kisegito OpenGL header fajlok
#include <GL/glew.h>

// SFML header fajlok
#include <SFML/Graphics.hpp>
#include <SFML/OpenGL.hpp>

// GLM header fajlok
#include <glm/glm.hpp>
#include <glm/ext.hpp>
//#include <glm/gtc/matrix_projection.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtx/transform2.hpp>

// Szabvany C++ header fajlok
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#define PI 3.14159265f


/////////////////////////////////////////////////////////////
///////////////////// Globalis valtozok /////////////////////
/////////////////////////////////////////////////////////////


GLint GL_err;				// Hiba allapot
GLuint vertexShaderObj;		// Vertex arnyalo objektum
GLuint fragmentShaderObj;	// Fragment arnyalo objektum
GLuint glProgram;			// OpenGL program
GLuint m_vbo;				// Vertex Buffer Object
GLuint m_vao;				// Vertex Array Object

glm::mat4 m_matWorld;		// Vilag matrix
glm::mat4 m_matView;		// Nezeti matrix
glm::mat4 m_matProj;		// Vetitesi matrix

glm::vec3 m_vecEye;

GLint worldMatrixLocation;		// Vilag matrix helye az arnyalo kodjaban
GLint viewMatrixLocation;		// Nezeti matrix helye az arnyalo kodjaban
GLint projectionMatrixLocation;	// Vetitesi matrix helye az arnyalo kodjaban

//////////////////////////////////////////////////////////////
/////////////////////// Adatstrukturak ///////////////////////
//////////////////////////////////////////////////////////////


struct Vertex
	{
		glm::vec3 p;
		glm::vec3 c;
	};


///////////////////////////////////////////////////////////////////
/////////////////////// Fuggveny definiciok ///////////////////////
///////////////////////////////////////////////////////////////////


// Hibakezelo fuggveny
inline void checkErr(int err, const char * name);

// OpenGL hibakezelo fuggveny
inline bool checkError(const char* Title);

// Arynalok betoltese
void loadShaders();