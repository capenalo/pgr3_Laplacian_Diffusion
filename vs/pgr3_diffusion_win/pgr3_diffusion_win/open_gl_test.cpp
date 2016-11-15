#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <iostream>
#include <thread>
#include <windows.h>
#include <math.h> 
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
//#include <SOIL.h>

GLFWwindow* window;
GLuint vao;
GLuint buffer;
GLuint vs;
GLuint fs;
GLuint sp;
GLuint uniColor;
GLuint ebo;
GLuint texture;
int fps;
double nextTime = 1;

float verticalAngle = 45;
float horizontalAngle = 45;

const char* vertexShaderData =
"#version 450 \n"
"layout(location = 0) in vec3 vp;"
"layout(location = 1) in vec3 color;"
"layout (location = 2) in vec2 texCoord;\n"
"out vec3 Color;"
"out vec2 TexCoords;\n"
"void main(){"
"Color=color;"
"TexCoords = texCoord;"
"gl_Position = vec4(vp, 1.0);"
"}";


const char* fragShaderData =
"#version 410\n"
//"uniform vec4 incolor;\n"
"in vec3 Color;"
"in vec2 TexCoords;\n"
"out vec4 outColor;\n"
"uniform sampler2D textureMap;\n"
"void main(){"
"outColor = vec4(Color, 1.0);"
//"outColor = texture(textureMap, TexCoords) * vec4(Color,1.0f);"
"}";

static GLfloat data[] = {

	0.5f, 0.5f, 0.5f,   1.0f, 0.0f, 0.0f,       0.5f, 0.5f,
	0.5f, -0.5f, 0.5f,     0.0f, 1.0f, 0.0f,    0.5f, -0.5f,
	-0.5f, -0.5f, 0.5f,    0.0f, 0.0f, 1.0f,    -0.5f, -0.5f,
	-0.5f, 0.5f, 0.5f,    0.0f, 1.0f, 1.0f,    -0.5f, 0.5f,

	0.5f, 0.5f, -0.5f,   1.0f, 1.0f, 0.0f,       0.5f, 0.5f,
	0.5f, -0.5f, -0.5f,     1.0f, 1.0f, 1.0f,    0.5f, -0.5f,
	-0.5f, -0.5f, -0.5f,    0.4f, 0.5f, 0.5f,    -0.5f, -0.5f,
	-0.5f, 0.5f, -0.5f,    0.0f, 0.5f, 0.5f,    -0.5f, 0.5f,
};


static GLuint indices[] = {
	0, 1, 2, 3, 2, 0, //front face
	0, 4, 1, 4, 5, 1, //right face
	3, 2, 7, 2, 6, 7, //left face
	4, 5, 6, 7, 6, 4, //back face
	3, 7, 0, 0, 4, 7, //top face
	2, 1, 5, 2, 6, 5  //bottom face
};



int init() {
	glfwInit();
	window = glfwCreateWindow(1000, 1000, "Window", NULL, NULL);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_SAMPLES, 4);
	glewExperimental = true;
	glfwMakeContextCurrent(window);
	glewInit();
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GLFW_TRUE);
	std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
	glEnable(GL_DEPTH_TEST);

	return 1;
}

/*void initTextures() {
	int width, height;
	unsigned char* image = SOIL_load_image("C:\\Users\\Sanjiv\\Desktop\\container.jpg", &width, &height, 0, SOIL_LOAD_RGB);
	if (0 == image) {
		std::cout << "error";
		printf("SOIL loading error: '%s'\n", SOIL_last_result());
	}
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);
	SOIL_free_image_data(image);
	glGenerateMipmap(GL_TEXTURE_2D);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);

	//glBindTexture(GL_TEXTURE_2D, 0);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);


}*/

void initShaders() {
	vs = glCreateShader(GL_VERTEX_SHADER);
	fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(vs, 1, &vertexShaderData, NULL);
	glShaderSource(fs, 1, &fragShaderData, NULL);
	glCompileShader(vs);
	glCompileShader(fs);
	sp = glCreateProgram();
	glBindFragDataLocation(sp, 0, "outColor");

	glAttachShader(sp, vs);
	glAttachShader(sp, fs);
	glLinkProgram(sp);
	glUseProgram(sp);
	GLint status;
	glGetShaderiv(vs, GL_COMPILE_STATUS, &status);
	//std::cout << (bool)status;
	char buffer[512];
	glGetShaderInfoLog(fs, 512, NULL, buffer);
	std::cout << buffer;

	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);
}

void storeData() {
	//glm::mat4 camera;
	//camera = glm::lookAt(glm::vec3(0.0f, 0.0f, -2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	for (int i = 0; i < 64; i += 8) {
		glm::mat4 trans;
		trans = glm::rotate(trans, glm::radians(horizontalAngle), glm::vec3(0.0, 1.0, 0.0));
		glm::mat4 trans2;
		trans2 = glm::rotate(trans, glm::radians(verticalAngle), glm::vec3(1.0, 0.0, 0.0));

		glm::vec4 p = glm::vec4(data[i], data[i + 1], data[i + 2], 1.0);
		p = trans2*trans*p;
		std::cout << p.x << ", " << p.y << ", " << p.z << std::endl;
		data[i] = p.x;
		data[i + 1] = p.y;
		data[i + 2] = p.z;
		//std::cout << data[i] << ", " << data[i + 1] << ", " << data[i + 2] << std::endl;
	}


	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);

	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, sizeof(GLuint) * 3, (GLvoid*)0);

	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	initShaders();



}

void draw(float color[]) {
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glUseProgram(sp);
	glBindVertexArray(vao);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

	//glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);

}

void handleKeys() {
	float x = 0;
	float y = 0;
	float z = 0;
	if (glfwGetKey(window, GLFW_KEY_RIGHT)) {
		x = -1;
	}
	if (glfwGetKey(window, GLFW_KEY_UP)) {
		y = 1;
	}
	if (glfwGetKey(window, GLFW_KEY_LEFT)) {
		x = 1;
	}
	if (glfwGetKey(window, GLFW_KEY_DOWN)) {
		y = -1;
	}
	if (glfwGetKey(window, GLFW_KEY_W)) {
		z = 1;
	}
	if (glfwGetKey(window, GLFW_KEY_S)) {
		z = -1;
	}
	for (int i = 0; i < 64; i += 8) {
		glm::mat4 trans;
		trans = glm::rotate(trans, glm::radians(x), glm::vec3(0.0, 1.0, 0.0));
		glm::mat4 trans2;
		trans2 = glm::rotate(trans, glm::radians(y), glm::vec3(1.0, 0.0, 0.0));
		glm::mat4 trans3;
		trans3 = glm::rotate(trans, glm::radians(z), glm::vec3(0.0, 0.0, 1.0));

		glm::vec4 p = glm::vec4(data[i], data[i + 1], data[i + 2], 1.0);
		p = trans3*trans2*trans*p;
		//std::cout << p.x << ", " << p.y << ", " << p.z << std::endl;
		data[i] = p.x;
		data[i + 1] = p.y;
		data[i + 2] = p.z;
	}
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
}

void startLoop() {
	glfwSetTime(0);
	float a = 0.0;
	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0) {
		a = sin(glfwGetTime() * 3) + 1;
		float color[] = { 1.0f, 1.0f, 0.0f, a };
		handleKeys();
		draw(color);
		glfwSwapBuffers(window);
		glfwPollEvents();
		fps++;
		if (glfwGetTime() > nextTime) {
			std::cout << "FPS: " << fps << std::endl;
			nextTime += 1;
			fps = 0;
		}

	}
}

int main() {
	init();
	storeData();
	//initTextures();
	startLoop();

	return 1;
}