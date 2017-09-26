#version 400

layout(location = 0) in vec4 data;

out vec3 colour;
out vec3 pos;
void main()
{
	colour = vec3(data.z, 0.0f, data.w);
	gl_Position = vec4(data.x, data.y, 0.5f, 1.0f);
	pos = vec3(data.x, data.y, 0.5f);
}
