#version 330

// VS locations
#define POSITION	0
#define COLOR		1

// FS locations
#define FRAG_COLOR	0

layout(location = POSITION) in vec3 in_Position;
layout(location = COLOR) in float in_Color;

out block
{
	vec4 Position;
	vec4 Color;
} VS_Out;

uniform mat4 mat_MVP;
uniform mat4 mat_M;

void main()
{
	gl_Position = mat_MVP * vec4(in_Position, 1.0);

	VS_Out.Position = mat_M * vec4(in_Position, 1.0);
	VS_Out.Color = vec4(in_Color, 1.0, 1.0, 1.0);
}

