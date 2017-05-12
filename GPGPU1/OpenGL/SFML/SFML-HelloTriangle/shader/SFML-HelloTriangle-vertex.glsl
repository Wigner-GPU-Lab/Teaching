#version 330

// VS locations
#define POSITION	0
#define COLOR		1

// FS locations
#define FRAG_COLOR	0

layout(location = POSITION) in vec3 in_Position;
layout(location = COLOR) in vec3 in_Color;

out block
{
	vec4 Position;
	vec4 Color;
} VS_Out;

uniform mat4 matWorld;
uniform mat4 matView;
uniform mat4 matProj;

void main()
{
	gl_Position = matProj * matView * matWorld * vec4(in_Position, 1.0);

	VS_Out.Position = matWorld * vec4(in_Position, 1.0);
	VS_Out.Color = vec4(in_Color, 1.0);
}
