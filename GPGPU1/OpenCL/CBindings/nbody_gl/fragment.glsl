#version 330

// FS locations
#define FRAG_COLOR	0
in block
{
	vec4 Position;
	vec3 Color;
} FS_In;

layout(location = FRAG_COLOR, index = 0) out vec3 Color;

void main()
{
	Color = FS_In.Color;
}