#version 330

// VS locations
#define POSITION	0
#define COLOR		1

// FS locations
#define FRAG_COLOR	0

in block
{
	vec4 Position;
	vec4 Color;
} FS_In;

// FS index = 0 kimenete a gl_FragColor, de igazabol foloseges igy allitani
//layout(location = FRAG_COLOR, index = 0) out vec4 Color;

void main()
{
	gl_FragColor = FS_In.Color;
//	Color = FS_In.Color;
}