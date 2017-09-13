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

void main()
{	
	float R = 0.01f * FS_In.Position.z;
	float G = (FS_In.Position.z < 0.f) ? FS_In.Position.z / 100.0f + 0.7f : -FS_In.Position.z / 100.f + 0.7f;
	float B = -0.01f * FS_In.Position.z;

	gl_FragColor = vec4(R,G,B,1.0f);
}

