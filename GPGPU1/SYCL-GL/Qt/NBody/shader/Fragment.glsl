#version 330

// VS locations
#define POSITION    0
#define TEXCOORD    1

// FS locations
#define FRAG_COLOR  0

in block
{
    vec2 TexCoord;
} FS_In;

out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0f,FS_In.Color,1.0f,1.0f);
}
