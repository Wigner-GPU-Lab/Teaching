#version 330

// VS locations
#define POSITION	0
// FS locations
#define FRAG_COLOR	0

layout(location = POSITION) in vec4 Pin;

out block
{
	vec4 Position;
	vec3 Color;
} VS_Out;

void main()
{
	gl_Position = vec4(Pin.xy/20.0f, 0.5f, 1.0f);
	VS_Out.Position = gl_Position;
	float div = Pin.z > 0.05 ? Pin.z : 1.0f;
	VS_Out.Color = vec3(0.1f/div, 0.1f/div, 0.1f/div);
	if( Pin.w > 100.0f )
	{
		VS_Out.Color = vec3(1.0f, 0.0f, 0.0f);
	}
}