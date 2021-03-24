#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iostream>

struct color_u8{ unsigned char r, g, b, a; };
struct color_f{ float r, g, b, a; };

int main()
{
    static const std::string input_filename  = "input.png";
    static const std::string output_filename = "output.png";

    int w  = 0;//width
    int h  = 0;//height
    int ch = 0;//number of components

    // Load image:
    color_u8* data0 = reinterpret_cast<color_u8*>(stbi_load(input_filename.c_str(), &w, &h, &ch, 4 /* we expect 4 components */));
    if(!data0)
    {
        std::cout << "Error: could not open input file: " << input_filename << "\n";
        return -1;
    }
    else
    {
        std::cout << "Image (" << input_filename << ") opened successfully. File had Width x Height x Components = " << w << " x " << h << " x " << ch << " but the input data always has 4 components\n";
    }
    // we asked 4 components, so the pixel data pointed by 'data0' has 4 components:
    ch = 4;

    std::vector<color_f> input(w*h);
    std::vector<color_f> output(w*h);
    std::transform(data0, data0+w*h, input.begin(), [](color_u8 c){ return color_f{c.r/255.0f, c.g/255.0f, c.b/255.0f, c.a/255.0f}; } );
    
    // data has been copyed from the input buffer, we can free it now:
    stbi_image_free(data0);

    // do some trivial transformation:
    // exchange red and blue channels, increase green level by 20%, clamp at maixmum 1.0f:
    std::transform(input.cbegin(), input.cend(), output.begin(), [](color_f c){ return color_f{c.b, std::min(c.g * 1.2f, 1.0f), c.r, c.a}; } );

    // convert back to unsigned char data:
    std::vector<color_u8> tmp(w*h*4);
    std::transform(output.cbegin(), output.cend(), tmp.begin(),
                [](color_f c){ return color_u8{ (unsigned char)(c.r*255.0f),
                                                (unsigned char)(c.g*255.0f),
                                                (unsigned char)(c.b*255.0f),
                                                (unsigned char)(c.a*255.0f) }; } );

    int res = stbi_write_png(output_filename.c_str(), w, h, 4, tmp.data(), w*4);
    if(res == 0){ std::cout << "Error writing output to file: " << output_filename << "\n"; }
    else        { std::cout << "Output written to file: " << output_filename << "\n"; }
	
	return 0;
}