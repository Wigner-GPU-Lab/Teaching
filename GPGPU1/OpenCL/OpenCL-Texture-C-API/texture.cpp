
#include <sstream>
#include <fstream>
#include <algorithm>
#include <memory>
#include <iostream>
#include <vector>
#include <CL\cl.h>

#define NOMINMAX
template<typename T> T min(T const& x, T const& y){ return x<y?x:y; }
template<typename T> T max(T const& x, T const& y){ return x<y?y:x; }
#include <windows.h>
#include <Windowsx.h>
#include <WinUser.h>
#include <gdiplus.h>

using namespace Gdiplus;
#pragma comment (lib,"Gdiplus.lib")

static int GDIPlus_GetEncoderClsid(const wchar_t* format, CLSID* pClsid)
{
	unsigned int num = 0;          // number of image encoders
	unsigned int size = 0;         // size of the image encoder array in bytes
	ImageCodecInfo* pImageCodecInfo = nullptr;
		
	GetImageEncodersSize(&num, &size);
	if(size == 0){ return -1; }  // Failure
		
	pImageCodecInfo = (ImageCodecInfo*)(new unsigned char[size]);
	if(!pImageCodecInfo){ return -1; } // Failure
		
	GetImageEncoders(num, size, pImageCodecInfo);
	for(int j=0; j<(int)num; ++j)
	{
		if( wcscmp(pImageCodecInfo[j].MimeType, format) == 0 )
		{
			*pClsid = pImageCodecInfo[j].Clsid;
			delete[] pImageCodecInfo;
			return j;  // Success
		}
	}
	delete[] pImageCodecInfo;
	return -1; // Failure
}

int main()
{
    GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
    GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, nullptr);
    

    Bitmap image( L"input.jpg" );
    size_t W = image.GetWidth();
    size_t H = image.GetHeight();

	cl_platform_id platform = NULL;
	auto status = clGetPlatformIDs(1, &platform, NULL);

	cl_device_id device = NULL;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

	cl_context_properties cps[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };
	auto context = clCreateContext(cps, 1, &device, 0, 0, &status);

	auto queue = clCreateCommandQueueWithProperties(context, device, nullptr, &status);

	std::ifstream file("sobel.cl");
	std::string source( std::istreambuf_iterator<char>(file), (std::istreambuf_iterator<char>()));
	size_t      sourceSize = source.size();
	const char* sourcePtr  = source.c_str();
	auto program = clCreateProgramWithSource(context, 1, &sourcePtr, &sourceSize, &status);
	
	status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
	if (status != CL_SUCCESS)
	{
		size_t len = 0;
		status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
		std::unique_ptr<char[]> log = std::make_unique<char[]>(len);
		status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, len, log.get(), nullptr);
		std::cout << log.get() << "\n";
		return -1;
	}

	auto kernel = clCreateKernel(program, "sobel", &status);
	
    std::vector<float> image_data(W*H*4);
    
    Rect rct; rct.X = 0; rct.Y = 0; rct.Width = W; rct.Height = H;
    BitmapData bmpdata;
    if( image.LockBits(&rct, ImageLockModeRead, PixelFormat32bppARGB, &bmpdata) == Status::Ok)
    {
        for(int y=0; y<H; ++y)
        {
            for(int x=0; x<W; ++x)
            {
                auto p = ((Color*)bmpdata.Scan0)[y * bmpdata.Stride / 4 + x];
                image_data[(y*W+x)*4 + 0] = (float)p.GetRed() / 255.0f;
                image_data[(y*W+x)*4 + 1] = (float)p.GetGreen() / 255.0f;
                image_data[(y*W+x)*4 + 2] = (float)p.GetBlue() / 255.0f;
                image_data[(y*W+x)*4 + 3] = (float)p.GetAlpha() / 255.0f;
            }
        }
    }
    image.UnlockBits(&bmpdata);
    

    cl_image_format format = { CL_RGBA, CL_FLOAT };
	cl_image_desc desc = {};
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width =  W; //x
	desc.image_height = H; //y
	desc.image_depth =  0;
	
	cl_mem img_src = clCreateImage(context, CL_MEM_READ_ONLY  | CL_MEM_COPY_HOST_PTR | CL_MEM_HOST_NO_ACCESS, &format, &desc, image_data.data(), &status);
    if (status != CL_SUCCESS){ printf("Image allocation failed!\n"); }
    cl_mem img_dst = clCreateImage(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,                        &format, &desc, image_data.data(), &status);
	if (status != CL_SUCCESS){ printf("Image allocation failed!\n"); }

	status = clSetKernelArg(kernel, 0, sizeof(img_src), &img_src);
	status = clSetKernelArg(kernel, 1, sizeof(img_dst), &img_dst);


    size_t kernel_dims[2] = {W, H};
	status = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, kernel_dims, nullptr, 0, nullptr, nullptr);
    status = clFinish(queue);
    size_t origin[3] = {0, 0, 0};
    size_t dims[3] = {W, H, 1};
	status = clEnqueueReadImage(queue, img_dst, false, origin, dims, 0, 0, image_data.data(), 0, nullptr, nullptr);//clEnqueueReadBuffer(queue, img_dst, false, 0, image_data.size() * sizeof(float), image_data.data(), 0, nullptr, nullptr);
	status = clFinish(queue);

    CLSID Clsid;
	GDIPlus_GetEncoderClsid(L"image/png", &Clsid);
	
    Bitmap& bmp = Bitmap(W, H, PixelFormat32bppARGB);

    BitmapData bmpdata2;
    if( bmp.LockBits(&rct, ImageLockModeWrite, PixelFormat32bppARGB, &bmpdata2) == Status::Ok )
    {
        for(int y=0; y<H; ++y)
        {
            for(int x=0; x<W; ++x)
            {
                auto r = (BYTE)(image_data[(y*W+x)*4 + 0] * 255.0);
                auto g = (BYTE)(image_data[(y*W+x)*4 + 1] * 255.0);
                auto b = (BYTE)(image_data[(y*W+x)*4 + 2] * 255.0);
                auto a = (BYTE)(image_data[(y*W+x)*4 + 3] * 255.0);

                ((Color*)bmpdata2.Scan0)[y * bmpdata2.Stride / 4 + x] = Color(a, r, g, b);
                
            }
        }
    }
    bmp.UnlockBits(&bmpdata2);
    bmp.Save( L"output.png", &Clsid );


    clReleaseMemObject(img_src);
    clReleaseMemObject(img_dst);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device);
	
	return 0;
}