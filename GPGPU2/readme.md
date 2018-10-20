# Notes on the GPGPU2 sample codes

This folder contains sample code presented and discussed during the second semester [lectures on Science Targeted Programming of Graphical Processors](http://gpu.wigner.mta.hu/hu/laboratory/teaching/science-targeted-programming-of-graphical-processors-2) (in Hungarian). The content is targeted at illustrating category theoretical, functional programming and generic programming concepts in modern C++ (14, 17) in some cases on the GPU via CUDA and SYCL with some focus on physics simulations.

Below are links and comments on the sample codes in the order they appear in the lectures.

1. Type level list manipulation in C++ [type_list.cpp](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/type_list.cpp)

2. Functor, Foldable and 'Zippable' concepts in action in case of a vector class [Vector.cpp](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/Vector.cpp)

3. Functor, Applicative and Monad concepts in C++ [functorapplicativemonad.cpp](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/functorapplicativemonad.cpp)

4. State monad example for OpenCL API calls [state_monad_demo_opencl.cpp](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/state_monad_demo_opencl.cpp)

5. Natural transformation example [NaturalTransformation.cpp](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/NaturalTransformation.cpp)

6. Tuple implementation [tuple.h](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/tuple.h)

7. Common abstraction over the Tuple and the Vector implementation [indexible.h](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/indexible.h)

7. Map, Zip and reduce on the GPU
 - map with CUDA [map.cu](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/map.cu)
 - map with SYCL [map_sycl.cpp](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/map_sycl.cpp)
 - zip with CUDA [zip.cu](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/zip.cu)
 - zip with SYCL [zip_sycl.cpp](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/zip_sycl.cpp)
 - reduce with CUDA [reduce.cu](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/reduce.cu)
 - reduce with SYCL [reduce_sycl.cpp](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/reduce_sycl.cpp)
 - reduce benchmark in CUDA [reduce_bench.cu](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/reduce_bench.cu)
 - the fusion of a map followed by a reduce in SYCL [reduce_map_sycl.cpp](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/reduce_map_sycl.cpp)
 - reduce map but with the thread index as the map input array, and its usage in numeric integration [reduce_map_idx_sycl.cpp](https://github.com/Wigner-GPU-Lab/Teaching/blob/master/GPGPU2/reduce_map_idx_sycl.cpp)
