# prAndroidYUVNative
=====================================================

The proposal of this aplication is to take raw images and transform them into other formats that are lighter or that allow us to make some useful aplications like edge detections

## >  YUV
The raw format that our mobile phone camera has is YUV. It encodes a color image taking human perception into account, allowing reduced bandwidth for chrominance components. The Y'UV model defines a color space in terms of one luma (Y') and two chrominance (UV) components. The Y'UV color model is used in the PAL and SECAM composite color video standards. The application will take raw images in YUV format and will transform them into RGB, GRAY and RGB through a convolution matrix. There will be options to do it with no optimizations, with native C and optionally with parallelism

## > YUV420 to RGB888 conversion
To make the YUV to RGB conversion we will use the Y'UV420 to RGB8888 conversion. Y'UV420p is a planar format, meaning that the Y', U, and V values are grouped together instead of interspersed. The reason for this is that by grouping the U and V values together, the image becomes much more compressible. When given an array of an image in the Y'UV420p format, all the Y' values come first, followed by all the U values, followed finally by all the V values

## > Native
To perform efficiency we will use native codification using de JNI (Java Native Interface). It allows us to write C/C++ code

## > Paralellism
Wether if we use the native code or the java version, we will implement a parallel version to optimize our application and make it run faster

## > Neon
The last modification we will make to our code is embedde assembler code with neon instructions.Citing official Android Website [The NDK supports the ARM Advanced SIMD, an optional instruction-set extension of the ARMv7 spec. NEON provides a set of scalar/vector instructions and registers (shared with the FPU) comparable to MMX/SSE/3DNow! in the x86 world. To function, it requires VFPv3-D32 (32 hardware FPU 64-bit registers, instead of the minimum of 16).](http://developer.android.com/intl/es/ndk/guides/cpu-arm-neon.html)
