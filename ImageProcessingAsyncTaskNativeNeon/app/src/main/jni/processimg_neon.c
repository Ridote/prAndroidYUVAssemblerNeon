//
// Created by Andrés Rodríguez Moreno on 28/11/15.
//

#include <jni.h>
#include <android/log.h>

// NEON Implementation

#if defined(__ARM_ARCH_7A__) && defined(__ARM_NEON__)
#include <arm_neon.h>

#define bytes_per_pixel 4 // RGB+alfa

    static int convertYUV420_NV21toRGB8888_NEON(const unsigned char * data, int * pixels, int width, int height)
    // pixels must have room for width*height ints, one per pixel
    //bool decode_yuv_neon(unsigned char* out, unsigned char const* yuv, int width, int height, unsigned char fill_alpha=0xff)
    {

        unsigned char* out = (unsigned char*) pixels;
        unsigned char const* yuv= data;
        unsigned char const fill_alpha=0xff;

    // pre-condition : width must be multiple of 8, height must be even
        if (0!=(width&7) || width<8 || 0!=(height&1) || height<2 || !out || !yuv)
            return 0;

    // Y' and UV pointers
        unsigned char const* y  = yuv;
        unsigned char const* uv = yuv + (width*height);

    // iteration counts
        int const itHeight = height>>1;
        int const itWidth = width>>3;

    // stride of each line
        int const stride = width*bytes_per_pixel;

    // total bytes of each write
        int const dst_pblock_stride = 8*bytes_per_pixel;

    // a block temporary stores consecutively 8 pixels
        uint8x8x4_t pblock; //Array of 4 vectors with 8 lines, 8 bits each
    // Last position of the array initialized with 8 lines equal to 0xFF
        pblock.val[3] = vdup_n_u8(fill_alpha); // alpha channel in the last

//R = [128 + 298(Y - 16) + 409(V - 128)] >> 8
//G = [128 + 298(Y - 16) - 100(V - 128) - 208(U - 128)] >> 8
//B = [128 + 298(Y - 16) + 516(U - 128)] >> 8

    // simd constants
        uint8x8_t const Yshift = vdup_n_u8(16); //this is the constant to substract to y
        int16x8_t const half = vdupq_n_s16(128); // this is the constant to substract to u and v
        int32x4_t const rounding = vdupq_n_s32(128); // this is the constant to round adding 128/256=0.5

    // tmp variable to load y and uv
        uint16x8_t t;
        int i,j;

        for ( j=0; j<itHeight; ++j, y+=width, out+=stride) {
            for ( i=0; i<itWidth; ++i, y+=8, uv+=8, out+=dst_pblock_stride) {
    // load u8x8 y values, substract 16 to each one and promote to u16x8:
                t = vmovl_u8(vqsub_u8(vld1_u8(y), Yshift));
    // splits the u16x8 in two u16x4 that are multiplied by 298 and promoted to u32x4
                int32x4_t const Y00 = vmulq_n_u32(vmovl_u16(vget_low_u16(t)), 298);
                int32x4_t const Y01 = vmulq_n_u32(vmovl_u16(vget_high_u16(t)), 298);
    // the same with the next row that also shares the u and v values
                t = vmovl_u8(vqsub_u8(vld1_u8(y+width), Yshift));
                int32x4_t const Y10 = vmulq_n_u32(vmovl_u16(vget_low_u16(t)), 298);
                int32x4_t const Y11 = vmulq_n_u32(vmovl_u16(vget_high_u16(t)), 298);

    // load uv pack 4 sets of uv into a uint8x8_t, layout : { u0,v0, u1,v1, u2,v2, u3,v3 }
    // u8x8 pack is promoted to u16x8 and then 128 is subtracted to each line
                t = vsubq_s16((int16x8_t)vmovl_u8(vld1_u8(uv)), half);

//Unzip operation to compute UV array
// 	    Low part of t	        High part of t
//  	u2 v2 u3 v3             u0 v0 u1 v1
// After unzip op:
// UV.val[0] : u0, u1, u2, u3
// UV.val[1] : v0, v1, v2, v3
                int16x4x2_t const UV = vuzp_s16(vget_low_s16(t), vget_high_s16(t));

    // tR : 128+409V
    // tG : 128-100U-208V
    // tB : 128+516U
    // int32x4_t  vmlal_s16(int32x4_t a, int16x4_t b, int16x4_t c);    // VMLAL.S16 q0,d0,d0

                int32x4_t const tR = vmlal_n_s16(rounding, UV.val[1], 409);
                int32x4_t const tG = vmlal_n_s16(vmlal_n_s16(rounding, UV.val[0], -100), UV.val[1], -208);
                int32x4_t const tB = vmlal_n_s16(rounding, UV.val[0], 516);
    // Dup TR to combine with two different y
                int32x4x2_t const R = vzipq_s32(tR, tR); // [tR0, tR0, tR1, tR1] [tR2, tR2, tR3, tR3]
                int32x4x2_t const G = vzipq_s32(tG, tG); // [tG0, tG0, tG1, tG1] [tG2, tG2, tG3, tG3]
                int32x4x2_t const B = vzipq_s32(tB, tB); // [tB0, tB0, tB1, tB1] [tB2, tB2, tB3, tB3]

/* The following intrinsics are:
// vaddq_s32  standard addition
// int32x4_t   vaddq_s32(int32x4_t a, int32x4_t b);     // VADD.I32 q0,q0,q0

// vqmovun_s32  Vector saturating narrow integer signed->unsigned
//uint16x4_t vqmovun_s32(int32x4_t a);                     // VQMOVUN.S32 d0,q0

// These intrinsics join two 64 bit vectors into a single 128 bit vector
//uint16x8_t  vcombine_u16(uint16x4_t low, uint16x4_t high);   // VMOV d0,d0

//Vector narrowing shift right by constant
// uint8x8_t  vshrn_n_u16(uint16x8_t a, __constrange(1,8) int b);  // VSHRN.I16 d0,q0,#8
*/
    // upper 8 pixels
    //store_pixel_block(out, pblock,
                pblock.val[0] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(R.val[0], Y00)), vqmovun_s32(vaddq_s32(R.val[1], Y01))), 8);
                pblock.val[1] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(G.val[0], Y00)), vqmovun_s32(vaddq_s32(G.val[1], Y01))), 8);
                pblock.val[2] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(B.val[0], Y00)), vqmovun_s32(vaddq_s32(B.val[1], Y01))), 8);
// Strided store: 4 is the stride factor
// pblock has a1 a2 a3 a4 a5 a6 a7 a8 b1 b2 b3 b4 b5 b6 b7 b8 g1 g2 g3 g4 g5 g6 g7 g8 r1 r2 r3 r4 r5 r6 r7 r8
// after vst4 --> a1 b1 g1 r1 a2 b2 g2 r2 ....
                vst4_u8(out, pblock);

//For the row below (+ width) same u and v values are also used.
    // lower 8 pixels
    //store_pixel_block(out+stride, pblock,
                pblock.val[0] =vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(R.val[0], Y10)), vqmovun_s32(vaddq_s32(R.val[1], Y11))), 8);
                pblock.val[1] =vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(G.val[0], Y10)), vqmovun_s32(vaddq_s32(G.val[1], Y11))), 8);
                pblock.val[2] =vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(B.val[0], Y10)), vqmovun_s32(vaddq_s32(B.val[1], Y11))), 8);
                vst4_u8(out+stride, pblock);
            }
        }
        return 1;
    }

    // Native function called from Java to process an image in parallel using nthreads threads
    void Java_es_uma_muii_apdm_ImageProcessingNative_MainActivity_YUVtoRGBNativeNEON( JNIEnv* env, jobject thiz,
                                                                                             jbyteArray data,
                                                                                             jintArray result,
                                                                                             jint width, jint height, jint nthreads)
    {
        unsigned char *cData;
        int *cResult;

        cData = (*env)->GetByteArrayElements(env,data,NULL); // While arrays of objects must be accessed one entry at a time, arrays of primitives can be read and written directly as if they were declared in C.   http://developer.android.com/training/articles/perf-jni.html
        if (cData==NULL) __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "Can't get data array reference");
        else
        {
            cResult= (*env)->GetIntArrayElements(env,result,NULL);
            if (cResult==NULL) __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "Can't get result array reference");
            else
            {
                omp_set_num_threads(nthreads);
                // operates on data
                convertYUV420_NV21toRGB8888_NEON(cData,cResult,width,height);
            }
        }
        if (cResult!=NULL) (*env)->ReleaseIntArrayElements(env,result,cResult,0); // care: errors in the type of the pointers are detected at runtime
        if (cData!=NULL) (*env)->ReleaseByteArrayElements(env, data, cData, 0); // release must be done even when the original array is not copied into cDATA
    }








static int convertYUV420_NV21toGREY8888_NEON(const unsigned char * data, int * pixels, int width, int height)
// pixels must have room for width*height ints, one per pixel
//bool decode_yuv_neon(unsigned char* out, unsigned char const* yuv, int width, int height, unsigned char fill_alpha=0xff)
{

    unsigned char* out = (unsigned char*) pixels;
    unsigned char const* yuv= data;
    unsigned char const fill_alpha=0xff;

    // pre-condition : width must be multiple of 8, height must be even
    if (0!=(width&7) || width<8 || 0!=(height&1) || height<2 || !out || !yuv)
        return 0;

    // Y' and UV pointers
    unsigned char const* y  = yuv;
    unsigned char const* uv = yuv + (width*height);
    // iteration counts
    int const itHeight = height>>1;
    int const itWidth = width>>3;

    // stride of each line
    int const stride = width*bytes_per_pixel;

    // total bytes of each write
    int const dst_pblock_stride = 8*bytes_per_pixel;

    int32x4_t const rounding = vdupq_n_s32(128);

    // a block temporary stores consecutively 8 pixels
    uint8x8x4_t pblock; //Array of 4 vectors with 8 lines, 8 bits each
    // Last position of the array initialized with 8 lines equal to 0xFF
    pblock.val[3] = vdup_n_u8(fill_alpha); // alpha channel in the last

    //R = [128 + 298(Y - 16) + 409(V - 128)] >> 8
    //G = [128 + 298(Y - 16) - 100(V - 128) - 208(U - 128)] >> 8
    //B = [128 + 298(Y - 16) + 516(U - 128)] >> 8

    // simd constants
    uint8x8_t const Yshift = vdup_n_u8(16); //this is the constant to substract to y

    // tmp variable to load y and uv
    uint16x8_t t;
    int i,j;

    for ( j=0; j<itHeight; ++j, y+=width, out+=stride) {
        for ( i=0; i<itWidth; ++i, y+=8, uv+=8, out+=dst_pblock_stride) {

            /*********************************************************************************/
            /**************************** 298(Y - 16) **************************************/
            /*********************************************************************************/

            // load u8x8 y values, substract 16 to each one and promote to u16x8:
            t = vmovl_u8(vqsub_u8(vld1_u8(y), Yshift));
            // splits the u16x8 in two u16x4 that are multiplied by 298 and promoted to u32x4
            int32x4_t const Y00 = vmulq_n_u32(vmovl_u16(vget_low_u16(t)), 298);
            int32x4_t const Y01 = vmulq_n_u32(vmovl_u16(vget_high_u16(t)), 298);
            // the same with the next row that also shares the u and v values
            t = vmovl_u8(vqsub_u8(vld1_u8(y+width), Yshift));
            int32x4_t const Y10 = vmulq_n_u32(vmovl_u16(vget_low_u16(t)), 298);
            int32x4_t const Y11 = vmulq_n_u32(vmovl_u16(vget_high_u16(t)), 298);


            /*********************************************************************************/
            /*********************************************************************************/
            /*********************************************************************************/

            /*********************************************************************************/
            /************************* (128 + LUMINANCIA) >> 8 *******************************/
            /*********************************************************************************/

            // upper 8 pixels
            //store_pixel_block(out, pblock,
            pblock.val[0] = vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(rounding, Y00)), vqmovun_s32(vaddq_s32(rounding, Y01))), 8);
            pblock.val[1] = pblock.val[0];
            pblock.val[2] = pblock.val[0];
            vst4_u8(out, pblock);

            //For the row below (+ width) same u and v values are also used.
            // lower 8 pixels
            //store_pixel_block(out+stride, pblock,
            pblock.val[0] =vshrn_n_u16(vcombine_u16(vqmovun_s32(vaddq_s32(rounding, Y10)), vqmovun_s32(vaddq_s32(rounding, Y11))), 8);
            pblock.val[1] = pblock.val[0];
            pblock.val[2] = pblock.val[0];
            vst4_u8(out+stride, pblock);
        }
    }
    return 1;
}





// Native function called from Java to process an image in parallel using nthreads threads
    void Java_es_uma_muii_apdm_ImageProcessingNative_MainActivity_YUVtoGREYNativeNEON( JNIEnv* env, jobject thiz,
                                                                                             jbyteArray data,
                                                                                             jintArray result,
                                                                                             jint width, jint height, jint nthreads)
    {
        unsigned char *cData;
        int *cResult;

        cData = (*env)->GetByteArrayElements(env,data,NULL); // While arrays of objects must be accessed one entry at a time, arrays of primitives can be read and written directly as if they were declared in C.   http://developer.android.com/training/articles/perf-jni.html
        if (cData==NULL) __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "Can't get data array reference");
        else
        {
            cResult= (*env)->GetIntArrayElements(env,result,NULL);
            if (cResult==NULL) __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "Can't get result array reference");
            else
            {
                omp_set_num_threads(nthreads);
                // operates on data
                convertYUV420_NV21toGREY8888_NEON(cData,cResult,width,height);
            }
        }
        if (cResult!=NULL) (*env)->ReleaseIntArrayElements(env,result,cResult,0); // care: errors in the type of the pointers are detected at runtime
        if (cData!=NULL) (*env)->ReleaseByteArrayElements(env, data, cData, 0); // release must be done even when the original array is not copied into cDATA
    }

    jboolean Java_es_uma_muii_apdm_ImageProcessingNative_MainActivity_isNEONSupported( JNIEnv* env, jobject thiz)
    {
        return JNI_TRUE;
    }







#else
// Native function called from Java to process an image in parallel using nthreads threads
void Java_es_uma_muii_apdm_ImageProcessingNative_MainActivity_YUVtoRGBNativeNEON( JNIEnv* env, jobject thiz,
                                                                                  jbyteArray data,
                                                                                  jintArray result,
                                                                                  jint width, jint height, jint nthreads)
{
    __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "NEON not supported");
}

// Native function called from Java to process an image in parallel using nthreads threads
void Java_es_uma_muii_apdm_ImageProcessingNative_MainActivity_YUVtoGREYNativeNEON( JNIEnv* env, jobject thiz,
                                                                                  jbyteArray data,
                                                                                  jintArray result,
                                                                                  jint width, jint height, jint nthreads)
{
    __android_log_print(ANDROID_LOG_INFO, "HOOKnative", "NEON not supported");
}



jboolean Java_es_uma_muii_apdm_ImageProcessingNative_MainActivity_isNEONSupported( JNIEnv* env, jobject thiz)
{
    return JNI_FALSE;
}
#endif

