package es.uma.muii.apdm.ImageProcessingNative;

/**
 * Created by corbera on 22/10/15.
 */
public class YUVtoRGB {

    private static int convertYUVtoRGB(int y, int u, int v)
    {
        int r,g,b;

        r = y + (int)1.14f*v;
        g = y - (int)(0.395f*u +0.581f*v);
        b = y + (int)2.033f*u;
        r = r>255? 255 : r<0 ? 0 : r;
        g = g>255? 255 : g<0 ? 0 : g;
        b = b>255? 255 : b<0 ? 0 : b;
        return 0xff000000 | (b<<16) | (g<<8) | r; // one byte for alpha, one for blue, one for green, one for red
    }

    public static void convertYUV420_NV21toRGB8888(byte [] data, int [] pixels, int width, int height)
    // pixels must have room for width*height ints, one per pixel. See https://en.wikipedia.org/wiki/YUV
    {
        int size = width*height;
        int u, v, y1, y2, y3, y4;

        // i traverses Y
        // k traverses U and V
        for(int i=0, k=0; i < size; i+=2, k+=2) { // each 4x4 region of the bitmap has the same (u,v) but different y1,y2,y3,y4
            y1 = data[i  ]&0xff;
            y2 = data[i+1]&0xff;
            y3 = data[width+i  ]&0xff;
            y4 = data[width+i+1]&0xff;

            u = data[size+k  ]&0xff;
            v = data[size+k+1]&0xff;
            u = u-128;
            v = v-128;

            pixels[i  ] = convertYUVtoRGB(y1, u, v);
            pixels[i+1] = convertYUVtoRGB(y2, u, v);
            pixels[width+i  ] = convertYUVtoRGB(y3, u, v);
            pixels[width+i+1] = convertYUVtoRGB(y4, u, v);

            if (i!=0 && (i+2)%width==0)
                i+=width;
        }
/*
        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
*/
    }
}
