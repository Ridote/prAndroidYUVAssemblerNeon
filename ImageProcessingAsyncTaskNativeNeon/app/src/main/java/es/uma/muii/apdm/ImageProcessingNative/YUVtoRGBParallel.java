package es.uma.muii.apdm.ImageProcessingNative;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by corbera on 23/10/15.
 */
public class YUVtoRGBParallel {

    private ExecutorService exSrv;
    private int nThreads;

    YUVtoRGBParallel(int _nthreads) {
        nThreads = _nthreads;
        exSrv = Executors.newFixedThreadPool(nThreads);
    }

    public void shutdown() {
        exSrv.shutdown();
    }

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

    private static void convertYUV420_NV21toRGB8888(byte [] data, int [] pixels, int width, int height, int my_id, int nThreads)
    // pixels must have room for width*height ints, one per pixel. See https://en.wikipedia.org/wiki/YUV
    {
        int size = width*height;
        int u=0, v=0, y1, y2, y3, y4;
        int myEnd, myOff;

        int myInitRow = (int)(((float)my_id/(float)nThreads)*((float)height/2.0f));
        int nextThreadRow = (int)(((float)(my_id+1)/(float)nThreads)*((float)height/2.0f));
        myOff = 2*width*myInitRow;
        myEnd = 2*width*nextThreadRow;
        int myUVOff = size+myInitRow*width;
        for(int i=myOff, k=myUVOff; i < myEnd; i+=2, k+=2) { // each 4x4 region of the bitmap has the same (u,v) but different y1,y2,y3,y4
            y1 = data[i  ]&0xff;
            y2 = data[i+1]&0xff;
            y3 = data[width+i  ]&0xff;
            y4 = data[width+i+1]&0xff;

            u = data[k  ]&0xff;
            v = data[k+1]&0xff;
            u = u-128;
            v = v-128;

            pixels[i  ] = convertYUVtoRGB(y1, u, v);
            pixels[i+1] = convertYUVtoRGB(y2, u, v);
            pixels[width+i  ] = convertYUVtoRGB(y3, u, v);
            pixels[width+i+1] = convertYUVtoRGB(y4, u, v);

            if (i!=0 && (i+2)%width==0)
                i+=width;
        }
    }

    private final class myJavaWorker implements Callable<Void> {
        private byte [] data;
        private int [] pixels;
        private int width;
        private int height;
        private int id;
        private int nth;

        public myJavaWorker(byte [] _data, int [] _pixels, int _width, int _height, int _th_id, int _nth) {
            data = _data;
            pixels = _pixels;
            width = _width;
            height = _height;
            id = _th_id;
            nth = _nth;
        }

        public Void call(){
            convertYUV420_NV21toRGB8888(data, pixels, width, height, id, nth);
            return null;
        }

    }

    public void convertYUV420_NV21toRGB8888_parallel(byte [] data, int [] pixels, int width, int height) {
        List<Callable<Void>> todoTasks;
        todoTasks = new ArrayList<Callable<Void>>(nThreads);

        for (int i=0; i < nThreads; i++) {
            todoTasks.add(i, new myJavaWorker(data, pixels, width, height, i, nThreads));
        }
        try {
            exSrv.invokeAll(todoTasks);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

}
