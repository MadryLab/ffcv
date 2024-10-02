#include <cstdint>
#include <Python.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include <turbojpeg.h>
#include <malloc.h>
#include <pthread.h>
#include <iostream>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h> 

#ifdef _WIN32
    typedef unsigned __int32 __uint32_t;
    typedef unsigned __int64 __uint64_t;
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

// #define _DEBUG
#ifdef _DEBUG
#define DBOUT std::cout // or any other ostream
#else
#define DBOUT 0 && std::cout
#endif

#include <utility> // For std::pair


int axis_to_image_boundaries(int a, int img_boundary, int mcuBlock) {
    int img_b = img_boundary - (img_boundary % mcuBlock);
    int delta_a = a % mcuBlock;
    // reduce the a to align the mcu block
    if (a > img_b) {
        a = img_b;

    } else {
        a -= delta_a;
    }
    return a;
}

struct Boundaries {
    int x;
    int y;
    int h;
    int w;
};

extern "C" {
    // a key use to point to the tjtransform instance
    static pthread_key_t key_tj_transformer;
    // a key use to point to the tjdecompressor instance
    static pthread_key_t key_tj_decompressor;
    static pthread_key_t key_share_buffer;
    static pthread_once_t key_once = PTHREAD_ONCE_INIT;

    // will make the keys to access the tj instances
    static void make_keys()
    {
        pthread_key_create(&key_tj_decompressor, NULL);
        pthread_key_create(&key_tj_transformer, NULL);
        pthread_key_create(&key_share_buffer, NULL);
    }

    EXPORT int cv_imdecode(uint8_t* buf, 
                            uint64_t buf_size, 
                            int64_t flag,
                            uint8_t* output_buffer){
        DBOUT << "imdecode called" << std::endl;
        cv::Mat bufArray(1, buf_size, CV_8UC1, buf);
        cv::Mat image;
        image = cv::imdecode(bufArray, flag);
        // Check for failure
        if (image.empty()) {
            std::cout << "Could not decode the image" << std::endl;
            return -1;
        }else{
            DBOUT << "Image decoded" << image.rows<<","<<image.cols<<std::endl;
            cv::Mat dest_matrix(image.rows, image.cols, CV_8UC3, output_buffer);
            image.copyTo(dest_matrix);
        }

        return 0;
    }

    EXPORT void resize(int64_t cresizer, int64_t source_p, int64_t sx, int64_t sy,
                int64_t start_row, int64_t end_row, int64_t start_col, int64_t end_col,
                int64_t dest_p, int64_t tx, int64_t ty, int64_t interpolation) {
        // TODO use proper arguments type

        cv::Mat source_matrix(sx, sy, CV_8UC3, (uint8_t*) source_p);
        cv::Mat dest_matrix(tx, ty, CV_8UC3, (uint8_t*) dest_p);
        cv::resize(source_matrix.colRange(start_col, end_col).rowRange(start_row, end_row),
                   dest_matrix, dest_matrix.size(), 0, 0, interpolation);
    }

    EXPORT void my_memcpy(void *source, void* dst, uint64_t size) {
        memcpy(dst, source, size);
    }

    EXPORT void my_fread(int64_t fp, int64_t offset, void *destination, int64_t size) {
        fseek((FILE *) fp, offset, SEEK_SET);
        fread(destination, 1, size, (FILE *) fp);
    }

    EXPORT int imdecode(unsigned char *input_buffer, __uint64_t input_size,
                      __uint32_t source_height, __uint32_t source_width,

                      unsigned char *output_buffer,
                      __uint32_t crop_height, __uint32_t crop_width,
                      __uint32_t offset_x, __uint32_t offset_y,
                      __uint32_t scale_num, __uint32_t scale_denom,
                      bool enable_crop,
                      bool hflip)
    {
        pthread_once(&key_once, make_keys);

        tjhandle tj_transformer;
        tjhandle tj_decompressor;
        if ((tj_transformer = pthread_getspecific(key_tj_transformer)) == NULL)
        {
            tj_transformer = tjInitTransform();
            pthread_setspecific(key_tj_transformer, tj_transformer);
        }
        if ((tj_decompressor = pthread_getspecific(key_tj_decompressor)) == NULL)
        {
            tj_decompressor = tjInitDecompress();
            pthread_setspecific(key_tj_decompressor, tj_decompressor);
        }

        tjtransform xform;
        tjscalingfactor scaling;
        memset(&xform, 0, sizeof(tjtransform));
        if (hflip) {
          xform.op = TJXOP_HFLIP;
        }
        xform.r.x = offset_x;
        xform.r.y = offset_y;
        xform.r.h = crop_height;
        xform.r.w = crop_width;
        xform.options |= TJXOPT_CROP;
        scaling.num = scale_num;
        scaling.denom = scale_denom;

        unsigned char *dstBuf = NULL;
        unsigned long dstSize = 0;

        bool do_transform = enable_crop || hflip;

        if (do_transform) {
            tjTransform(tj_transformer, input_buffer, input_size, 1, &dstBuf,
                        &dstSize, &xform, TJFLAG_FASTDCT);
        } else {
            dstBuf = input_buffer;
            dstSize = input_size;
        }
        int result =  tjDecompress2(tj_decompressor, dstBuf, dstSize, output_buffer,
                TJSCALED(crop_width, scaling), 0, TJSCALED(crop_height, scaling),
                TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_NOREALLOC);

        if (do_transform) {
             tjFree(dstBuf);
        }
        return result;
    }

    static PyMethodDef libffcvMethods[] = {
        {NULL, NULL, 0, NULL}
    };


    static struct PyModuleDef libffcvmodule = {
        PyModuleDef_HEAD_INIT,
        "libffcv",
        "This is a dummy python extension, the real code is available through ctypes",
        -1,
        libffcvMethods
    };

    PyMODINIT_FUNC PyInit__libffcv(void) {
        return PyModule_Create(&libffcvmodule);
    }


    EXPORT int imcropresizedecode(unsigned char *input_buffer, __uint64_t input_size, 
                      unsigned char *output_buffer,
                      __uint32_t tar_height, __uint32_t tar_width,
                      __uint32_t crop_height, __uint32_t crop_width,
                      __uint32_t offset_y, __uint32_t offset_x,
                      __uint32_t interpolation
                      )
    {
        pthread_once(&key_once, make_keys);
        tjhandle tj_decompressor;
        if ((tj_decompressor = pthread_getspecific(key_tj_decompressor)) == NULL)
        {
            tj_decompressor = tj3Init(TJINIT_DECOMPRESS);
            pthread_setspecific(key_tj_decompressor, tj_decompressor);
        }
        int result ;

        // get info about the cropped image
        result =  tj3DecompressHeader(tj_decompressor, input_buffer, input_size);

        int subsamp  = tj3Get(tj_decompressor, TJPARAM_SUBSAMP);
        int height = tj3Get(tj_decompressor, TJPARAM_JPEGHEIGHT);
        int width = tj3Get(tj_decompressor, TJPARAM_JPEGWIDTH);

        // int MCU_block = tjMCUWidth[subsamp];
        int x_boundaries = axis_to_image_boundaries(offset_x, width, tjMCUWidth[subsamp]);
        int y_boundaries = axis_to_image_boundaries(offset_y, height, tjMCUHeight[subsamp]);

        crop_width = offset_x + crop_width - x_boundaries;
        crop_height = offset_y + crop_height - y_boundaries;

        tjregion region = {x_boundaries, y_boundaries, crop_width, crop_height};
        // crop the input image
        result = tj3SetCroppingRegion(tj_decompressor, region);
        if (result == -1) {
            const char* error_message =  tj3GetErrorStr(tj_decompressor);
            // Handle the error
            std::cerr << "Error in tj3SetCroppingRegion: " << error_message << std::endl;
            return -1;
        }
        // decompress the cropped image
        unsigned char *tmp_buffer = NULL;
        // size_t buf_size=tjBufSize(crop_width, crop_height, TJPF_RGB);
        size_t buf_size=tj3JPEGBufSize(width, height, TJPF_RGB);
        
        tmp_buffer = (unsigned char *)malloc(buf_size);
        
        result =  tj3Decompress8(tj_decompressor, input_buffer, input_size, tmp_buffer,
                 0,  TJPF_RGB);
        
        if (result == -1) {
            const char* error_message =  tj3GetErrorStr(tj_decompressor);
            // Handle the error
            std::cerr << "Error in tj3Decompress8: " << error_message << std::endl;
            return -1;
        }

        // resize the cropped image
        cv::Mat source_matrix(crop_height, crop_width, CV_8UC3, (uint8_t*) tmp_buffer);
        cv::Mat dest_matrix(tar_height, tar_width, CV_8UC3, (uint8_t*) output_buffer);
        int dx = offset_x-x_boundaries;
        int dy = offset_y-y_boundaries;
        cv::resize((source_matrix
            .colRange(dx,crop_width)
            .rowRange(dy,crop_height)
            ),
                   dest_matrix, dest_matrix.size(), 0, 0, interpolation);
        free(tmp_buffer);
        return result;
    }
}
