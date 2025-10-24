#pragma once

#ifdef __CUDACC__

#else
    #define __host__
    #define __device__
#endif