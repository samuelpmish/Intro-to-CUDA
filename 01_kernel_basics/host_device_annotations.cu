////////////////////////////////////////////////////////////////////////////////

// requires --extended-lambda flag
constexpr double constexpr_function(double x) {
    return x * x;
}

__global__ void f0(double * out, double * in) {
    *out = constexpr_function(*in);
}

////////////////////////////////////////////////////////////////////////////////

__device__ double free_function(double x) {
    return x * x;
}

__global__ void f1(double * out, double * in) {
    *out = free_function(*in);
}

////////////////////////////////////////////////////////////////////////////////

struct stateless_function_object {
    __device__ double operator()(double x) {
        return x * x;
    }
};

__global__ void f2(double * out, double * in) {
    stateless_function_object fn_obj;
    *out = fn_obj(*in);
}

////////////////////////////////////////////////////////////////////////////////

struct stateful_function_object {
    __device__ double operator()(double x) {
        return scale * x;
    }

    double scale;
};

__global__ void f3(double * out, double * in) {
    stateful_function_object fn_obj{3.14};
    *out = fn_obj(*in);
}

__global__ void f4(double * out, double * in, stateful_function_object fn_obj) {
    *out = fn_obj(*in);
}

////////////////////////////////////////////////////////////////////////////////

template < typename callable > 
__global__ void f5(double * out, double * in, callable fn) {
    *out = fn(*in);
}

////////////////////////////////////////////////////////////////////////////////

struct base_class {
    __device__ virtual double operator()(double x) = 0;
};

struct derived_class : public base_class {
     __device__ double operator() (double x) final {
        return x * x;
    };
};

// "It is not allowed to pass as an argument to a __global__ function 
//      an object of a class derived from virtual base classes."
//                  - CUDA Programming Guide
__global__ void f6(double * out, double * in, base_class * fn_obj) {
    *out = (*fn_obj)(*in);
}

__global__ void f7(double * out, double * in) {
    derived_class fn_obj;
    base_class * base = &fn_obj;
    *out = (*base)(*in);
}

////////////////////////////////////////////////////////////////////////////////

int main() {

    double * out;
    double * in;

    cudaMalloc(&out, sizeof(double));
    cudaMalloc(&in, sizeof(double));

    // ✅ constexpr function in kernel
    f0<<<1,1>>>(out, in);
    cudaDeviceSynchronize();

    // ✅ free function in kernel
    f1<<<1,1>>>(out, in);
    cudaDeviceSynchronize();

    // ✅ stateless function object in kernel
    f2<<<1,1>>>(out, in);
    cudaDeviceSynchronize();

    // ✅ stateful function object in kernel
    f3<<<1,1>>>(out, in);
    cudaDeviceSynchronize();

    // ✅ trivially copiable function object in kernel
    f4<<<1,1>>>(out, in, stateful_function_object{3.14});
    cudaDeviceSynchronize();

    // ✅ lambda function in kernel (note: requires -extended-lambda flag)
    f5<<<1,1>>>(out, in, [] __device__ (double x) { return x * x; });
    cudaDeviceSynchronize();

#if 0
    // ❌ host-instantiated object derived from virtual base in kernel 
    derived_class fn_obj;
    f6<<<1,1>>>(out, in, &fn_obj);
    cudaDeviceSynchronize();

    // ❌ host-instantiated, deep-copied object derived from virtual base in kernel  
    derived_class * d_fn_obj;
    cudaMalloc(&d_fn_obj, sizeof(derived_class));
    cudaMemcpy(&d_fn_obj, &fn_obj, sizeof(derived_class), cudaMemcpyDeviceToHost);
    f6<<<1,1>>>(out, in, d_fn_obj);
    cudaDeviceSynchronize();
    cudaFree(d_fn_obj);
#endif

    // ✅ kernel-instantiated class in kernel
    f7<<<1,1>>>(out, in);
    cudaDeviceSynchronize();

    cudaFree(in);
    cudaFree(out);

}
