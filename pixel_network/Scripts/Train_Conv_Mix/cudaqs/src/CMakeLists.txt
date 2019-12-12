set(SRC_CPP
    tensor.cu
    spring_system.cu
    newton_opt.cpp
    cg_solve.cpp
    cudaqs_utils.cu
    cudaqs_forward.cpp
    backward_opt.cpp
    cudaqs_backward.cpp
    opt_data.cpp
)

set(SRC_H
    cudaqs.h
)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_61")
if(CMAKE_BUILD_TYPE EQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

cuda_add_library(SRC_LIB STATIC ${SRC_CPP} ${SRC_H})
target_compile_features(SRC_LIB PUBLIC cxx_std_11)
set_target_properties( SRC_LIB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(SRC_LIB INTERFACE ./)
link_directories(/usr/local/cuda/lib64/)
set(LIBS SRC_LIB cublas ${LIBS} PARENT_SCOPE)
