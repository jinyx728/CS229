cmake_minimum_required( VERSION 3.1 )
mark_as_advanced(CMAKE_BACKWARDS_COMPATIBILITY)

set(PROJECT_NAME cudaqs)
find_package(CUDA REQUIRED)
# enable_language("CUDA")

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DLDL_LONG -DDLONG -fPIC -std=c++11 -pthread -D_GLIBCXX_USE_CXX11_ABI=0")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_61 --expt-extended-lambda")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_75,code=sm_75")

if($ENV{USER} MATCHES "dansj")
    set(TORCH_DIRECTORY "/data/dansj/anaconda3/envs/cudaqs/lib/python3.6/site-packages/torch")
elseif($ENV{USER} MATCHES "zhenglin")
    set(TORCH_DIRECTORY "/data/zhenglin/Tools/anaconda3/envs/p36/lib/python3.6/site-packages/torch")
else()
    message(FATAL_ERROR "Torch directory not found.")
endif()

set(CUDA_INCLUDE_DIRECTORY "/usr/local/cuda/include")

set(TORCH_INCLUDE
    ${TORCH_DIRECTORY}
    ${TORCH_DIRECTORY}/lib
    ${TORCH_DIRECTORY}/include/torch/csrc/api/include
    ${TORCH_DIRECTORY}/include
    ${TORCH_DIRECTORY}/include/TH
    ${TORCH_DIRECTORY}/include/THC
    )

if($ENV{USER} MATCHES "dansj")
    include_directories(${TORCH_INCLUDE} ${CUDA_INCLUDE_DIRECTORY} /usr/include/python3.6m)
elseif($ENV{USER} MATCHES "zhenglin")
    include_directories(${TORCH_INCLUDE} ${CUDA_INCLUDE_DIRECTORY} /usr/include/python3.6m)
endif()


add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_subdirectory(src)
add_subdirectory(include)

#add_executable(run_test ./tests/test.cpp)
#target_link_libraries(run_test PRIVATE ${LIBS})

IF(NOT CMAKE_DEBUG_POSTFIX)
    SET(CMAKE_DEBUG_POSTFIX -debug)
ENDIF()

link_directories(/usr/local/cuda/lib64/)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/tests/bin)
cuda_add_executable(test_forward ./tests/test_forward.cu ./tests/io_utils.cpp)
target_link_libraries(test_forward ${LIBS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/tests/bin)
cuda_add_executable(test_backward ./tests/test_backward.cu ./tests/io_utils.cpp)
target_link_libraries(test_backward ${LIBS})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/tests/bin)
cuda_add_executable(test_interface ./tests/test_interface.cu ./tests/io_utils.cpp)
set(CMAKE_PREFIX_PATH ${TORCH_DIRECTORY}/share/cmake/Torch ${CMAKE_PREFIX_PATH})
# find_package(Torch REQUIRED)
set(TORCH_LIBRARIES ${TORCH_DIRECTORY}/lib/libtorch.so
                    ${TORCH_DIRECTORY}/lib/libc10.so
                    ${TORCH_DIRECTORY}/lib/libc10_cuda.so
                    # ${TORCH_DIRECTORY}/lib//libcaffe2.so
                    # ${TORCH_DIRECTORY}/lib/libcaffe2_gpu.so
    )
target_link_libraries(test_interface ${LIBS} ${TORCH_LIBRARIES})
