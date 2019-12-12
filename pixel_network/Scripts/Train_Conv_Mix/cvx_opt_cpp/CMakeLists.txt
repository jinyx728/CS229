CMAKE_MINIMUM_REQUIRED( VERSION 3.1 )
MARK_AS_ADVANCED(CMAKE_BACKWARDS_COMPATIBILITY)

#Set a project name
# SET(CMAKE_CXX_STANDARD 11)
SET(PROJECT_NAME cvx_opt)
PROJECT(${PROJECT_NAME})

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DLDL_LONG -DDLONG -fPIC -std=c++11")
#Manually list your cpp and h files here - do NOT use GLOB/GLOB_RECURSE
SET(${PROJECT_NAME}_cpp
        ${PROJECT_SOURCE_DIR}/cvx_opt_utils.cpp
        ${PROJECT_SOURCE_DIR}/cvx_opt_forward.cpp
        ${PROJECT_SOURCE_DIR}/cvx_opt_backward.cpp
        ${PROJECT_SOURCE_DIR}/ecos_opt.cpp
        ${PROJECT_SOURCE_DIR}/forward_opt.cpp
        ${PROJECT_SOURCE_DIR}/backward_opt.cpp)

SET(${PROJECT_NAME}_h
        ${PROJECT_SOURCE_DIR}/cvx_opt_global.h
        ${PROJECT_SOURCE_DIR}/ecos_opt.h
        ${PROJECT_SOURCE_DIR}/forward_opt.h
        ${PROJECT_SOURCE_DIR}/backward_opt.h
        )

if($ENV{USER} MATCHES "dansj")
    SET(TORCH_DIRECTORY "/data/dansj/anaconda3/envs/newtontolerance/lib/python3.6/site-packages/torch")
elseif($ENV{USER} MATCHES "zhenglin")
    SET(TORCH_DIRECTORY "/data/zhenglin/Tools/anaconda3/envs/p36/lib/python3.6/site-packages/torch")
else()
    message(FATAL_ERROR "Torch directory not found.")
endif()

# SET(TORCH_DIRECTORY "/home/njin19/environments/python3.6_env/lib64/python3.6/site-packages/torch")
# SET(TORCH_DIRECTORY "/data/zhenglin/Tools/anaconda3/envs/p36/lib/python3.6/site-packages/torch")
SET(TORCH_INCLUDE ${TORCH_DIRECTORY}/include/torch/csrc/api/include
                  ${TORCH_DIRECTORY}/include
                  ${TORCH_DIRECTORY}/include/TH
                  ${TORCH_DIRECTORY}/include/THC
                  )
ADD_DEFINITIONS(-D_GLIBCXX_USE_CXX11_ABI=0) # Important!
# ECOS
# Set ECOS_DIRECTORY in .bashrc
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}
                    "../ecos/include"
                    "../ecos/external/ldl/include"
                    "../ecos/external/amd/include"
                    "../ecos/external/SuiteSparse_config"
                    ${TORCH_INCLUDE}
                    "../eigen")

IF(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    MESSAGE("debug")
    LINK_DIRECTORIES("../ecos/build_dbg")
ELSE()
    MESSAGE("Release")
    LINK_DIRECTORIES("../ecos/build_cmk")
ENDIF()

SET(LIBRARIES_TO_LINK ecos)

ADD_LIBRARY(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_cpp})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${LIBRARIES_TO_LINK})

IF(NOT CMAKE_DEBUG_POSTFIX)
    SET(CMAKE_DEBUG_POSTFIX -debug)
ENDIF()
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/tests/bin)
ADD_EXECUTABLE(test_forward tests/src/test_forward.cpp
                          tests/src/io_utils.cpp
                )
SET_TARGET_PROPERTIES(test_forward PROPERTIES
                        DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
TARGET_LINK_LIBRARIES(test_forward ${PROJECT_NAME})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/tests/bin)
ADD_EXECUTABLE(test_backward tests/src/test_backward.cpp
                          tests/src/io_utils.cpp
                )
SET_TARGET_PROPERTIES(test_backward PROPERTIES
                        DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
TARGET_LINK_LIBRARIES(test_backward ${PROJECT_NAME})


