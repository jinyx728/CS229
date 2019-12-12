CMAKE_MINIMUM_REQUIRED( VERSION 3.1 )
MARK_AS_ADVANCED(CMAKE_BACKWARDS_COMPATIBILITY)

#Set a project name
SET(CMAKE_CXX_STANDARD 11)
SET(PROJECT_NAME ecos)
PROJECT(${PROJECT_NAME})

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DLDL_LONG -DDLONG -fPIC -std=c++11")
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DLDL_LONG -DDLONG -fPIC")

SET(LDL_c ${PROJECT_SOURCE_DIR}/external/ldl/src/ldl.c)

SET(AMD_c ${PROJECT_SOURCE_DIR}/external/amd/src/amd_1.c
             ${PROJECT_SOURCE_DIR}/external/amd/src/amd_2.c
             ${PROJECT_SOURCE_DIR}/external/amd/src/amd_aat.c
             ${PROJECT_SOURCE_DIR}/external/amd/src/amd_defaults.c
             ${PROJECT_SOURCE_DIR}/external/amd/src/amd_dump.c
             ${PROJECT_SOURCE_DIR}/external/amd/src/amd_global.c
             ${PROJECT_SOURCE_DIR}/external/amd/src/amd_info.c
             ${PROJECT_SOURCE_DIR}/external/amd/src/amd_order.c
             ${PROJECT_SOURCE_DIR}/external/amd/src/amd_post_tree.c
             ${PROJECT_SOURCE_DIR}/external/amd/src/amd_postorder.c
             ${PROJECT_SOURCE_DIR}/external/amd/src/amd_preprocess.c
             ${PROJECT_SOURCE_DIR}/external/amd/src/amd_valid.c
            )

SET(${PROJECT_NAME}_cpp
        ${PROJECT_SOURCE_DIR}/src/cone.c
        ${PROJECT_SOURCE_DIR}/src/ctrlc.c
        ${PROJECT_SOURCE_DIR}/src/ecos.c
        ${PROJECT_SOURCE_DIR}/src/equil.c
        ${PROJECT_SOURCE_DIR}/src/expcone.c
        ${PROJECT_SOURCE_DIR}/src/kkt.c
        ${PROJECT_SOURCE_DIR}/src/preproc.c
        ${PROJECT_SOURCE_DIR}/src/runecos.c
        ${PROJECT_SOURCE_DIR}/src/runecos_exp.c
        ${PROJECT_SOURCE_DIR}/src/spla.c
        ${PROJECT_SOURCE_DIR}/src/splamm.c
        ${PROJECT_SOURCE_DIR}/src/timer.c
        ${PROJECT_SOURCE_DIR}/src/wright_omega.c
        ${PROJECT_SOURCE_DIR}/src/kkt_lu.cpp
        ${PROJECT_SOURCE_DIR}/src/full_sym_grad.c
        )

SET(${PROJECT_NAME}_h
        ${PROJECT_SOURCE_DIR}/include/cone.h
        ${PROJECT_SOURCE_DIR}/include/ctrlc.h
        ${PROJECT_SOURCE_DIR}/include/data.h
        ${PROJECT_SOURCE_DIR}/include/ecos.h
        ${PROJECT_SOURCE_DIR}/include/ecos_bb.h
        ${PROJECT_SOURCE_DIR}/include/equil.h
        ${PROJECT_SOURCE_DIR}/include/expcone.h
        ${PROJECT_SOURCE_DIR}/include/glblotps.h
        ${PROJECT_SOURCE_DIR}/include/kkt.h
        ${PROJECT_SOURCE_DIR}/include/spla.h
        ${PROJECT_SOURCE_DIR}/include/splamm.h
        ${PROJECT_SOURCE_DIR}/include/timer.h
        ${PROJECT_SOURCE_DIR}/include/wright_omega.h
        ${PROJECT_SOURCE_DIR}/include/kkt_lu.h
        ${PROJECT_SOURCE_DIR}/src/full_sym_grad.h
        )

ADD_DEFINITIONS(-D_GLIBCXX_USE_CXX11_ABI=0) # Important!
# ECOS
INCLUDE_DIRECTORIES("include"
                    "external/ldl/include"
                    "external/amd/include"
                    "external/SuiteSparse_config"
                    "../eigen")


ADD_LIBRARY(${PROJECT_NAME} STATIC ${LDL_c} ${AMD_c} ${${PROJECT_NAME}_cpp} )

