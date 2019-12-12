CMAKE_MINIMUM_REQUIRED( VERSION 3.1 )
MARK_AS_ADVANCED(CMAKE_BACKWARDS_COMPATIBILITY)

#Set a project name
SET(CMAKE_CXX_STANDARD 11)
SET(PROJECT_NAME acos)
PROJECT(${PROJECT_NAME})

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -fPIC -std=c++11")
SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -fPIC")

SET(LDL_c ${PROJECT_SOURCE_DIR}/solvers/external/ldl/src/ldl.c)

SET(AMD_c    ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_1.c
             ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_2.c
             ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_aat.c
             ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_defaults.c
             ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_dump.c
             ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_global.c
             ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_info.c
             ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_order.c
             ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_post_tree.c
             ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_postorder.c
             ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_preprocess.c
             ${PROJECT_SOURCE_DIR}/solvers/external/amd/src/amd_valid.c
            )
SET(LA_h ${PROJECT_SOURCE_DIR}/la/la_eigen.h
         ${PROJECT_SOURCE_DIR}/la/krylov_system_base.h
         ${PROJECT_SOURCE_DIR}/la/conjugate_gradient.h
         )
SET(LA_cpp ${PROJECT_SOURCE_DIR}/la/la_eigen.cpp
           ${PROJECT_SOURCE_DIR}/la/conjugate_gradient.cpp
           )

SET(LDL_3X3_cpp ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/impl/cone.cpp
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/impl/kkt.cpp
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/impl/spla.cpp
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/impl/splamm.cpp
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/kkt_ldl_3x3.cpp
            )
SET(LDL_2X2_cpp ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_2x2/kkt_ldl_2x2.cpp)

SET(LDL_3X3_h ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/impl/cone.h
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/impl/kkt.h
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/impl/spla.h
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/impl/splamm.h
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/kkt_ldl_3x3.h
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/pwork_ldl_3x3.h
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3/policy_ldl_3x3.h
            )
SET(LDL_2X2_h   ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_2x2/kkt_ldl_2x2.h
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_2x2/pwork_ldl_2x2.h
                ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_2x2/policy_ldl_2x2.h
            )
SET(CG_EIGEN_cpp ${PROJECT_SOURCE_DIR}/solvers/kkt_cg_eigen/kkt_cg_eigen.cpp
                 ${PROJECT_SOURCE_DIR}/solvers/kkt_cg_eigen/system_cg_eigen.cpp
            )
SET(CG_EIGEN_h ${PROJECT_SOURCE_DIR}/solvers/kkt_cg_eigen/kkt_cg_eigen.h
                   ${PROJECT_SOURCE_DIR}/solvers/kkt_cg_eigen/system_cg_eigen.h
                   ${PROJECT_SOURCE_DIR}/solvers/kkt_cg_eigen/pwork_cg_eigen.h
            )

SET(${PROJECT_NAME}_cpp
        ${PROJECT_SOURCE_DIR}/pwork.cpp
        ${PROJECT_SOURCE_DIR}/solver.cpp
    )

SET(${PROJECT_NAME}_h
        ${PROJECT_SOURCE_DIR}/acos.h
        ${PROJECT_SOURCE_DIR}/glblotps.h
        ${PROJECT_SOURCE_DIR}/kkt_data.h
        ${PROJECT_SOURCE_DIR}/cone_data.h
        ${PROJECT_SOURCE_DIR}/pwork.h
        ${PROJECT_SOURCE_DIR}/solver.h
        ${PROJECT_SOURCE_DIR}/la/la_eigen.h
        ${PROJECT_SOURCE_DIR}/solvers/kkt_base.h
    )

SET(test_cloth_cpp
    ${PROJECT_SOURCE_DIR}/tests/src/io_utils.cpp
    ${PROJECT_SOURCE_DIR}/tests/src/test_cloth.cpp)

IF(NOT CMAKE_DEBUG_POSTFIX)
    SET(CMAKE_DEBUG_POSTFIX -debug)
ENDIF()

ADD_DEFINITIONS(-D_GLIBCXX_USE_CXX11_ABI=0) # Important!
# ACOS
INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}
                    ${PROJECT_SOURCE_DIR}/../eigen
                    ${PROJECT_SOURCE_DIR}/solvers/external/ldl/include
                    ${PROJECT_SOURCE_DIR}/solvers/external/amd/include
                    ${PROJECT_SOURCE_DIR}/solvers/external/SuiteSparse_config
                    ${PROJECT_SOURCE_DIR}/solvers/
                    ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_3x3
                    ${PROJECT_SOURCE_DIR}/solvers/kkt_ldl_2x2
                    ${PROJECT_SOURCE_DIR}/solvers/kkt_cg_eigen
                    ${PROJECT_SOURCE_DIR}/la/
                    )

ADD_LIBRARY(${PROJECT_NAME} STATIC ${${PROJECT_NAME}_cpp} ${LA_cpp} ${LDL_3X3_cpp} ${LDL_2X2_cpp} ${CG_EIGEN_cpp} ${LDL_c} ${AMD_c})

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/tests/bin)
ADD_EXECUTABLE(test_cloth ${test_cloth_cpp})
SET_TARGET_PROPERTIES(test_cloth PROPERTIES
                        DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
TARGET_LINK_LIBRARIES(test_cloth acos)

