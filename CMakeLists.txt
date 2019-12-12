CMAKE_MINIMUM_REQUIRED( VERSION 3.1 )
MARK_AS_ADVANCED(CMAKE_BACKWARDS_COMPATIBILITY)
###########################################################################
#Project-specific Configuration
###########################################################################

SET (CMAKE_CXX_STANDARD 11)

#Set a project name
SET(PROJECT_NAME tex)
PROJECT(${PROJECT_NAME})

#Manually list your cpp and h files here - do NOT use GLOB/GLOB_RECURSE
SET(${PROJECT_NAME}_cpp
        ${PROJECT_SOURCE_DIR}/texture_extrapolation_diffusion.cpp
)

SET(${PROJECT_NAME}_h
        ${PROJECT_SOURCE_DIR}/GRID_TEST.h
        ${PROJECT_SOURCE_DIR}/FLAT_CLOTH_TEST.h
)

#List the libraries you want to link against - including a higher-level
#library automatically includes the lower-level ones
SET(LIBRARIES_TO_LINK PhysBAM_Rendering-OpenGL)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-but-set-variable")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-narrowing")
INCLUDE(${PROJECT_SOURCE_DIR}/../../Scripts/CMake/Project.cmake)
