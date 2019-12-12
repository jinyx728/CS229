add_library(ctpl_lib INTERFACE)
target_include_directories(ctpl_lib INTERFACE ./)
set(LIBS ctpl_lib ${LIBS} PARENT_SCOPE)
