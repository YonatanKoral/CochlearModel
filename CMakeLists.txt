cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
# I am leaving it here for compiling on unix, this will not work on visual studio
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
set(LIB_NAME "AudioLabCM")
set(TARGET_ARCHITECTURE "30")
set(TARGET_COMPUTE "30")
#set( CMAKE_CUDA_FLAGS "--ptxas-options=-v;-rdc=true;--keep;-gencode arch=compute_${TARGET_COMPUTE},code=sm_${TARGET_ARCHITECTURE}" CACHE STRING "" )


OPTION(DYNAMIC_LINKING "Set to ON to build the module for dynamic linking.  Use OFF for static." ON)
set(cudaNVCCFlags "-gencode=arch=compute_${TARGET_COMPUTE},code=sm_${TARGET_ARCHITECTURE};-cudart shared" CACHE STRING "")
set(CUDA_SDK_ROOT_DIR "$ENV{NVCUDASAMPLES_ROOT}" CACHE STRING "")
set(CUDA_SDK_INCLUDE_DIR "${CUDA_SDK_ROOT_DIR}/common/inc")
set(CUDA_SDK_LIB_DIR "${CUDA_SDK_ROOT_DIR}/common/lib/x64")

set(CUDA_SDK_ROOT_DIR2 "$ENV{CUDA_PATH}" CACHE STRING "")
set(CUDA_SDK_INCLUDE_DIR2 "${CUDA_SDK_ROOT_DIR2}/include")
set(CUDA_SDK_LIB_DIR2 "${CUDA_SDK_ROOT_DIR2}/lib/x64")
project(${LIB_NAME} LANGUAGES CUDA CXX C)

#enable_language(CUDA)
FIND_PACKAGE(Matlab COMPONENTS MX_LIBRARY ENG_LIBRARY MAT_LIBRARY)
FIND_PACKAGE(Eigen3 REQUIRED)
message("Eigen3 include: ${EIGEN3_INCLUDE_DIR}")
message("CUDA_SDK_INCLUDE_DIR: ${CUDA_SDK_INCLUDE_DIR}, ${CUDA_SDK_INCLUDE_DIR2}")
#FIND_PACKAGE(CUDA)
if ( CMAKE_CUDA_FLAGS )
	set(temp_cuda_flags ${CMAKE_CUDA_FLAGS})
	#list(REMOVE_ITEM temp_cuda_flags "-cudart static")
	#list(REMOVE_ITEM temp_cuda_flags "-cudart shared")
	string(REPLACE " -cudart shared" "" temp_cuda_flags ${CMAKE_CUDA_FLAGS})
	string(REPLACE " -cudart static" "" temp_cuda_flags ${temp_cuda_flags})
	string(APPEND temp_cuda_flags " -cudart shared")
	foreach(nvccFlag ${cudaNVCCFlags})
	message("nvcc flag: ${nvccFlag}")
	string(REPLACE " ${nvccFlag}" "" temp_cuda_flags ${temp_cuda_flags})
	string(APPEND temp_cuda_flags " ${nvccFlag}")
	endforeach(nvccFlag)
	set(CMAKE_CUDA_FLAGS ${temp_cuda_flags} CACHE STRING "" FORCE)
endif()

# add modules to cmake module path
############################
# We have some custom .cmake scripts not in the official distribution.
SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${${LIB_NAME}_SOURCE_DIR}/CMakeModules")

# Make the comfortable  cmake modules of CMake 2.6.3 also available for users of older
# cmake versions instead of requiring CMake >= 2.6.3
#ET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${${LIB_NAME}_SOURCE_DIR}/CMakeModules/osg")
set(CMAKE_INSTALL_PREFIX "D:/documents/Cochlea_model/Matlab/matlab_files/cochlea_gui" CACHE STRING "" FORCE)
# set custom .cmake CUDA scripts also to the module path
#SET(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${${LIB_NAME}_SOURCE_DIR}/CMakeModules/cuda")
# Make the headers visible to everything
IF(NOT ${PROJECT_BINARY_DIR} EQUAL ${PROJECT_SOURCE_DIR})
   INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR}/include)
ENDIF(NOT ${PROJECT_BINARY_DIR} EQUAL ${PROJECT_SOURCE_DIR})

SET(HEADER_PATH ${PROJECT_SOURCE_DIR}/include)
FIND_PACKAGE(Boost 1.69)
INCLUDE_DIRECTORIES(
    ${HEADER_PATH}
	${HEADER_PATH}/IowaHillsFilters
	${HEADER_PATH}/firpm
	${HEADER_PATH}/mexplus
	${Boost_INCLUDE_DIRS}
	${Matlab_INCLUDE_DIRS}
	${CUDA_TOOLKIT_INCLUDE}
	${CUDA_INCLUDE_DIRS}
	${EIGEN3_INCLUDE_DIR}
	${CUDA_SDK_INCLUDE_DIR}
	${CUDA_SDK_INCLUDE_DIR2}
)



################################################################################
# Platform specific definitions
################################################################################

IF(WIN32)
    IF(MSVC)
        ADD_DEFINITIONS(-D_SCL_SECURE_NO_WARNINGS)
        ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE)
        SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Zi")
        SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Gy")
    ENDIF(MSVC)
ENDIF(WIN32)

################################################################################
# Create bin and lib directories in build path
################################################################################

# Create bin and lib directories if required
IF("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
   FILE(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin ${CMAKE_BINARY_DIR}/lib)
ENDIF("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")


################################################################################
# Installation stuff
################################################################################

SET(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "add a postfix, usually d on windows")
ADD_DEFINITIONS(-DDEBUG_POSTFIX="${CMAKE_DEBUG_POSTFIX}")


# Make the bin/ and lib/ folders
SET(OUTPUT_BINDIR ${PROJECT_BINARY_DIR}/bin)
MAKE_DIRECTORY(${OUTPUT_BINDIR})

SET(OUTPUT_LIBDIR ${PROJECT_BINARY_DIR}/lib)
MAKE_DIRECTORY(${OUTPUT_LIBDIR})



################################################################################
# User Options
################################################################################

############################
# Dynamic vs Static Linking
############################

IF   (DYNAMIC_LINKING)
    SET(LINKING_USER_DEFINED_DYNAMIC_OR_STATIC "SHARED")
ELSE (DYNAMIC_LINKING)
    SET(LINKING_USER_DEFINED_DYNAMIC_OR_STATIC "STATIC")
ENDIF(DYNAMIC_LINKING)


# setting afx dll to acommodate inner definitions
IF(DYNAMIC_LINKING)
	add_definitions(-D_AFXDLL)
	set(CMAKE_MFC_FLAG 2)
ENDIF(DYNAMIC_LINKING)

OPTION(MEX_PROJECT "Set to ON to compile as mex prject" ON)
if(MEX_PROJECT)
ADD_DEFINITIONS(-DCUDA_MEX_PROJECT="1")
ADD_DEFINITIONS(-DMATLAB_MEX_FILE="1")
endif(MEX_PROJECT)

ADD_DEFINITIONS(-DMATLAB_MEX_FILE="1")
############################
# Include important macro
############################
INCLUDE(MacroUtils)


file(GLOB ${LIB_NAME}_SRC
    "${PROJECT_SOURCE_DIR}/src/*.cu"
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
	"${PROJECT_SOURCE_DIR}/src/IowaHillsFilters/*.cpp"
	"${PROJECT_SOURCE_DIR}/src/firpm/*.cpp"
	"${PROJECT_SOURCE_DIR}/assets/*.def"
)
set (CMAKE_MODULE_LINKER_FLAGS   "${CMAKE_MODULE_LINKER_FLAGS} \"/DEF:${CMAKE_CURRENT_SOURCE_DIR}/assets/${LIB_NAME}.def\"")

file(GLOB ${LIB_NAME}_HEADERS
    "${HEADER_PATH}/*.cuh"
    "${HEADER_PATH}/*.h"
	"${HEADER_PATH}/firpm/*.h"
	"${HEADER_PATH}/mexplus/*.h"
	"${HEADER_PATH}/IowaHillsFilters/*.h"
)
set(CUDA_VERBOSE_BUILD ON CACHE BOOL "override verobose" FORCE)
link_directories(${OUTPUT_LIBDIR} ${CUDA_SDK_LIB_DIR} ${CUDA_SDK_LIB_DIR2})
message("CUDA_TOOLKIT_ROOT_DIR: ${CUDA_TOOLKIT_ROOT_DIR}")
message("CUDA_SDK_ROOT_DIR: ${CUDA_SDK_ROOT_DIR}")
message("OUTPUT_LIBDIR: ${OUTPUT_LIBDIR}")
message("cuda nvcc flags are: ${CUDA_NVCC_FLAGS}")
add_library(${LIB_NAME} ${LINKING_USER_DEFINED_DYNAMIC_OR_STATIC} ${${LIB_NAME}_HEADERS} ${${LIB_NAME}_SRC})
target_compile_features(${LIB_NAME} PUBLIC cxx_std_11)
set_property(TARGET ${LIB_NAME} PROPERTY CUDA_STANDARD 11)
set_target_properties(${LIB_NAME} PROPERTIES
    PREFIX ""
    SUFFIX  ".${Matlab_MEX_EXTENSION}"
)
set_target_properties(${LIB_NAME} PROPERTIES
	CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties( ${LIB_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "" FORCE)
message("Matlab libraries are: ${Matlab_MEX_LIBRARY}  ${Matlab_MX_LIBRARY}  ${Matlab_ENG_LIBRARY} ${Matlab_MAT_LIBRARY}")
message("Cuda Libraries are: ${CUDA_LIBRARIES}")
LINK_WITH_VARIABLES(${LIB_NAME} CUDA_LIBRARIES)
string(REPLACE " " ";" cudaLoopLibraries ${CMAKE_CUDA_STANDARD_LIBRARIES})
foreach(cstl ${cudaLoopLibraries})
message("link ${cstl}")
LINK_WITH_VARIABLES(${LIB_NAME} cstl)
endforeach(cstl)
LINK_WITH_VARIABLES(${LIB_NAME}
Matlab_MEX_LIBRARY 
Matlab_MX_LIBRARY 
Matlab_ENG_LIBRARY
Matlab_MAT_LIBRARY)

INSTALL(
    TARGETS ${LIB_NAME}
    RUNTIME DESTINATION "."
)