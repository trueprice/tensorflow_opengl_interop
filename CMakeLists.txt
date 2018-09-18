cmake_minimum_required(VERSION 2.8)
project(simple_blend_network)

# C++11 compatibility
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
  message(ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include(${PROJECT_SOURCE_DIR}/cmake/SelectCudaComputeArch.cmake)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")

# find packages

IF (WIN32)
  
  set(GLEW_FOUND 1)
  set(GLEW_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/glew-1.10.0/include" CACHE PATH "Path to GLEW")
  set(GLEW_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/glew-1.10.0/include" CACHE PATH "Path to GLEW")
  set(GLEW_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/glew-1.10.0/lib64/glew32.lib" CACHE PATH "Path to GLEW")
  set(GLEW_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/glew-1.10.0/lib64/glew32.lib" CACHE PATH "Path to GLEW")  

  set(GLFW3_FOUND 1)
  set(GLFW3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/GLFW/include" CACHE PATH "Path to GLFW3") 
  set(GLFW3_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/GLFW/lib/glfw3.lib" CACHE PATH "Path to GLFW3") 
  set(GLFW3_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/extlibs/GLFW/include" CACHE PATH "Path to GLFW3")

  include_directories("${CMAKE_SOURCE_DIR}/extlibs/tfinterop/eigen3/include/eigen3")

  set(TensorFlow_FOUND 1)
  set(TensorFlow_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/extlibs/tfinterop/tensorflow/include/" CACHE PATH "Path to tf") 
  set(TensorFlow_LIBRARIES "${CMAKE_SOURCE_DIR}/extlibs/tfinterop/tensorflow/lib/tensorflow.lib" CACHE PATH "Path to tf") 

  set(Protobuf_FOUND 1)
  set(Protobuf_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/extlibs/tfinterop/protobuf/include/" CACHE PATH "Path to ptbf") 
  set(Protobuf_LIBRARIES "${CMAKE_SOURCE_DIR}/extlibs/tfinterop/protobuf/lib/libprotobuf.lib" CACHE PATH "Path to ptbf") 
ELSE()
  find_package(TensorFlow REQUIRED)
  find_package(Eigen REQUIRED)
  find_package(Protobuf REQUIRED)
ENDIF()

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW3 REQUIRED)

#set(CUDA_SEPARABLE_COMPILATION ON)

IF (WIN32)
  #Need to pass the C stdlib linking mode (dynamic (MD) instead of static (MT)).
  set(CUDA_PROPAGATE_HOST_FLAGS ON)
ELSE()
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
ENDIF()

CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_ARCH_FLAGS Auto)
set(CUDA_NVCC_FLAGS
  "${CUDA_NVCC_FLAGS}"
  -std=c++11
  --expt-relaxed-constexpr
  "-Wno-deprecated-gpu-targets"
  ${CUDA_ARCH_FLAGS})

message(STATUS "Enabling CUDA support (version: ${CUDA_VERSION_STRING},"
               " archs: ${CUDA_ARCH_FLAGS_readable})")

IF (WIN32)
# The Visual Studio 2017 toolchain is not compatible with CUDA 9.0.
# So install the "VS2015 toolchain for VS2017" package from the VS Installer, and point to it.
  set(CUDA_HOST_COMPILER "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin/amd64/cl.exe" CACHE PATH "Compiler path" FORCE)
ENDIF()


set(PROJECT_INCLUDE_DIRS
    src/
    ${TensorFlow_INCLUDE_DIRS}
    ${Eigen_INCLUDE_DIRS}
    ${Protobuf_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GLFW3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_INCLUDE_DIRS})

file(GLOB CUSTOM_OP_FILES
  src/custom_ops/*.cc
  src/custom_ops/*.cu
  src/custom_ops/*.h
)

cuda_add_library(custom_ops_lib ${CUSTOM_OP_FILES})

set(PROJECT_LIBRARIES
    custom_ops_lib
    ${TensorFlow_LIBRARIES}
    ${Protobuf_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OpenGL_glu_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GLFW3_LIBRARIES}
    opencv_core
    opencv_highgui
    opencv_imgcodecs)

file(GLOB SOURCE_FILES
  src/simple_blend_network.cc
  src/*.h
  src/gl_wrappers/*.cpp
  src/gl_wrappers/*.h
  src/custom_ops/*.h
)

add_executable(simple_blend_network ${SOURCE_FILES})
target_link_libraries(simple_blend_network ${PROJECT_LIBRARIES})
