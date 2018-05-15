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
find_package(TensorFlow REQUIRED)
find_package(Eigen REQUIRED)
find_package(Protobuf REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLFW3 REQUIRED)

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}" -std=c++11 --expt-relaxed-constexpr)

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
  #src/custom_ops/*.cc
  #src/custom_ops/*.cu
  #src/custom_ops/*.h
  src/custom_ops/CopyToTextureOp.cc
  src/custom_ops/CopyToTextureOp.cu
  src/custom_ops/CopyToTextureOp.h
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
    ${OpenCV_LIBRARIES})

file(GLOB SOURCE_FILES
  src/simple_blend_network.cc
  src/*.h
  src/gl_wrappers/*.cpp
  src/gl_wrappers/*.h
  src/custom_ops/*.h
)

add_executable(simple_blend_network ${SOURCE_FILES})
target_link_libraries(simple_blend_network ${PROJECT_LIBRARIES})
