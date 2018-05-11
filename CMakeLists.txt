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
set(SOURCE_FILES
    src/simple_blend_network.cc
    src/OfflineDataLoader.h)
set(EXECUTABLE simple_blend_network)

# add modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")

# find packages
find_package(TensorFlow REQUIRED)
find_package(Eigen REQUIRED)
find_package(Protobuf REQUIRED)

set(PROJECT_INCLUDE_DIRS
		${TensorFlow_INCLUDE_DIRS}
		${Eigen_INCLUDE_DIRS}
		${Protobuf_INCLUDE_DIRS})
set(PROJECT_LIBRARIES
		${TensorFlow_LIBRARIES}
		${Protobuf_LIBRARIES})

include_directories(${PROJECT_INCLUDE_DIRS})
add_executable(${EXECUTABLE} ${SOURCE_FILES})
target_link_libraries(${EXECUTABLE} ${PROJECT_LIBRARIES})
