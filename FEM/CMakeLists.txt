cmake_minimum_required(VERSION 3.10)
project(Composite)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -march=native -ffast-math")

set(CMAKE_BUILD_TYPE Release)

find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, using local version")
    set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../eigen3")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

set(SOURCES src/element.cpp src/material.cpp src/mesh.cpp src/solver.cpp src/main.cpp src/meshReader.cpp src/config.cpp src/tests.cpp)

add_executable(run ${SOURCES})
if(Eigen3_FOUND)
    target_link_libraries(run Eigen3::Eigen)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(run OpenMP::OpenMP_CXX)
    target_compile_definitions(run PRIVATE EIGEN_USE_OPENMP)
endif()

install(TARGETS run DESTINATION bin)
