cmake_minimum_required(VERSION 3.10)
project(Composite)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

set(CMAKE_BUILD_TYPE Debug)

find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, install using 'sudo apt-get install libeigen3-dev' on linux")
endif()
include_directories(${EIGEN3_INCLUDE_DIR})

set(SOURCES src/Material.cpp src/Mesh.cpp src/Solver.cpp src/main.cpp src/MeshReader.cpp src/Config.cpp src/Tests.cpp)

add_executable(run ${SOURCES})
if(Eigen3_FOUND)
    target_link_libraries(run Eigen3::Eigen)
endif()

set_target_properties(run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

install(TARGETS run DESTINATION bin)
