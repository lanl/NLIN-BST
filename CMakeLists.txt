cmake_minimum_required(VERSION 3.16)
project(nlin_wave_bs LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MPI REQUIRED)
option(NLIN_WAVE_OPENMP "Enable OpenMP inside each MPI rank" ON)
if (NLIN_WAVE_OPENMP)
    find_package(OpenMP)
endif()
file(GLOB SRC CONFIGURE_DEPENDS src/*.cpp)
add_executable(nlin_wave_bs ${SRC})
target_include_directories(nlin_wave_bs PRIVATE include)

target_link_libraries(nlin_wave_bs PRIVATE MPI::MPI_CXX)
if (NLIN_WAVE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(nlin_wave_bs PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(nlin_wave_bs PRIVATE NLIN_WAVE_USE_OPENMP=1)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(nlin_wave_bs PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
