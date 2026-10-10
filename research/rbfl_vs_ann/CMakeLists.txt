cmake_minimum_required(VERSION 3.16)
project(rbfl_vs_ann VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(BFL_SNAPSHOT "" CACHE PATH "Path to an extracted BFL commit 53c2f46")
if(NOT EXISTS "${BFL_SNAPSHOT}/src/routed_binary.cpp" OR
   NOT EXISTS "${BFL_SNAPSHOT}/include/bfl/routed_binary.hpp")
    message(FATAL_ERROR "Set BFL_SNAPSHOT to an extracted BFL commit 53c2f46")
endif()

option(RBFL_NATIVE "Optimize for the current CPU" OFF)

add_library(rbfl_core STATIC src/rbfl.cpp src/greedy.cpp)
target_include_directories(rbfl_core PUBLIC src)

add_executable(rbfl_benchmark
    src/benchmark.cpp src/ann_baseline.cpp
    "${BFL_SNAPSHOT}/src/routed_binary.cpp")
target_include_directories(rbfl_benchmark PRIVATE "${BFL_SNAPSHOT}/include")
target_link_libraries(rbfl_benchmark PRIVATE rbfl_core)

add_executable(rbfl_tests src/test_rbfl.cpp)
target_link_libraries(rbfl_tests PRIVATE rbfl_core)

foreach(target rbfl_core rbfl_benchmark rbfl_tests)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        if(RBFL_NATIVE)
            target_compile_options(${target} PRIVATE -march=native)
        endif()
    endif()
endforeach()

enable_testing()
add_test(NAME rbfl_tests COMMAND rbfl_tests)
