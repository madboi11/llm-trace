cmake_minimum_required(VERSION 3.14)
project(llm-trace CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(GGML_NATIVE        ON CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp)

add_executable(llm-trace-probe probe/probe.cpp)
target_link_libraries(llm-trace-probe PRIVATE llama-common llama)
target_compile_features(llm-trace-probe PRIVATE cxx_std_17)
