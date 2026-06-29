cmake_minimum_required(VERSION 3.14)
project(llm-trace CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_BUILD_COMMON ON CACHE BOOL "" FORCE)
set(GGML_NATIVE        ON CACHE BOOL "" FORCE)

set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS    OFF CACHE BOOL "" FORCE)

find_package(Threads REQUIRED)

add_subdirectory(third_party/llama.cpp)
add_subdirectory(third_party/ftxui)

add_executable(llm-trace-probe probe/probe.cpp)
target_link_libraries(llm-trace-probe PRIVATE llama-common llama)
target_compile_features(llm-trace-probe PRIVATE cxx_std_17)

add_executable(llm-trace src/main.cpp)
target_link_libraries(llm-trace PRIVATE llama-common llama component Threads::Threads)
target_compile_features(llm-trace PRIVATE cxx_std_17)
