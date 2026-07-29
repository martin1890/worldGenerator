cmake_minimum_required(VERSION 3.20)

project(world_generator LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    minifb
    GIT_REPOSITORY https://github.com/emoon/minifb.git
    GIT_TAG master
)

FetchContent_MakeAvailable(minifb)

add_executable(world_generator
    main.cpp
    helper.cpp
    region_distribution.cpp
)

target_link_libraries(world_generator PRIVATE minifb)

if(MSVC)
    target_compile_options(world_generator PRIVATE /O2 /arch:AVX2)
else()
    target_compile_options(world_generator PRIVATE -O3 -mavx2)
endif()