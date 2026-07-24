cmake_minimum_required(VERSION 3.20)
project(world_generator LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
    minifb
    GIT_REPOSITORY https://github.com/emoon/minifb.git
    GIT_TAG master
)

FetchContent_MakeAvailable(minifb)

add_executable(world_generator main.cpp)
target_link_libraries(world_generator PRIVATE minifb)