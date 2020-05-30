cmake_minimum_required(VERSION 3.0)
project(ch8)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 4.0 REQUIRED)
find_package(Eigen3 3.3.0 REQUIRED)


include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
)

add_executable(optical_flow optical_flow.cpp)
target_link_libraries(optical_flow ${OpenCV_LIBS})

add_executable()