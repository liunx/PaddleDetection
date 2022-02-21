cmake_minimum_required(VERSION 3.0)
project(PaddleObjectDetector CXX C)

SET(PADDLE_DIR "" CACHE PATH "Location of libraries")
SET(PADDLE_LIB_NAME "" CACHE STRING "libpaddle_light_api_shared")

include_directories("${CMAKE_SOURCE_DIR}/")

set(SRCS
    src/object_detector.cc
    src/keypoint_postprocess.cc
    src/preprocess_op.cc
    src/config_parser.cc
    src/utils.cc
    src/picodet_postprocess.cc
    src/keypoint_detector.cc
    src/main.cc)


if (NOT DEFINED PADDLE_DIR OR ${PADDLE_DIR} STREQUAL "")
    message(FATAL_ERROR "please set PADDLE_DIR with -DPADDLE_DIR=/path/paddle_influence_dir")
endif()
message("PADDLE_DIR IS:" ${PADDLE_DIR})

include_directories("${CMAKE_SOURCE_DIR}/")
include_directories("${PADDLE_DIR}/")

link_directories("${PADDLE_DIR}/cxx/lib/")
link_directories("${CMAKE_CURRENT_BINARY_DIR}")

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(JSONCPP jsoncpp)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${JSONCPP_INCLUDE_DIRS})
include_directories("${PADDLE_DIR}/cxx/include")
link_directories("${PADDLE_DIR}/cxx/lib")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -o2 -fopenmp -std=c++11")
set(CMAKE_STATIC_LIBRARY_PREFIX "")


set(DEPS ${PADDLE_DIR}/cxx/lib/${PADDLE_LIB_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX})

message("PADDLE_LIB_NAME:" ${PADDLE_LIB_NAME})

set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread -ljsoncpp")
set(DEPS ${DEPS} ${EXTERNAL_LIB})

set(DEPS ${DEPS} ${OpenCV_LIBS} ${JSONCPP_LIBS})

message("DEPS:" $DEPS)
add_executable(main ${SRCS})
target_link_libraries(main ${DEPS})
