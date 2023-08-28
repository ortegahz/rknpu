cmake_minimum_required(VERSION 3.4.1)

project(rknn_yolov5_demo_linux)

set(CMAKE_SYSTEM_NAME Linux)

# location of the target environment
set(CMAKE_FIND_ROOT_PATH /home/manu/nfs/opencv_install_rv1126)

# adjust the default behavior of the FIND_XXX() commands:
# search for headers and libraries in the target environment,
# search for programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -s -O3")

if (CMAKE_C_COMPILER MATCHES "aarch64")
  set(LIB_ARCH lib64)
else()
  set(LIB_ARCH lib)
endif()

# rga
set(RGA_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/rga)
include_directories(${RGA_DIR}/include)

# drm
set(DRM_DIR ${CMAKE_SOURCE_DIR}/../3rdparty/drm)
include_directories(${DRM_DIR}/include)
include_directories(${DRM_DIR}/include/libdrm)

include_directories(${CMAKE_SOURCE_DIR}/include)

# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/../../librknn_api)
include_directories(${RKNN_API_PATH}/include)
set(RKNN_API_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknn_api.so)

#stb
include_directories(${CMAKE_SOURCE_DIR}/../3rdparty/)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(rknn_yolov5_demo
        src/drm_func.c
        src/rga_func.c
        src/postprocess.cc
        src/main.cc
        )

target_link_libraries(rknn_yolov5_demo
	${RKNN_API_LIB}
  ${OpenCV_LIBS}
  dl
)

# install target and libraries
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_yolov5_demo)
install(TARGETS rknn_yolov5_demo DESTINATION ./)
install(DIRECTORY model DESTINATION ./)
install(PROGRAMS ${RKNN_API_LIB} DESTINATION lib)

# SET(CMAKE_BUILD_TYPE "Debug")
