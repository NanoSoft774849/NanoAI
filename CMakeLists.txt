cmake_minimum_required(VERSION 3.20)
project(NanoAI)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Output
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "F:/CubeBox/ns_ai")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})   # ignored by MSVC (DLLs follow RUNTIME)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})  # put ns_infer.lib next to ns_infer.dll

# Paths to your libraries
set(ONNXRUNTIME_ROOT "U:/nano_local_ai/onnxruntime")
set(LIBTORCH_ROOT "U:/nano_local_ai/libtorch/libtorch/")
set(OpenCV_ROOT "U:/nano_local_ai/opencv/build")
set(FFMPEG_ROOT "U:/nano_local_ai/ffmpeg")

# Find Qt
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Network Sql Concurrent Multimedia)

# ─── GLOBAL TARGET ───
add_library(global_deps INTERFACE)

# Includes
target_include_directories(global_deps INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/ns_onnex
    ${OpenCV_ROOT}/include
    ${FFMPEG_ROOT}/include
    ${ONNXRUNTIME_ROOT}/include
    ${LIBTORCH_ROOT}/include
)

# Libraries
target_link_libraries(global_deps INTERFACE
    Qt6::Core Qt6::Gui Qt6::Widgets Qt6::Network Qt6::Sql Qt6::Concurrent Qt6::Multimedia
    ${OpenCV_ROOT}/x64/vc16/lib/opencv_world490.lib
    ${FFMPEG_ROOT}/lib/avformat.lib
    ${FFMPEG_ROOT}/lib/avcodec.lib
    ${FFMPEG_ROOT}/lib/avutil.lib
    ${FFMPEG_ROOT}/lib/swresample.lib
    ${FFMPEG_ROOT}/lib/swscale.lib
    ${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib
)

# Defines
target_compile_definitions(global_deps INTERFACE
    USE_ONNX_BACKEND
    _CRT_SECURE_NO_WARNINGS
    NOMINMAX
    WIN32_LEAN_AND_MEAN
)

# Subdirectories
add_subdirectory(ns_onnex)
add_subdirectory(examples/test01)
# add_subdirectory(inference)
# add_subdirectory(app)