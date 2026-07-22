# examples/test01/CMakeLists.txt
# ---------------------------------------------------------------------------
# Smoke test for ns_infer.
#
# add_subdirectory(examples/test01) is added in the root CMakeLists.txt so
# this builds automatically as part of `./prepare.bat` + `./build.bat`.
# ---------------------------------------------------------------------------

add_executable(test01 main.cpp)

# ns_infer provides the project headers (OrtEngine.h, etc.) publicly.
# global_deps provides ONNX Runtime + OpenCV + FFmpeg headers (the example
# uses cv::Mat in transform() and Ort::Exception for error handling).
target_link_libraries(test01 PRIVATE ns_infer global_deps)

# Drop test01.exe next to ns_infer.dll + its sibling runtime DLLs so it can
# be launched directly from F:/CubeBox/ns_ai/Release/ without any extra
# PATH setup.
set_target_properties(test01 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/"
)
