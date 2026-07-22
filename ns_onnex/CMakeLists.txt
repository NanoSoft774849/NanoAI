

# Sources
set(SOURCES
    # src/main.cpp
    src/utils.cpp
    src/types.cpp
    src/OrtEngine.cpp

)

set(HEADERS

    include/ns_infer_export.h
    include/ns_headers.h
    include/utils.h
    include/types.h
    include/OrtEngine.h
    include/constants.h
   
)

# ─────────────────────────────────────────────
#  DLL (shared library)
# ─────────────────────────────────────────────
add_library(ns_infer SHARED ${SOURCES} ${HEADERS})

# Mark symbols as dllexport when building this DLL,
# so consumers of the DLL get dllimport via the same headers.
target_compile_definitions(ns_infer PRIVATE NS_INFER_EXPORTS)

# Produce ns_infer.dll alongside the import library ns_infer.lib
set_target_properties(ns_infer PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# ─────────────────────────────────────────────
#  JUST LINK global_deps - GETS EVERYTHING
# ─────────────────────────────────────────────
target_link_libraries(ns_infer PRIVATE
    global_deps  # <-- THIS ONE LINE DOES EVERYTHING
)

# ─────────────────────────────────────────────
#  Inference-specific includes (PUBLIC so consumers see them)
# ─────────────────────────────────────────────
target_include_directories(ns_infer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# ─────────────────────────────────────────────
#  CUDA (if needed)
# ─────────────────────────────────────────────
if(USE_CUDA)
    find_package(CUDA REQUIRED)
    target_compile_definitions(ns_infer PRIVATE USE_CUDA)
    target_link_libraries(ns_infer PRIVATE ${CUDA_LIBRARIES})
endif()

# ─────────────────────────────────────────────
#  MSVC flags
# ─────────────────────────────────────────────
if(MSVC)
    target_compile_options(ns_infer PRIVATE
        /W4
        /MP
        /Zc:__cplusplus
        /permissive-
        $<$<CONFIG:Release>:/O2 /Oi /GL>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
    target_link_options(ns_infer PRIVATE
        $<$<CONFIG:Release>:/LTCG>
    )
endif()

# ─────────────────────────────────────────────
#  Copy DLLs
# ─────────────────────────────────────────────
if(WIN32)
    add_custom_command(TARGET ns_infer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${FFMPEG_ROOT}/bin/avcodec-63.dll"
            "${FFMPEG_ROOT}/bin/avformat-63.dll"
            "${FFMPEG_ROOT}/bin/avutil-61.dll"
            "${FFMPEG_ROOT}/bin/swresample-7.dll"
            "${FFMPEG_ROOT}/bin/swscale-10.dll"
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
            "${OpenCV_ROOT}/x64/vc16/bin/opencv_world490.dll"
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>
    )
endif()