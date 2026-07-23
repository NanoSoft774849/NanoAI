

# Sources
set(SOURCES
    # src/main.cpp
    src/utils.cpp
    src/types.cpp
    src/OrtEngine.cpp
    src/CasDecoding.cpp
    src/CasDenseEngine.cpp
    src/CasScoreEngine.cpp
    src/CasOasisDetector.cpp

)

set(HEADERS

    include/ns_infer_export.h
    include/ns_headers.h
    include/utils.h
    include/types.h
    include/OrtEngine.h
    include/constants.h
    include/CasDetectorOptions.h
    include/CasDecoding.h
    include/CasDenseEngine.h
    include/CasScoreEngine.h
    include/CasOasisDetector.h

)

# ─────────────────────────────────────────────
#  DLL (shared library)
# ─────────────────────────────────────────────
add_library(ns_infer SHARED ${SOURCES} ${HEADERS})

# Mark symbols as dllexport when building this DLL,
# so consumers of the DLL get dllimport via the same headers.
target_compile_definitions(ns_infer PRIVATE NS_INFER_EXPORTS)

# Produce ns_infer.dll alongside the import library ns_infer.lib.
# WINDOWS_EXPORT_ALL_SYMBOLS is intentionally OFF: the DLL uses explicit
# __declspec(dllexport) via NS_INFER_API. Turning the auto-export on triggers
# CMake's __create_def helper, which invokes `link.exe /DUMP` and segfaults
# when Git's link.exe (a symlink utility) precedes MSVC's link.exe on PATH.
# set_target_properties(ns_infer PROPERTIES
#     WINDOWS_EXPORT_ALL_SYMBOLS ON
# )

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
#  CUDA execution provider (optional, OFF by default)
#  When -DUSE_CUDA=ON is passed:
#    * define USE_CUDA so OrtEngine.cpp forwards the session options into
#      OrtSessionOptionsAppendExecutionProvider_CUDA (declared manually
#      because ORT 1.24+ dropped the public header).
#    * link against onnxruntime.lib (already linked) — that .lib provides
#      the import address for OrtSessionOptionsAppendExecutionProvider_CUDA.
#      We do NOT link against the CUDA toolkit itself: the actual EP code
#      lives in onnxruntime_providers_cuda.dll which ORT loads at runtime.
# ─────────────────────────────────────────────
if(USE_CUDA)
    target_compile_definitions(ns_infer PRIVATE USE_CUDA)
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
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
            "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_cuda.dll"
            "${OpenCV_ROOT}/x64/vc16/bin/opencv_world490.dll"
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>
    )
endif()