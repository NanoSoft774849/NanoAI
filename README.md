# ns_ai — `ns_infer` ONNX inference DLL

A small Windows DLL that wraps ONNX Runtime, OpenCV and FFmpeg behind a
couple of lightweight inference base classes. Designed to be consumed by a
larger Qt-based desktop app — drop `ns_infer.dll` next to your binary, link
against `ns_infer.lib`, include the headers under `ns_onnex/include/`, and
derive your models from `ns_ort_engine` (single input) or
`BasicMultiOrtHandler` (multi input).

## What you get

- **`ns_infer.dll`** — the inference runtime. Exports the public C++ surface
  declared in `ns_onnex/include/` via the `NS_INFER_API` macro.
- **`ns_infer.lib`** — the import library. Link against this from your
  consumer.
- **Sibling runtime DLLs** (copied next to `ns_infer.dll` by the post-build
  step): `onnxruntime.dll`, `opencv_world490.dll`, and the FFmpeg set
  (`avcodec-*.dll`, `avformat-*.dll`, `avutil-*.dll`, `swresample-*.dll`,
  `swscale-*.dll`).

All of those land in `F:/CubeBox/ns_ai/Release/` by default.

## Repository layout

```
ns_ai/
├── CMakeLists.txt            # top-level — global_deps, output dirs, subdirs
├── prepare.bat               # cmake configure step
├── build.bat                 # cmake build step (MSBuild / Ninja)
├── .gitignore                # excludes build/, Release/, .claude/, VS noise
└── ns_onnex/
    ├── CMakeLists.txt        # builds ns_infer as a SHARED library
    ├── include/
    │   ├── ns_infer_export.h # NS_INFER_API dllimport/dllexport switch
    │   ├── ns_headers.h      # stdlib + OpenCV umbrella
    │   ├── types.h           # lite::types (Boxf, Landmarks, EulerAngles, …)
    │   ├── utils.h           # lite::utils (drawing, NMS, matting, math)
    │   ├── constants.h       # project-wide constants
    │   └── OrtEngine.h       # ns_ort_engine + BasicMultiOrtHandler bases
    └── src/
        ├── utils.cpp
        ├── types.cpp
        └── OrtEngine.cpp     # session creation, I/O metadata caching
```

## Build

Two scripts wrap the standard CMake workflow:

```bash
./prepare.bat     # cmake . -B build
./build.bat       # cmake --build build  (uses Ninja if present, else MSBuild)
```

`build.bat` already sets `/O2 /Ob2 /Oi /Ot /GL` and `/LTCG` for Release, so
it will only stay fast on a machine with multiple cores — the
`NUMBER_OF_PROCESSORS` env var is honored.

### Prerequisites

| Component       | Where it lives on this machine                  |
|-----------------|------------------------------------------------|
| Visual Studio   | `F:/Vs2022/VC/Tools/MSVC/14.43.34808/...` (2022) |
| CMake ≥ 3.20    | `U:/cmake/bin/cmake.exe`                        |
| Qt 6            | `F:/Qt6/6.11.1/msvc2022_64`                     |
| ONNX Runtime    | `U:/nano_local_ai/onnxruntime`                  |
| OpenCV 4.9.0    | `U:/nano_local_ai/opencv/build`                 |
| FFmpeg          | `U:/nano_local_ai/ffmpeg`                       |
| (optional)      | `U:/nano_local_ai/libtorch/libtorch/` — declared but not yet linked |

If any of those paths differ on your box, override the cache var on the
command line:

```bash
cmake . -B build -DOpenCV_ROOT=/path/to/opencv -DFFMPEG_ROOT=/path/to/ffmpeg
```

…or edit the defaults in [CMakeLists.txt](CMakeLists.txt).

## Consuming the DLL

```cpp
#include "OrtEngine.h"   // pulls NS_INFER_API + the public base classes

class MyDetector : public ns_ort_engine {
public:
    explicit MyDetector(const std::string& model_path)
        : ns_ort_engine(model_path, /*num_threads=*/4) {}

protected:
    Ort::Value transform(const cv::Mat& mat) override {
        // your preprocessing here — return a tensor
    }
};
```

Then in your build system:

```cmake
target_link_libraries(myapp PRIVATE
    F:/CubeBox/ns_ai/Release/ns_infer.lib
)
target_include_directories(myapp PRIVATE
    F:/CubeBox/ns_ai/ns_onnex/include
)
```

…and drop every DLL in `F:/CubeBox/ns_ai/Release/` next to your `.exe` at
deploy time.

## Status

- Builds clean: 0 errors, 0 warnings on MSVC 2022.
- Public surface: `ns_ort_engine`, `BasicMultiOrtHandler`, the `lite::types`
  types and `lite::utils` free functions are tagged with `NS_INFER_API`.
- `prepare.bat` line:
  `-DCMAKE_PREFIX_PATH=F:/Qt6/6.11.1/msvc2022_64` is commented out — Qt is
  picked up automatically through `find_package(Qt6 …)` because its `bin`
  directory is on `PATH`.

## Where things go next

- Pull out a separate `app_deps` interface so the inference DLL stops
  inheriting `Qt6::Gui`, `Qt6::Widgets`, `Qt6::Sql`, `Qt6::Multimedia`,
  `Qt6::Network` — see the `#1` issue in the CMakeLists.txt review.
- Re-enable `add_subdirectory(inference)` and `add_subdirectory(app)` once
  those trees exist.
