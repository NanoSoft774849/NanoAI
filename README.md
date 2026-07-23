# ns_ai — `ns_infer` ONNX inference DLL

A small Windows DLL that wraps ONNX Runtime, OpenCV and FFmpeg behind a
couple of lightweight inference base classes. Designed to be consumed by a
larger Qt-based desktop app — drop `ns_infer.dll` next to your binary, link
against `ns_infer.lib`, include the headers under `ns_onnex/include/`, and
derive your models from `ns_ort_engine` (single input) or
`BasicMultiOrtHandler` (multi input).

The DLL also ships with the **CAS-Oasis line detector** — a two-model
pipeline (`cas_dense.onnx` → `cas_score.onnx`) split into independent
engines and glued together by a thin façade class.

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
├── CMakeLists.txt                  # top-level — global_deps, output dirs, subdirs
├── prepare.bat                     # cmake configure step
├── build.bat                       # cmake build step (MSBuild / Ninja)
├── .gitignore                      # excludes build/, Release/, .claude/, VS noise
├── ns_onnex/                       # the inference DLL
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── ns_infer_export.h       # NS_INFER_API dllimport/dllexport switch
│   │   ├── ns_headers.h            # stdlib + OpenCV umbrella
│   │   ├── types.h                 # lite::types (Boxf, Landmarks, EulerAngles, …)
│   │   ├── utils.h                 # lite::utils (drawing, NMS, matting, math)
│   │   ├── constants.h             # project-wide constants
│   │   ├── OrtEngine.h             # ns_ort_engine + BasicMultiOrtHandler bases
│   │   ├── CasDetectorOptions.h    # CasDetectorOptions POD (shared by the engines)
│   │   ├── CasDecoding.h           # cas::decoding::Junction / Proposal + decoders
│   │   ├── CasDenseEngine.h        # cas_dense.onnx wrapper (public)
│   │   ├── CasScoreEngine.h        # cas_score.onnx wrapper (public)
│   │   └── CasOasisDetector.h      # façade composing both engines
│   └── src/
│       ├── utils.cpp
│       ├── types.cpp
│       ├── OrtEngine.cpp           # session creation, I/O metadata caching
│       ├── CasDecoding.cpp         # NMS, junction/proposal extraction
│       ├── CasDenseEngine.cpp
│       ├── CasScoreEngine.cpp
│       └── CasOasisDetector.cpp    # pipeline + persistent scratch buffers
└── examples/
    ├── test01/                     # smoke test: print model I/O metadata
    │   ├── CMakeLists.txt
    │   └── main.cpp
    └── DetectorTest/               # live camera + single-image line detector
        ├── CMakeLists.txt
        └── main.cpp
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

### Roll your own single-input model

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

### Use the CAS-Oasis line detector

`CasOasisDetector` is a façade that owns one `CasDenseEngine` (image →
features + heatmaps) and one `CasScoreEngine` (features + proposals →
per-line scores), with shared CPU-side decoding in `cas::decoding`. Each
engine is independently usable; the detector just orchestrates them.

```cpp
#include "CasOasisDetector.h"

CasDetectorOptions options;
options.junctionThreshold = 0.008f;
options.intraOpThreads    = 0;   // 0 = let ONNX Runtime pick

CasOasisDetector detector(
    "models/cas_dense.onnx",
    "models/cas_score.onnx",
    options
);

detector.warmUp();   // optional: prime ORT kernels before the first frame

cv::Mat frame = /* … grab a frame … */;
CasDetectionResult result = detector.detect(frame, /*threshold=*/0.80f);

for (const CasLine& line : result.lines) {
    cv::line(frame, line.start, line.end, cv::Scalar(0, 165, 255), 2);
}
```

Run the bundled demo:

```bash
# Camera
build\Release\Detector.exe models\cas_dense.onnx models\cas_score.onnx 0 0.80

# Single image
build\Release\Detector.exe models\cas_dense.onnx models\cas_score.onnx --image test.jpg 0.80
```

### Linker / include setup

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

## CAS-Oasis pipeline (realtime notes)

The detector is designed for repeated calls from a real-time loop:

- **Persistent scratch buffers** — `CasOasisDetector` keeps `indexScratch_`,
  `proposalDataScratch_`, `scoresScratch_`, `linesScratch_`,
  `junctionsScratch_`, and `proposalsScratch_` as members and reuses them
  across `detect()` calls. The hot path allocates nothing.
- **Skip the score session** — if decoding produces no proposals, the
  score model is not invoked.
- **Live `features` pass-through** — `CasDenseEngine::run()` returns the
  dense output tensors as a struct; `CasScoreEngine::score()` consumes the
  `features` `Ort::Value` without copying.
- **`warmUp()`** — runs a zero-image inference on each engine once so the
  first real frame doesn't pay ORT's kernel-compile tax.

## Profiling / benchmarking

`CasDenseEngine::Output` and `CasDetectionResult` expose a per-stage
breakdown, not just a single dense total:

| Field                          | What it covers                                  |
|--------------------------------|-------------------------------------------------|
| `densePreprocessMilliseconds`  | resize + BGR→RGB + normalize + plane memcpy     |
| `denseInferenceMilliseconds`   | `session().Run(...)` on `cas_dense.onnx`        |
| `decodeMilliseconds`           | `cas::decoding::decodeJunctions/Proposals` NMS  |
| `scoreMilliseconds`            | `session().Run(...)` on `cas_score.onnx`        |
| `denseMilliseconds`            | sum of preprocess + inference (back-compat)     |

The `Detector.exe` example ships with a benchmark mode:

```bash
build\Release\Detector.exe models\cas_dense.onnx models\cas_score.onnx \
    --bench test1.jpg 50 0.80
```

It runs one warm-up iteration, then N iterations, and prints mean / min /
max + per-stage share of total time. Sample output (10 cores / 8 threads,
`test1.jpg`, N = 10):

```text
Per-stage timings (10 samples):
  dense preprocess       mean=    4.04 ms  min=    3.15 ms  max=    6.34 ms  (  0.1 %)
  dense inference        mean= 1812.52 ms  min= 1619.63 ms  max= 1967.63 ms  ( 66.4 %)
  decode                 mean=    1.76 ms  min=    1.51 ms  max=    2.49 ms  (  0.1 %)
  score                  mean=  910.86 ms  min=  758.65 ms  max= 1093.61 ms  ( 33.4 %)
  TOTAL                  mean= 2729.18 ms  min= 2553.56 ms  max= 2971.33 ms  (100.0 %)
```

### Reading the profile

- Preprocess and decode are essentially free (≈ 0.1 % each). They are not
  the bottleneck and do not need optimization.
- The two CNN forward-passes dominate (66 % dense + 33 % score = 99 % of
  total). This is pure ONNX Runtime CPU execution.
- To reach real-time on CPU you would need a much smaller / lower-resolution
  model. The cleanest single-step win is GPU / DirectML execution providers
  — add an EP at session construction time in both engines (out of scope
  for this repo today).

## Status

- Builds clean: **0 errors**. A handful of pre-existing `C4251` (DLL-interface
  on STL members) and `C4267` (signed/unsigned conversion) warnings remain
  in `ns_ort_engine`, `ns_ort_utils`, and `CasOasisDetector` — all harmless
  for in-process use; they only matter if a consumer of the DLL holds
  pointers to STL members across the DLL boundary.
- Public surface: `ns_ort_engine`, `BasicMultiOrtHandler`, the `lite::types`
  types and `lite::utils` free functions are tagged with `NS_INFER_API`.
  `CasDenseEngine`, `CasScoreEngine`, `CasOasisDetector`, `CasLine`, and
  `CasDetectionResult` are exported the same way.
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
- Bind the score session's `proposals` input to a pre-allocated tensor
  via ORT's IO Binding API to skip one allocation per frame.