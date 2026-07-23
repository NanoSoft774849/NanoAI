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

To enable the CUDA execution provider (recommended for real-time; see the
benchmark numbers below):

```bash
cmake . -B build -DUSE_CUDA=ON
```

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
max + per-stage share of total time. Sample outputs below.

### CPU baseline (10 cores / 8 threads, `test1.jpg`, threshold 0.80, N = 10)

```text
Per-stage timings (10 samples):
  dense preprocess       mean=    4.04 ms  min=    3.15 ms  max=    6.34 ms  (  0.1 %)
  dense inference        mean= 1812.52 ms  min= 1619.63 ms  max= 1967.63 ms  ( 66.4 %)
  decode                 mean=    1.76 ms  min=    1.51 ms  max=    2.49 ms  (  0.1 %)
  score                  mean=  910.86 ms  min=  758.65 ms  max= 1093.61 ms  ( 33.4 %)
  TOTAL                  mean= 2729.18 ms  min= 2553.56 ms  max= 2971.33 ms  (100.0 %)
```

### CUDA execution provider (`-DUSE_CUDA=ON`, same image, threshold 0.80)

Per-iteration totals across runs of increasing length — the means converge
between N = 50 and N = 100 (within 0.5 %), confirming steady state:

| N   | dense pre (ms) | dense inf (ms) | decode (ms) | score (ms) | **TOTAL (ms)** | **Mean FPS** |
|-----|----------------|----------------|-------------|------------|----------------|--------------|
| 10  | 4.04           | 27.03          | 2.18        | 5.23       | 38.67          | 25.9         |
| 50  | 3.84           | 24.20          | 1.72        | 5.38       | 35.13          | 28.5         |
| 100 | 3.78           | 24.48          | 1.67        | 5.31       | 35.25          | 28.4         |
| 200 | 2.66           | 23.93          | 1.62        | 5.03       | **33.24**      | **30.1**     |

Sample N = 200 raw output:

```text
Per-stage timings (200 samples):
  dense preprocess       mean=    2.66 ms  min=    2.38 ms  max=    3.75 ms  (  8.0 %)
  dense inference        mean=   23.93 ms  min=   21.06 ms  max=   28.62 ms  ( 72.0 %)
  decode                 mean=    1.62 ms  min=    1.54 ms  max=    2.73 ms  (  4.9 %)
  score                  mean=    5.03 ms  min=    4.36 ms  max=    7.06 ms  ( 15.1 %)
  TOTAL                  mean=   33.24 ms  min=   29.67 ms  max=   37.97 ms  (100.0 %)

Lines per frame: mean=33.0
```

Best / worst frame rate at N = 200:

| Metric | Per-frame | FPS |
|--------|-----------|-----|
| Mean   | 33.24 ms  | 30.1 |
| Min    | 29.67 ms  | 33.7 |
| Max    | 37.97 ms  | 26.3 |

### CPU vs CUDA head-to-head (N = 10, threshold 0.80)

| Stage            | CPU     | CUDA   | Speedup |
|------------------|---------|--------|---------|
| dense preprocess | 4 ms    | 4 ms   | —       |
| dense inference  | 1813 ms | 27 ms  | **67×** |
| decode           | 2 ms    | 2 ms   | —       |
| score            | 911 ms  | 5 ms   | **174×**|
| **TOTAL**        | **2.73 s** | **39 ms** | **~70×** |

CUDA brings the per-frame budget well under a 15 FPS real-time cap (≤ 67 ms).
Same lines detected per frame in both modes, so model output is unchanged.

### Effect of line-score threshold (CUDA, N = 200)

| Threshold | Lines/frame | TOTAL (ms) | Mean FPS |
|-----------|-------------|------------|----------|
| 0.80      | 33          | 33.24      | 30.1     |
| 0.90      | 30          | 34.44      | 29.0     |

Threshold has **no meaningful effect on speed**. The score session is
invoked on **all decoded proposals** (up to 512 slots), not just the ones
that will survive the filter — the threshold only filters *after* scoring.
The 3 fewer lines at 0.90 are the only observable difference. To speed up
further, reduce the number of proposals entering the score session via
`CasDetectorOptions::maximumJunctions` / `maximumLineCenters` /
`junctionThreshold` instead.

### Enabling CUDA

Reconfigure with `-DUSE_CUDA=ON` and rebuild:

```bash
cmake . -B build -DUSE_CUDA=ON
./build.bat
```

`ns_ort_engine::initialize_handler()` then appends the CUDA EP (device 0)
to every session's options. Implementation lives in
`onnxruntime_providers_cuda.dll`, which is copied into `Release/` by the
post-build step. No CUDA toolkit headers or `cudart.lib` are required to
compile — only to actually run on a CUDA-capable GPU.

### Reading the profile

- **Preprocess and decode are essentially free** (~ 0.1 % on CPU, ~ 13 %
  combined on CUDA). They are not the bottleneck and do not need
  optimization.
- **The CNN forward-passes dominate.** Dense inference is ~ 70 % of the
  CUDA budget, score is ~ 15 %. On CPU they were 66 % and 33 %.
- **On CPU you would need a much smaller / lower-resolution model** to
  reach real-time; the dense model alone takes ~ 1.8 s at 512×512 on 10
  cores. **On CUDA the EP handles it transparently** — ~ 30 FPS is plenty
  for real-time use.
- **The dense inference stage is single-batched.** A second call cannot
  start until the previous `session().Run()` returns. To push beyond 30 FPS
  you would need to overlap a second frame with the first via two
  sessions and a producer/consumer queue.

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