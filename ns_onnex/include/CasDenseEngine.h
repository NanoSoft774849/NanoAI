// CasDenseEngine.h
// -----------------------------------------------------------------------------
// Thin wrapper around the cas_dense.onnx model. Owns the dense ONNX Runtime
// session, validates its input shape, and exposes a single `run()` that
// returns the model's six output tensors as an Output struct.
//
// This engine is intentionally independent of the score model and the
// post-processing — it can be used standalone for any task that wants the
// dense features or heatmaps.
// -----------------------------------------------------------------------------

#pragma once

#include "OrtEngine.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <vector>

#include "CasDetectorOptions.h"
#include "ns_infer_export.h"

class NS_INFER_API CasDenseEngine : public ns_ort_engine {
public:
    // Output ordering matches the dense model's export order:
    //   features, jmap, joff, cmap, coff, lvec.
    struct Output {
        Ort::Value features;        // [1, 256, 128, 128]
        Ort::Value junctionMap;     // [1, 1,   128, 128]
        Ort::Value junctionOffsets; // [1, 2,   128, 128]
        Ort::Value centerMap;       // [1, 1,   128, 128]
        Ort::Value centerOffsets;   // [1, 2,   128, 128]
        Ort::Value lineVectors;     // [1, 4,   128, 128]
    };

    CasDenseEngine(
        const std::filesystem::path& modelPath,
        CasDetectorOptions options = {}
    );

    // Preprocess + run. Throws on shape mismatch or ORT errors.
    Output run(const cv::Mat& bgrImage);

    // Run a single zero-image inference to compile ORT kernels. Optional;
    // call once after construction if low first-frame latency matters.
    void warmUp();

    int inputWidth()  const noexcept { return inputWidth_;  }
    int inputHeight() const noexcept { return inputHeight_; }

    // The dense network uses stride 4: heatmap dims = input / 4.
    int heatmapWidth()  const noexcept { return heatmapWidth_;  }
    int heatmapHeight() const noexcept { return heatmapHeight_; }

protected:
    // ns_ort_engine contract: convert one cv::Mat into the dense session's
    // input tensor.
    Ort::Value transform(const cv::Mat& bgrImage) override;

private:
    std::vector<float> preprocess(const cv::Mat& bgrImage) const;

    int inputWidth_   = 0;
    int inputHeight_  = 0;
    int heatmapWidth_  = 0;
    int heatmapHeight_ = 0;
};