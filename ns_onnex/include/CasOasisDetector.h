#pragma once

#include "CasDecoding.h"
#include "CasDenseEngine.h"
#include "CasScoreEngine.h"

#include <opencv2/core.hpp>

#include <cstddef>
#include <filesystem>
#include <vector>
#include "ns_infer_export.h"

// Re-export so existing consumers of CasOasisDetector.h keep compiling.
#include "CasDetectorOptions.h"

struct CasLine {
    cv::Point2f start;
    cv::Point2f end;
    float score = 0.0f;
};

struct CasDetectionResult {
    std::vector<CasLine> lines;
    // Aggregate timing for the whole dense stage (preprocess + inference).
    // Kept for backward compatibility; equals
    // densePreprocessMilliseconds + denseInferenceMilliseconds.
    double denseMilliseconds           = 0.0;
    // Breakdown of the dense stage so callers can tell where time goes.
    double densePreprocessMilliseconds = 0.0;
    double denseInferenceMilliseconds  = 0.0;
    double decodeMilliseconds          = 0.0;
    double scoreMilliseconds           = 0.0;
};

// CasOasisDetector is a thin façade over two engines:
//   * CasDenseEngine — runs cas_dense.onnx on the input image
//   * CasScoreEngine — scores the decoded line proposals
//
// All heavy work is delegated to the engines; this class owns the decoding
// state and a set of persistent scratch buffers so repeated calls from a
// real-time loop allocate as little as possible.
class NS_INFER_API CasOasisDetector {
public:
    CasOasisDetector(
        const std::filesystem::path& denseModelPath,
        const std::filesystem::path& scoreModelPath,
        CasDetectorOptions options = {}
    );

    CasDetectionResult detect(
        const cv::Mat& bgrImage,
        float lineScoreThreshold = 0.80f,
        std::size_t maximumOutputLines = 160
    );

    int inputWidth()    const noexcept { return dense_.inputWidth();  }
    int inputHeight()   const noexcept { return dense_.inputHeight(); }
    int proposalSlots() const noexcept { return score_.proposalSlots(); }

    // Run one dummy inference on each engine to compile ORT kernels before
    // the first real frame. Useful for stable first-frame latency.
    void warmUp();

private:
    CasDenseEngine    dense_;
    CasScoreEngine    score_;
    CasDetectorOptions options_;

    // Persistent scratch buffers — reused across detect() calls so the
    // hot path allocates nothing.
    std::vector<int>                          indexScratch_;
    std::vector<float>                        proposalDataScratch_;
    std::vector<float>                        scoresScratch_;
    std::vector<CasLine>                      linesScratch_;
    std::vector<cas::decoding::Junction>      junctionsScratch_;
    std::vector<cas::decoding::Proposal>      proposalsScratch_;
};