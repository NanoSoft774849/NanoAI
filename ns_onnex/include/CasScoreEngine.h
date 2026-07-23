// CasScoreEngine.h
// -----------------------------------------------------------------------------
// Thin wrapper around the cas_score.onnx model. Owns the score ONNX Runtime
// session, validates its input shapes, and exposes a single `score()` that
// takes the dense model's features tensor + a flat proposal buffer and
// writes per-line scores into a caller-owned vector.
//
// This engine is intentionally independent of the dense model and the
// decoding logic — it can be used with any feature tensor of the right
// shape and any proposal buffer.
// -----------------------------------------------------------------------------

#pragma once

#include "OrtEngine.h"

#include <filesystem>
#include <vector>

#include "CasDetectorOptions.h"
#include "ns_infer_export.h"

class NS_INFER_API CasScoreEngine : public ns_ort_engine {
public:
    CasScoreEngine(
        const std::filesystem::path& modelPath,
        CasDetectorOptions options = {}
    );

    // Run the score model.
    //   features:      [1, 256, H, W] dense feature map (any compatible shape)
    //   proposalData:  flat float buffer of length proposalSlots() * 4
    //                  (x0, y0, x1, y1) per proposal
    //   scoresOut:     caller-owned vector; resized to proposalSlots() and
    //                  filled with one score per proposal
    void score(
        const Ort::Value& features,
        const float* proposalData,
        std::size_t proposalDataCount,
        std::vector<float>& scoresOut
    );

    // Run a single dummy inference to compile ORT kernels. Optional.
    void warmUp();

    int proposalSlots()         const noexcept { return proposalSlots_;         }
    int proposalControlPoints() const noexcept { return proposalControlPoints_; }

protected:
    // ns_ort_engine contract — not used here. The score engine uses
    // session().Run(...) directly because its inputs are tensors, not a
    // single cv::Mat.
    Ort::Value transform(const cv::Mat& bgrImage) override;

private:
    int proposalSlots_         = 0;
    int proposalControlPoints_ = 2;
};