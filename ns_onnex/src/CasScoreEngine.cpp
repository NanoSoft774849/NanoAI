#include "CasScoreEngine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>

CasScoreEngine::CasScoreEngine(
    const std::filesystem::path& modelPath,
    CasDetectorOptions options
)
    // 0 → "let ORT pick" (uses hardware_concurrency). Negative values are
    // clamped to 0 so they degrade to auto-pick rather than silently
    // pinning to 1 thread.
    : ns_ort_engine(
          modelPath.string(),
          static_cast<unsigned int>(std::max(0, options.intraOpThreads))
      )
{
    if (!std::filesystem::is_regular_file(modelPath)) {
        throw std::runtime_error(
            "Score ONNX model not found: " + modelPath.string()
        );
    }
    if (count_inputs() != 2) {
        throw std::runtime_error(
            "cas_score.onnx must have exactly two inputs (features, proposals)."
        );
    }
    if (count_outputs() != 1) {
        throw std::runtime_error(
            "cas_score.onnx must have exactly one output (scores)."
        );
    }

    const auto& proposalShape = input_shapes()[1];
    if (proposalShape.size() != 4 ||
        proposalShape[0] != 1 ||
        proposalShape[1] <= 0 ||
        proposalShape[2] <= 0 ||
        proposalShape[3] != 2) {
        throw std::runtime_error(
            "cas_score.onnx proposal input must have shape [1, R, P, 2]."
        );
    }

    proposalSlots_         = static_cast<int>(proposalShape[1]);
    proposalControlPoints_ = static_cast<int>(proposalShape[2]);
}

Ort::Value CasScoreEngine::transform(const cv::Mat& /*bgrImage*/)
{
    // The score engine uses session().Run(...) directly — its inputs are
    // tensors, not a single cv::Mat, so the inherited transform() is
    // intentionally unreachable.
    throw std::runtime_error(
        "CasScoreEngine::transform is not used; call score() instead."
    );
}

void CasScoreEngine::score(
    const Ort::Value& features,
    const float* proposalData,
    std::size_t proposalDataCount,
    std::vector<float>& scoresOut
)
{
    if (proposalDataCount != static_cast<std::size_t>(proposalSlots_) * 4U) {
        throw std::invalid_argument(
            "proposalData must contain proposalSlots() * 4 floats."
        );
    }

    // Build input tensors. The features tensor is passed through (we do not
    // copy it); the proposal tensor wraps the caller's flat buffer.
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    std::array<int64_t, 4> proposalShape = {
        1,
        static_cast<int64_t>(proposalSlots_),
        static_cast<int64_t>(proposalControlPoints_),
        2
    };

    Ort::Value featureTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        const_cast<float*>(features.GetTensorData<float>()),
        features.GetTensorTypeAndShapeInfo().GetElementCount(),
        features.GetTensorTypeAndShapeInfo().GetShape().data(),
        features.GetTensorTypeAndShapeInfo().GetShape().size()
    );

    Ort::Value proposalTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        const_cast<float*>(proposalData),
        proposalDataCount,
        proposalShape.data(),
        proposalShape.size()
    );

    std::array<Ort::Value, 2> inputs = {
        std::move(featureTensor),
        std::move(proposalTensor)
    };

    const auto& inputNames  = input_names_raw();
    const auto& outputNames = output_names_raw();

    std::vector<Ort::Value> outputs = session().Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        inputs.data(),
        inputs.size(),
        outputNames.data(),
        outputNames.size()
    );

    if (outputs.size() != 1) {
        throw std::runtime_error(
            "cas_score.onnx returned an unexpected output count."
        );
    }

    const float* scores = outputs[0].GetTensorData<float>();

    scoresOut.assign(scores, scores + proposalSlots_);
}

void CasScoreEngine::warmUp()
{
    // Zero features + zero proposals to compile fused kernels.
    const std::size_t proposalFloats =
        static_cast<std::size_t>(proposalSlots_) * 4U;
    std::vector<float> dummyProposals(proposalFloats, 0.0f);

    // features shape from the model: [1, C, H, W].
    const auto& featShape = input_shapes()[0];
    const std::size_t featFloats = [&]() {
        std::size_t n = 1;
        for (int64_t d : featShape) n *= static_cast<std::size_t>(d);
        return n;
    }();
    std::vector<float> dummyFeatures(featFloats, 0.0f);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    Ort::Value featureTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        dummyFeatures.data(),
        dummyFeatures.size(),
        featShape.data(),
        featShape.size()
    );

    std::array<int64_t, 4> proposalShape = {
        1,
        static_cast<int64_t>(proposalSlots_),
        static_cast<int64_t>(proposalControlPoints_),
        2
    };
    Ort::Value proposalTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        dummyProposals.data(),
        dummyProposals.size(),
        proposalShape.data(),
        proposalShape.size()
    );

    std::array<Ort::Value, 2> inputs = {
        std::move(featureTensor),
        std::move(proposalTensor)
    };

    const auto& inputNames  = input_names_raw();
    const auto& outputNames = output_names_raw();

    std::vector<Ort::Value> outputs = session().Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        inputs.data(),
        inputs.size(),
        outputNames.data(),
        outputNames.size()
    );
    (void)outputs;
}