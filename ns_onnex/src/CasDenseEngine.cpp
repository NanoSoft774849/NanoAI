#include "CasDenseEngine.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <utility>

CasDenseEngine::CasDenseEngine(
    const std::filesystem::path& modelPath,
    CasDetectorOptions options
)
    // 0 → "let ORT pick" (uses hardware_concurrency). Negative values
    // (which would be a user error) are clamped to 0 so they degrade
    // gracefully to auto-pick rather than silently pinning to 1 thread.
    : ns_ort_engine(
          modelPath.string(),
          static_cast<unsigned int>(std::max(0, options.intraOpThreads))
      )
{
    if (!std::filesystem::is_regular_file(modelPath)) {
        throw std::runtime_error(
            "Dense ONNX model not found: " + modelPath.string()
        );
    }

    const auto& shape = input_shapes().front();
    if (shape.size() != 4 ||
        shape[0] != 1 ||
        shape[1] != 3 ||
        shape[2] <= 0 ||
        shape[3] <= 0) {
        throw std::runtime_error(
            "cas_dense.onnx must have input shape [1, 3, H, W]."
        );
    }

    inputHeight_  = static_cast<int>(shape[2]);
    inputWidth_   = static_cast<int>(shape[3]);
    heatmapHeight_ = inputHeight_  / 4;
    heatmapWidth_  = inputWidth_   / 4;

    if (heatmapHeight_ <= 0 || heatmapWidth_ <= 0) {
        throw std::runtime_error(
            "cas_dense.onnx heatmap dims must be positive (input / 4)."
        );
    }
}

std::vector<float> CasDenseEngine::preprocess(const cv::Mat& bgrImage) const
{
    if (bgrImage.empty()) {
        throw std::invalid_argument("Input image is empty.");
    }
    if (bgrImage.channels() != 3) {
        throw std::invalid_argument("Input image must contain three BGR channels.");
    }

    cv::Mat resized;
    cv::resize(
        bgrImage,
        resized,
        cv::Size(inputWidth_, inputHeight_),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(rgb, channels);

    constexpr std::array<float, 3> mean = {
        0.43031373f,
        0.40718431f,
        0.38698431f
    };
    constexpr std::array<float, 3> standardDeviation = {
        0.08735294f,
        0.08676078f,
        0.09109412f
    };

    const std::size_t planeSize =
        static_cast<std::size_t>(inputWidth_) * inputHeight_;
    std::vector<float> tensor(3U * planeSize);

    for (int channel = 0; channel < 3; ++channel) {
        channels[channel] =
            (channels[channel] - mean[channel]) / standardDeviation[channel];

        if (!channels[channel].isContinuous()) {
            channels[channel] = channels[channel].clone();
        }

        std::memcpy(
            tensor.data() + static_cast<std::size_t>(channel) * planeSize,
            channels[channel].ptr<float>(),
            planeSize * sizeof(float)
        );
    }

    return tensor;
}

Ort::Value CasDenseEngine::transform(const cv::Mat& bgrImage)
{
    // preprocess() returns a fresh vector each call; assign it into the
    // engine-owned buffer so the Ort::Value's data pointer stays valid
    // after transform() returns. The next run() / warmUp() overwrites it.
    preprocessBuffer_ = preprocess(bgrImage);
    return create_tensor<float>(preprocessBuffer_, input_shapes().front());
}

CasDenseEngine::Output CasDenseEngine::run(const cv::Mat& bgrImage)
{
    // Time preprocess (resize + colorspace + normalize + planarity + memcpy)
    // and inference (session().Run) independently so the caller can pinpoint
    // where the budget goes. Use steady_clock to match the rest of the
    // detector timings.
    using Clock = std::chrono::steady_clock;

    const auto preprocessStart = Clock::now();
    Ort::Value inputTensor = transform(bgrImage);
    const auto preprocessEnd = Clock::now();

    const auto inferenceStart = Clock::now();
    const auto& inputNames  = input_names_raw();
    const auto& outputNames = output_names_raw();

    std::vector<Ort::Value> outputs = session().Run(
        Ort::RunOptions{nullptr},
        inputNames.data(),
        &inputTensor,
        inputNames.size(),
        outputNames.data(),
        outputNames.size()
    );

    const auto inferenceEnd = Clock::now();

    if (outputs.size() != 6) {
        throw std::runtime_error(
            "cas_dense.onnx returned an unexpected output count: " +
            std::to_string(outputs.size())
        );
    }

    Output result;
    result.features         = std::move(outputs[0]);
    result.junctionMap      = std::move(outputs[1]);
    result.junctionOffsets  = std::move(outputs[2]);
    result.centerMap        = std::move(outputs[3]);
    result.centerOffsets    = std::move(outputs[4]);
    result.lineVectors      = std::move(outputs[5]);

    result.preprocessMilliseconds =
        std::chrono::duration<double, std::milli>(
            preprocessEnd - preprocessStart
        ).count();
    result.inferenceMilliseconds =
        std::chrono::duration<double, std::milli>(
            inferenceEnd - inferenceStart
        ).count();
    return result;
}

void CasDenseEngine::warmUp()
{
    // A single zero-image run forces ORT to compile fused kernels once.
    cv::Mat zeroImage(inputHeight_, inputWidth_, CV_8UC3, cv::Scalar(0, 0, 0));
    (void)run(zeroImage);
}