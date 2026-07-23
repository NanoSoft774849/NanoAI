#include "CasOasisDetector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>

namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

CasOasisDetector::CasOasisDetector(
    const std::filesystem::path& denseModelPath,
    const std::filesystem::path& scoreModelPath,
    CasDetectorOptions options
)
    : dense_(denseModelPath, options),
      score_(scoreModelPath, options),
      options_(options)
{
    if (options_.nmsKernelSize <= 0 || options_.nmsKernelSize % 2 == 0) {
        throw std::invalid_argument("NMS kernel size must be a positive odd number.");
    }

    if (score_.proposalControlPoints() != 2) {
        throw std::runtime_error(
            "This implementation targets CasLarg straight-line proposals only "
            "and therefore expects two control points."
        );
    }
}

void CasOasisDetector::warmUp()
{
    dense_.warmUp();
    score_.warmUp();
}

CasDetectionResult CasOasisDetector::detect(
    const cv::Mat& bgrImage,
    float lineScoreThreshold,
    std::size_t maximumOutputLines
)
{
    if (lineScoreThreshold < 0.0f || lineScoreThreshold > 1.0f) {
        throw std::invalid_argument("Line score threshold must be in [0, 1].");
    }

    CasDetectionResult result;

    // ---- 1. Dense inference ------------------------------------------------
    const auto denseStart = Clock::now();
    CasDenseEngine::Output denseOut = dense_.run(bgrImage);
    const auto denseEnd = Clock::now();
    result.denseMilliseconds =
        std::chrono::duration<double, std::milli>(denseEnd - denseStart).count();

    // ---- 2. Decode junctions + proposals ----------------------------------
    const int heatmapHeight = dense_.heatmapHeight();
    const int heatmapWidth  = dense_.heatmapWidth();

    const float* junctionMap     = denseOut.junctionMap.GetTensorData<float>();
    const float* junctionOffsets = denseOut.junctionOffsets.GetTensorData<float>();
    const float* centerMap       = denseOut.centerMap.GetTensorData<float>();
    const float* centerOffsets   = denseOut.centerOffsets.GetTensorData<float>();
    const float* lineVectors     = denseOut.lineVectors.GetTensorData<float>();

    const auto decodeStart = Clock::now();

    junctionsScratch_ = cas::decoding::decodeJunctions(
        junctionMap,
        junctionOffsets,
        heatmapHeight,
        heatmapWidth,
        options_.junctionThreshold,
        options_.maximumJunctions,
        options_.nmsKernelSize,
        indexScratch_
    );

    static thread_local std::unordered_set<std::uint64_t> pairScratch;
    proposalsScratch_ = cas::decoding::decodeProposals(
        centerMap,
        centerOffsets,
        lineVectors,
        heatmapHeight,
        heatmapWidth,
        junctionsScratch_,
        options_.maximumLineCenters,
        score_.proposalSlots(),
        indexScratch_,
        pairScratch
    );

    const auto decodeEnd = Clock::now();
    result.decodeMilliseconds =
        std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();

    // ---- 3. Score (skip if no proposals) ----------------------------------
    if (proposalsScratch_.empty()) {
        return result;
    }

    const std::size_t proposalValueCount =
        static_cast<std::size_t>(score_.proposalSlots()) * 4U;

    proposalDataScratch_.assign(proposalValueCount, 0.0f);
    for (std::size_t index = 0; index < proposalsScratch_.size(); ++index) {
        const std::size_t base = index * 4U;
        proposalDataScratch_[base + 0U] = proposalsScratch_[index].x0;
        proposalDataScratch_[base + 1U] = proposalsScratch_[index].y0;
        proposalDataScratch_[base + 2U] = proposalsScratch_[index].x1;
        proposalDataScratch_[base + 3U] = proposalsScratch_[index].y1;
    }

    const auto scoreStart = Clock::now();
    score_.score(
        denseOut.features,
        proposalDataScratch_.data(),
        proposalDataScratch_.size(),
        scoresScratch_
    );
    const auto scoreEnd = Clock::now();
    result.scoreMilliseconds =
        std::chrono::duration<double, std::milli>(scoreEnd - scoreStart).count();

    // ---- 4. Filter + scale lines back to image space ----------------------
    struct RankedLine {
        std::size_t proposalIndex;
        float score;
    };

    std::vector<RankedLine> ranked;
    ranked.reserve(proposalsScratch_.size());
    for (std::size_t index = 0; index < proposalsScratch_.size(); ++index) {
        const float score = scoresScratch_[index];
        if (std::isfinite(score) && score >= lineScoreThreshold) {
            ranked.push_back({index, score});
        }
    }

    std::sort(
        ranked.begin(),
        ranked.end(),
        [](const RankedLine& left, const RankedLine& right) {
            return left.score > right.score;
        }
    );

    if (maximumOutputLines > 0 && ranked.size() > maximumOutputLines) {
        ranked.resize(maximumOutputLines);
    }

    const float scaleX =
        static_cast<float>(bgrImage.cols) / static_cast<float>(heatmapWidth);
    const float scaleY =
        static_cast<float>(bgrImage.rows) / static_cast<float>(heatmapHeight);

    linesScratch_.clear();
    linesScratch_.reserve(ranked.size());
    for (const RankedLine& item : ranked) {
        const auto& proposal = proposalsScratch_[item.proposalIndex];

        CasLine line;
        line.start.x = std::clamp(
            proposal.x0 * scaleX - 0.5f,
            0.0f,
            static_cast<float>(bgrImage.cols - 1)
        );
        line.start.y = std::clamp(
            proposal.y0 * scaleY - 0.5f,
            0.0f,
            static_cast<float>(bgrImage.rows - 1)
        );
        line.end.x = std::clamp(
            proposal.x1 * scaleX - 0.5f,
            0.0f,
            static_cast<float>(bgrImage.cols - 1)
        );
        line.end.y = std::clamp(
            proposal.y1 * scaleY - 0.5f,
            0.0f,
            static_cast<float>(bgrImage.rows - 1)
        );
        line.score = item.score;
        linesScratch_.push_back(line);
    }

    result.lines = linesScratch_;
    return result;
}