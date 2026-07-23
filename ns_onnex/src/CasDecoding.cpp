#include "CasDecoding.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace cas::decoding {

namespace {

// Run NMS via max-dilate and threshold on the heatmap, then partial-sort
// the surviving indices by score and truncate to `maximumCount`.
std::vector<int> selectTopIndices(
    const float* values,
    int valueCount,
    float threshold,
    int maximumCount,
    std::vector<int>& scratch
)
{
    struct Entry {
        float score;
        int index;
    };

    scratch.clear();
    scratch.reserve(static_cast<std::size_t>(valueCount));
    for (int index = 0; index < valueCount; ++index) {
        const float score = values[index];
        if (std::isfinite(score) && score >= threshold) {
            scratch.push_back(index);
        }
    }

    // Convert scratch into (score, index) entries for sorting.
    std::vector<Entry> entries;
    entries.reserve(scratch.size());
    for (const int index : scratch) {
        entries.push_back({values[index], index});
    }
    scratch.clear();

    const auto descending = [](const Entry& left, const Entry& right) {
        if (left.score == right.score) {
            return left.index < right.index;
        }
        return left.score > right.score;
    };

    if (maximumCount > 0 &&
        entries.size() > static_cast<std::size_t>(maximumCount)) {
        std::partial_sort(
            entries.begin(),
            entries.begin() + maximumCount,
            entries.end(),
            descending
        );
        entries.resize(static_cast<std::size_t>(maximumCount));
    } else {
        std::sort(entries.begin(), entries.end(), descending);
    }

    std::vector<int> indices;
    indices.reserve(entries.size());
    for (const Entry& entry : entries) {
        indices.push_back(entry.index);
    }
    return indices;
}

inline float tensorAt(const float* data, int channel, int index, int planeSize)
{
    return data[channel * planeSize + index];
}

inline float clampCoordinate(float value, int dimension)
{
    const float maximum = static_cast<float>(dimension) - 1.0e-4f;
    return std::clamp(value, 0.0f, maximum);
}

inline std::uint64_t pairKey(int first, int second)
{
    const auto low  = static_cast<std::uint32_t>(std::min(first, second));
    const auto high = static_cast<std::uint32_t>(std::max(first, second));
    return (static_cast<std::uint64_t>(low) << 32U) |
           static_cast<std::uint64_t>(high);
}

}  // namespace

std::vector<Junction> decodeJunctions(
    const float* junctionMap,
    const float* junctionOffsets,
    int heatmapHeight,
    int heatmapWidth,
    float threshold,
    int maximumJunctions,
    int nmsKernelSize,
    std::vector<int>& scratch
)
{
    const int planeSize = heatmapHeight * heatmapWidth;

    // NMS via max-dilate: a pixel is a local max iff it equals the dilated
    // value at that pixel.
    cv::Mat junctionMatrix(
        heatmapHeight,
        heatmapWidth,
        CV_32F,
        const_cast<float*>(junctionMap)
    );
    cv::Mat localMaximum;
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(nmsKernelSize, nmsKernelSize)
    );
    cv::dilate(junctionMatrix, localMaximum, kernel);

    std::vector<float> suppressed(static_cast<std::size_t>(planeSize), 0.0f);
    for (int index = 0; index < planeSize; ++index) {
        const int y = index / heatmapWidth;
        const int x = index % heatmapWidth;
        const float score = junctionMap[index];
        const float neighborhoodMaximum = localMaximum.at<float>(y, x);
        if (score == neighborhoodMaximum) {
            suppressed[static_cast<std::size_t>(index)] = score;
        }
    }

    const std::vector<int> indices = selectTopIndices(
        suppressed.data(),
        planeSize,
        threshold,
        maximumJunctions,
        scratch
    );

    std::vector<Junction> junctions;
    junctions.reserve(indices.size());

    for (const int index : indices) {
        const int gridY = index / heatmapWidth;
        const int gridX = index % heatmapWidth;

        Junction junction;
        junction.x = clampCoordinate(
            static_cast<float>(gridX) +
                tensorAt(junctionOffsets, 0, index, planeSize) + 0.5f,
            heatmapWidth
        );
        junction.y = clampCoordinate(
            static_cast<float>(gridY) +
                tensorAt(junctionOffsets, 1, index, planeSize) + 0.5f,
            heatmapHeight
        );
        junction.score = suppressed[static_cast<std::size_t>(index)];
        junctions.push_back(junction);
    }

    return junctions;
}

std::vector<Proposal> decodeProposals(
    const float* centerMap,
    const float* centerOffsets,
    const float* lineVectors,
    int heatmapHeight,
    int heatmapWidth,
    const std::vector<Junction>& junctions,
    int maximumLineCenters,
    int proposalSlots,
    std::vector<int>& indexScratch,
    std::unordered_set<std::uint64_t>& pairScratch
)
{
    std::vector<Proposal> proposals;
    if (junctions.size() < 2) {
        return proposals;
    }

    const int planeSize = heatmapHeight * heatmapWidth;
    const std::vector<int> centerIndices = selectTopIndices(
        centerMap,
        planeSize,
        0.0f,
        maximumLineCenters,
        indexScratch
    );

    proposals.reserve(
        std::min<std::size_t>(
            centerIndices.size(),
            static_cast<std::size_t>(proposalSlots)
        )
    );

    const auto nearestJunction = [&junctions](float x, float y) {
        int nearestIndex = -1;
        float nearestDistance = std::numeric_limits<float>::max();

        for (std::size_t junctionIndex = 0;
             junctionIndex < junctions.size();
             ++junctionIndex) {
            const float dx = x - junctions[junctionIndex].x;
            const float dy = y - junctions[junctionIndex].y;
            const float distance = dx * dx + dy * dy;
            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestIndex = static_cast<int>(junctionIndex);
            }
        }
        return nearestIndex;
    };

    pairScratch.clear();
    pairScratch.reserve(static_cast<std::size_t>(proposalSlots) * 2U);

    for (const int index : centerIndices) {
        const int gridY = index / heatmapWidth;
        const int gridX = index % heatmapWidth;

        const float centerX =
            static_cast<float>(gridX) +
            tensorAt(centerOffsets, 0, index, planeSize) + 0.5f;
        const float centerY =
            static_cast<float>(gridY) +
            tensorAt(centerOffsets, 1, index, planeSize) + 0.5f;

        const float endpoint0X = clampCoordinate(
            centerX + tensorAt(lineVectors, 0, index, planeSize),
            heatmapWidth
        );
        const float endpoint0Y = clampCoordinate(
            centerY + tensorAt(lineVectors, 1, index, planeSize),
            heatmapHeight
        );
        const float endpoint1X = clampCoordinate(
            centerX + tensorAt(lineVectors, 2, index, planeSize),
            heatmapWidth
        );
        const float endpoint1Y = clampCoordinate(
            centerY + tensorAt(lineVectors, 3, index, planeSize),
            heatmapHeight
        );

        const int junction0 = nearestJunction(endpoint0X, endpoint0Y);
        const int junction1 = nearestJunction(endpoint1X, endpoint1Y);

        if (junction0 < 0 || junction1 < 0 || junction0 == junction1) {
            continue;
        }

        const std::uint64_t key = pairKey(junction0, junction1);
        if (!pairScratch.insert(key).second) {
            continue;
        }

        const Junction& firstJunction =
            junctions[static_cast<std::size_t>(std::min(junction0, junction1))];
        const Junction& secondJunction =
            junctions[static_cast<std::size_t>(std::max(junction0, junction1))];

        Proposal proposal{
            firstJunction.x,
            firstJunction.y,
            secondJunction.x,
            secondJunction.y
        };

        // Match the original orientation convention: lower y first.
        if (proposal.y0 > proposal.y1) {
            std::swap(proposal.x0, proposal.x1);
            std::swap(proposal.y0, proposal.y1);
        }

        proposals.push_back(proposal);
        if (proposals.size() >= static_cast<std::size_t>(proposalSlots)) {
            break;
        }
    }

    return proposals;
}

}  // namespace cas::decoding