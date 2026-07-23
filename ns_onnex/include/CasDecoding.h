// CasDecoding.h
// -----------------------------------------------------------------------------
// Shared post-processing for the CAS-Oasis line detector.
//
// The dense ONNX model emits six tensors: a feature map plus five small
// heatmap / offset maps that describe line junctions and line centers. Those
// tensors are turned into a list of (junction) points and (proposal) line
// segments here, on the CPU.
//
// Both decoders are designed for repeated use inside a real-time loop:
// they take scratch vectors by reference so the caller can keep them alive
// across frames and avoid per-frame heap allocations.
// -----------------------------------------------------------------------------

#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

namespace cas::decoding {

struct Junction {
    float x = 0.0f;
    float y = 0.0f;
    float score = 0.0f;
};

struct Proposal {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
};

// Extract up to `maximumJunctions` local-maxima from `junctionMap`, refining
// their sub-pixel positions via `junctionOffsets`.
//
// `scratch` is reused for the intermediate index list. The caller owns it
// and is expected to `clear()` it (or reuse it) between frames.
std::vector<Junction> decodeJunctions(
    const float* junctionMap,
    const float* junctionOffsets,
    int heatmapHeight,
    int heatmapWidth,
    float threshold,
    int maximumJunctions,
    int nmsKernelSize,
    std::vector<int>& scratch
);

// Convert `centerMap` into line proposals by pairing the nearest junctions
// to each line-center's two endpoints. Duplicate pairs (a, b) ≡ (b, a) are
// removed via `pairScratch`. Up to `proposalSlots` proposals are returned.
//
// `indexScratch` and `pairScratch` are caller-owned scratch buffers used to
// avoid per-frame heap allocations.
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
);

}  // namespace cas::decoding