// CasDetectorOptions.h
// -----------------------------------------------------------------------------
// Configuration shared by the CAS-Oasis engines and the detector façade.
// Kept in its own header so CasDenseEngine, CasScoreEngine, and
// CasOasisDetector can all include it without pulling in each other's
// implementation headers.
// -----------------------------------------------------------------------------

#pragma once

struct CasDetectorOptions {
    float junctionThreshold = 0.008f;
    int   maximumJunctions  = 300;
    int   maximumLineCenters = 5000;
    int   nmsKernelSize     = 3;
    int   intraOpThreads    = 0;   // 0 lets ONNX Runtime choose.
};