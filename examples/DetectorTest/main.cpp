#include "CasOasisDetector.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void drawResult(cv::Mat& image, const CasDetectionResult& result)
{
    for (const CasLine& line : result.lines) {
        const cv::Point start(
            cvRound(line.start.x),
            cvRound(line.start.y)
        );
        const cv::Point end(
            cvRound(line.end.x),
            cvRound(line.end.y)
        );

        cv::line(
            image,
            start,
            end,
            cv::Scalar(0, 165, 255),
            2,
            cv::LINE_AA
        );
        cv::circle(
            image,
            start,
            3,
            cv::Scalar(255, 255, 0),
            cv::FILLED,
            cv::LINE_AA
        );
        cv::circle(
            image,
            end,
            3,
            cv::Scalar(255, 255, 0),
            cv::FILLED,
            cv::LINE_AA
        );
    }

    const double totalMilliseconds =
        result.denseMilliseconds +
        result.decodeMilliseconds +
        result.scoreMilliseconds;

    std::ostringstream status;
    status << std::fixed << std::setprecision(1)
           << "Lines: " << result.lines.size()
           << " | Dense(pre/in): "
           << result.densePreprocessMilliseconds << " / "
           << result.denseInferenceMilliseconds << " ms"
           << " | Decode: " << result.decodeMilliseconds << " ms"
           << " | Score: " << result.scoreMilliseconds << " ms"
           << " | Total: " << totalMilliseconds << " ms";

    cv::rectangle(
        image,
        cv::Rect(0, 0, image.cols, 38),
        cv::Scalar(20, 20, 20),
        cv::FILLED
    );
    cv::putText(
        image,
        status.str(),
        cv::Point(12, 25),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        1,
        cv::LINE_AA
    );
}

void printUsage(const char* program)
{
    std::cout
        << "CAS-Oasis ONNX Runtime demo\n\n"
        << "Live camera:\n  "
        << program
        << " <cas_dense.onnx> <cas_score.onnx> [camera_index] [threshold]\n\n"
        << "Single image:\n  "
        << program
        << " <cas_dense.onnx> <cas_score.onnx> --image <image_path> [threshold]\n\n"
        << "Benchmark (runs N iterations, prints mean/min/max per stage):\n  "
        << program
        << " <cas_dense.onnx> <cas_score.onnx> --bench <image_path> [N] [threshold]\n\n"
        << "Examples:\n  "
        << program
        << " models\\cas_dense.onnx models\\cas_score.onnx 0 0.80\n  "
        << program
        << " models\\cas_dense.onnx models\\cas_score.onnx --image test.jpg 0.80\n  "
        << program
        << " models\\cas_dense.onnx models\\cas_score.onnx --bench test.jpg 50 0.80\n";
}

float parseThreshold(const char* value)
{
    const float threshold = std::stof(value);
    if (threshold < 0.0f || threshold > 1.0f) {
        throw std::invalid_argument("Threshold must be between 0 and 1.");
    }
    return threshold;
}

int runImage(
    CasOasisDetector& detector,
    const std::filesystem::path& imagePath,
    float threshold
)
{
    cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Could not read image: " + imagePath.string());
    }

    CasDetectionResult result = detector.detect(image, threshold, 160);
    drawResult(image, result);

    const std::filesystem::path outputPath =
        imagePath.parent_path() /
        (imagePath.stem().string() + "_cas_oasis.jpg");

    if (!cv::imwrite(outputPath.string(), image)) {
        throw std::runtime_error("Could not save output image.");
    }

    std::cout << "Saved: " << outputPath << '\n';
    cv::imshow("CAS-Oasis Line Detection", image);
    cv::waitKey(0);
    return 0;
}

namespace {

// Stage summary: mean/min/max + per-stage percent of total. Computed from
// a fixed-capacity vector of per-iteration samples so it allocates once.
struct StageStats {
    double sum = 0.0;
    double min = 1e18;
    double max = 0.0;
};

void update(StageStats& stats, double value)
{
    stats.sum += value;
    if (value < stats.min) stats.min = value;
    if (value > stats.max) stats.max = value;
}

void printStatsLine(
    const std::string& label,
    const StageStats& stats,
    std::size_t count,
    double totalMean
)
{
    const double mean = count > 0 ? stats.sum / static_cast<double>(count) : 0.0;
    const double share = totalMean > 0.0 ? 100.0 * mean / totalMean : 0.0;

    std::cout
        << "  " << std::left << std::setw(22) << label
        << std::right << std::fixed << std::setprecision(2)
        << " mean=" << std::setw(8) << mean << " ms"
        << "  min=" << std::setw(8) << stats.min << " ms"
        << "  max=" << std::setw(8) << stats.max << " ms"
        << "  (" << std::setprecision(1) << std::setw(5) << share << " %)\n";
}

}  // namespace

int runBench(
    CasOasisDetector& detector,
    const std::filesystem::path& imagePath,
    int iterations,
    float threshold
)
{
    if (iterations < 1) {
        throw std::invalid_argument("Iteration count must be >= 1.");
    }

    cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Could not read image: " + imagePath.string());
    }

    std::cout
        << "Benchmark: " << imagePath << '\n'
        << "  image:    " << image.cols << 'x' << image.rows << '\n'
        << "  input:    " << detector.inputWidth() << 'x'
        << detector.inputHeight() << '\n'
        << "  iters:    " << iterations << '\n'
        << "  threshold:" << threshold << '\n'
        << "  warming up..." << std::flush;

    // Single warm-up so kernel-compile cost does not skew the first sample.
    (void)detector.detect(image, threshold, 160);

    std::cout << " done\n\n";

    std::vector<CasDetectionResult> samples;
    samples.reserve(static_cast<std::size_t>(iterations));

    for (int i = 0; i < iterations; ++i) {
        samples.push_back(detector.detect(image, threshold, 160));
    }

    StageStats pre, inf, dec, sc, tot;
    std::size_t lineCount = 0;
    for (const CasDetectionResult& r : samples) {
        const double t = r.densePreprocessMilliseconds
                       + r.denseInferenceMilliseconds
                       + r.decodeMilliseconds
                       + r.scoreMilliseconds;
        update(pre, r.densePreprocessMilliseconds);
        update(inf, r.denseInferenceMilliseconds);
        update(dec, r.decodeMilliseconds);
        update(sc,  r.scoreMilliseconds);
        update(tot, t);
        lineCount += r.lines.size();
    }

    const double totalMean =
        samples.empty() ? 0.0 : tot.sum / static_cast<double>(samples.size());

    std::cout << "Per-stage timings (" << samples.size() << " samples):\n";
    printStatsLine("dense preprocess", pre, samples.size(), totalMean);
    printStatsLine("dense inference",  inf, samples.size(), totalMean);
    printStatsLine("decode",           dec, samples.size(), totalMean);
    printStatsLine("score",            sc,  samples.size(), totalMean);
    printStatsLine("TOTAL",            tot, samples.size(), totalMean);

    std::cout
        << "\nLines per frame: mean="
        << (samples.empty() ? 0.0
            : static_cast<double>(lineCount) / static_cast<double>(samples.size()))
        << "\n";

    return 0;
}

int runCamera(
    CasOasisDetector& detector,
    int cameraIndex,
    float threshold
)
{
#ifdef _WIN32
    cv::VideoCapture camera(cameraIndex, cv::CAP_DSHOW);
#else
    cv::VideoCapture camera(cameraIndex);
#endif

    if (!camera.isOpened()) {
        throw std::runtime_error(
            "Could not open camera index " + std::to_string(cameraIndex)
        );
    }

    camera.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    camera.set(cv::CAP_PROP_FPS, 15);
    camera.set(cv::CAP_PROP_BUFFERSIZE, 1);

    std::cout
        << "Camera started. Press Esc or Q to stop.\n"
        << "Model input: " << detector.inputWidth()
        << 'x' << detector.inputHeight() << '\n'
        << "Proposal slots: " << detector.proposalSlots() << '\n';

    cv::namedWindow("CAS-Oasis Line Detection", cv::WINDOW_NORMAL);

    for (;;) {
        cv::Mat frame;
        if (!camera.read(frame) || frame.empty()) {
            std::cerr << "Camera frame read failed.\n";
            continue;
        }

        try {
            CasDetectionResult result = detector.detect(frame, threshold, 160);
            drawResult(frame, result);
        } catch (const std::exception& error) {
            cv::putText(
                frame,
                std::string("Detection error: ") + error.what(),
                cv::Point(12, 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.55,
                cv::Scalar(0, 0, 255),
                2,
                cv::LINE_AA
            );
            std::cerr << "Detection error: " << error.what() << '\n';
        }

        cv::imshow("CAS-Oasis Line Detection", frame);
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q' || key == 'Q') {
            break;
        }
    }

    camera.release();
    cv::destroyAllWindows();
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        if (argc < 3) {
            printUsage(argv[0]);
            return 1;
        }

        const std::filesystem::path denseModelPath = argv[1];
        const std::filesystem::path scoreModelPath = argv[2];

        CasDetectorOptions options;
        options.intraOpThreads = 0;

        CasOasisDetector detector(
            denseModelPath,
            scoreModelPath,
            options
        );

        if (argc >= 5 && std::string(argv[3]) == "--image") {
            const std::filesystem::path imagePath = argv[4];
            const float threshold = argc >= 6 ? parseThreshold(argv[5]) : 0.80f;
            return runImage(detector, imagePath, threshold);
        }

        if (argc >= 5 && std::string(argv[3]) == "--bench") {
            const std::filesystem::path imagePath = argv[4];
            const int    iterations = argc >= 6 ? std::stoi(argv[5]) : 20;
            const float  threshold  = argc >= 7 ? parseThreshold(argv[6]) : 0.80f;
            return runBench(detector, imagePath, iterations, threshold);
        }

        const int cameraIndex = argc >= 4 ? std::stoi(argv[3]) : 0;
        const float threshold = argc >= 5 ? parseThreshold(argv[4]) : 0.80f;
        return runCamera(detector, cameraIndex, threshold);
    }
    catch (const Ort::Exception& error) {
        std::cerr << "ONNX Runtime error: " << error.what() << '\n';
        return 2;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 3;
    }
}
