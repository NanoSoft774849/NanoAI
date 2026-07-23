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
           << " | Dense: " << result.denseMilliseconds << " ms"
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
        << "Examples:\n  "
        << program
        << " models\\cas_dense.onnx models\\cas_score.onnx 0 0.80\n  "
        << program
        << " models\\cas_dense.onnx models\\cas_score.onnx --image test.jpg 0.80\n";
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
