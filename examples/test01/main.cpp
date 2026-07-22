// examples/test01/main.cpp
// ---------------------------------------------------------------------------
// Smoke test for ns_infer: derive from ns_ort_engine, load an ONNX model,
// and dump its I/O metadata. No inference is performed.
//
// Usage:
//   test01                     # uses ./model.onnx if present, otherwise prints help
//   test01 path/to/foo.onnx    # loads the given model
//
// Exit codes:
//   0  model loaded and metadata printed (or no model supplied — help printed)
//   1  ORT / loading error
//   2  bad CLI (too many args)
// ---------------------------------------------------------------------------

#include "OrtEngine.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Lightweight derived class whose only job is to construct the base and
// expose the cached metadata. transform() is never called in this smoke test.
class ModelPrinter : public ns_ort_engine
{
public:
    explicit ModelPrinter(const std::string& onnx_path)
        : ns_ort_engine(onnx_path, /*num_threads=*/1)
    {}

    void print_summary(const std::string& display_path) const
    {
        std::cout << "Loaded: " << display_path << "\n";

        const auto& in_shapes  = input_shapes();
        const auto& out_shapes = output_shapes();
        const auto& in_names   = input_names_raw();
        const auto& out_names  = output_names_raw();

        std::cout << "\nInputs (" << count_inputs() << "):\n";
        for (std::size_t i = 0; i < count_inputs(); ++i) {
            std::cout << "  [" << i << "] " << in_names[i] << "  shape=[";
            print_dims(std::cout, in_shapes[i]) << "]\n";
        }

        std::cout << "\nOutputs (" << count_outputs() << "):\n";
        for (std::size_t i = 0; i < count_outputs(); ++i) {
            std::cout << "  [" << i << "] " << out_names[i] << "  shape=[";
            print_dims(std::cout, out_shapes[i]) << "]\n";
        }
    }

protected:
    Ort::Value transform(const cv::Mat& /*mat*/) override
    {
        // The smoke test never asks for inference, so this is unreachable.
        throw std::runtime_error(
            "ModelPrinter::transform called — should never happen in smoke test");
    }

private:
    static std::ostream& print_dims(std::ostream& os, const std::vector<int64_t>& dims)
    {
        for (std::size_t k = 0; k < dims.size(); ++k) {
            if (dims[k] < 0) os << "?";
            else             os << dims[k];
            if (k + 1 < dims.size()) os << " x ";
        }
        return os;
    }
};

static void print_help(const char* exe)
{
    std::cout
        << "ns_infer — model-info smoke test\n"
        << "\n"
        << "Usage:\n"
        << "  " << exe << "                       # try ./model.onnx\n"
        << "  " << exe << " path/to/model.onnx    # load a specific file\n"
        << "\n"
        << "If no path is supplied and ./model.onnx is absent, this help is\n"
        << "printed and the test exits 0. The point is to prove that ns_infer.dll\n"
        << "loads cleanly and that the ONNX Runtime session can be created.\n";
}

int main(int argc, char** argv)
{
    if (argc > 2) {
        std::cerr << "too many arguments\n";
        print_help(argv[0]);
        return 2;
    }

    const std::string path = (argc == 2) ? argv[1] : "model.onnx";

    std::error_code ec;
    if (!fs::exists(path, ec)) {
        std::cout << "(no model at \"" << path << "\" — help follows)\n\n";
        print_help(argv[0]);
        return 0;
    }

    try {
        ModelPrinter printer(path);
        printer.print_summary(path);
        std::cout << "\nOK\n";
    }
    catch (const Ort::Exception& e) {
        std::cerr << "ORT error: " << e.what() << "\n";
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
