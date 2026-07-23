
#include "OrtEngine.h"
#include "utils.h"

// Note: OrtSessionOptionsAppendExecutionProvider_CUDA is declared in
// onnxruntime_c_api.h itself in ORT 1.24+ (the dedicated
// onnxruntime_providers_cuda.h header was dropped). Implementation lives in
// onnxruntime_providers_cuda.dll which ORT loads at runtime.

ns_ort_engine::ns_ort_engine(
    const std::string& _onnx_path, unsigned int _num_threads) :
    log_id(_onnx_path.data()), num_threads(_num_threads)
{
    std::wstring _w_onnx_path(lite::utils::to_wstring(_onnx_path));
    onnx_path = _w_onnx_path.data();
    initialize_handler();
}


void ns_ort_engine::initialize_handler()
{
    ort_env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, log_id);

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetLogSeverityLevel(4);

#ifdef USE_CUDA
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
#endif

    ort_session = new Ort::Session(ort_env, onnx_path, session_options);

    if (ort_session)
        std::cout << "[OrtEngine] Session created successfully\n";

    Ort::Allocator allocator(*ort_session, memory_info_handler);

    //----------------------------------------
    // Inputs
    //----------------------------------------

    num_inputs = ort_session->GetInputCount();

    input_node_names.clear();
    input_node_names_.clear();
    input_node_dims.clear();
    input_tensor_sizes.clear();
    input_values_handler.clear();

    input_node_names.resize(num_inputs);
    input_node_names_.resize(num_inputs);
    input_node_dims.resize(num_inputs);
    input_tensor_sizes.resize(num_inputs);
    input_values_handler.resize(num_inputs);

    for (size_t i = 0; i < num_inputs; ++i)
    {
        auto name = ort_session->GetInputNameAllocated(i, allocator);

        input_node_names_[i] = name.get();
        input_node_names[i] = input_node_names_[i].c_str();

        auto type_info = ort_session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        input_node_dims[i] = tensor_info.GetShape();

        size_t tensor_size = 1;

        for (auto dim : input_node_dims[i])
        {
            if (dim > 0)
                tensor_size *= static_cast<size_t>(dim);
        }

        input_tensor_sizes[i] = tensor_size;
        input_values_handler[i].resize(tensor_size);
    }

    //----------------------------------------
    // Outputs
    //----------------------------------------

    num_outputs = ort_session->GetOutputCount();

    output_node_names.clear();
    output_node_names_.clear();
    output_node_dims.clear();

    output_node_names.resize(num_outputs);
    output_node_names_.resize(num_outputs);

    for (size_t i = 0; i < num_outputs; ++i)
    {
        auto name = ort_session->GetOutputNameAllocated(i, allocator);

        output_node_names_[i] = name.get();
        output_node_names[i] = output_node_names_[i].c_str();

        auto type_info = ort_session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        output_node_dims.push_back(tensor_info.GetShape());
    }

    print_debug_string();
}

ns_ort_engine::~ns_ort_engine()
{
    if (ort_session)
        delete ort_session;
    ort_session = nullptr;
}


void ns_ort_engine::print_debug_string()
{
    std::wcout << L"NANOAI LogId: " << onnx_path << L"\n";

    std::cout << "=============== Input-Dims ==============\n";

    for (size_t i = 0; i < num_inputs; ++i)
    {
        std::cout << "Input: " << i
            << " Name: " << input_node_names[i] << "\n";

        for (auto dim : input_node_dims[i])
        {
            if (dim == -1)
                std::cout << "    dynamic\n";
            else
                std::cout << "    " << dim << "\n";
        }

        std::cout << "Tensor Size : "
            << input_tensor_sizes[i]
            << "\n\n";
    }

    std::cout << "=============== Output-Dims ==============\n";

    for (size_t i = 0; i < num_outputs; ++i)
    {
        std::cout << "Output: " << i
            << " Name: " << output_node_names[i]
            << "\n";

            for (auto dim : output_node_dims[i])
            {
                if (dim == -1)
                    std::cout << "    dynamic\n";
                else
                    std::cout << "    " << dim << "\n";
            }

            std::cout << "\n";
    }

    std::cout << "========================================\n";
}


