// OrtEngine.h
// -----------------------------------------------------------------------------
// Lightweight ONNX Runtime base classes used by every inference module in
// ns_onnex. Two flavours are provided:
//
//   ns_ort_engine         - one cv::Mat input,  N tensor outputs
//   BasicMultiOrtHandler  - N cv::Mat inputs,    N tensor outputs
//
// Both classes:
//   * load a model from disk and hold a single ONNX Runtime session
//   * pre-allocate per-input value buffers and cache the I/O metadata
//     (names, shapes, tensor sizes)
//   * expose a pure-virtual `transform(...)` that derived classes override
//     to do model-specific preprocessing
//
// Derived classes live in their own headers (e.g. face_aligner.h, etc.) and
// implement `transform`. The session management and metadata are reused.
// -----------------------------------------------------------------------------

#ifndef LITE_AI_ORT_CORE_ORT_HANDLER_H
#define LITE_AI_ORT_CORE_ORT_HANDLER_H

#include "ns_infer_export.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "utils.h"

#define LITEORT_CHAR wchar_t

// ---------------------------------------------------------------------------
//  Single-input, multi-output ONNX Runtime base class.
// ---------------------------------------------------------------------------
class NS_INFER_API ns_ort_engine
{
public:
    // Build an Ort tensor backed by an owned vector<T>. Buffer lifetime is
    // tied to `data`; copy the result before `data` goes out of scope.
    template <typename T>
    Ort::Value create_tensor(std::vector<T>& data,
                             const std::vector<int64_t>& shape)
    {
        return Ort::Value::CreateTensor<T>(
            memory_info_handler,
            data.data(), data.size(),
            shape.data(), shape.size());
    }

    // ---- Read-only accessors -------------------------------------------------
    std::size_t count_inputs()  const noexcept { return num_inputs;  }
    std::size_t count_outputs() const noexcept { return num_outputs; }

    const std::vector<std::vector<int64_t>>& input_shapes()  const noexcept { return input_node_dims;  }
    const std::vector<std::vector<int64_t>>& output_shapes() const noexcept { return output_node_dims; }

    const std::vector<const char*>& input_names_raw()  const noexcept { return input_node_names;  }
    const std::vector<const char*>& output_names_raw() const noexcept { return output_node_names; }

    Ort::Session&       session()       noexcept { return *ort_session; }
    const Ort::Session& session() const noexcept { return *ort_session; }

protected:
    // Derived classes provide their own preprocessing.
    virtual Ort::Value transform(const cv::Mat& mat) = 0;

    explicit ns_ort_engine(const std::string& _onnx_path,
                           unsigned int       _num_threads = 1);
    virtual ~ns_ort_engine();

    // Non-copyable, non-movable: Ort::Session and Ort::Env own state that
    // must not be transferred.
    ns_ort_engine(const ns_ort_engine&)            = delete;
    ns_ort_engine(ns_ort_engine&&)                 = delete;
    ns_ort_engine& operator=(const ns_ort_engine&) = delete;
    ns_ort_engine& operator=(ns_ort_engine&&)      = delete;

protected:
    // Handle to the ORT environment created in initialize_handler().
    // Exposed so derived classes that wrap more than one session (e.g. a
    // dense + score pair) can construct their additional sessions against
    // the same environment instead of spinning up a second one.
    Ort::Env                       ort_env;

private:
    // Loads the model, configures session options, and caches I/O metadata.
    void initialize_handler();

    // Verbose dump of I/O metadata to stdout. Kept always available; the
    // implementation gates its own output on NS_INFER_DEBUG_LOG.
    void print_debug_string();

    // --- ONNX Runtime state ---------------------------------------------------
    Ort::Session*                  ort_session = nullptr;  // owned; freed in dtor
    const char*                    input_name  = nullptr;
    Ort::MemoryInfo                memory_info_handler = Ort::MemoryInfo::CreateCpu(
                                                          OrtArenaAllocator, OrtMemTypeDefault);
    unsigned int                   num_threads = 1;

    // --- Cached I/O metadata --------------------------------------------------
    std::size_t                    num_inputs  = 0;
    std::size_t                    num_outputs = 1;

    std::vector<std::vector<int64_t>> input_node_dims;
    std::vector<std::vector<int64_t>> output_node_dims;

    std::vector<std::string>          input_node_names_;   // owns the names
    std::vector<std::string>          output_node_names_;  // owns the names
    std::vector<const char*>          input_node_names;    // view into _names_
    std::vector<const char*>          output_node_names;   // view into _names_

    // --- Per-input pre-allocated buffers (filled by derived transform) -------
    std::vector<std::size_t>          input_tensor_sizes;
    std::vector<std::vector<float>>   input_values_handler;

    // --- Path / log identifiers (string views into ctor arguments) -----------
    const LITEORT_CHAR* onnx_path = nullptr;
    const char*         log_id    = nullptr;
};

// ---------------------------------------------------------------------------
//  Multi-input, multi-output ONNX Runtime base class.
// ---------------------------------------------------------------------------
class NS_INFER_API BasicMultiOrtHandler
{
public:
    template <typename T>
    Ort::Value create_tensor(std::vector<T>& data,
                             const std::vector<int64_t>& shape)
    {
        return Ort::Value::CreateTensor<T>(
            memory_info_handler,
            data.data(), data.size(),
            shape.data(), shape.size());
    }

    // ---- Read-only accessors -------------------------------------------------
    std::size_t count_inputs()  const noexcept { return num_inputs;  }
    std::size_t count_outputs() const noexcept { return num_outputs; }

    const std::vector<std::vector<int64_t>>& input_shapes()  const noexcept { return input_node_dims;  }
    const std::vector<std::vector<int64_t>>& output_shapes() const noexcept { return output_node_dims; }

    const std::vector<const char*>& input_names_raw()  const noexcept { return input_node_names;  }
    const std::vector<const char*>& output_names_raw() const noexcept { return output_node_names; }

    Ort::Session&       session()       noexcept { return *ort_session; }
    const Ort::Session& session() const noexcept { return *ort_session; }

protected:
    virtual std::vector<Ort::Value> transform(const std::vector<cv::Mat>& mats) = 0;

    explicit BasicMultiOrtHandler(const std::string& _onnx_path,
                                  unsigned int       _num_threads = 1);
    virtual ~BasicMultiOrtHandler() = default;

    BasicMultiOrtHandler(const BasicMultiOrtHandler&)            = delete;
    BasicMultiOrtHandler(BasicMultiOrtHandler&&)                 = delete;
    BasicMultiOrtHandler& operator=(const BasicMultiOrtHandler&) = delete;
    BasicMultiOrtHandler& operator=(BasicMultiOrtHandler&&)      = delete;

private:
    void initialize_handler();

    // Verbose dump of I/O metadata to stdout. Implementation gates output on
    // NS_INFER_DEBUG_LOG; declaration is always present so the .cpp can
    // define the body unconditionally.
    void print_debug_string();

    Ort::Env                       ort_env;                 // assigned in initialize_handler
    Ort::Session*                  ort_session = nullptr;
    Ort::MemoryInfo                memory_info_handler = Ort::MemoryInfo::CreateCpu(
                                                          OrtArenaAllocator, OrtMemTypeDefault);
    unsigned int                   num_threads = 1;

    std::size_t                    num_inputs  = 1;
    std::size_t                    num_outputs = 1;

    std::vector<std::vector<int64_t>> input_node_dims;
    std::vector<std::vector<int64_t>> output_node_dims;

    std::vector<std::string>          input_node_names_;
    std::vector<std::string>          output_node_names_;
    std::vector<const char*>          input_node_names;
    std::vector<const char*>          output_node_names;

    // One pre-allocated buffer per input tensor (multi-input variant).
    std::vector<std::vector<float>>   input_values_handlers;

    const LITEORT_CHAR* onnx_path = nullptr;
    const char*         log_id    = nullptr;
};

#endif // LITE_AI_ORT_CORE_ORT_HANDLER_H
