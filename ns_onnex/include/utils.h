//
// Created by DefTruth on 2021/10/7.
//

#ifndef NS_AI_UTILS
#define NS_AI_UTILS

#include "ns_infer_export.h"
#include "types.h"
#include <onnxruntime_cxx_api.h>
namespace lite
{
    namespace utils
    {
        // String Utils
         NS_INFER_API std::wstring to_wstring(const std::string& str);
         NS_INFER_API std::string to_string(const std::wstring& wstr);
        // Drawing Utils
         NS_INFER_API cv::Mat draw_axis(const cv::Mat& mat, const types::EulerAngles& euler_angles, float size = 50.f, int thickness = 2);
         NS_INFER_API cv::Mat draw_boxes(const cv::Mat& mat, const std::vector<types::Boxf>& boxes);
         NS_INFER_API cv::Mat draw_landmarks(const cv::Mat& mat, types::Landmarks& landmarks);
         NS_INFER_API cv::Mat draw_age(const cv::Mat& mat, types::Age& age);
         NS_INFER_API cv::Mat draw_gender(const cv::Mat& mat, types::Gender& gender);
         NS_INFER_API cv::Mat draw_emotion(const cv::Mat& mat, types::Emotions& emotions);
         NS_INFER_API cv::Mat draw_boxes_with_landmarks(const cv::Mat& mat, const std::vector<types::BoxfWithLandmarks>& boxes_kps, bool text = false);
         NS_INFER_API void draw_boxes_inplace(cv::Mat& mat_inplace, const std::vector<types::Boxf>& boxes);
         NS_INFER_API void draw_axis_inplace(cv::Mat& mat_inplace, const types::EulerAngles& euler_angles, float size = 50.f, int thickness = 2);
         NS_INFER_API void draw_landmarks_inplace(cv::Mat& mat, types::Landmarks& landmarks);
         NS_INFER_API void draw_age_inplace(cv::Mat& mat_inplace, types::Age& age);
         NS_INFER_API void draw_gender_inplace(cv::Mat& mat_inplace, types::Gender& gender);
         NS_INFER_API void draw_emotion_inplace(cv::Mat& mat_inplace, types::Emotions& emotions);
         NS_INFER_API void draw_boxes_with_landmarks_inplace(cv::Mat& mat_inplace, const std::vector<types::BoxfWithLandmarks>& boxes_kps, bool text = false);
        // Object Detection Utils
         NS_INFER_API void hard_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output, float iou_threshold, unsigned int topk);
         NS_INFER_API void blending_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output, float iou_threshold, unsigned int topk);
         NS_INFER_API void offset_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output, float iou_threshold, unsigned int topk);
        // Matting & Segmentation Utils
         NS_INFER_API void swap_background(const cv::Mat& fgr_mat, const cv::Mat& pha_mat, const cv::Mat& bgr_mat, cv::Mat& out_mat, bool fgr_is_already_mul_pha = false);
         NS_INFER_API void remove_small_connected_area(cv::Mat& alpha_pred, float threshold = 0.05f);

        namespace math
        {
            template<typename T = float>
            std::vector<float> softmax(const std::vector<T>& logits, unsigned int& max_id);
            template  std::vector<float> softmax(const std::vector<float>& logits, unsigned int& max_id);

            template<typename T = float>
            std::vector<float> softmax(const T* logits, unsigned int _size, unsigned int& max_id);
            template  std::vector<float> softmax(const float* logits, unsigned int _size, unsigned int& max_id);

            template<typename T = float>
            std::vector<unsigned int> argsort(const std::vector<T>& arr);
            template  std::vector<unsigned int> argsort(const std::vector<float>& arr);

            template<typename T = float>
            std::vector<unsigned int> argsort(const T* arr, unsigned int _size);
            template  std::vector<unsigned int> argsort(const float* arr, unsigned int _size);

            template<typename T = float>
            float cosine_similarity(const std::vector<T>& a, const std::vector<T>& b);
            template  float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);
        } // NAMESPACE MATH
    } // NAMESPACE UTILS
}

template<typename T> std::vector<float> lite::utils::math::softmax(
    const T* logits, unsigned int _size, unsigned int& max_id)
{
    types::__assert_type<T>();
    if (_size == 0 || logits == nullptr) return {};
    float max_prob = 0.f, total_exp = 0.f;
    std::vector<float> softmax_probs(_size);
    for (unsigned int i = 0; i < _size; ++i)
    {
        softmax_probs[i] = std::exp((float)logits[i]);
        total_exp += softmax_probs[i];
    }
    for (unsigned int i = 0; i < _size; ++i)
    {
        softmax_probs[i] = softmax_probs[i] / total_exp;
        if (softmax_probs[i] > max_prob)
        {
            max_id = i;
            max_prob = softmax_probs[i];
        }
    }
    return softmax_probs;
}

template<typename T> std::vector<float> lite::utils::math::softmax(
    const std::vector<T>& logits, unsigned int& max_id)
{
    types::__assert_type<T>();
    if (logits.empty()) return {};
    const unsigned int _size = logits.size();
    float max_prob = 0.f, total_exp = 0.f;
    std::vector<float> softmax_probs(_size);
    for (unsigned int i = 0; i < _size; ++i)
    {
        softmax_probs[i] = std::exp((float)logits[i]);
        total_exp += softmax_probs[i];
    }
    for (unsigned int i = 0; i < _size; ++i)
    {
        softmax_probs[i] = softmax_probs[i] / total_exp;
        if (softmax_probs[i] > max_prob)
        {
            max_id = i;
            max_prob = softmax_probs[i];
        }
    }
    return softmax_probs;
}

template<typename T> std::vector<unsigned int> lite::utils::math::argsort(
    const std::vector<T>& arr)
{
    types::__assert_type<T>();
    if (arr.empty()) return {};
    const unsigned int _size = arr.size();
    std::vector<unsigned int> indices;
    for (unsigned int i = 0; i < _size; ++i) indices.push_back(i);
    std::sort(indices.begin(), indices.end(),
        [&arr](const unsigned int a, const unsigned int b)
    { return arr[a] > arr[b]; });
    return indices;
}

template<typename T> std::vector<unsigned int> lite::utils::math::argsort(
    const T* arr, unsigned int _size)
{
    types::__assert_type<T>();
    if (_size == 0 || arr == nullptr) return {};
    std::vector<unsigned int> indices;
    for (unsigned int i = 0; i < _size; ++i) indices.push_back(i);
    std::sort(indices.begin(), indices.end(),
        [arr](const unsigned int a, const unsigned int b)
    { return arr[a] > arr[b]; });
    return indices;
}

template<typename T> float lite::utils::math::cosine_similarity(
    const std::vector<T>& a, const std::vector<T>& b)
{
    types::__assert_type<T>();
    float zero_vale = 0.f;
    if (a.empty() || b.empty() || (a.size() != b.size())) return zero_vale;
    const unsigned int _size = a.size();
    float mul_a = zero_vale, mul_b = zero_vale, mul_ab = zero_vale;
    for (unsigned int i = 0; i < _size; ++i)
    {
        mul_a += (float)a[i] * (float)a[i];
        mul_b += (float)b[i] * (float)b[i];
        mul_ab += (float)a[i] * (float)b[i];
    }
    if (mul_a == zero_vale || mul_b == zero_vale) return zero_vale;
    return (mul_ab / (std::sqrt(mul_a) * std::sqrt(mul_b)));
}


// abdulbary 





//*************************************** ortcv::utils **********************************************//

namespace ns_ort_utils
{
    //public:
    enum transform
    {
        CHW = 0, HWC = 1
    };

    static Ort::Value create_tensor(const cv::Mat& mat,
        const std::vector<int64_t>& tensor_dims,
        const Ort::MemoryInfo& memory_info_handler,
        std::vector<float>& tensor_value_handler,
        unsigned int data_format)
    {
        const unsigned int rows = mat.rows;
        const unsigned int cols = mat.cols;
        const unsigned int channels = mat.channels();

        cv::Mat mat_ref;
        if (mat.type() != CV_32FC(channels)) mat.convertTo(mat_ref, CV_32FC(channels));
        else mat_ref = mat; // reference only. zero-time cost. support 1/2/3/... channels

        if (tensor_dims.size() != 4) throw std::runtime_error("dims mismatch.");
        if (tensor_dims.at(0) != 1) throw std::runtime_error("batch != 1");

        // CXHXW
        if (data_format == transform::CHW)
        {

            const unsigned int target_height = tensor_dims.at(2);
            const unsigned int target_width = tensor_dims.at(3);
            const unsigned int target_channel = tensor_dims.at(1);
            const unsigned int target_tensor_size = target_channel * target_height * target_width;
            if (target_channel != channels) throw std::runtime_error("channel mismatch.");

            tensor_value_handler.resize(target_tensor_size);

            cv::Mat resize_mat_ref;
            if (target_height != rows || target_width != cols)
                cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
            else resize_mat_ref = mat_ref; // reference only. zero-time cost.

            std::vector<cv::Mat> mat_channels;
            cv::split(resize_mat_ref, mat_channels);
            // CXHXW
            for (unsigned int i = 0; i < channels; ++i)
                std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
                    mat_channels.at(i).data, target_height * target_width * sizeof(float));

            return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
                target_tensor_size, tensor_dims.data(),
                tensor_dims.size());
        }

        // HXWXC
        const unsigned int target_height = tensor_dims.at(1);
        const unsigned int target_width = tensor_dims.at(2);
        const unsigned int target_channel = tensor_dims.at(3);
        const unsigned int target_tensor_size = target_channel * target_height * target_width;
        if (target_channel != channels) throw std::runtime_error("channel mismatch!");
        tensor_value_handler.resize(target_tensor_size);

        cv::Mat resize_mat_ref;
        if (target_height != rows || target_width != cols)
            cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
        else resize_mat_ref = mat_ref; // reference only. zero-time cost.

        std::memcpy(tensor_value_handler.data(), resize_mat_ref.data, target_tensor_size * sizeof(float));

        return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
            target_tensor_size, tensor_dims.data(),
            tensor_dims.size());
    }

    static cv::Mat normalize(const cv::Mat& mat, float mean, float scale)
    {
        cv::Mat matf;
        if (mat.type() != CV_32FC3) mat.convertTo(matf, CV_32FC3);
        else matf = mat; // reference
        return (matf - mean) * scale;
    }

    static cv::Mat normalize(const cv::Mat& mat, const float* mean, const float* scale)
    {
        cv::Mat mat_copy;
        if (mat.type() != CV_32FC3) mat.convertTo(mat_copy, CV_32FC3);
        else mat_copy = mat.clone();
        for (unsigned int i = 0; i < mat_copy.rows; ++i)
        {
            cv::Vec3f* p = mat_copy.ptr<cv::Vec3f>(i);
            for (unsigned int j = 0; j < mat_copy.cols; ++j)
            {
                p[j][0] = (p[j][0] - mean[0]) * scale[0];
                p[j][1] = (p[j][1] - mean[1]) * scale[1];
                p[j][2] = (p[j][2] - mean[2]) * scale[2];
            }
        }
        return mat_copy;
    }

    static void normalize(const cv::Mat& inmat, cv::Mat& outmat,
        float mean, float scale)
    {
        outmat = normalize(inmat, mean, scale);
    }

    static void normalize_inplace(cv::Mat& mat_inplace, float mean, float scale)
    {
        if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
        normalize(mat_inplace, mat_inplace, mean, scale);
    }

    static void normalize_inplace(cv::Mat& mat_inplace, const float* mean, const float* scale)
    {
        if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
        for (unsigned int i = 0; i < mat_inplace.rows; ++i)
        {
            cv::Vec3f* p = mat_inplace.ptr<cv::Vec3f>(i);
            for (unsigned int j = 0; j < mat_inplace.cols; ++j)
            {
                p[j][0] = (p[j][0] - mean[0]) * scale[0];
                p[j][1] = (p[j][1] - mean[1]) * scale[1];
                p[j][2] = (p[j][2] - mean[2]) * scale[2];
            }
        }
    }
};


#endif //LITE_AI_TOOLKIT_UTILS_H
