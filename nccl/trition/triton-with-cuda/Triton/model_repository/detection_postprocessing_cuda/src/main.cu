#include "postprocess.cuh"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/backend/backend_output_responder.h"
#include "triton/core/tritonbackend.h"

namespace triton {
namespace backend {
namespace detection_postprocessing_cuda {

class ModelState : public BackendModel {
  public:
    static TRITONSERVER_Error *Create(TRITONBACKEND_Model *triton_model, ModelState **model_state);
    virtual ~ModelState() = default;

    std::string const &DetectionsTensorName() const { return detections_name_; }
    std::string const &ShapeTensorName() const { return shape_name_; }
    std::string const &BBoxesTensorName() const { return bboxes_name_; }
    std::string const &ScoresTensorName() const { return scores_name_; }
    std::string const &ClassIdsTensorName() const { return class_ids_name_; }

    std::vector<int64_t> const &DetectionsDims() const { return detections_dims_; }
    std::vector<int64_t> const &ShapeDims() const { return shape_dims_; }

    TRITONSERVER_DataType BBoxesDataType() const { return bboxes_dtype_; }
    TRITONSERVER_DataType ScoresDataType() const { return scores_dtype_; }
    TRITONSERVER_DataType ClassIdsDataType() const { return class_ids_dtype_; }

    TRITONSERVER_Error *ValidateModelConfig();

  private:
    ModelState(TRITONBACKEND_Model *triton_model);

    std::string detections_name_, shape_name_;
    std::vector<int64_t> detections_dims_, shape_dims_;

    std::string bboxes_name_, scores_name_, class_ids_name_;
    TRITONSERVER_DataType bboxes_dtype_, scores_dtype_, class_ids_dtype_;
};

ModelState::ModelState(TRITONBACKEND_Model *triton_model) : BackendModel(triton_model) {
    THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
}

TRITONSERVER_Error *ModelState::Create(TRITONBACKEND_Model *triton_model, ModelState **model_state) {
    try {
        *model_state = new ModelState(triton_model);
    } catch (BackendModelException const &ex) {
        RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                             std::string("Error creating ModelState"));
        RETURN_IF_ERROR(ex.err_);
    }
    return nullptr; // success
}

TRITONSERVER_Error *ModelState::ValidateModelConfig() {
    // If verbose logging is enabled, print the model configuration
    if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
        common::TritonJson::WriteBuffer buffer;
        RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
        LOG_MESSAGE(TRITONSERVER_LOG_VERBOSE,
                    (std::string("Model configuration:\n") + buffer.Contents()).c_str());
    }

    common::TritonJson::Value inputs, outputs;
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));

    RETURN_ERROR_IF_FALSE(inputs.ArraySize() == 2, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("Model configuration must have 2 inputs"));
    RETURN_ERROR_IF_FALSE(outputs.ArraySize() == 3, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("Model configuration must have 3 outputs"));

    common::TritonJson::Value detections, shape, bboxes, scores, class_ids;
    RETURN_IF_ERROR(inputs.IndexAsObject(0, &detections));
    RETURN_IF_ERROR(inputs.IndexAsObject(1, &shape));
    RETURN_IF_ERROR(outputs.IndexAsObject(0, &bboxes));
    RETURN_IF_ERROR(outputs.IndexAsObject(1, &scores));
    RETURN_IF_ERROR(outputs.IndexAsObject(2, &class_ids));

    // Get the input and output names in the model state.
    const char *detections_name, *shape_name, *bboxes_name, *scores_name, *class_ids_name;
    size_t detections_name_len, shape_name_len, bboxes_name_len, scores_name_len, class_ids_name_len;
    RETURN_IF_ERROR(detections.MemberAsString("name", &detections_name, &detections_name_len));
    RETURN_IF_ERROR(shape.MemberAsString("name", &shape_name, &shape_name_len));
    RETURN_IF_ERROR(bboxes.MemberAsString("name", &bboxes_name, &bboxes_name_len));
    RETURN_IF_ERROR(scores.MemberAsString("name", &scores_name, &scores_name_len));
    RETURN_IF_ERROR(class_ids.MemberAsString("name", &class_ids_name, &class_ids_name_len));

    detections_name_ = std::string(detections_name);
    shape_name_ = std::string(shape_name);
    bboxes_name_ = std::string(bboxes_name);
    scores_name_ = std::string(scores_name);
    class_ids_name_ = std::string(class_ids_name);

    // Data type of the input and output tensors
    std::string detections_dtype, shape_dtype, bboxes_dtype, scores_dtype, class_ids_dtype;
    RETURN_IF_ERROR(detections.MemberAsString("data_type", &detections_dtype));
    RETURN_IF_ERROR(shape.MemberAsString("data_type", &shape_dtype));
    RETURN_IF_ERROR(bboxes.MemberAsString("data_type", &bboxes_dtype));
    RETURN_IF_ERROR(scores.MemberAsString("data_type", &scores_dtype));
    RETURN_IF_ERROR(class_ids.MemberAsString("data_type", &class_ids_dtype));

    // Validate that the data types are supported by this backend.
    RETURN_ERROR_IF_FALSE(detections_dtype == "TYPE_FP32", TRITONSERVER_ERROR_UNSUPPORTED,
                          std::string("Only supports TYPE_FP32 for detections, got ") + detections_dtype);
    RETURN_ERROR_IF_FALSE(shape_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_UNSUPPORTED,
                          std::string("Only supports TYPE_UINT32 for shape, got ") + shape_dtype);
    RETURN_ERROR_IF_FALSE(bboxes_dtype == "TYPE_FP32", TRITONSERVER_ERROR_UNSUPPORTED,
                          std::string("Only supports TYPE_FP32 for bboxes, got ") + bboxes_dtype);
    RETURN_ERROR_IF_FALSE(scores_dtype == "TYPE_FP32", TRITONSERVER_ERROR_UNSUPPORTED,
                          std::string("Only supports TYPE_FP32 for scores, got ") + scores_dtype);
    RETURN_ERROR_IF_FALSE(class_ids_dtype == "TYPE_UINT32", TRITONSERVER_ERROR_UNSUPPORTED,
                          std::string("Only supports TYPE_UINT32 for class ids, got ") + class_ids_dtype);

    bboxes_dtype_ = ModelConfigDataTypeToTritonServerDataType(bboxes_dtype);
    scores_dtype_ = ModelConfigDataTypeToTritonServerDataType(scores_dtype);
    class_ids_dtype_ = ModelConfigDataTypeToTritonServerDataType(class_ids_dtype);

    // Reshape is not supported for any of the tensors.
    triton::common::TritonJson::Value reshape;
    RETURN_ERROR_IF_TRUE(detections.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
                         std::string("reshape not supported for input_detections tensor"));
    RETURN_ERROR_IF_TRUE(shape.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
                         std::string("reshape not supported for input_shape tensor"));
    RETURN_ERROR_IF_TRUE(bboxes.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
                         std::string("reshape not supported for output_bboxes tensor"));
    RETURN_ERROR_IF_TRUE(scores.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
                         std::string("reshape not supported for output_scores tensor"));
    RETURN_ERROR_IF_TRUE(class_ids.Find("reshape", &reshape), TRITONSERVER_ERROR_UNSUPPORTED,
                         std::string("reshape not supported for output_class_ids tensor"));

    std::vector<int64_t> detections_dims, shape_dims, bboxes_dims, scores_dims, class_ids_dims;
    RETURN_IF_ERROR(backend::ParseShape(detections, "dims", &detections_dims));
    RETURN_IF_ERROR(backend::ParseShape(shape, "dims", &shape_dims));
    RETURN_IF_ERROR(backend::ParseShape(bboxes, "dims", &bboxes_dims));
    RETURN_IF_ERROR(backend::ParseShape(scores, "dims", &scores_dims));
    RETURN_IF_ERROR(backend::ParseShape(class_ids, "dims", &class_ids_dims));

    RETURN_ERROR_IF_FALSE(
        detections_dims.size() == 3 && detections_dims[0] == 1 && detections_dims[1] == 84 &&
            detections_dims[2] == 8400,
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("Expected detections to have 3 dimensions of shape [1, 84, 8400], got ") +
            backend::ShapeToString(detections_dims));
    RETURN_ERROR_IF_FALSE(shape_dims.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("Expected shape to have 1 dimension of shape [2], got ") +
                              backend::ShapeToString(shape_dims));
    RETURN_ERROR_IF_FALSE(bboxes_dims.size() == 2 && bboxes_dims[1] == 4, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("Expected bboxes to have 2 dimensions with shape [-1, 4], got ") +
                              backend::ShapeToString(bboxes_dims));
    RETURN_ERROR_IF_FALSE(scores_dims.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("Expected scores to have 1 dimension with shape [-1], got ") +
                              backend::ShapeToString(scores_dims));
    RETURN_ERROR_IF_FALSE(class_ids_dims.size() == 1, TRITONSERVER_ERROR_INVALID_ARG,
                          std::string("Expected class_ids to have 1 dimension with shape [-1], got ") +
                              backend::ShapeToString(class_ids_dims));

    detections_dims_ = detections_dims;
    shape_dims_ = shape_dims;

    return nullptr; // success
}

extern "C" {

TRITONSERVER_Error *TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *triton_model) {
    ModelState *model_state;
    RETURN_IF_ERROR(ModelState::Create(triton_model, &model_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(triton_model, reinterpret_cast<void *>(model_state)));
    return nullptr; // success
}
TRITONSERVER_Error *TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *triton_model) {
    void *vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(triton_model, &vstate));
    ModelState *model_state = reinterpret_cast<ModelState *>(vstate);
    delete model_state;
    return nullptr; // success
}
} // extern "C"

//
// ModelInstanceState
//
class ModelInstanceState : public BackendModelInstance {
  public:
    static TRITONSERVER_Error *Create(ModelState *model_state,
                                      TRITONBACKEND_ModelInstance *triton_model_instance,
                                      ModelInstanceState **model_instance_state);
    virtual ~ModelInstanceState() = default;

    ModelState *StateForModel() const { return model_state_; }
    PostProcess<float> *GetPostProcessor() { return post_processor_; }

  private:
    ModelInstanceState(ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance,
                       PostProcess<float> *post_processor)
        : BackendModelInstance(model_state, triton_model_instance), model_state_(model_state),
          post_processor_(post_processor) {}

    ModelState *model_state_;
    PostProcess<float> *post_processor_;
};

TRITONSERVER_Error *ModelInstanceState::Create(ModelState *model_state,
                                               TRITONBACKEND_ModelInstance *triton_model_instance,
                                               ModelInstanceState **model_instance_state) {
    try {
        PostProcess<float> *post_processor =
            new PostProcess<float>(model_state->DetectionsDims()[1] - 4, model_state->DetectionsDims()[2]);
        *model_instance_state = new ModelInstanceState(model_state, triton_model_instance, post_processor);
    } catch (const BackendModelInstanceException &ex) {
        RETURN_ERROR_IF_TRUE(ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
                             std::string("Error creating ModelInstanceState"));
        RETURN_IF_ERROR(ex.err_);
    }
    return nullptr; // success
}

extern "C" {

TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *triton_model_instance) {
    TRITONBACKEND_Model *triton_model;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(triton_model_instance, &triton_model));

    void *vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelState(triton_model, &vstate));
    ModelState *model_state = reinterpret_cast<ModelState *>(vstate);

    ModelInstanceState *model_instance_state;
    RETURN_IF_ERROR(ModelInstanceState::Create(model_state, triton_model_instance, &model_instance_state));
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(triton_model_instance,
                                                        reinterpret_cast<void *>(model_instance_state)));
    return nullptr; // success
}

TRITONSERVER_Error *TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *triton_model_instance) {
    void *vstate;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(triton_model_instance, &vstate));
    ModelInstanceState *model_instance_state = reinterpret_cast<ModelInstanceState *>(vstate);
    delete model_instance_state->GetPostProcessor();
    delete model_instance_state;
    return nullptr; // success
}

TRITONSERVER_Error *TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance *instance,
                                                       TRITONBACKEND_Request **requests,
                                                       uint32_t const request_count) {

    ModelInstanceState *instance_state;
    RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, reinterpret_cast<void **>(&instance_state)));
    ModelState *model_state = instance_state->StateForModel();

    std::vector<TRITONBACKEND_Response *> responses;
    responses.reserve(request_count);
    for (uint32_t r = 0; r < request_count; ++r) {
        TRITONBACKEND_Request *request = requests[r];
        TRITONBACKEND_Response *response;
        RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
        responses.push_back(response);
    }

    // Collect input tensors
    BackendInputCollector collector(requests, request_count, &responses, model_state->TritonMemoryManager(),
                                    false /* pinned_enabled */, nullptr /* stream*/);
    // Set allowed memory types
    std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> allowed_input_types = {
        {TRITONSERVER_MEMORY_GPU, 0}};

    // Process input: detections
    const char *detections_buffer;
    size_t detections_buffer_byte_size;
    TRITONSERVER_MemoryType detections_memory_type;
    int64_t detections_memory_type_id;

    // Get the input detections tensor
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        collector.ProcessTensor(model_state->DetectionsTensorName().c_str(), nullptr /* existing_buffer */,
                                0 /* existing_buffer_byte_size */, allowed_input_types, &detections_buffer,
                                &detections_buffer_byte_size, &detections_memory_type,
                                &detections_memory_type_id));

    // Validate detections input size
    size_t expected_detections_size = model_state->DetectionsDims()[0] * model_state->DetectionsDims()[1] *
                                      model_state->DetectionsDims()[2] * sizeof(float);
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        (detections_buffer_byte_size == expected_detections_size)
            ? nullptr
            : TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                                    (std::string("detections has incorrect size, expected ") +
                                     std::to_string(expected_detections_size) + " bytes, got " +
                                     std::to_string(detections_buffer_byte_size) + " bytes")
                                        .c_str()));

    // Process input: shape
    const char *shape_buffer;
    size_t shape_buffer_byte_size;
    TRITONSERVER_MemoryType shape_memory_type;
    int64_t shape_memory_type_id;

    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        collector.ProcessTensor(model_state->ShapeTensorName().c_str(), nullptr /* existing_buffer */,
                                0 /* existing_buffer_byte_size */, allowed_input_types, &shape_buffer,
                                &shape_buffer_byte_size, &shape_memory_type, &shape_memory_type_id));

    // Finalize the collector
    const bool need_cuda_input_sync = collector.Finalize();
    if (need_cuda_input_sync) {
        // Synchronize CUDA stream for GPU-only operation
        cudaError_t sync_err = cudaStreamSynchronize(nullptr);
        RESPOND_ALL_AND_SET_NULL_IF_ERROR(
            responses, request_count,
            (sync_err == cudaSuccess)
                ? nullptr
                : TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INTERNAL,
                      ("GPU synchronization failed: " + std::string(cudaGetErrorString(sync_err))).c_str()));
    }

    // Ensure inputs are on GPU (this backend only supports GPU computation)
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(
        responses, request_count,
        (detections_memory_type == TRITONSERVER_MEMORY_GPU && shape_memory_type == TRITONSERVER_MEMORY_GPU)
            ? nullptr
            : TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                                    "GPU-only backend: all inputs must be on GPU memory"));

    // Perform the actual detection post-processing
    float iou_threshold = 0.45f;
    float score_threshold = 0.25f;
    int max_num_boxes = 300;
    PostProcess<float> *post_processor = instance_state->GetPostProcessor();
    post_processor->run(reinterpret_cast<float const *>(detections_buffer),
                        reinterpret_cast<int const *>(shape_buffer), iou_threshold, score_threshold,
                        max_num_boxes);

    bool supports_first_dim_batching;
    RESPOND_ALL_AND_SET_NULL_IF_ERROR(responses, request_count,
                                      model_state->SupportsFirstDimBatching(&supports_first_dim_batching));

    BackendOutputResponder responder(requests, request_count, &responses, model_state->TritonMemoryManager(),
                                     supports_first_dim_batching, false /* pinned_enabled */,
                                     nullptr /* stream*/);

    std::vector<int64_t> bboxes_dims{(int64_t)post_processor->num_selected_boxes, 4};
    TRITONSERVER_MemoryType bboxes_memory_type = TRITONSERVER_MEMORY_GPU;
    int64_t bboxes_memory_type_id = detections_memory_type_id;
    responder.ProcessTensor(model_state->BBoxesTensorName().c_str(), model_state->BBoxesDataType(),
                            bboxes_dims, reinterpret_cast<char const *>(post_processor->bboxes),
                            bboxes_memory_type, bboxes_memory_type_id);

    std::vector<int64_t> scores_dims{(int64_t)post_processor->num_selected_boxes};
    TRITONSERVER_MemoryType scores_memory_type = TRITONSERVER_MEMORY_GPU;
    int64_t scores_memory_type_id = detections_memory_type_id;
    responder.ProcessTensor(model_state->ScoresTensorName().c_str(), model_state->ScoresDataType(),
                            scores_dims, reinterpret_cast<char const *>(post_processor->scores),
                            scores_memory_type, scores_memory_type_id);

    std::vector<int64_t> class_ids_dims{(int64_t)post_processor->num_selected_boxes};
    TRITONSERVER_MemoryType class_ids_memory_type = TRITONSERVER_MEMORY_GPU;
    int64_t class_ids_memory_type_id = shape_memory_type_id;
    responder.ProcessTensor(model_state->ClassIdsTensorName().c_str(), model_state->ClassIdsDataType(),
                            class_ids_dims, reinterpret_cast<char const *>(post_processor->class_ids),
                            class_ids_memory_type, class_ids_memory_type_id);

    // Finalize the responder
    const bool need_cuda_output_sync = responder.Finalize();
    if (need_cuda_output_sync) {
        // Synchronize CUDA stream for GPU-only operation
        cudaError_t sync_err = cudaStreamSynchronize(nullptr);
        RESPOND_ALL_AND_SET_NULL_IF_ERROR(
            responses, request_count,
            (sync_err == cudaSuccess) ? nullptr
                                      : TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL,
                                                              ("GPU output synchronization failed: " +
                                                               std::string(cudaGetErrorString(sync_err)))
                                                                  .c_str()));
    }

    for (auto &response : responses) {
        if (response != nullptr) {
            LOG_IF_ERROR(TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
                         "Failed to send response");
        }
    }

    // Report statistics for each request, and then release the request.
    for (uint32_t r = 0; r < request_count; ++r) {
        auto &request = requests[r];

        LOG_IF_ERROR(TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
                     "Failed releasing request");
    }

    return nullptr; // success
}

} // extern "C"

} // namespace detection_postprocessing_cuda
} // namespace backend
} // namespace triton
