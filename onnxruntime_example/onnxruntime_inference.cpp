#include "onnxruntime_inference.h"



void OnnxInferenceBase::SetSessionOptions(bool useCUDA) {
	sessionOptions.SetInterOpNumThreads(1);
	sessionOptions.SetIntraOpNumThreads(1);
	// Optimization will take time and memory during startup
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
	// CUDA options. If used.
	if (useCUDA)
	{
		SetUseCuda();
	}
}
void OnnxInferenceBase::SetUseCuda() {
	cuda_options.device_id = 0;  //GPU_ID
	cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
	cuda_options.arena_extend_strategy = 0;
	// May cause data race in some condition
	cuda_options.do_copy_in_default_stream = 0;
	sessionOptions.AppendExecutionProvider_CUDA(cuda_options); // Add CUDA options to session options
}

bool OnnxInferenceBase::LoadWeights(OnnxENV* Env, const wchar_t* ModelPath) {
	try {
		// Model path is const wchar_t*
		session = Ort::Session(Env->env, ModelPath, sessionOptions);
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ", Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}
	try {	// For allocating memory for input tensors
		memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ", Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}
	return true;
}

void OnnxInferenceBase::SetInputNodeNames(std::vector<const char*>* names) {
	input_node_names = names;
}

void OnnxInferenceBase::SetOutputNodeNames(std::vector<const char*>* names) {
	output_node_names = names;
}

void OnnxInferenceBase::SetInputDemensions(std::vector<int64_t> Dims) {
	input_node_dims = Dims;
}

bool YOLOv7::PreProcess(cv::Mat frame, std::vector<Ort::Value>& inputTensor) {
	// this will make the input into 1,3,640,640
	float dw = (640-frame.cols)/2.0f, dh = (640 - frame.rows) / 2.0f;
	cv::copyMakeBorder(frame, frame, roundf(dh - 0.1), roundf(dh + 0.1), 0, 0, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));  //# add border
	blob = cv::dnn::blobFromImage(frame, 1 / 255.0, cv::Size(640, 640), (0, 0, 0), true, false);
	size_t input_tensor_size = blob.total();
	try {
		inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, input_tensor_size, input_node_dims.data(), input_node_dims.size()));
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
		return false;
	}
	return true;
}

int YOLOv7::Inference(cv::Mat frame, std::vector<Ort::Value>& OutputTensor) {
	std::vector<Ort::Value>InputTensor;
	bool error = PreProcess(frame, InputTensor);
	if (!error) return NULL;
	try {
		OutputTensor = session.Run(Ort::RunOptions{ nullptr }, input_node_names->data(), InputTensor.data(), InputTensor.size(), output_node_names->data(), 1);
	}
	catch (Ort::Exception oe) {
		std::cout << "ONNX exception caught: " << oe.what() << ". Code: " << oe.GetOrtErrorCode() << ".\n";
		return -1;
	}
	return OutputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount();	// Number of elements in output. Num_of-detected * 7
}