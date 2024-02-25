#pragma once
#include "onnxruntime_cxx_api.h"
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
// This have to be the first thing called
struct OnnxENV {
	Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
};

// Common functions
class OnnxInferenceBase {
public:
	// Settings
	void SetSessionOptions(bool UseCuda);
	void SetUseCuda();
	// Create session
	bool LoadWeights(OnnxENV* Env, const wchar_t* ModelPath);
	void SetInputNodeNames(std::vector<const char*>* input_node_names);
	void SetInputDemensions(std::vector<int64_t> input_node_dims);
	void SetOutputNodeNames(std::vector<const char*>* input_node_names);
protected:
	Ort::Session session = Ort::Session(nullptr);
	Ort::SessionOptions sessionOptions;
	OrtCUDAProviderOptions cuda_options;
	Ort::MemoryInfo memory_info{ nullptr };					// Used to allocate memory for input
	std::vector<const char*>* output_node_names = nullptr;	// output node names
	std::vector<const char*>* input_node_names = nullptr;	// Input node names
	std::vector<int64_t> input_node_dims;					// Input node dimension
	cv::Mat blob;											// Converted input. In this case for the (1,3,640,640)
};

// model specifics
class YOLOv7 :public OnnxInferenceBase {
public:
	bool PreProcess(cv::Mat frame, std::vector<Ort::Value>& inputTensor);
	// Return is number of results;
	int Inference(cv::Mat Frame, std::vector<Ort::Value> &OutputTensor);
};