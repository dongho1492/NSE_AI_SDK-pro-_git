#include "pch.h"
#include "Deep.h"

#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;

namespace NSE
{
	Utils::Utils e_Utils;
	std::mutex NSE::AI::GpuCla::globalInferenceMutex;

	CV::BaseIsOpencv e_Opencv;

	Utils::SingleTimer e_Timer1;
	Utils::SingleTimer e_Timer2;


	namespace AI {
		// ONNX 객체들을 포함하는 구조체 정의 (캡슐화)
		struct NSE::AI::GpuCla::OrtObjects {
			Ort::Env env;
			Ort::SessionOptions session_options;
			OrtCUDAProviderOptions options;
			std::unique_ptr<Ort::Session> session;

			OrtObjects() : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime") {}  // 초기화
		};

		GpuCla::GpuCla()
			: ortObjects(std::make_unique<OrtObjects>())
		{
			e_Utils.logMessagePrint("GpuCla instance created.");

			Utils::ConfigManager::getInstance().loadSettings();

			if (e_configImageSaveMode) {
				Utils::ImageSaver::getInstance().startImageSaveThread();
			}

			if (e_configCSVLogSaveMode) {
				Utils::CSVLogSaver::getInstance().startCSVLogSaveThread();
			}

			const std::string buildInfo = std::string("Build Version: ") + RELEASE_VERSION;
			const std::string buildDate = std::string("Build Date: ") + BUILD_DATE;
			e_Utils.logMessagePrint(buildInfo);
			e_Utils.logMessagePrint(buildDate);

			e_Utils.logMessagePrint("[log_level] (0=OFF, 1=Print, 2=LogSave)  : " + std::to_string(e_configLogLevel));
			e_Utils.logMessagePrint("[execution_environment] ((0=Laptop, 1=Equip)  : " + std::to_string(e_configExeEnv));
			e_Utils.logMessagePrint("[procces_mode] (0=cpu, 1=gpu)  : " + std::to_string(e_configProccesMode));
			e_Utils.logMessagePrint("[image_save_mode] (0=OFF, 1=JPG, 2=PNG, 3=ALL)  : " + std::to_string(e_configImageSaveMode));
			e_Utils.logMessagePrint("[threshold_cla] (0->ProgramBase)  : " + std::to_string(e_configThresholdCla));
			e_Utils.logMessagePrint("[heatmap_cla] (1->ON)  : " + std::to_string(e_configHeatmapCla));
			e_Utils.logMessagePrint("[CSVLog_save_mode] (0->OFF, 1->ON)  : " + std::to_string(e_configCSVLogSaveMode));
			e_Utils.logMessagePrint("[ontimer_mode] (0->OFF, 1->ON)  : " + std::to_string(e_configOntimerMode));

			//Utils::checkProgramAdmin();
			//Utils::runNvidiaSMI();

			// 폴더 생성
			e_Utils.makeFolder(e_pathDefault);
			e_Utils.makeFolder(e_pathLog);
			e_Utils.makeFolder(e_pathLogError);
			e_Utils.makeFolder(e_pathLogRecord);
			e_Utils.makeFolder(e_pathLogUsed);
			e_Utils.makeFolder(e_pathSaveFolder);

			flagInferIndepend = TRUE;
		}


		GpuCla::~GpuCla() {
			ortObjects.reset();

			e_Utils.logMessagePrint("GpuCla instance destroyed.");

			if (e_configImageSaveMode)
				Utils::ImageSaver::getInstance().stopImageSaveThread();

			if (e_configCSVLogSaveMode)
				Utils::CSVLogSaver::getInstance().stopCSVLogSaveThread();
		}

		int GpuCla::initialAIModel()
		{
			try {
				e_Utils.logMessagePrint("Initializing model...");

				std::string model_path = "D:\\Intellisense_AI\\Model\\Model_main.onnx";

				if (!e_Utils.isFileExists(model_path)) {
					e_Utils.logMessagePrint("[ERROR] Model file not found: " + model_path);
				}

				std::wstring Wmodel_path = std::wstring(model_path.begin(), model_path.end());
				const wchar_t* WCmodel_path = Wmodel_path.c_str();

				ortObjects->options.device_id = 0;
				ortObjects->options.arena_extend_strategy = 0;
				ortObjects->options.gpu_mem_limit = 1 * 512 * 512 * 3;
				ortObjects->options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
				ortObjects->options.do_copy_in_default_stream = 1;

				OrtSessionOptionsAppendExecutionProvider_CUDA(ortObjects->session_options, 0);
				ortObjects->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

				ortObjects->session = std::make_unique<Ort::Session>(ortObjects->env, WCmodel_path, ortObjects->session_options); // 수정됨 (메모리 누수 방지)
				
				if (ortObjects->session) {
					e_Utils.logMessagePrint("Model loaded successfully.");
				}
				else {
					e_Utils.logMessagePrint("Model loaded falied.");
				}
			
				int stateInfer = useGpuPeriod();

				if (e_configOntimerMode) {
					e_Timer1.start(1000, []() {
						std::cout << "1초 타이머 실행\n";
						});

					e_Timer2.start(500, []() {
						std::cout << "0.5초 타이머 실행\n";
						});

					std::this_thread::sleep_for(std::chrono::seconds(5));

					e_Timer1.stop();
					std::cout << "1초 타이머 정지\n";

					std::this_thread::sleep_for(std::chrono::seconds(10));

					e_Timer2.stop();
					std::cout << "0.5초 타이머 정지\n";

				}

				return 0;

			} catch (const std::exception& e) {	e_Utils.handleException(e, "Error in initialAIModel"); }
		}

		int GpuCla::inferenceAIModel(cv::Mat& image, cv::Mat& heatmap, float probThresh)
		{
			try {
				if (!ortObjects->session) {
					std::cerr << "Error: Model not initialized!" << std::endl;
				}

				double startTime = e_Utils.getCurrentTime();

				cv::Size dnnInputSize = cv::Size(512, 512);
				//cv::Scalar mean = cv::Scalar(0.485, 0.456, 0.406);
				//bool swapRB = true;

				cv::Mat blob;
				// ONNX: (N x 3 x H x W)
				cv::dnn::blobFromImage(image, blob, (1.0 / 255.0), dnnInputSize, cv::Scalar(), TRUE, false, CV_32F);
				std::vector<int> order = { 0,2,3,1 };
				cv::transposeND(blob, order, blob);

				// Output 이름 추가
				//*************************************************************************
				// print model input layer (node names, types, shape etc.)
				Ort::AllocatorWithDefaultOptions allocator;

				size_t num_output_nodes = ortObjects->session->GetOutputCount();
				std::vector<char*> outputNames;
				for (size_t i = 0; i < num_output_nodes; ++i)
				{
					//char* name = session.GetOutputName(i, allocator);
					char* name = ortObjects->session->GetOutputName(i, allocator);

					//std::cout << "output: " << name << std::endl;
					outputNames.push_back(name);
				}

				// Input 개수
				// print number of model input nodes
				size_t num_input_nodes = ortObjects->session->GetInputCount();
				std::vector<const char*> input_node_names(num_input_nodes);
				std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
														// Otherwise need vector<vector<>>

				//printf("Number of inputs = %zu\n", num_input_nodes);

				// input_node_dims 정보 넣기(입력 텐서의 구조 정보)
				// iterate over all input nodes
				for (int i = 0; i < num_input_nodes; i++) {
					// print input node names
					char* input_name = ortObjects->session->GetInputName(i, allocator);
					//printf("Input %d : name=%s\n", i, input_name);
					input_node_names[i] = input_name;
					allocator.Free(input_name); // 추가됨 (메모리 해제)
					// print input node types
					Ort::TypeInfo type_info = ortObjects->session->GetInputTypeInfo(i);
					auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

					ONNXTensorElementDataType type = tensor_info.GetElementType();
					//printf("Input %d : type=%d\n", i, type);

					// print input shapes/dims
					input_node_dims = tensor_info.GetShape();
					input_node_dims[0] = 1;
					//printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
					//for (int j = 0; j < input_node_dims.size(); j++)
						//printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
				}

				// 데이터를 실수형 벡터에 복사
				size_t input_tensor_size = blob.total();

				std::vector<float> input_tensor_values(input_tensor_size);
				for (size_t i = 0; i < input_tensor_size; ++i)
				{
					input_tensor_values[i] = blob.at<float>(i);
				}
				std::vector<const char*> output_node_names = { outputNames.front() };

				// 입력 텐서에 데이터 넣기
				// create input tensor object from data values
				auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
				Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
				assert(input_tensor.IsTensor());

				// session.Run. Inference 결과 생성. 텐서의 벡터 형태로 리턴
				// score model & input tensor, get back output tensor
				auto output_tensors = ortObjects->session->Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
				assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

				// 출력한 텐서 벡터 첫 번째 요소의 포인터 반환
				// Get pointer to output tensor float values
				float* floatarr = output_tensors.front().GetTensorMutableData<float>();
				assert(abs(floatarr[0] - 0.000045) < 1e-6);

				// kdh 240409 출력 class
				// 출력 텐서의 형태를 얻습니다.
				auto output_tensor_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();
				// 출력 텐서 형태의 마지막 요소(차원)는 클래스의 수를 나타냅니다.
				int num_classes = output_tensor_shape.back();

				// Mat 타입으로 변환 후 가장 큰 confidence를 가진 Id를 최종 결과로 출력
				cv::Mat1f result = cv::Mat1f(num_classes, 1, floatarr);

				cv::Point classIdPoint;
				double confidence = 0;
				minMaxLoc(result, 0, &confidence, 0, &classIdPoint);
				int classId = classIdPoint.y;
				//std::cout << "confidence: " << confidence << std::endl;
				//std::cout << "Cla_infer CLASS: " << classId << std::endl;

				classId = checkProbabilityThreshold(classId, confidence, probThresh);

				image.copyTo(heatmap);

				// 분리된 이미지 저장 함수 호출
				if (e_configImageSaveMode) {
					saveInferenceImage(image, classId);
				}

				double endTime = e_Utils.getCurrentTime();
				double duration = e_Utils.logExecutionTime("Inference Time: ", startTime, endTime);

				return classId;

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in inferenceAIModel"); }
		}

		int GpuCla::inferenceAIModel1D(unsigned char* Img_Buf, int w, int h, float probThresh)
		{
			try {
				std::lock_guard<std::mutex> lock(globalInferenceMutex);

				int result_cla = 0;

				cv::Mat image = convert8UC1ToMat(Img_Buf, w, h);
				cv::Mat heatmap_image;


				if (!e_configHeatmapCla)
					result_cla = inferenceAIModel(image, heatmap_image, probThresh);
				else 
					result_cla = inferenceAIModelHeatmap(image, heatmap_image, probThresh);

				/*std::string message = "flagInferIndepend != TRUE";
				e_Utils.logMessagePrint(message);*/

				return result_cla;

			} catch (const std::exception& e) { e_Utils.handleException(e, "Error in inferenceAIModel1D"); }
		}

		int GpuCla::inferenceAIModelHeatmap(cv::Mat& image, cv::Mat& heatmap, float probThresh)
		{
			try {
				if (!ortObjects->session) {
					std::cerr << "Error: Model not initialized!" << std::endl;
					return -1;
				}

				double startTime = e_Utils.getCurrentTime();

				cv::Size dnnInputSize = cv::Size(512, 512);
				cv::Mat blob;
				cv::dnn::blobFromImage(image, blob, (1.0 / 255.0), dnnInputSize, cv::Scalar(), TRUE, false, CV_32F);
				std::vector<int> order = { 0,2,3,1 };
				cv::transposeND(blob, order, blob);

				Ort::AllocatorWithDefaultOptions allocator;
				std::vector<const char*> input_node_names;
				std::vector<int64_t> input_node_dims;
				size_t num_input_nodes = ortObjects->session->GetInputCount();
				input_node_names.resize(num_input_nodes);

				for (int i = 0; i < num_input_nodes; ++i) {
					char* input_name = ortObjects->session->GetInputName(i, allocator);
					input_node_names[i] = input_name;
					Ort::TypeInfo type_info = ortObjects->session->GetInputTypeInfo(i);
					auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
					input_node_dims = tensor_info.GetShape();
					input_node_dims[0] = 1;
					allocator.Free(input_name);
				}

				size_t input_tensor_size = blob.total();
				std::vector<float> input_tensor_values(input_tensor_size);
				for (size_t i = 0; i < input_tensor_size; ++i)
					input_tensor_values[i] = blob.at<float>(i);

				std::vector<const char*> output_node_names = { "block14_sepconv2_act", "dense_1" };

				auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
				Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

				auto output_tensors = ortObjects->session->Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);

				// 디버깅 정보 출력
				auto conv_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
				auto softmax_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
				std::cout << "[DEBUG] Conv Feature Shape: ";
				for (auto s : conv_shape) std::cout << s << " "; std::cout << std::endl;
				std::cout << "[DEBUG] Softmax Output Shape: ";
				for (auto s : softmax_shape) std::cout << s << " "; std::cout << std::endl;

				float* softmax_output = output_tensors[1].GetTensorMutableData<float>();
				std::cout << "[DEBUG] Softmax Values: ";
				for (int i = 0; i < std::min(10, (int)softmax_shape.back()); ++i)
					std::cout << softmax_output[i] << " ";
				std::cout << std::endl;

				cv::Mat fc_weights;
				cv::FileStorage fs("D:\\Intellisense_AI\\Model\\fc_weights_cam.yml", cv::FileStorage::READ);
				//std::string model_path = "D:\\Intellisense_AI\\Model\\Model_main.onnx";

				fs["weights"] >> fc_weights;
				fs.release();
				std::cout << "[DEBUG] FC Weights Shape: " << fc_weights.rows << " x " << fc_weights.cols << std::endl;
				if (fc_weights.cols == softmax_shape.back() && fc_weights.rows == conv_shape[1]) {
					cv::transpose(fc_weights, fc_weights);
					std::cout << "[DEBUG] Transposed FC Weights\n";
				}

				// 최종 결과 클래스
				int num_classes = softmax_shape.back();
				cv::Mat1f result = cv::Mat1f(num_classes, 1, softmax_output);
				cv::Point classIdPoint;
				double confidence = 0;
				cv::minMaxLoc(result, 0, &confidence, 0, &classIdPoint);
				int classId = classIdPoint.y;
				std::cout << "Predicted class: " << classId << ", confidence: " << confidence << std::endl;
				classId = checkProbabilityThreshold(classId, confidence, probThresh);

				float* conv_output_data = output_tensors[0].GetTensorMutableData<float>();
				int featmap_w = conv_shape[3];
				int featmap_h = conv_shape[2];
				int channels = conv_shape[1];

				cv::Mat cam = cv::Mat::zeros(featmap_h, featmap_w, CV_32F);
				for (int c = 0; c < channels; ++c) {
					float weight = fc_weights.at<float>(classId, c);
					for (int y = 0; y < featmap_h; ++y) {
						for (int x = 0; x < featmap_w; ++x) {
							int idx = y * featmap_w * channels + x * channels + c;
							cam.at<float>(y, x) += conv_output_data[idx] * weight;
						}
					}
				}

				cv::threshold(cam, cam, 0, 0, cv::THRESH_TOZERO);
				cv::normalize(cam, cam, 0, 255, cv::NORM_MINMAX);  // 이 줄은 일단 유지
				cam.convertTo(cam, CV_8U);
				cv::resize(cam, cam, image.size());

				// Contrast 보정 옵션 (추가로 테스트)
				//cv::equalizeHist(cam, cam);  // 또는 cv::convertScaleAbs(cam, cam, 1.5, 0);

				// ColorMap 적용
				cv::applyColorMap(cam, heatmap, cv::COLORMAP_JET);
				cv::addWeighted(image, 0.5, heatmap, 0.5, 0, heatmap);

				// 이미지 저장 함수 호출
				if (e_configImageSaveMode) {
					saveHeatmapImage(image, heatmap, classId);
				}

				// CSVLog 저장 함수 호출
				if (e_configCSVLogSaveMode) {
					saveCSVLog(classId);
				}

				double minVal, maxVal;
				cv::minMaxLoc(cam, &minVal, &maxVal);
				std::cout << "CAM raw min/max: " << minVal << ", " << maxVal << std::endl;

				double endTime = e_Utils.getCurrentTime();
				e_Utils.logExecutionTime("Inference Time: ", startTime, endTime);

				return classId;
			}
			catch (std::exception& e) {
				e_Utils.handleException(e, "Error in inferenceAIModelHeatmap");
				return -1;
			}
		}

		void GpuCla::check_fc_weights_shape(const std::string& weight_path, int expected_rows, int expected_cols) {
			cv::Mat fc_weights;
			cv::FileStorage fs(weight_path, cv::FileStorage::READ);

			if (!fs.isOpened()) {
				std::cerr << "[X] File open failed: " << weight_path << std::endl;
				return;
			}

			fs["weights"] >> fc_weights;
			fs.release();

			std::cout << "[INFO] Loaded weight shape: "
				<< fc_weights.rows << " x " << fc_weights.cols << std::endl;

			if (fc_weights.rows == expected_rows && fc_weights.cols == expected_cols) {
				std::cout << "[O] Weight shape matches expected ("
					<< expected_rows << " x " << expected_cols << ")" << std::endl;
			}
			else {
				std::cerr << "[X] Weight shape mismatch!" << std::endl;
				std::cerr << "      Expected: " << expected_rows << " x " << expected_cols << std::endl;
				std::cerr << "      Loaded:   " << fc_weights.rows << " x " << fc_weights.cols << std::endl;
			}
		}





		// 중복 코드 제거를 위한 전처리 함수
		cv::Mat GpuCla::convert8UC1ToMat(unsigned char* Img_Buf, int w, int h)
		{
			try {
				cv::Mat image(h, w, CV_8UC1, Img_Buf);

				if (e_configExeEnv)
					cv::flip(image, image, 0); // 0은 x축을 기준으로 뒤집음

				if (image.channels() == 1)
					cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
				if (image.cols != 512)
					cv::resize(image, image, cv::Size(512, 512), cv::INTER_AREA);
				if (image.rows != 512)
					cv::resize(image, image, cv::Size(512, 512), cv::INTER_AREA);

				return image;

			} catch (std::exception& e) { e_Utils.handleException(e, "Error to convert8UC1ToMat"); }
		}

		int GpuCla::useGpuPeriod()
		{
			try {
				e_Utils.logMessagePrint("[GPU 예열] 잠시만 기다려주세요.");

				cv::Mat dummyImage = cv::Mat::zeros(512, 512, CV_8UC3);
				cv::Mat heatmapImage;
				int result_cla = 0;

				auto originalImageSaveMode = e_configImageSaveMode;
				e_configImageSaveMode = 0;
				result_cla = inferenceAIModel(dummyImage, heatmapImage, 0.5f);
				e_configImageSaveMode = originalImageSaveMode;

				return result_cla;

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in useGpuPeriod"); }
		}

		void GpuCla::saveInferenceImage(const cv::Mat& image, int result_cla) {
			try {
				bool shouldSave = false;

				switch (e_configImageSaveMode)
				{
				case 1:
				case 2:
					if (result_cla != 0)  // 결과값이 0이 아닐 때 저장
						shouldSave = true;
					break;
				case 3:
					shouldSave = true;  // 결과값과 관계없이 저장
					break;
				}

				if (shouldSave)
				{
					// 오늘 날짜 (YYYY-MM-DD) 기반 폴더 경로 생성
					std::string time_yearmonday = e_Utils.makeTime("yearmonday");
					std::string path_save_folder_day = e_pathSaveFolder + time_yearmonday + "\\";
					e_Utils.makeFolder(path_save_folder_day);

					std::string secMinTime = e_Utils.makeTime("all");
					std::string path_save_image = path_save_folder_day + secMinTime + "_C" + std::to_string(result_cla);
					e_vecAllInspImage.push_back(std::make_tuple(path_save_image, image));
				}
			}

			catch (const std::exception& e) { e_Utils.handleException(e, "Error in saveInferenceImage"); }
		}

		void GpuCla::saveHeatmapImage(const cv::Mat& image, const cv::Mat& heatmap, int result_cla) {
			try {
				bool shouldSave = false;

				switch (e_configImageSaveMode)
				{
				case 1:
				case 2:
					if (result_cla != 0)  // 결과값이 0이 아닐 때 저장
						shouldSave = true;
					break;
				case 3:
					shouldSave = true;  // 결과값과 관계없이 저장
					break;
				}

				if (shouldSave)
				{
					// 오늘 날짜 (YYYY-MM-DD) 기반 폴더 경로 생성
					std::string time_yearmonday = e_Utils.makeTime("yearmonday");
					std::string path_save_folder_day = e_pathSaveFolder + time_yearmonday + "\\";
					e_Utils.makeFolder(path_save_folder_day);

					std::string secMinTime = e_Utils.makeTime("all");
					std::string path_save_image = path_save_folder_day + secMinTime + "_C" + std::to_string(result_cla);
					e_vecAllInspImage.push_back(std::make_tuple(path_save_image, image));

					if (e_configHeatmapCla)
					{
						std::string path_save_folder_day_heat = e_pathSaveFolder + time_yearmonday + "_Heat" + "\\";
						e_Utils.makeFolder(path_save_folder_day_heat);

						std::string path_save_image_heat = path_save_folder_day_heat + secMinTime + "_C" + std::to_string(result_cla);
						e_vecAllInspImage.push_back(std::make_tuple(path_save_image_heat, heatmap));
					}
				}
			}

			catch (const std::exception& e) { e_Utils.handleException(e, "Error in saveHeatmapImage"); }
		}

		void GpuCla::saveCSVLog(int classId) {

			if (e_configCSVLogSaveMode) {
				// 오늘 날짜 (YYYY-MM-DD) 기반 폴더 경로 생성
				std::string secMinTime = e_Utils.makeTime("time");

				Utils::CSVLogSaver::getInstance().pushCSVLogEntry(secMinTime, "product", classId, 0, 0);
			}
		
		}
		
		// 확률 임계값 검사 함수
		int GpuCla::checkProbabilityThreshold(int classId, double confidence, float probThresh)
		{
			if (e_configThresholdCla != 0)
			{
				probThresh = static_cast<float>(e_configThresholdCla) / 100.0f;
			}

			if ((classId >= 1) && (confidence <= probThresh))
			{
				//e_Utils.logMessagePrint("Probabilty Threashold : " + std::to_string(confidence));
				
				classId = 0; // 임계값 미달 시
			}

			//cout << "probThresh : " << probThresh << endl;

			return classId;  // 정상일 경우 원래 classId 반환
		}

	}

	namespace CV {
		BaseIsOpencv::BaseIsOpencv() {} // 생성자 구현
		BaseIsOpencv::~BaseIsOpencv() {} // 소멸자 구현

		Mat BaseIsOpencv::standTransImage(const Mat& input_image) {
			if (input_image.empty()) {
				cerr << "Error: Cannot load image!" << endl;
				return Mat();
			}

			Mat standardized_image;

			// 1. 8-bit 변환 (16-bit 또는 32-bit 이미지 처리)
			if (input_image.depth() != CV_8U) {
				input_image.convertTo(input_image, CV_8U, 255.0 / 65535.0);
			}

			// 2. 채널 처리 (1채널 → 3채널, 4채널 → 3채널 변환)
			if (input_image.channels() == 1) {
				cvtColor(input_image, standardized_image, COLOR_GRAY2BGR);
			}
			else if (input_image.channels() == 4) {
				cvtColor(input_image, standardized_image, COLOR_BGRA2BGR);
			}
			else {
				standardized_image = input_image.clone();
			}

			// 3. Resize to 512x512
			resize(standardized_image, standardized_image, Size(512, 512));

			return standardized_image;
		}

		double BaseIsOpencv::detectSpoidLength(const Mat& input_image) {
			if (input_image.empty()) {
				cerr << "Error: Cannot load image!" << endl;
				return -1;
			}

			// 1. 입력 이미지를 통일된 형식으로 변환 (512x512, 8-bit grayscale)
			Mat ori_image = input_image.clone()
				;
			if (input_image.type() != CV_8UC1) {
				cvtColor(input_image, ori_image, COLOR_BGR2GRAY);
			}

			Mat equalHisted;
			equalizeHist(ori_image, equalHisted);

			// 2. 병 내부 영역을 검출하기 위한 대비 조정 및 전처리
			Mat blurred;
			GaussianBlur(equalHisted, blurred, Size(5, 5), 0);

			Mat binary;
			threshold(blurred, binary, 100, 255, THRESH_BINARY_INV);

			// 병 외곽 윤곽선 검출
			vector<vector<Point>> contours;
			findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

			if (contours.empty()) {
				cerr << "Error: No bottle detected!" << endl;
				return -1;
			}

			// 가장 큰 윤곽선(병) 찾기
			vector<Point> largest_contour = *max_element(contours.begin(), contours.end(),
				[](const vector<Point>& a, const vector<Point>& b) { return contourArea(a) < contourArea(b); });

			// 병 전체 영역을 포함하도록 ROI 수정
			Rect boundingBox = boundingRect(largest_contour);
			int roi_x = boundingBox.x + boundingBox.width / 4;
			int roi_width = boundingBox.width / 2;
			int roi_y = boundingBox.y;
			int roi_height = boundingBox.height;
			Rect spoidROI(roi_x, roi_y, roi_width, roi_height);
			if (spoidROI.width > 0 && spoidROI.height > 0) {
				ori_image = ori_image(spoidROI).clone();
			}
			else {
				cerr << "Error: Invalid bounding box detected!" << endl;
				return -1;
			}

			// 3. 병 내부에서 스포이드 검출 수행
			Mat roi_gray;
			Sobel(ori_image, roi_gray, CV_8U, 1, 0, 3);

			Mat roi_equalized;
			equalizeHist(roi_gray, roi_equalized);

			Mat roi_blurred;
			GaussianBlur(roi_equalized, roi_blurred, Size(5, 5), 0);

			Mat roi_edges;
			Canny(roi_blurred, roi_edges, 50, 150);

			// 4. Hough Transform을 사용하여 스포이드의 수직 직선 검출
			vector<Vec4i> roi_lines;
			HoughLinesP(roi_edges, roi_lines, 1, CV_PI / 180, 50, 30, 5);

			// 가장 긴 수직 직선 찾기 (스포이드만 감지, 상하 방향만 허용)
			double max_spoid_length = 0;
			Vec4i best_line;

			for (const auto& l : roi_lines) {
				double length = sqrt(pow(l[2] - l[0], 2) + pow(l[3] - l[1], 2));
				int mid_x = (l[0] + l[2]) / 2;
				double angle = atan2(abs(l[3] - l[1]), abs(l[2] - l[0])) * 180 / CV_PI;

				// 수직 방향 선 필터링 (80~100도 범위의 수직선만 허용)
				if (angle > 80 && angle < 100 &&
					mid_x > ori_image.cols / 4 && mid_x < 3 * ori_image.cols / 4) {
					if (length > max_spoid_length) {
						max_spoid_length = length;
						best_line = l;
					}
				}
			}

			// 5. 결과 출력
			cout << "Detected Spoid Length: " << max_spoid_length << " pixels" << endl;

			// 6. 원본 이미지에 결과 표시
			Mat spoid_result = input_image.clone();
			if (spoid_result.channels() == 1) {
				cvtColor(spoid_result, spoid_result, COLOR_GRAY2BGR);
			}

			if (max_spoid_length > 0) {
				// ROI 내부 좌표를 원본 좌표로 변환
				Point start_pt(best_line[0] + spoidROI.x, best_line[1] + spoidROI.y);
				Point end_pt(best_line[2] + spoidROI.x, best_line[3] + spoidROI.y);

				// 원본 이미지에 직선 표시
				line(spoid_result, start_pt, end_pt, Scalar(0, 0, 255), 2);
			}

			imshow("Detected Spoid", spoid_result);
			waitKey(0);

			return max_spoid_length;
		}

		Mat BaseIsOpencv::correctImageRotation(const Mat& input_image)
		{
			Mat gray, edged;
			cvtColor(input_image, gray, COLOR_BGR2GRAY);
			GaussianBlur(gray, gray, Size(5, 5), 0);
			Canny(gray, edged, 50, 150);

			// 가장 큰 외곽선을 찾기
			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(edged, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

			if (contours.empty()) {
				cerr << "No contours found!" << endl;
				return input_image;
			}

			// 가장 큰 컨투어 찾기
			size_t largestContourIdx = 0;
			double maxArea = 0;
			for (size_t i = 0; i < contours.size(); i++) {
				double area = contourArea(contours[i]);
				cout << "Contour " << i << " Area: " << area << endl; // 디버깅 출력
				if (area > maxArea) {
					maxArea = area;
					largestContourIdx = i;
				}
			}

			// 디버깅: 모든 컨투어를 원본 이미지에 그리기
			Mat contourDebugImage = input_image.clone();
			for (size_t i = 0; i < contours.size(); i++) {
				Scalar color = (i == largestContourIdx) ? Scalar(0, 0, 255) : Scalar(0, 255, 0); // 빨강: 가장 큰 컨투어
				drawContours(contourDebugImage, contours, (int)i, color, 2);
			}
			imwrite("contour_debug.jpg", contourDebugImage);
			cout << "Contour debug image saved as 'contour_debug.jpg'" << endl;

			// 최소 경계 사각형 찾기
			RotatedRect minRect = minAreaRect(contours[largestContourIdx]);

			Point2f rectPoints[4];
			minRect.points(rectPoints);
			for (int i = 0; i < 4; i++) {
				line(contourDebugImage, rectPoints[i], rectPoints[(i + 1) % 4], Scalar(255, 0, 0), 2);
			}
			// 기울기 계산 (긴 변이 세로가 되도록 보정)
			double angle = minRect.angle;
			if (minRect.size.width > minRect.size.height) {
				angle += 90; // 긴 변이 수직이 되도록 조정
			}

			cout << "Detected angle: " << angle << " degrees" << endl;

			// 회전 변환 행렬 생성 (중심 기준 회전)
			Mat rotationMatrix = getRotationMatrix2D(minRect.center, angle, 1.0);

			// 이미지 회전
			Mat rotated;
			warpAffine(input_image, rotated, rotationMatrix, input_image.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

			return rotated;
		}

#include <opencv2/opencv.hpp>
#include <iostream>

		// 함수: 기울어진 앰플병을 수직으로 정렬
		Mat BaseIsOpencv::straightenAmpoule(const Mat& input_image) {
			Mat gray, edged;
			cvtColor(input_image, gray, COLOR_BGR2GRAY);
			GaussianBlur(gray, gray, Size(5, 5), 0);
			// 모폴로지 연산을 추가하여 선을 연결 (Closing)
			Mat morphKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
			morphologyEx(gray, gray, MORPH_CLOSE, morphKernel);

			Canny(gray, edged, 50, 150);

			// 모폴로지 연산을 추가하여 작은 간격을 메움
			morphologyEx(edged, edged, MORPH_CLOSE, morphKernel);
			// 가장 큰 외곽선을 찾기
			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(edged, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

			if (contours.empty()) {
				cerr << "No contours found!" << endl;
				return input_image;
			}

			// 가장 큰 컨투어 찾기 (Area, Bounding Box, Point 개수 조합)
			vector<vector<Point>> filteredContours;
			double maxArea = 0;
			double maxBoundingBoxArea = 0;
			size_t maxPointCount = 0;

			for (size_t i = 0; i < contours.size(); i++) {
				double area = contourArea(contours[i]);
				Rect boundingBox = boundingRect(contours[i]);
				double boundingBoxArea = boundingBox.area();
				size_t pointCount = contours[i].size();

				// 디버깅 출력
				cout << "Contour " << i
					<< " - Contour Area: " << area
					<< ", Bounding Box Area: " << boundingBoxArea
					<< ", Point Count: " << pointCount << endl;

				// 작은 잡음 제거 (모든 기준 적용)
				if (area < 30 || boundingBoxArea < 10000 || pointCount < 50) {
					cout << "Skipping small contour " << i << endl;
					continue;
				}

				// 이미지 경계에 닿아 있는 컨투어 제거
				if (boundingBox.x == 0 || boundingBox.y == 0 ||
					boundingBox.x + boundingBox.width >= input_image.cols ||
					boundingBox.y + boundingBox.height >= input_image.rows) {
					cout << "Skipping contour " << i << " (Touches image boundary)" << endl;
					continue;
				}

				// 가장 큰 컨투어 선택 (Bounding Box & Point 개수 고려)
				if (boundingBoxArea > maxBoundingBoxArea && pointCount > maxPointCount) {
					maxBoundingBoxArea = boundingBoxArea;
					maxPointCount = pointCount;
					filteredContours.clear();
					filteredContours.push_back(contours[i]);
				}
			}

			if (filteredContours.empty()) {
				cerr << "Error: No valid contours found!" << endl;
				return input_image;
			}

			// 디버깅: 가장 큰 컨투어를 원본 이미지에 그리기
			Mat contourDebugImage = input_image.clone();
			drawContours(contourDebugImage, filteredContours, 0, Scalar(0, 0, 255), 2);
			cout << "Contour debug image saved as 'contour_debug.jpg'" << endl;

			// 최소 경계 사각형 찾기
			RotatedRect minRect = minAreaRect(filteredContours[0]);
			Point2f rectPoints[4];
			minRect.points(rectPoints);
			for (int i = 0; i < 4; i++) {
				line(contourDebugImage, rectPoints[i], rectPoints[(i + 1) % 4], Scalar(255, 0, 0), 2);
			}

			// 기울기 계산 (긴 변이 세로가 되도록 보정)
			double angle = minRect.angle;
			if (minRect.size.width > minRect.size.height) {
				angle += 90;
			}

			cout << "Detected angle: " << angle << " degrees" << endl;

			// 회전 변환 행렬 생성 (중심 기준 회전)
			Mat rotationMatrix = getRotationMatrix2D(minRect.center, angle, 1.0);
			Mat rotated;
			warpAffine(input_image, rotated, rotationMatrix, input_image.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(255, 255, 255));

			// --- 머리 방향 확인 후 180도 추가 회전 ---
			vector<Point> largestContour = filteredContours[0];
			int minY = INT_MAX, maxY = INT_MIN;
			Point topPoint, bottomPoint;

			for (const Point& p : largestContour) {
				if (p.y < minY) {
					minY = p.y;
					topPoint = p;
				}
				if (p.y > maxY) {
					maxY = p.y;
					bottomPoint = p;
				}
			}

			// 머리가 아래쪽을 향하면 추가 180도 회전
			if (topPoint.y > bottomPoint.y) {
				cout << "Head is upside down. Rotating 180 degrees." << endl;
				rotate(rotated, rotated, ROTATE_180);
			}

			return rotated;
		}
	}

///////////////////////////////////////////////////////////////////////////////////////////

	namespace Utils {

		// OpenCV imwrite()를 활용한 파일 저장 함수
		bool ImageSaver::saveImageWithOpenCV(const std::string& filename, const cv::Mat& image) 
		{
			try {
				std::string fullPath;
				if (e_configImageSaveMode == 1)
					fullPath = filename + ".jpg";
				else if (e_configImageSaveMode == 2)
					fullPath = filename + ".png";
				else if (e_configImageSaveMode == 3)
					fullPath = filename + ".jpg";

				bool success = cv::imwrite(fullPath, image);  // OpenCV를 사용하여 이미지 저장
				return success;

			} catch (const cv::Exception& e) { e_Utils.handleException(e, "Error to saveImageWithOpenCV"); return false; }
		}

		// 파일 저장 확인 함수
		bool ImageSaver::verifyFileSave(const std::string& filename, int maxRetries, int sleepTime) 
		{
			try {
				std::string fullPath;
				if (e_configImageSaveMode == 1)
					fullPath = filename + ".jpg";
				else if (e_configImageSaveMode == 2)
					fullPath = filename + ".png";
				else if (e_configImageSaveMode == 3)
					fullPath = filename + ".jpg";

				for (int i = 0; i < maxRetries; ++i) {
					if (std::filesystem::exists(fullPath) && std::filesystem::file_size(fullPath) > 0) {
						return true;  // 파일이 정상적으로 저장됨
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(sleepTime));
				}
				return false;  // 최종적으로 파일 저장 실패

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in logMessagePrint"); return false; }

		}

		// 이미지 저장 스레드 실행
		void ImageSaver::runImageSaveThread() 
		{
			try {
				while (keepRunning.load()) {
					if (!e_vecAllInspImage.empty()) {
						std::string filename = std::get<0>(e_vecAllInspImage.front());
						cv::Mat image = std::get<1>(e_vecAllInspImage.front());

						// OpenCV imwrite()를 사용하여 이미지 저장
						if (!saveImageWithOpenCV(filename, image)) {
							e_Utils.logMessagePrint("이미지 저장 실패: " + filename);
						}

						// 저장 후 파일이 정상적으로 생성되었는지 확인
						if (!verifyFileSave(filename, 20, 5)) {
							//Utils::logMessagePrint("파일 확인 실패: " + filename);
						}

						e_vecAllInspImage.erase(e_vecAllInspImage.begin());  // 큐에서 이미지 제거
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(2)); // CPU 사용량 최적화
				}
			} catch (std::exception& e) { e_Utils.handleException(e, "Error in runImageSaveThread"); }
		}

		// 스레드 시작
		void ImageSaver::startImageSaveThread() 
		{
			try {
				if (imageSaveThread.joinable()) {
					return;  // 이미 실행 중이면 다시 시작하지 않음
				}

				keepRunning.store(true);
				imageSaveThread = std::thread(&ImageSaver::runImageSaveThread, &ImageSaver::getInstance());

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in startImageSaveThread"); }
		}

		// 스레드 종료
		void ImageSaver::stopImageSaveThread() 
		{
			try {
				keepRunning.store(false);
				if (imageSaveThread.joinable()) {
					imageSaveThread.join();  // 스레드 종료 대기
				}
			} catch (std::exception& e) { e_Utils.handleException(e, "Error in stopImageSaveThread"); }
		}



		void CSVLogSaver::startCSVLogSaveThread() {
			if (running.load()) {
				return;
			}

			running.store(true);
			logThread = std::thread(&CSVLogSaver::runCSVLogSaveThread, this);
		}

		void CSVLogSaver::stopCSVLogSaveThread() {
			if (!running.load()) {
				return;
			}

			running.store(false);
			dataCV.notify_one();  // 스레드 깨우기
			if (logThread.joinable()) 
				logThread.join();
		}

		void CSVLogSaver::runCSVLogSaveThread() {
			while (running.load()) {
				LogEntry entry;

				{
					std::unique_lock<std::mutex> lock(queueMutex);
					// 큐에 데이터가 없으면 대기
					dataCV.wait(lock, [this] { return !logQueue.empty() || !running; });

					if (!running && logQueue.empty()) 
						break;

					// 큐에서 데이터 꺼내기
					entry = logQueue.front();
					logQueue.pop_front();
				}

				// CSV 라인 구성 및 저장
				std::ostringstream oss;
				oss << entry.timestamp << "," << entry.productName << ","
					<< entry.total << "," << entry.good << "," << entry.bad;

				writeCSVLogToFile(oss.str(), entry.timestamp.substr(0, 10));  // 날짜로 파일명 결정
			}
		}

		void CSVLogSaver::pushCSVLogEntry(const std::string& currentTime, const std::string& productName, int total, int ok, int ng) {
			std::lock_guard<std::mutex> lock(queueMutex);
			logQueue.push_back({
				currentTime,
				productName,
				total,
				ok,
				ng
				});
			dataCV.notify_one();  // 데이터 추가됨 알림
		}

		bool CSVLogSaver::writeCSVLogToFile(const std::string& line, const std::string& date) {

			std::string time_yearmonday = e_Utils.makeTime("yearmonday");
			std::string filePath = e_pathLogRecord + "\\" + time_yearmonday + ".csv";

			// 파일이 존재하는지 확인
			bool fileExists = std::filesystem::exists(filePath);
			std::ofstream ofs(filePath, std::ios::app);

			if (!ofs.is_open()) 
				return false;

			// 헤더 자동 추가
			if (!fileExists) {
				ofs << "시간,제품명,Result,empty,empty\n";
			}

			ofs << line << "\n";
			ofs.close();
			return true;
		}





		double Utils::getCurrentTime()
		{
			try {
				auto now = std::chrono::high_resolution_clock::now();
				return std::chrono::duration<double>(now.time_since_epoch()).count();

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in getCurrentTime"); }
		}

		double Utils::logExecutionTime(const std::string& prefix, double startTime, double endTime)
		{
			try {
				// 경과 시간 계산
				double duration = endTime - startTime;

				// 소수점 3자리까지만 유지 (버림)
				duration = std::round(duration * 1000) / 1000.0;

				// stringstream을 사용하여 소수점 3자리까지만 출력
				std::ostringstream stream;
				stream.precision(3);
				stream << std::fixed << duration;

				// 최종 로그 출력
				std::cout << "[LOG] " << prefix << stream.str() << " sec" << std::endl;

				return duration;

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in logExecutionTime"); }
		}

		void Utils::runNvidiaSMI()
		{
			try {
				STARTUPINFO si = { sizeof(STARTUPINFO) };
				PROCESS_INFORMATION pi;

				// NVIDIA SMI 실행 명령어
				const char* cmd = "C:\\Windows\\System32\\cmd.exe /C \"nvidia-smi -lgc 3100\"";

				if (CreateProcessA(NULL, (LPSTR)cmd, NULL, NULL, FALSE, CREATE_NO_WINDOW, NULL, NULL, &si, &pi))
				{
					// 프로세스 핸들 정리
					CloseHandle(pi.hProcess);
					CloseHandle(pi.hThread);
					logMessagePrint("NVIDIA SMI Command Executed.");
				}
				else
				{
					logMessagePrint("[ERROR] Failed to execute NVIDIA SMI command.");
				}
			} catch (std::exception& e) { e_Utils.handleException(e, "Error in runNvidiaSMI"); }
		}

		bool Utils::checkProgramAdmin()
		{
			try {
				BOOL isAdmin = FALSE;
				HANDLE hToken = NULL;

				// 현재 프로세스의 액세스 토큰 가져오기
				if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &hToken))
				{
					TOKEN_ELEVATION elevation;
					DWORD dwSize = sizeof(TOKEN_ELEVATION);

					// 관리자 권한 여부 확인
					if (GetTokenInformation(hToken, TokenElevation, &elevation, sizeof(elevation), &dwSize))
					{
						isAdmin = elevation.TokenIsElevated;
					}
					CloseHandle(hToken);
				}

				// 관리자 여부 출력
				if (isAdmin)
				{
					logMessagePrint("프로그램이 관리자 권한으로 실행되었습니다.");
				}
				else
				{
					logMessagePrint("프로그램이 관리자 권한이 아닙니다. 관리자 권한으로 다시 실행하세요.");
				}
				return isAdmin;

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in logMessagePrint"); }
		}

		std::string Utils::makeTime(const std::string& format) 
		{
			try {
				// 현재 시간 가져오기
				auto now = std::chrono::system_clock::now();
				std::time_t tt = std::chrono::system_clock::to_time_t(now);
				std::tm tm;
				localtime_s(&tm, &tt);

				std::stringstream ss;

				if (format == "yearmonday") {
					// 연-월-일 형식 (YYYY-MM-DD)
					ss << tm.tm_year + 1900 << "-"
						<< std::setfill('0') << std::setw(2) << tm.tm_mon + 1 << "-"
						<< std::setfill('0') << std::setw(2) << tm.tm_mday;
				}
				else if (format == "time") {
					// 시-분-초 형식 (HH:MM:SS)
					ss << std::setfill('0') << std::setw(2) << tm.tm_hour << "-"
						<< std::setfill('0') << std::setw(2) << tm.tm_min << "-"
						<< std::setfill('0') << std::setw(2) << tm.tm_sec;
				}
				else {
					// 연-월-일-시-분-초-밀리초 형식 (YYYY-MM-DD-HHMMSSmmm)
					ss << tm.tm_year + 1900 << "-"
						<< std::setfill('0') << std::setw(2) << tm.tm_mon + 1 << "-"
						<< std::setfill('0') << std::setw(2) << tm.tm_mday << "-"
						<< std::setfill('0') << std::setw(2) << tm.tm_hour
						<< std::setfill('0') << std::setw(2) << tm.tm_min
						<< std::setfill('0') << std::setw(2) << tm.tm_sec
						<< std::setfill('0') << std::setw(3)
						<< std::chrono::duration_cast<std::chrono::milliseconds>(now - std::chrono::system_clock::from_time_t(tt)).count();
				}

				return ss.str();

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in makeTime"); }
		}

		void Utils::makeFolder(const std::string& path) 
		{
			try	{
				if (!std::filesystem::exists(path)) {
					std::filesystem::create_directories(path);
				}
			}  catch (std::exception& e) { e_Utils.handleException(e, "Error in logMessagePrint"); }
		}

		void Utils::logMessagePrint(const std::string& message)
		{
			try {
				// 폴더가 없으면 생성
				Utils::makeFolder(e_pathLogUsed);

				if (e_configLogLevel == 0) {
					return;
				}

				// 콘솔 출력 (LOG_LEVEL 1 이상)
				if (e_configLogLevel >= 1) {
					std::cout << "[LOG] " << Utils::makeTime("time") << "  " << message << std::endl;
				}

				// 파일 로그 저장 (LOG_LEVEL 2 이상)
				if (e_configLogLevel >= 2) {
					std::string log_file = e_pathLogUsed + "\\" + Utils::makeTime("yearmonday") + "_Used.txt";

					// 로그 파일 저장
					std::ofstream logStream(log_file, std::ios::app);
					if (logStream.is_open()) {
						logStream << "[LOG] " << Utils::makeTime("time") << "  " << message << std::endl;
						logStream.close();
					}
				}

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in logMessagePrint"); }
		}

		void Utils::logError(const std::string& context, const std::string& message) 
		{
			try {
				Utils::makeFolder(e_pathLogError);

				std::string log_file = e_pathLogError + "\\" + Utils::makeTime("yearmonday") + "_Error.txt";

				std::ofstream logStream(log_file, std::ios::app);
				if (logStream.is_open()) {
					logStream << "[LOG] " << Utils::makeTime("time") << "  " << context << " : " << message << std::endl;
					logStream.close();
				}
				else {
					std::cerr << "[ERROR] Unable to open log file." << std::endl;
				}
			} catch (std::exception& e) { e_Utils.handleException(e, "Error in logError"); }
		}

		void Utils::logErrorCustom(const std::string& message) {
			Utils::logError("[Custom Error]", message);
			throw;
		}

		void Utils::handleException(const std::exception& e, const std::string& context) {
			std::cerr << "[ERROR] " << context << " - " << e.what() << std::endl;
			Utils::logError(context, e.what());
			throw;
		}

		bool Utils::isFileExists(const std::string& path) {
			try {
				// 경로가 존재하고, 파일인지 확인
				return std::filesystem::exists(path) && std::filesystem::is_regular_file(path);

			} catch (const std::exception& e) {	e_Utils.handleException(e, "Error in isFileExists");}
		}


		std::string Utils::pathSelectFolder() {
			std::string selected_folder_path;

			// COM 객체 생성
			CComPtr<IFileDialog> pFileDialog;
			HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pFileDialog));

			if (SUCCEEDED(hr))
			{
				// 폴더 선택 옵션 설정
				DWORD dwOptions;
				pFileDialog->GetOptions(&dwOptions);
				pFileDialog->SetOptions(dwOptions | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM);

				// 다이얼로그 실행
				hr = pFileDialog->Show(nullptr);  // DLL이라면 nullptr 사용
				if (SUCCEEDED(hr))
				{
					CComPtr<IShellItem> pItem;
					hr = pFileDialog->GetResult(&pItem);
					if (SUCCEEDED(hr))
					{
						PWSTR pszFilePath = nullptr;
						pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

						// 💡 WideChar → MultiByte 문자열 변환
						char pathA[MAX_PATH] = { 0 };
						WideCharToMultiByte(CP_ACP, 0, pszFilePath, -1, pathA, MAX_PATH, nullptr, nullptr);
						selected_folder_path = std::string(pathA);

						CoTaskMemFree(pszFilePath);
					}
				}
			}

			return selected_folder_path;
		}

	// ================================= ConfigManager (설정 관리) =================================
	// 설정 값 저장
		void ConfigManager::setSetting(const std::string& key, const std::string& value) 
		{
			try {
				configs[key] = value;
				std::cout << "[Settings] " << key << " updated to " << value << std::endl;
			} catch (std::exception& e) { e_Utils.handleException(e, "Error in setSetting"); }
		}

		// 설정 값 불러오기 (기본값 존재)
		std::string ConfigManager::getSetting(const std::string& key, const std::string& defaultValue) 
		{
			try {
				if (configs.find(key) != configs.end()) {
					return configs[key];
				}
				return defaultValue;

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in getSetting"); }
		}

		// 설정 파일 저장
		void ConfigManager::saveSettings() 
		{
			try {
				std::ofstream file(e_pathConfigFile);
				if (!file.is_open()) {
					std::cerr << "[ERROR] Cannot open configs file: " << e_pathConfigFile << std::endl;
					return;
				}

				file << "[General]" << std::endl;
				for (const auto& pair : configs) {
					file << pair.first << " = " << pair.second << std::endl;
				}

				file.close();
				std::cout << "[INFO] Settings saved to " << e_pathConfigFile << std::endl;

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in saveSettings"); }
		}

		// 설정 파일 로드
		void ConfigManager::loadSettings() 
		{
			try {
				std::ifstream file(e_pathConfigFile);
				if (!file.is_open()) {
					e_Utils.logMessagePrint("[WARNING] No configs file found. Using default values.");
					return;
				}

				std::string line;
				while (std::getline(file, line)) {
					if (line.empty() || line[0] == '[') continue;  // 빈 줄과 섹션 무시

					std::istringstream iss(line);
					std::string key, value;
					if (std::getline(iss, key, '=') && std::getline(iss, value)) {
						key.erase(key.find_last_not_of(" \t") + 1);  // 공백 제거
						value.erase(0, value.find_first_not_of(" \t"));
						configs[key] = value;
					}
				}
				file.close();

				e_configLogLevel = std::stoi(configs["log_level"]);
				e_configExeEnv = std::stoi(configs["execution_environment"]);
				e_configProccesMode = std::stoi(configs["procces_mode"]);
				e_configImageSaveMode = std::stoi(configs["image_save_mode"]);
				e_configThresholdCla = std::stoi(configs["threshold_cla"]);
				e_configHeatmapCla = std::stoi(configs["heatmap_cla"]);
				e_configCSVLogSaveMode = std::stoi(configs["CSVLog_save_mode"]);
				e_configOntimerMode = std::stoi(configs["ontimer_mode"]);

			} catch (std::exception& e) { e_Utils.handleException(e, "Error in loadSettings"); }
		}

		// 설정 값 출력
		void ConfigManager::printAllSettings() 
		{
			try {
				std::cout << "====== Current Settings ======" << std::endl;
				for (const auto& pair : configs) {
					std::cout << pair.first << " : " << pair.second << std::endl;
				}
			} catch (std::exception& e) { e_Utils.handleException(e, "Error in printAllSettings"); }
		}


		// ================================= TimerPool (타이머 부분) =================================

		SingleTimer::SingleTimer() {}

		SingleTimer::~SingleTimer() {
			stop();
		}

		void SingleTimer::start(int interval_ms, std::function<void()> callback) {
			if (running.load()) return;

			interval = interval_ms;
			task = callback;
			running = true;

			worker = std::thread(&SingleTimer::timerLoop, this);
		}

		void SingleTimer::stop() {
			running = false;
			if (worker.joinable())
				worker.join();
		}

		bool SingleTimer::isRunning() const {
			return running.load();
		}

		void SingleTimer::timerLoop() {
			while (running.load()) {
				auto startTime = std::chrono::steady_clock::now();
				if (task) 
					task();

				std::this_thread::sleep_until(startTime + std::chrono::milliseconds(interval));
			}
		}
	}
}

