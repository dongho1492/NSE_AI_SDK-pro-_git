#ifdef DEEP_EXPORTS
#define DEEP_API __declspec(dllexport)
#else
#define DEEP_API __declspec(dllimport)
#endif

#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <time.h>
#include <ctime>
#include <chrono>
#include <fstream>
#include <exception>
#include <sstream>
#include <thread>
#include <atomic>
#include <windows.h>
#include <cmath>
#include <iomanip>
#include <time.h>
#include <mutex>

#include <shobjidl.h>  // IFileDialog 관련
#include <atlbase.h>    // CComPtr
#include <atlconv.h>    // CT2A

using namespace cv;
using namespace std;

namespace NSE
{
	namespace AI {

		// class GpuCla : GPU Classification
		class DEEP_API GpuCla
		{
		public:
			GpuCla();
			~GpuCla();

			bool flagInferIndepend;
			int initialAIModel();
			int inferenceAIModel(cv::Mat& Frame, cv::Mat& heatmap, float probThresh = 0.5f);
			int inferenceAIModel1D(unsigned char* Img_Buf, int w, int h, float probThresh = 0.5f);
			int inferenceAIModelHeatmap(cv::Mat& Frame, cv::Mat& heatmap, float probThresh = 0.5f);
			void check_fc_weights_shape(const std::string& weight_path, int expected_rows, int expected_cols);

			int useGpuPeriod();
			void saveInferenceImage(const cv::Mat& image, int result_cla);
			void saveHeatmapImage(const cv::Mat& image, const cv::Mat& heatmap, int result_cla);
			void saveCSVLog(int classId);

			int checkProbabilityThreshold(int classId, double confidence, float probThresh);

			void TEST1();
			void TEST2();
			void TEST3();

		private:
			struct OrtObjects;
			std::unique_ptr<OrtObjects> ortObjects;

			static std::mutex globalInferenceMutex;

			cv::Mat convert8UC1ToMat(unsigned char* Img_Buf, int w, int h);
		};
	}
	
	namespace CV {
		class DEEP_API BaseIsOpencv
		{
		public:
			BaseIsOpencv();  // 생성자 추가 (객체 생성 필요)
			~BaseIsOpencv(); // 소멸자 추가

			Mat standTransImage(const Mat& input_image);
			double detectSpoidLength(const Mat& input_image);
			Mat correctImageRotation(const Mat& input_image);
			Mat straightenAmpoule(const Mat& input_image);
			
			void TEST1();
			void TEST2();
			void TEST3();

		private:

		};
	}

	namespace Utils {

		// class Utils : 각종 유틸리티 기능
		class DEEP_API Utils
		{
		public:
			void logMessagePrint(const std::string& message);
			double getCurrentTime();
			double logExecutionTime(const std::string& prefix, double startTime, double endTime);
			void runNvidiaSMI();
			bool checkProgramAdmin();

			std::string makeTime(const std::string& format = "yearmonday");
			void makeFolder(const std::string& path);
			void logError(const std::string& context, const std::string& message);
			void logErrorCustom(const std::string& message);
			void handleException(const std::exception& e, const std::string& context);

			bool isFileExists(const std::string& path);

			std::string pathSelectFolder();

			void TEST1();
			void TEST2();
			void TEST3();

		private:

		};

		// class ImageSaver : 이미지 저장 기능
		class DEEP_API ImageSaver
		{
		public:
			static ImageSaver& getInstance() {
				static ImageSaver instance;
				return instance;
			}

			void startImageSaveThread();
			void stopImageSaveThread();
			bool saveImageWithOpenCV(const std::string& filename, const cv::Mat& image);
			bool verifyFileSave(const std::string& filename, int maxRetries = 40, int sleepTime = 5);

		private:
			ImageSaver() = default;
			~ImageSaver() { stopImageSaveThread(); }

			ImageSaver(const ImageSaver&) = delete;
			ImageSaver& operator=(const ImageSaver&) = delete;
			ImageSaver(ImageSaver&&) = delete;
			ImageSaver& operator=(ImageSaver&&) = delete;

			void runImageSaveThread();

			std::atomic<bool> keepRunning{ true };
			std::thread imageSaveThread;
		};

		// class ImageSaver : 이미지 저장 기능
		class DEEP_API CSVLogSaver
		{
		public:
			static CSVLogSaver& getInstance() {
				static CSVLogSaver instance;
				return instance;
			}

			void startCSVLogSaveThread();
			void stopCSVLogSaveThread();
			
			// 검사 스레드에서 로그 데이터 추가
			void pushCSVLogEntry(const std::string& currentTime, const std::string& productName, int total, int ok, int ng);

		private:
			CSVLogSaver() = default;
			~CSVLogSaver() { stopCSVLogSaveThread(); }

			CSVLogSaver(const CSVLogSaver&) = delete;
			CSVLogSaver& operator=(const CSVLogSaver&) = delete;
			CSVLogSaver(CSVLogSaver&&) = delete;
			CSVLogSaver& operator=(CSVLogSaver&&) = delete;

			void runCSVLogSaveThread();
			bool writeCSVLogToFile(const std::string& line, const std::string& date);

			// 로그 데이터 구조
			struct LogEntry {
				std::string timestamp;
				std::string productName;
				int total;
				int good;
				int bad;
			};

			// 공유 리소스
			std::deque<LogEntry> logQueue;
			std::mutex queueMutex;
			std::condition_variable dataCV;

			std::atomic<bool> running{ false };
			std::thread logThread;
		};

		// class ConfigManager : 설정 값 관리 (Config.ini 파일 저장 및 로드)
		class DEEP_API ConfigManager
		{
		private:
			std::unordered_map<std::string, std::string> configs;

			ConfigManager() = default;
			~ConfigManager() = default;

			ConfigManager(const ConfigManager&) = delete;
			ConfigManager& operator=(const ConfigManager&) = delete;

		public:
			static ConfigManager& getInstance() {
				static ConfigManager instance;
				return instance;
			}

			void setSetting(const std::string& key, const std::string& value);
			std::string getSetting(const std::string& key, const std::string& defaultValue = "");
			void saveSettings();
			void loadSettings();
			void printAllSettings();
		};

		class DEEP_API SingleTimer {
		public:
			SingleTimer();
			~SingleTimer();

			void start(int interval_ms, std::function<void()> callback);
			void stop();
			bool isRunning() const;

		private:
			void timerLoop();

			std::thread worker;
			std::atomic<bool> running{false};
			int interval{0};
			std::function<void()> task;
		};


	}

}