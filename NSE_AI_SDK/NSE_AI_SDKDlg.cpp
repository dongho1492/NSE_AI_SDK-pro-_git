
// NSE_AI_SDKDlg.cpp: 구현 파일
//

#include "pch.h"
#include "framework.h"
#include "NSE_AI_SDK.h"
#include "NSE_AI_SDKDlg.h"
#include "afxdialogex.h"

#include <iostream>
#include <filesystem>
#include <vector>
#include <time.h>
#include <cmath>

#include <list>
#include <iostream>
#include <filesystem>
#include <vector>
#include <time.h>
#include <cmath>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

#include <ShObjIdl.h>  // CFolderPickerDialog

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


using namespace cv;
//namespace fs = std::filesystem;

std::string Input_image_dir;

std::vector<std::string> img_names;

void parse_image_names_from_directory()
{
	for (const auto& entry : std::filesystem::directory_iterator(Input_image_dir)) {
		auto& curpath = entry.path();
		std::string ext = curpath.extension().string();
		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
		if (ext == ".jpg" || ext == ".png" || ext == ".bmp")
			img_names.push_back(curpath.filename().string());
	}
}

std::string select_model_path() {
	std::string model_path;
	std::cout << "Please enter the Model path: ";
	std::getline(std::cin, model_path);

	return model_path;
}

std::string select_folder_path() {
	std::string folder_path;
	std::cout << "Please enter the folder path: ";
	std::getline(std::cin, folder_path);

	return folder_path;
}






// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CNSEAISDKDlg 대화 상자



CNSEAISDKDlg::CNSEAISDKDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_NSE_AI_SDK_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CNSEAISDKDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CNSEAISDKDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_BUTTON1, &CNSEAISDKDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CNSEAISDKDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CNSEAISDKDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CNSEAISDKDlg::OnBnClickedButton4)
	ON_BN_CLICKED(IDC_BUTTON5, &CNSEAISDKDlg::OnBnClickedButton5)
	ON_BN_CLICKED(IDC_BUTTON6, &CNSEAISDKDlg::OnBnClickedButton6)
	ON_BN_CLICKED(IDC_BUTTON7, &CNSEAISDKDlg::OnBnClickedButton7)
	ON_BN_CLICKED(IDC_BUTTON8, &CNSEAISDKDlg::OnBnClickedButton8)
	ON_BN_CLICKED(IDC_BUTTON9, &CNSEAISDKDlg::OnBnClickedButton9)
	ON_BN_CLICKED(IDC_BUTTON10, &CNSEAISDKDlg::OnBnClickedButton10)
	ON_BN_CLICKED(IDC_BUTTON11, &CNSEAISDKDlg::OnBnClickedButton11)
	ON_BN_CLICKED(IDC_BUTTON12, &CNSEAISDKDlg::OnBnClickedButton12)
	ON_BN_CLICKED(IDC_BUTTON13, &CNSEAISDKDlg::OnBnClickedButton13)
	ON_BN_CLICKED(IDC_BUTTON14, &CNSEAISDKDlg::OnBnClickedButton14)
	ON_BN_CLICKED(IDC_BUTTON15, &CNSEAISDKDlg::OnBnClickedButton15)
END_MESSAGE_MAP()


// CNSEAISDKDlg 메시지 처리기

BOOL CNSEAISDKDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.

	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}

void CNSEAISDKDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 애플리케이션의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CNSEAISDKDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CNSEAISDKDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CNSEAISDKDlg::OnBnClickedButton1()
{
	
}


void CNSEAISDKDlg::OnBnClickedButton2()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	
}


void CNSEAISDKDlg::OnBnClickedButton3()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

}


void CNSEAISDKDlg::OnBnClickedButton4()
{

}


void CNSEAISDKDlg::OnBnClickedButton5()
{

}


void CNSEAISDKDlg::OnBnClickedButton6()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	try {
		e_GpuCla = std::make_unique<NSE::AI::GpuCla>();  // 명시적으로 초기화
	} catch (const std::exception& e) {
		AfxMessageBox(CString("[DLL Error] GpuCla is Explicit initialization \n") + CString(e.what()));
	}

	try {
		int result = e_GpuCla->initialAIModel();
	} catch (const std::exception& e) {
		AfxMessageBox(CString("[DLL Error] AI Model Initialization \n"));
	}
}


void CNSEAISDKDlg::OnBnClickedButton7()
{
	
}


void CNSEAISDKDlg::OnBnClickedButton8()
{
	// 이미지 경로를 자동으로 입력받는 방식
	//selected_folder_path = "D:\\Intellisense_AI\\Image\\test\\";

	std::string selected_folder_path = e_Utils->pathSelectFolder();

	if (!selected_folder_path.empty() && selected_folder_path.back() != '\\')
		selected_folder_path += "\\";

	Input_image_dir = selected_folder_path;

	img_names.clear();
	parse_image_names_from_directory();

	double duration_sum = 0;

	for (int i = 0; i < img_names.size(); i++) {
		cv::Mat Image_test = cv::imread(Input_image_dir + img_names[i], cv::IMREAD_GRAYSCALE);
		std::cout << img_names[i] << std::endl;

		int Img_width = Image_test.cols;
		int Img_height = Image_test.rows;
		int ch = Image_test.channels();
		int imagesize = Img_width * Img_height * ch;

		// KDH 230323 : n차원 포인터 배열 선언 (Input Data의 형태)
		// KDH 230503 : g_Img_Buf에 input image 입력
		unsigned char* g_Img_Buf;
		g_Img_Buf = new unsigned char[Img_height * Img_width * ch];

		for (int i = 0; i < imagesize; i++)
		{
			g_Img_Buf[i] = 255;
		}

		// KDH 230503 : [테스트용]
		std::memcpy(g_Img_Buf, Image_test.data, imagesize);

		int result_infer;

		//double startTime = m_Utils.getCurrentTime();

		// KDH 230503 : Deep_Run_buf_ONNX 에 n차원 배열로 전송
		try {
			//float probThresh = 0.7f;
			result_infer = e_GpuCla->inferenceAIModel1D(g_Img_Buf, Img_width, Img_height/*, probThresh*/);
		}
		catch (const std::exception& e) {
			AfxMessageBox(CString("[DLL Error] AI Model is Inference \n") + CString(e.what()));
		}

		std::cout << "result_infer : " << result_infer << std::endl;

		//double endTime = m_Utils.getCurrentTime();

		/*double duration = endTime - startTime;
		duration = std::round(duration * 1000) / 1000.0;

		if (i >= 3) {
			duration_sum += duration;
		}*/

		delete[] g_Img_Buf;
	}

	std::ostringstream stream;
	stream.precision(3);
	stream << std::fixed << duration_sum / (img_names.size() - 3);

	std::cout << "[Avg] Infer Speed : " << stream.str() << " sec" << std::endl;
}

void CNSEAISDKDlg::startInferenceThreads() {
	stopThread = false;  // 쓰레드 실행 가능하도록 설정

	// 2개의 쓰레드 시작
	thread1 = std::thread(&CNSEAISDKDlg::inferenceLoop, this, 0);
	thread2 = std::thread(&CNSEAISDKDlg::inferenceLoop, this, 1);
}


void CNSEAISDKDlg::stopInferenceThreads() {
	stopThread = true;  // 쓰레드 종료 요청

	if (thread1.joinable()) {
		thread1.join();  // thread1 종료 대기
	}
	if (thread2.joinable()) {
		thread2.join();  // thread2 종료 대기
	}
}


void CNSEAISDKDlg::inferenceLoop(int threadIndex) {
	int i = threadIndex;  // 각 쓰레드가 다른 이미지부터 시작하도록 설정

	while (!stopThread) {
		//auto startLoop = std::chrono::high_resolution_clock::now();

		if (!img_names.empty()) {
			cv::Mat Image_test = cv::imread(Input_image_dir + img_names[i], cv::IMREAD_GRAYSCALE);
			std::cout << "Thread " << threadIndex << " processing: " << img_names[i] << std::endl;

			int Img_width = Image_test.cols;
			int Img_height = Image_test.rows;
			int ch = Image_test.channels();
			int imagesize = Img_width * Img_height * ch;

			unsigned char* g_Img_Buf = new unsigned char[imagesize];
			std::memcpy(g_Img_Buf, Image_test.data, imagesize);

			//double startTime = m_Utils.getCurrentTime();
			int result_infer = e_GpuCla->inferenceAIModel1D(g_Img_Buf, Img_width, Img_height, 0.5f);
			//double endTime = m_Utils.getCurrentTime();

			/*std::cout << "Thread " << threadIndex << " Result: " << result_infer
				<< " | Time: " << (endTime - startTime) << " sec" << std::endl;*/

			delete[] g_Img_Buf;

			// 이미지 순차 처리 (각 쓰레드는 서로 다른 인덱스를 가짐)
			i = (i + 2) % img_names.size();
		}

		//// 실행 시간이 10ms보다 짧으면 추가 대기
		//auto endLoop = std::chrono::high_resolution_clock::now();
		//auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(endLoop - startLoop).count();

		//int waitTime = 10 - elapsedTime;
		//if (waitTime > 0) {
		//	std::this_thread::sleep_for(std::chrono::milliseconds(waitTime));
		//}
	}
}


void CNSEAISDKDlg::OnBnClickedButton9()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	Input_image_dir = "D:\\Intellisense_AI\\Image\\test\\";
	img_names.clear();
	parse_image_names_from_directory();  // 이미지 파일 목록 가져오기

	startInferenceThreads();  // 1초마다 실행하는 쓰레드 시작
}


void CNSEAISDKDlg::OnBnClickedButton10()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	stopInferenceThreads();  // 쓰레드 종료
}


void CNSEAISDKDlg::OnBnClickedButton11()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	




}


void CNSEAISDKDlg::OnBnClickedButton12()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	std::string selected_folder_path = "D:\\Intellisense_AI\\Image\\test\\";

	Input_image_dir = selected_folder_path;

	img_names.clear();
	parse_image_names_from_directory();

	for (int i = 0; i < img_names.size(); i++) {
		cv::Mat image = cv::imread(Input_image_dir + img_names[i], cv::IMREAD_UNCHANGED);
		std::cout << img_names[i] << std::endl;

		double result_infer;

		try {
			Mat input_image = e_Opencv->standTransImage(image).clone();

			Mat input_image2 = e_Opencv->straightenAmpoule(input_image).clone();

			result_infer = e_Opencv->detectSpoidLength(input_image2);
		}
		catch (const std::exception& e) {
			AfxMessageBox(CString("[DLL Error] AI Model is Inference \n") + CString(e.what()));
		}

		std::cout << "result_infer : " << result_infer << std::endl;
	}

}


void CNSEAISDKDlg::OnBnClickedButton13()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	std::string fc_weight_path = "D:/Intellisense_AI/Model/fc_weights_cam.yml";

	// XceptionNet + Dense(2048→50→2) 기준
	int expected_channels = 2048;
	int expected_classes = 2;

	e_GpuCla->check_fc_weights_shape(fc_weight_path, expected_channels, expected_classes);

}


void CNSEAISDKDlg::OnBnClickedButton14()
{
	std::string selected_folder_path = e_Utils->pathSelectFolder();

	if (!selected_folder_path.empty() && selected_folder_path.back() != '\\')
		selected_folder_path += "\\";

	Input_image_dir = selected_folder_path;

	img_names.clear();
	parse_image_names_from_directory();

	for (int i = 0; i < img_names.size(); i++) {
		cv::Mat image = cv::imread(Input_image_dir + img_names[i], cv::IMREAD_UNCHANGED);
		std::cout << img_names[i] << std::endl;

		int inferResult;
		cv::Mat heatmapImage;

		try {
			Mat inputImage = e_Opencv->standTransImage(image).clone();

			inferResult = e_GpuCla->inferenceAIModelHeatmap(inputImage, heatmapImage, 0.5f);

		}
		catch (const std::exception& e) {
			AfxMessageBox(CString("[DLL Error] AI Model is Inference \n") + CString(e.what()));
		}
	}
}


void CNSEAISDKDlg::OnBnClickedButton15()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.



}
