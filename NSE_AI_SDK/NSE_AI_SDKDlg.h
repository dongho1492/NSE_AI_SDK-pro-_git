
// NSE_AI_SDKDlg.h: 헤더 파일
//

#pragma once

#include <thread>
#include <atomic>
#include <condition_variable>

// CNSEAISDKDlg 대화 상자
class CNSEAISDKDlg : public CDialogEx
{
// 생성입니다.
public:
	CNSEAISDKDlg(CWnd* pParent = nullptr);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_NSE_AI_SDK_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

public:
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton4();
	afx_msg void OnBnClickedButton5();
	afx_msg void OnBnClickedButton6();

	//NSEAI::GpuCla GetGpuCla() { return m_GpuCla; };
	afx_msg void OnBnClickedButton7();
	afx_msg void OnBnClickedButton8();
	afx_msg void OnBnClickedButton9();
	afx_msg void OnBnClickedButton10();

	void startInferenceThreads();  // 2개 쓰레드 시작
	void stopInferenceThreads();   // 쓰레드 종료
	void inferenceLoop(int threadIndex);  // 쓰레드 실행 함수

private:
	std::thread thread1, thread2;  // 2개의 독립적인 쓰레드
	std::atomic_bool stopThread;   // 쓰레드 종료 플래그
public:
	afx_msg void OnBnClickedButton11();
	afx_msg void OnBnClickedButton12();
	afx_msg void OnBnClickedButton13();
	afx_msg void OnBnClickedButton14();
	afx_msg void OnBnClickedButton15();
};
