// pch.cpp: 미리 컴파일된 헤더에 해당하는 소스 파일

#include "pch.h"

// 미리 컴파일된 헤더를 사용하는 경우 컴파일이 성공하려면 이 소스 파일이 필요합니다.

std::unique_ptr<NSE::AI::GpuCla> e_GpuCla;  // nullptr 초기화

std::unique_ptr<NSE::CV::BaseIsOpencv> e_Opencv;  // nullptr 초기화

std::unique_ptr<NSE::Utils::Utils> e_Utils;  // nullptr 초기화

//NSE::AI::GpuCla e_GpuCla;
