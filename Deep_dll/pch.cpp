// pch.cpp: 미리 컴파일된 헤더에 해당하는 소스 파일

#include "pch.h"

// 미리 컴파일된 헤더를 사용하는 경우 컴파일이 성공하려면 이 소스 파일이 필요합니다.


std::vector<std::tuple<std::string, cv::Mat>> e_vecAllInspImage;

const std::string e_pathDefault = "D:\\Intellisense_AI";
const std::string e_pathLog = "D:\\Intellisense_AI\\Log";
const std::string e_pathLogError = "D:\\Intellisense_AI\\Log\\Error";
const std::string e_pathLogRecord = "D:\\Intellisense_AI\\Log\\Record";
const std::string e_pathLogUsed = "D:\\Intellisense_AI\\Log\\Used";
const std::string e_pathSaveFolder = "D:\\Intellisense_AI\\Save_image\\";
const std::string e_pathConfigFile = "D:\\Intellisense_AI\\Config\\config.ini";

int e_configLogLevel;
int e_configExeEnv;
int e_configProccesMode;
int e_configImageSaveMode;
int e_configThresholdCla;
int e_configHeatmapCla;
int e_configCSVLogSaveMode;