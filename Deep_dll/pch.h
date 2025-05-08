// pch.h: 미리 컴파일된 헤더 파일입니다.
// 아래 나열된 파일은 한 번만 컴파일되었으며, 향후 빌드에 대한 빌드 성능을 향상합니다.
// 코드 컴파일 및 여러 코드 검색 기능을 포함하여 IntelliSense 성능에도 영향을 미칩니다.
// 그러나 여기에 나열된 파일은 빌드 간 업데이트되는 경우 모두 다시 컴파일됩니다.
// 여기에 자주 업데이트할 파일을 추가하지 마세요. 그러면 성능이 저하됩니다.

#ifndef PCH_H
#define PCH_H

// 여기에 미리 컴파일하려는 헤더 추가
#include "framework.h"
#include <vector>
#include <tuple>
#include <string>
#include <opencv2/opencv.hpp>


////////////////////////////////////////////////////////////////////////////////////////////////
#define PROJECT_NAME    "NSE_AI_SDK(pro)"
#define VERSION_MAJOR   1
#define VERSION_MINOR   3
#define VERSION_PATCH   0

#define RELEASE_VERSION PROJECT_NAME " v" \
                        + std::to_string(VERSION_MAJOR) + "." \
                        + std::to_string(VERSION_MINOR) + "." \
                        + std::to_string(VERSION_PATCH)

#define BUILD_DATE      __DATE__
#define BUILD_TIME      __TIME__
////////////////////////////////////////////////////////////////////////////////////////////////


extern std::vector<std::tuple<std::string, cv::Mat>> e_vecAllInspImage;

extern const std::string e_pathDefault;
extern const std::string e_pathLog;
extern const std::string e_pathLogError;
extern const std::string e_pathLogRecord;
extern const std::string e_pathLogUsed;
extern const std::string e_pathSaveFolder;
extern const std::string e_pathConfigFile;

extern int e_configLogLevel;
extern int e_configExeEnv;
extern int e_configProccesMode;
extern int e_configImageSaveMode;
extern int e_configThresholdCla;
extern int e_configHeatmapCla;
extern int e_configCSVLogSaveMode;




#endif //PCH_H
