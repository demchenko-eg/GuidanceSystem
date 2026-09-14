#define NOMINMAX
#include <iostream>
#include <chrono>
#include <cmath>
#include <vector>
#include <thread>
#include <atomic>
#include <sstream>
#include <mutex>
#include <algorithm>
#include <cctype>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "Ws2_32.lib")
    using socket_t = SOCKET;
    #define CLOSE_SOCKET(s) closesocket(s)
#else
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    using socket_t = int;
    #define INVALID_SOCKET (-1)
    #define SOCKET_ERROR   (-1)
    #define CLOSE_SOCKET(s) close(s)
#endif

struct Config {
    int cameraIndex = 0;
    int frameWidth = 640;
    int frameHeight = 480;
    std::string windowName = "Guidance System Simulation v3.2 + Stable INS";
    int udpPort = 5555;
};

enum class SystemState {
    SEARCHING,
    TRACKING,
    INERTIAL,
    LOST
};

struct ImuData {
    float angleX = 0.0f;
    float angleY = 0.0f;
};

class UdpSensorReceiver {
private:
    socket_t sock = INVALID_SOCKET;
    std::thread rxThread;
    std::atomic<bool> isRunning{false};

    ImuData imuData;
    std::mutex dataMutex;

public:
    UdpSensorReceiver() = default;

    void rxLoop() {
        char buffer[1024];
        sockaddr_in clientAddr{};
        socklen_t clientLen = sizeof(clientAddr);

        while (isRunning) {
            int bytesReceived = recvfrom(sock, buffer, sizeof(buffer) - 1, 0, (sockaddr*)&clientAddr, &clientLen);

            if (bytesReceived > 0) {
                buffer[bytesReceived] = '\0';

                char* endPtr = nullptr;
                float x = strtof(buffer, &endPtr);
                if (endPtr != buffer && *endPtr == ',') {
                    float y = strtof(endPtr + 1, nullptr);

                    std::lock_guard<std::mutex> lock(dataMutex);
                    imuData.angleX = x;
                    imuData.angleY = y;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    bool start(int port) {
#ifdef _WIN32
        WSADATA wsaData;
        if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
            return false;
        }
#endif

        sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (sock == INVALID_SOCKET) {
#ifdef _WIN32
            WSACleanup();
#endif
            return false;
        }

#ifdef _WIN32
        DWORD timeout = 200;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
#else
        struct timeval timeout;
        timeout.tv_sec = 0;
        timeout.tv_usec = 200000;
        setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
#endif

        sockaddr_in serverAddr{};
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_port = htons(port);
        serverAddr.sin_addr.s_addr = INADDR_ANY;

        if (bind(sock, (sockaddr*)&serverAddr, sizeof(serverAddr)) == SOCKET_ERROR) {
            CLOSE_SOCKET(sock);
            sock = INVALID_SOCKET;
#ifdef _WIN32
            WSACleanup();
#endif
            return false;
        }

        isRunning = true;
        rxThread = std::thread(&UdpSensorReceiver::rxLoop, this);
        std::cout << "[INS] Sensor Fusion Server active on port " << port << std::endl;
        return true;
    }

    ImuData getImuData() {
        std::lock_guard<std::mutex> lock(dataMutex);
        return imuData;
    }

    void stop() {
        if (!isRunning) return;
        isRunning = false;

        if (sock != INVALID_SOCKET) {
            CLOSE_SOCKET(sock);
            sock = INVALID_SOCKET;
        }

        if (rxThread.joinable()) {
            rxThread.join();
        }
#ifdef _WIN32
        WSACleanup();
#endif
    }

    ~UdpSensorReceiver() { stop(); }
};

class PIDController {
private:
    double kp, ki, kd;
    double integral = 0.0;
    double prevError = 0.0;

public:
    PIDController(double p, double i, double d) : kp(p), ki(i), kd(d) {}

    double calculate(double error, double dt) {
        if (dt <= 0.001) dt = 0.001;
        if (dt > 0.1) dt = 0.1;

        double pOut = kp * error;
        integral += error * dt;

        if (integral > 100.0) integral = 100.0;
        if (integral < -100.0) integral = -100.0;

        double iOut = ki * integral;
        double derivative = (error - prevError) / dt;
        double dOut = kd * derivative;
        prevError = error;
        return pOut + iOut + dOut;
    }

    void reset() { integral = 0.0; prevError = 0.0; }
};

class Camera {
private:
    cv::VideoCapture cap;
    Config config;

public:
    explicit Camera(const Config& cfg) : config(cfg) {}

    bool initialize() {
        cap.open(config.cameraIndex, cv::CAP_ANY);
        if (!cap.isOpened()) return false;
        cap.set(cv::CAP_PROP_FRAME_WIDTH, config.frameWidth);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.frameHeight);
        return true;
    }

    bool captureFrame(cv::Mat& frame) { cap >> frame; return !frame.empty(); }
    void release() { if (cap.isOpened()) cap.release(); }
};

class GuidanceSystem {
private:
    SystemState state = SystemState::SEARCHING;
    cv::Rect targetBox;
    cv::Rect searchGatingBox;

    cv::Scalar targetHsvMin;
    cv::Scalar targetHsvMax;
    const int hTolerance = 15;
    const int sTolerance = 60;
    const int vTolerance = 70;

    cv::KalmanFilter kalmanFilter;
    bool kalmanInitialized = false;

    double baseTargetArea = 0.0;
    double lastValidArea = 0.0;
    double baseAspectRatio = 1.0;
    double baseFillRatio = 1.0;

    // 2D Гістограма еталона (Hue + Saturation)
    cv::Mat refHistogram;
    const int hBins = 16;
    const int sBins = 16;

    float prevAngleX = 0.0f;
    float prevAngleY = 0.0f;
    bool isFirstAngleReading = true;
    bool isFirstFrameAfterLock = true;

    float smoothedDx = 0.0f;
    float smoothedDy = 0.0f;
    const float lpfAlpha = 0.25f;

    PIDController pidX{0.6, 0.05, 0.1};
    PIDController pidY{0.6, 0.05, 0.1};
    double controlCommandX = 0.0;
    double controlCommandY = 0.0;

    void initKalman(float initX, float initY) {
        kalmanFilter.init(4, 2, 2);

        kalmanFilter.transitionMatrix = (cv::Mat_<float>(4, 4) <<
            1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);

        kalmanFilter.controlMatrix = (cv::Mat_<float>(4, 2) <<
            1, 0,
            0, 1,
            0, 0,
            0, 0);

        kalmanFilter.statePre.at<float>(0) = initX;
        kalmanFilter.statePre.at<float>(1) = initY;
        kalmanFilter.statePre.at<float>(2) = 0.0f;
        kalmanFilter.statePre.at<float>(3) = 0.0f;

        kalmanFilter.statePost.at<float>(0) = initX;
        kalmanFilter.statePost.at<float>(1) = initY;
        kalmanFilter.statePost.at<float>(2) = 0.0f;
        kalmanFilter.statePost.at<float>(3) = 0.0f;

        cv::setIdentity(kalmanFilter.measurementMatrix);
        cv::setIdentity(kalmanFilter.processNoiseCov, cv::Scalar::all(1e-4));
        cv::setIdentity(kalmanFilter.measurementNoiseCov, cv::Scalar::all(1e-2));
        cv::setIdentity(kalmanFilter.errorCovPost, cv::Scalar::all(0.1));

        kalmanInitialized = true;
    }

    void updateSearchGatingBox(cv::Point center, int frameWidth, int frameHeight) {
        int gateSize = 280;
        searchGatingBox = cv::Rect(center.x - gateSize / 2, center.y - gateSize / 2, gateSize, gateSize);
        searchGatingBox &= cv::Rect(0, 0, frameWidth, frameHeight);
    }

    // Модернізований розрахунок двовимірної HS гістограми
    cv::Mat calculateHsHistogram(const cv::Mat& hsvSrc, const cv::Mat& mask) {
        cv::Mat hist;
        int channels[] = {0, 1};
        int histSize[] = {hBins, sBins};
        float hRanges[] = {0, 180};
        float sRanges[] = {0, 256};
        const float* ranges[] = {hRanges, sRanges};

        cv::calcHist(&hsvSrc, 1, channels, mask, hist, 2, histSize, ranges, true, false);
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
        return hist;
    }

public:
    GuidanceSystem() = default;

    void setTarget(const cv::Rect& roi, const cv::Mat& currentFrame, float currentAngleX, float currentAngleY) {
        if (roi.width < 10 || roi.height < 10) return;

        cv::Rect safeRoi = roi & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);
        if (safeRoi.width < 5 || safeRoi.height < 5) return;

        int subW = std::max(6, safeRoi.width * 30 / 100);
        int subH = std::max(6, safeRoi.height * 30 / 100);
        int subX = safeRoi.x + (safeRoi.width - subW) / 2;
        int subY = safeRoi.y + (safeRoi.height - subH) / 2;
        cv::Rect coreRoi = cv::Rect(subX, subY, subW, subH) & cv::Rect(0, 0, currentFrame.cols, currentFrame.rows);

        if (coreRoi.area() <= 0) return;

        cv::Mat roiFrame = currentFrame(coreRoi);
        cv::Mat hsvRoi;
        cv::cvtColor(roiFrame, hsvRoi, cv::COLOR_BGR2HSV);
        cv::Scalar meanHsv = cv::mean(hsvRoi);

        targetHsvMin = cv::Scalar(std::max(0.0, meanHsv[0] - hTolerance), std::max(40.0, meanHsv[1] - sTolerance), std::max(40.0, meanHsv[2] - vTolerance));
        targetHsvMax = cv::Scalar(std::min(179.0, meanHsv[0] + hTolerance), std::min(255.0, meanHsv[1] + sTolerance), std::min(255.0, meanHsv[2] + vTolerance));

        cv::Mat fullHsv, tempMask;
        cv::cvtColor(currentFrame(safeRoi), fullHsv, cv::COLOR_BGR2HSV);
        cv::inRange(fullHsv, targetHsvMin, targetHsvMax, tempMask);

        std::vector<std::vector<cv::Point>> initialContours;
        cv::findContours(tempMask, initialContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double maxContourArea = 0.0;
        for (const auto& c : initialContours) {
            double a = cv::contourArea(c);
            if (a > maxContourArea) maxContourArea = a;
        }

        baseTargetArea = (maxContourArea > 10.0) ? maxContourArea : safeRoi.area() * 0.5;
        lastValidArea = baseTargetArea;
        baseAspectRatio = static_cast<double>(safeRoi.width) / safeRoi.height;
        baseFillRatio = baseTargetArea / safeRoi.area();

        // Генеруємо еталонну 2D HS карту об'єкта
        refHistogram = calculateHsHistogram(fullHsv, tempMask);

        targetBox = safeRoi;
        state = SystemState::TRACKING;
        isFirstFrameAfterLock = true;

        prevAngleX = currentAngleX;
        prevAngleY = currentAngleY;
        isFirstAngleReading = false;

        smoothedDx = 0.0f;
        smoothedDy = 0.0f;

        cv::Point initCenter(safeRoi.x + safeRoi.width / 2, safeRoi.y + safeRoi.height / 2);
        updateSearchGatingBox(initCenter, currentFrame.cols, currentFrame.rows);

        pidX.reset(); pidY.reset();
        initKalman(static_cast<float>(initCenter.x), static_cast<float>(initCenter.y));

        cv::Mat measurement = (cv::Mat_<float>(2, 1) << static_cast<float>(initCenter.x), static_cast<float>(initCenter.y));
        kalmanFilter.correct(measurement);
        std::cout << "[SYSTEM] Stable 2D lock established." << std::endl;
    }

    void processFrame(const cv::Mat& inputFrame, cv::Mat& outputFrame, double dt, float currentAngleX, float currentAngleY) {
        inputFrame.copyTo(outputFrame);

        int centerX = outputFrame.cols / 2;
        int centerY = outputFrame.rows / 2;
        cv::Point predictedCenter(-1, -1);
        cv::Point finalTargetPoint(centerX, centerY);
        float dx = 0.0f;
        float dy = 0.0f;

        if (isFirstAngleReading) {
            prevAngleX = currentAngleX;
            prevAngleY = currentAngleY;
            isFirstAngleReading = false;
        } else {
            float deltaX = currentAngleX - prevAngleX;
            float deltaY = currentAngleY - prevAngleY;

            if (std::abs(deltaY) > 0.0001f || std::abs(deltaX) > 0.0001f) {
                float rawDx = -deltaY * 1200.0f;
                float rawDy = deltaX * 1200.0f;

                smoothedDx = smoothedDx + lpfAlpha * (rawDx - smoothedDx);
                smoothedDy = smoothedDy + lpfAlpha * (rawDy - smoothedDy);

                dx = smoothedDx;
                dy = smoothedDy;
            } else {
                smoothedDx *= 0.85f;
                smoothedDy *= 0.85f;
                dx = smoothedDx;
                dy = smoothedDy;
            }
            prevAngleX = currentAngleX;
            prevAngleY = currentAngleY;
        }

        if (kalmanInitialized && (state == SystemState::TRACKING || state == SystemState::INERTIAL)) {
            cv::Mat prediction;
            if (isFirstFrameAfterLock) {
                cv::Mat zeroControl = (cv::Mat_<float>(2, 1) << 0.0f, 0.0f);
                prediction = kalmanFilter.predict(zeroControl);
                isFirstFrameAfterLock = false;
            } else {
                cv::Mat control = (cv::Mat_<float>(2, 1) << dx, dy);
                prediction = kalmanFilter.predict(control);

                if (state == SystemState::INERTIAL) {
                    kalmanFilter.statePost.at<float>(0) = prediction.at<float>(0);
                    kalmanFilter.statePost.at<float>(1) = prediction.at<float>(1);
                }
            }

            predictedCenter.x = static_cast<int>(prediction.at<float>(0));
            predictedCenter.y = static_cast<int>(prediction.at<float>(1));

            updateSearchGatingBox(predictedCenter, outputFrame.cols, outputFrame.rows);
        }

        bool objectFound = false;
        cv::Point detectedCenter(-1, -1);
        cv::Rect detectedBox;

        cv::Mat hsvFrame;
        cv::cvtColor(inputFrame, hsvFrame, cv::COLOR_BGR2HSV);
        cv::Mat fullMask;
        cv::inRange(hsvFrame, targetHsvMin, targetHsvMax, fullMask);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(fullMask, fullMask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(fullMask, fullMask, cv::MORPH_CLOSE, kernel);

        cv::Mat searchArea;
        if (state == SystemState::TRACKING && searchGatingBox.width > 0 && searchGatingBox.height > 0) {
            searchGatingBox &= cv::Rect(0, 0, fullMask.cols, fullMask.rows);
            searchArea = cv::Mat(fullMask, searchGatingBox);
        } else {
            searchArea = fullMask;
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(searchArea, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        double maxArea = 0;
        int maxIdx = -1;
        cv::Rect bestRect;

        double currentMinAreaThreshold = (state == SystemState::TRACKING) ? (lastValidArea * 0.35) : (baseTargetArea * 0.30);
        if (currentMinAreaThreshold < 20.0) currentMinAreaThreshold = 20.0;

        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > maxArea && area > currentMinAreaThreshold) {
                cv::Rect r = cv::boundingRect(contours[i]);
                if (state == SystemState::TRACKING) {
                    r = cv::Rect(r.x + searchGatingBox.x, r.y + searchGatingBox.y, r.width, r.height);
                }

                double aspRatio = static_cast<double>(r.width) / r.height;
                double aspRatioDiff = std::max(aspRatio / baseAspectRatio, baseAspectRatio / aspRatio);
                if (aspRatioDiff > 2.0) continue; // Збільшено допуск деформації геометричної форми

                double fillRatio = area / r.area();
                double fillRatioDiff = std::max(fillRatio / baseFillRatio, baseFillRatio / fillRatio);
                if (fillRatioDiff > 2.5) continue;

                cv::Rect clampedRect = r & cv::Rect(0, 0, hsvFrame.cols, hsvFrame.rows);
                if (clampedRect.area() <= 0) continue;

                cv::Mat candHsv = hsvFrame(clampedRect);
                cv::Mat candMask = fullMask(clampedRect);

                // Рахуємо нову 2D гістограму кандидата
                cv::Mat candHist = calculateHsHistogram(candHsv, candMask);

                double histMatch = cv::compareHist(refHistogram, candHist, cv::HISTCMP_BHATTACHARYYA);

                // ОПТИМІЗАЦІЯ: поріг піднято до 0.58.
                // Це припинить постійні зриви при нахилах чохла, але волосся все одно не пройде.
                if (histMatch > 0.58) {
                    continue;
                }

                maxArea = area;
                maxIdx = static_cast<int>(i);
                bestRect = r;
            }
        }

        if (maxIdx != -1) {
            bestRect &= cv::Rect(0, 0, outputFrame.cols, outputFrame.rows);
            cv::Moments mu = cv::moments(contours[maxIdx]);
            if (mu.m00 != 0) {
                int cx = static_cast<int>(mu.m10 / mu.m00);
                int cy = static_cast<int>(mu.m01 / mu.m00);

                if (state == SystemState::TRACKING) {
                    detectedCenter = cv::Point(cx + searchGatingBox.x, cy + searchGatingBox.y);
                } else {
                    detectedCenter = cv::Point(cx, cy);
                }
                detectedBox = bestRect;
                lastValidArea = maxArea;
                objectFound = true;
            }
        }

        if (objectFound) {
            cv::Mat measurement = (cv::Mat_<float>(2, 1) << static_cast<float>(detectedCenter.x), static_cast<float>(detectedCenter.y));
            kalmanFilter.correct(measurement);

            targetBox = detectedBox;
            finalTargetPoint = detectedCenter;
            updateSearchGatingBox(detectedCenter, outputFrame.cols, outputFrame.rows);

            cv::rectangle(outputFrame, targetBox, cv::Scalar(0, 0, 255), 2);
            cv::circle(outputFrame, detectedCenter, 4, cv::Scalar(0, 0, 255), -1);
            cv::line(outputFrame, cv::Point(centerX, centerY), detectedCenter, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);

            state = SystemState::TRACKING;
        } else if (state == SystemState::TRACKING) {
            state = SystemState::INERTIAL;
        }

        if (state == SystemState::INERTIAL) {
            if (predictedCenter.x < 0 || predictedCenter.x >= outputFrame.cols ||
                predictedCenter.y < 0 || predictedCenter.y >= outputFrame.rows) {
                state = SystemState::LOST;
            } else {
                finalTargetPoint = predictedCenter;

                cv::circle(outputFrame, predictedCenter, 10, cv::Scalar(0, 140, 255), 2);
                cv::line(outputFrame, predictedCenter + cv::Point(-10, 0), predictedCenter + cv::Point(10, 0), cv::Scalar(0, 140, 255), 1);
                cv::line(outputFrame, predictedCenter + cv::Point(0, -10), predictedCenter + cv::Point(0, 10), cv::Scalar(0, 140, 255), 1);
                cv::line(outputFrame, cv::Point(centerX, centerY), predictedCenter, cv::Scalar(0, 140, 255), 1, cv::LINE_8);
            }
        }

        if (state == SystemState::TRACKING || state == SystemState::INERTIAL) {
            double errorX = finalTargetPoint.x - centerX;
            double errorY = centerY - finalTargetPoint.y;
            controlCommandX = pidX.calculate(errorX, dt);
            controlCommandY = pidY.calculate(errorY, dt);
        } else {
            controlCommandX = 0.0; controlCommandY = 0.0;
        }

        int crosshairSize = 20;
        cv::line(outputFrame, cv::Point(centerX - crosshairSize, centerY), cv::Point(centerX + crosshairSize, centerY), cv::Scalar(0, 255, 0), 2);
        cv::line(outputFrame, cv::Point(centerX, centerY - crosshairSize), cv::Point(centerX, centerY + crosshairSize), cv::Scalar(0, 255, 0), 2);
    }

    SystemState getState() const { return state; }
    double getControlX() const { return controlCommandX; }
    double getControlY() const { return controlCommandY; }

    void reset() {
        state = SystemState::SEARCHING;
        kalmanInitialized = false;
        controlCommandX = 0.0;
        controlCommandY = 0.0;
        isFirstAngleReading = true;
        baseTargetArea = 0.0;
        lastValidArea = 0.0;
        refHistogram.release();
    }
};

cv::Point startPoint;
cv::Point endPoint;
bool isSelecting = false;
bool needsTargetUpdate = false;

void onMouse(int event, int x, int y, int flags, void* param) {
    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
            isSelecting = true;
            startPoint = cv::Point(x, y);
            endPoint = startPoint;
            break;
        case cv::EVENT_MOUSEMOVE:
            if (isSelecting) endPoint = cv::Point(x, y);
            break;
        case cv::EVENT_LBUTTONUP:
            if (isSelecting) {
                endPoint = cv::Point(x, y);
                isSelecting = false;
                needsTargetUpdate = true;
            }
            break;
    }
}

class Application {
private:
    Config config;
    Camera camera;
    GuidanceSystem guidance;
    UdpSensorReceiver imu;

    int frameCount = 0;
    double fps = 0.0;
    std::chrono::time_point<std::chrono::steady_clock> lastFrameTime;
    std::chrono::time_point<std::chrono::steady_clock> lastFpsUpdateTime;

public:
    Application() : camera(config) {
        lastFrameTime = std::chrono::steady_clock::now();
        lastFpsUpdateTime = lastFrameTime;
    }

    int run() {
        if (!camera.initialize()) {
            std::cerr << "[ERROR] Camera initialization failed." << std::endl;
            return -1;
        }

        if (!imu.start(config.udpPort)) {
            std::cerr << "[ERROR] UDP binding failed." << std::endl;
        }

        cv::namedWindow(config.windowName, cv::WINDOW_AUTOSIZE);
        cv::setMouseCallback(config.windowName, onMouse, nullptr);

        cv::Mat rawFrame, processedFrame;

        while (true) {
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> frameDuration = now - lastFrameTime;
            lastFrameTime = now;
            double dt = frameDuration.count();

            if (!camera.captureFrame(rawFrame)) break;

            ImuData currentImu = imu.getImuData();

            if (needsTargetUpdate) {
                cv::Rect box(std::min(startPoint.x, endPoint.x), std::min(startPoint.y, endPoint.y), std::abs(startPoint.x - endPoint.x), std::abs(startPoint.y - endPoint.y));
                guidance.setTarget(box, rawFrame, currentImu.angleX, currentImu.angleY);
                needsTargetUpdate = false;
            }

            guidance.processFrame(rawFrame, processedFrame, dt, currentImu.angleX, currentImu.angleY);

            if (isSelecting) {
                cv::Rect currentSelection(std::min(startPoint.x, endPoint.x), std::min(startPoint.y, endPoint.y), std::abs(startPoint.x - endPoint.x), std::abs(startPoint.y - endPoint.y));
                currentSelection &= cv::Rect(0, 0, processedFrame.cols, processedFrame.rows);
                if (currentSelection.width > 0 && currentSelection.height > 0) {
                    cv::rectangle(processedFrame, currentSelection, cv::Scalar(255, 0, 0), 2);
                }
            }

            frameCount++;
            std::chrono::duration<double> elapsedFpsTime = now - lastFpsUpdateTime;
            if (elapsedFpsTime.count() >= 1.0) {
                fps = frameCount / elapsedFpsTime.count();
                frameCount = 0;
                lastFpsUpdateTime = now;
            }

            std::string fpsText = "FPS: " + std::to_string(static_cast<int>(fps));
            cv::putText(processedFrame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            std::string statusText = "STATUS: ";
            cv::Scalar statusColor;
            switch (guidance.getState()) {
                case SystemState::SEARCHING:
                    statusText += "SEARCHING";
                    statusColor = cv::Scalar(0, 255, 255);
                    break;
                case SystemState::TRACKING:
                    statusText += "LOCK ON TARGET (OPTICAL)";
                    statusColor = cv::Scalar(0, 0, 255);
                    break;
                case SystemState::INERTIAL:
                    statusText += "INERTIAL (INS ACTIVE)";
                    statusColor = cv::Scalar(0, 140, 255);
                    break;
                case SystemState::LOST:
                    statusText += "SEARCHING (TARGET LOST)";
                    statusColor = cv::Scalar(0, 255, 255);
                    break;
            }
            cv::putText(processedFrame, statusText, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, statusColor, 2);

            std::string imuText = "INS MOTION VECTOR VALUE: " + std::to_string(currentImu.angleY);
            cv::putText(processedFrame, imuText, cv::Point(10, 140), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 145, 255), 1);

            std::string cmdX = "SURFACE COMMAND X (YAW): " + std::to_string(guidance.getControlX());
            std::string cmdY = "SURFACE COMMAND Y (PITCH): " + std::to_string(guidance.getControlY());
            cv::putText(processedFrame, cmdX, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
            cv::putText(processedFrame, cmdY, cv::Point(10, 110), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            cv::putText(processedFrame, "R: Reset | ESC: Exit", cv::Point(10, processedFrame.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

            cv::imshow(config.windowName, processedFrame);

            char key = static_cast<char>(cv::waitKey(1));
            if (key == 27) break;
            if (key == 'r' || key == 'R') guidance.reset();
        }
        imu.stop();
        camera.release();
        cv::destroyAllWindows();
        return 0;
    }
};

int main() {
    Application app;
    return app.run();
}