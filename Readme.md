# Guidance System Simulation v3.2 + Stable INS

Real-time computer vision and inertial navigation system prototype written in C++ using OpenCV and socket networking. The application simulates an optical seeker head or an automated targeting turret tracking a dynamic visual object while compensating for platform rotation via external IMU telemetry.

---

## Core Components

* **UDP Telemetry Receiver (`UdpSensorReceiver`):**
  Listens on UDP port 5555 on a dedicated background thread. Parses incoming orientation deltas and caches the latest readings inside a thread-safe container. Uses Winsock2 on Windows and POSIX sockets on macOS/Linux.
* **Optical Tracking Engine (`GuidanceSystem`):**
  Extracts the dominant color palette from a manually selected bounding box in HSV space. Computes a 2D Hue-Saturation reference histogram (16x16 bins) to validate candidate contours via the Bhattacharyya distance metric (threshold $\le 0.58$). Dynamically constrains search regions using a predictive bounding gate.
* **Inertial Target Extrapolation:**
  Maintains a 4D state vector (position and velocity) via `cv::KalmanFilter`. When visual acquisition fails due to occlusion or sensor movement, the system switches to inertial extrapolation, shifting predicted target coordinates using low-pass-filtered IMU deltas.
* **Flight Surface Closed-Loop Control (`PIDController`):**
  Runs two independent discrete PID controllers for yaw and pitch axes. Clamps integral windup to $[-100.0, 100.0]$ and converts positional errors into mechanical actuator displacement commands.

---

## Finite State Machine

| State | Condition | Visual Indicator |
| :--- | :--- | :--- |
| **`SEARCHING`** | No active lock. Awaiting target acquisition. | Yellow status label with green crosshair. |
| **`TRACKING`** | Visual contour verified against geometric criteria and 2D histogram. | Red target bounding box, lock center marker, yellow error vector. |
| **`INERTIAL`** | Visual lock dropped. Target position estimated using Kalman prediction and IMU deltas. | Orange target prediction circle and heading vector. |
| **`LOST`** | Extrapolated coordinates moved outside camera frame boundaries. | Reverts to search state. |

---

## Build Instructions

### Prerequisites
* C++17 compiler (Clang, Apple Clang, GCC, or MSVC)
* CMake 3.20 or newer
* OpenCV 4.x runtime and development headers (`core`, `highgui`, `imgproc`, `video`)

### Installing Dependencies

**macOS (Homebrew):**
```bash
brew install cmake opencv
```

**Ubuntu / Debian:**
```bash
sudo apt-get update && sudo apt-get install -y cmake g++ libopencv-dev
```

**Windows (vcpkg):**
```bash
vcpkg install opencv4:x64-windows
```

### Build & Run

```bash
git clone https://github.com/demchenko-eg/GuidanceSystem.git
cd GuidanceSystem
mkdir build && cd build
cmake ..
cmake --build . --config Release
./GuidanceSystem
```