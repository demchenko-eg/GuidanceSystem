cmake_minimum_required(VERSION 3.20)
project(GuidanceSystem CXX)

set(CMAKE_CXX_STANDARD 17)

# Знаходимо OpenCV
find_package(OpenCV REQUIRED)

# Вказуємо шляхи до бібліотек та заголовків Homebrew (для Apple Silicon)
link_directories(/opt/homebrew/lib)
include_directories(/opt/homebrew/include)

add_executable(GuidanceSystem main.cpp)

# Лінкуємо OpenCV та системну бібліотеку потоків
target_link_libraries(GuidanceSystem PRIVATE
        ${OpenCV_LIBS}
        pthread
)