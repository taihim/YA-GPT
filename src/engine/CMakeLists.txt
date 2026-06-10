cmake_minimum_required(VERSION 3.20)
project(translate-server CXX)
set(CMAKE_CXX_STANDARD 20)

include(FetchContent)

# cpp-httplib (HTTP server/client)
FetchContent_Declare(httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib
    GIT_TAG        master
)
FetchContent_MakeAvailable(httplib)

# nlohmann/json (JSON parser)
FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

add_executable(model-server src/main.cpp)

target_link_libraries(model-server PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
)