#include <httplib.h>
#include <nlohmann/json.hpp>

#include <csignal>
#include <iostream>

using json = nlohmann::json;

httplib::Server svr;

void signal_handler(int sig) {
    if (sig == SIGINT || sig == SIGTERM) {
        std::cout << "\nReceived signal, shutting down gracefully...\n";
        svr.stop();
    }
}

int main() {
    // Log requests and responses
    svr.set_logger([](const auto &req, const auto &res) {
        std::cout << req.method << " " << req.path << " -> " << res.status << std::endl;
    });

    svr.Get("/health", [](const auto &, auto &res) {
        res.set_content(json{{"status", "ok"}}.dump(), "application/json");
    });


    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::cout << "Listening on http://127.0.0.1:8080" << std::endl;
    svr.listen("127.0.0.1", 8080);
}