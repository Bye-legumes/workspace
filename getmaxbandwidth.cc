#include <iostream>
#include <memory>
#include <stdexcept>
#include <array>
#include <string>
#include <chrono>
#include <thread>

std::string exec(const char* cmd) {
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
        result += buffer.data();
    }
    return result;
}

std::string getDefaultInterface() {
    std::string commandOutput = exec("ip route | grep default | awk '{print $5}'");
    return commandOutput;
}

long getInterfaceRxBytes(const std::string& interface) {
    std::string command = "cat /sys/class/net/" + interface + "/statistics/rx_bytes";
    std::string result = exec(command.c_str());
    return std::stol(result);
}

double GetMaxBandwidth() {
    std::string interface = getDefaultInterface();
    long initialBytes = getInterfaceRxBytes(interface);

    // Measure for a short period
    std::this_thread::sleep_for(std::chrono::seconds(1));

    long finalBytes = getInterfaceRxBytes(interface);
    long bytesPerSecond = finalBytes - initialBytes;

    // Convert bytes per second to megabits per second
    double bandwidthMbps = (bytesPerSecond * 8) / 1e6;
    return bandwidthMbps;
}

int main() {
    try {
        double bandwidth = GetMaxBandwidth();
        std::cout << "Estimated Max Bandwidth: " << bandwidth << " Mbps" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
