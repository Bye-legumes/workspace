#include <iostream>
#include <memory>
#include <stdexcept>
#include <array>
#include <string>

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

double getMaxBandwidth() {
    std::string interface = getDefaultInterface();
    std::string command = "ethtool " + interface + " | grep Speed | awk '{print $2}'";
    std::string result = exec(command.c_str());

    if (result.empty()) {
        throw std::runtime_error("Failed to get network speed.");
    }

    // Assuming the result is something like "1000Mb/s"
    size_t mbpsPos = result.find("Mb/s");
    if (mbpsPos == std::string::npos) {
        throw std::runtime_error("Unexpected format of network speed.");
    }

    std::string speedStr = result.substr(0, mbpsPos);
    double speedMbps = std::stod(speedStr);
    return speedMbps;
}

int main() {
    try {
        double maxBandwidth = getMaxBandwidth();
        std::cout << "Maximum Bandwidth: " << maxBandwidth << " Mbps" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    return 0;
}
