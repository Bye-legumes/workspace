#include <iostream>
#include <fstream>
#include <stdexcept>
#include <array>
#include <memory>
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
    if (commandOutput.empty()) {
        throw std::runtime_error("Default network interface not found.");
    }
    commandOutput.pop_back(); // Remove the newline character at the end
    return commandOutput;
}

double getMaxBandwidth() {
    std::string interface = getDefaultInterface();
    std::string filePath = "/sys/class/net/" + interface + "/speed";

    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filePath);
    }

    double speed;
    file >> speed;
    if (file.fail()) {
        throw std::runtime_error("Failed to read speed from file: " + filePath);
    }

    return speed; // Speed is in Mbps
}

int main() {
    try {
        double maxBandwidth = getMaxBandwidth();
        std::cout << "Maximum Bandwidth of " << getDefaultInterface() << ": " << maxBandwidth << " Mbps" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
