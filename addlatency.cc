void ObjectManager::SendPullRequest(const ObjectID &object_id, const NodeID &client_id) {
    auto start_time = std::chrono::steady_clock::now();
    int timeout_ms = 10000; // 10 seconds timeout

    while (true) {
        double networkUsage = GetCurrentNetworkUsage(); // Placeholder for actual network usage check
        if (networkUsage < 95.0) {
            // Network usage is below threshold, proceed with pull request
            // Existing implementation to send pull request...
            break;
        } else {
            // Check if timeout has been reached
            auto current_time = std::chrono::steady_clock::now();
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();
            if (elapsed_ms > timeout_ms) {
                // Timeout reached, break the loop
                break;
            }

            // Network usage is too high, wait before retrying
            std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait for 100 milliseconds before retrying
        }
    }
}
