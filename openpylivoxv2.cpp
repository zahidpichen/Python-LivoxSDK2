#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>
#include <string>
#include <cstring>            // ← for strcmp
#include <iostream>
#include <fstream>
#include <atomic>

// Add cross-platform socket headers
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#endif

// Include Livox SDK headers
#include "Livox-SDK2/include/livox_lidar_api.h"
#include "Livox-SDK2/include/livox_lidar_def.h"

namespace py = pybind11;

// Global state
std::mutex device_mutex, connect_mutex, work_mode_mutex, point_cloud_mutex;
std::vector<LivoxLidarInfo> discovered_devices;
std::vector<uint32_t> connected_handles;
bool connect_success = false;
std::string connect_error_message;
bool work_mode_changed = false;
std::string work_mode_error_message;
std::atomic<bool> is_recording(false);
std::vector<std::vector<float>> point_cloud_buffer;
std::vector<uint8_t> reflectivity_buffer;
std::string recording_error_message;

// Global constant for the configuration file path
const std::string CONFIG_FILE_PATH = "./livox_config.json";

// Helper function to create the configuration file
void create_config_file(const std::string& host_ip = "192.168.1.10") {
    std::ofstream config_file(CONFIG_FILE_PATH);
    if (!config_file.is_open()) {
        throw std::runtime_error("Failed to create configuration file");
    }

    config_file << "{\n";
    config_file << "  \"lidar_log_path\": \"./\",\n";
    config_file << "  \"MID360\": {\n";
    config_file << "    \"lidar_net_info\": {\n";
    config_file << "      \"cmd_data_port\": 56100,\n";
    config_file << "      \"push_msg_port\": 56200,\n";
    config_file << "      \"point_data_port\": 56300,\n";
    config_file << "      \"imu_data_port\": 56400,\n";
    config_file << "      \"log_data_port\": 56500\n";
    config_file << "    },\n";
    config_file << "    \"host_net_info\": [\n";
    config_file << "      {\n";
    config_file << "        \"host_ip\": \"" << host_ip << "\",\n";
    config_file << "        \"cmd_data_port\": 56101,\n";
    config_file << "        \"push_msg_port\": 56201,\n";
    config_file << "        \"point_data_port\": 56301,\n";
    config_file << "        \"imu_data_port\": 56401,\n";
    config_file << "        \"log_data_port\": 56501\n";
    config_file << "      }\n";
    config_file << "    ]\n";
    config_file << "  }\n";
    config_file << "}\n";

    config_file.close();
}

// Callback function for device discovery
void DeviceInfoChangeCallback(uint32_t handle, const LivoxLidarInfo* info, void*) {
    if (!info) return;
    std::lock_guard<std::mutex> lk(device_mutex);
    discovered_devices.push_back(*info);
    std::cout << "Discovered LiDAR - Handle: " << handle
              << " Type: " << static_cast<int>(info->dev_type)
              << " SN: " << info->sn
              << " IP: " << info->lidar_ip << std::endl;
}


void WorkModeCallback(livox_status status, uint32_t, LivoxLidarAsyncControlResponse* res, void*) {
    std::lock_guard<std::mutex> lk(work_mode_mutex);
    if (status == kLivoxLidarStatusSuccess && res && res->ret_code == 0) {
        work_mode_changed = true;
    } else {
        work_mode_changed = false;
        work_mode_error_message = res
            ? "Failed to change work mode: Error code " + std::to_string(res->ret_code)
              + ", Error key " + std::to_string(res->error_key)
            : "Failed to change work mode: Unknown error";
    }
}


// Connection callback
void ConnectionCallback(uint32_t handle, const LivoxLidarInfo* info, void*) {
    if (!info) return;
    {
        std::lock_guard<std::mutex> lk(device_mutex);
        bool found = false;
        for (auto& dev : discovered_devices) {
            if (std::strcmp(dev.sn, info->sn) == 0) {
                found = true;
                break;
            }
        }
        if (!found) discovered_devices.push_back(*info);
    }
    {
        std::lock_guard<std::mutex> lk(connect_mutex);
        connected_handles.push_back(handle);
        connect_success = true;
    }
    // Start measurements
    SetLivoxLidarWorkMode(handle, kLivoxLidarNormal, WorkModeCallback, nullptr);
}

// Point-cloud callback
void PointCloudRecordCallback(uint32_t, uint8_t, LivoxLidarEthernetPacket* pkt, void*) {
    if (!pkt || !is_recording.load()) return;
    std::lock_guard<std::mutex> lk(point_cloud_mutex);
    if (pkt->data_type == kLivoxLidarCartesianCoordinateHighData) {
        auto *p = reinterpret_cast<LivoxLidarCartesianHighRawPoint*>(pkt->data);
        for (uint32_t i = 0; i < pkt->dot_num; i++) {
            std::vector<float> pt = {
                p[i].x / 1000.0f,
                p[i].y / 1000.0f,
                p[i].z / 1000.0f
            };
            point_cloud_buffer.push_back(pt);
            reflectivity_buffer.push_back(p[i].reflectivity);
        }
    } else if (pkt->data_type == kLivoxLidarCartesianCoordinateLowData) {
        auto *p = reinterpret_cast<LivoxLidarCartesianLowRawPoint*>(pkt->data);
        for (uint32_t i = 0; i < pkt->dot_num; i++) {
            std::vector<float> pt = {
                p[i].x / 100.0f,
                p[i].y / 100.0f,
                p[i].z / 100.0f
            };
            point_cloud_buffer.push_back(pt);
            reflectivity_buffer.push_back(p[i].reflectivity);
        }
    } else if (pkt->data_type == kLivoxLidarSphericalCoordinateData) {
        auto *p = reinterpret_cast<LivoxLidarSpherPoint*>(pkt->data);
        constexpr float to_rad = 3.14159265358979323846f / 180.0f / 100.0f;
        for (uint32_t i = 0; i < pkt->dot_num; i++) {
            float depth = p[i].depth / 1000.0f;
            float theta = p[i].theta * to_rad;
            float phi   = p[i].phi   * to_rad;
            std::vector<float> pt = {
                depth * std::sin(theta) * std::cos(phi),
                depth * std::sin(theta) * std::sin(phi),
                depth * std::cos(theta)
            };
            point_cloud_buffer.push_back(pt);
            reflectivity_buffer.push_back(p[i].reflectivity);
        }
    }
}

// Function to discover Livox LiDAR devices
py::list discover(std::string host_ip = "", int timeout_seconds = 5) {
    // Clear any previous discoveries
    {
        std::lock_guard<std::mutex> lock(device_mutex);
        discovered_devices.clear();
    }
    
    create_config_file(host_ip.empty() ? "192.168.1.10" : host_ip);
    std::cout << "Initializing SDK..." << std::endl;
    if (!LivoxLidarSdkInit(CONFIG_FILE_PATH.c_str())) {
        std::cerr << "Failed to initialize Livox SDK" << std::endl;
        throw std::runtime_error("Failed to initialize Livox SDK");
    }
    std::cout << "SDK initialized successfully." << std::endl;

    SetLivoxLidarInfoChangeCallback(DeviceInfoChangeCallback, nullptr);
    std::cout << "Callback registered." << std::endl;

    if (!LivoxLidarSdkStart()) {
        std::cerr << "Failed to start Livox SDK" << std::endl;
        LivoxLidarSdkUninit();
        throw std::runtime_error("Failed to start Livox SDK");
    }
    std::cout << "SDK started successfully." << std::endl;
    
    // Wait for the specified timeout period
    std::this_thread::sleep_for(std::chrono::seconds(timeout_seconds));
    
    // Prepare the Python list to return with discovery results
    py::list devices;
    {
        std::lock_guard<std::mutex> lock(device_mutex);
        for (const auto& device : discovered_devices) {
            // Convert each device to a Python dictionary
            py::dict device_info;
            device_info["dev_type"] = static_cast<int>(device.dev_type);
            device_info["sn"] = device.sn;
            device_info["ip"] = device.lidar_ip;
            
            // Get device type name
            std::string device_type_name;
            switch (device.dev_type) {
                case kLivoxLidarTypeHub:
                    device_type_name = "Hub";
                    break;
                case kLivoxLidarTypeMid40:
                    device_type_name = "Mid-40";
                    break;
                case kLivoxLidarTypeTele:
                    device_type_name = "Tele";
                    break;
                case kLivoxLidarTypeHorizon:
                    device_type_name = "Horizon";
                    break;
                case kLivoxLidarTypeMid70:
                    device_type_name = "Mid-70";
                    break;
                case kLivoxLidarTypeAvia:
                    device_type_name = "Avia";
                    break;
                case kLivoxLidarTypeMid360:
                    device_type_name = "Mid-360";
                    break;
                case kLivoxLidarTypeIndustrialHAP:
                    device_type_name = "Industrial HAP";
                    break;
                case kLivoxLidarTypeHAP:
                    device_type_name = "HAP";
                    break;
                case kLivoxLidarTypePA:
                    device_type_name = "PA";
                    break;
                default:
                    device_type_name = "Unknown";
                    break;
            }
            device_info["dev_type_name"] = device_type_name;
            
            devices.append(device_info);
        }
    }
    
    // Cleanup SDK resources
    LivoxLidarSdkUninit();
    
    return devices;
}

// Function to automatically connect to the first available Livox LiDAR
py::dict auto_connect(std::string host_ip = "", int timeout_seconds = 5) {
    // Clear previous connection information
    {
        std::lock_guard<std::mutex> lock(connect_mutex);
        connect_success = false;
        connect_error_message.clear();
        connected_handles.clear();
    }
    
    create_config_file(host_ip.empty() ? "192.168.1.10" : host_ip);
    std::cout << "Initializing SDK..." << std::endl;
    if (!LivoxLidarSdkInit(CONFIG_FILE_PATH.c_str())) {
        std::cerr << "Failed to initialize Livox SDK" << std::endl;
        throw std::runtime_error("Failed to initialize Livox SDK");
    }
    std::cout << "SDK initialized successfully." << std::endl;

    SetLivoxLidarInfoChangeCallback(ConnectionCallback, nullptr);
    std::cout << "Callback registered." << std::endl;

    if (!LivoxLidarSdkStart()) {
        std::cerr << "Failed to start Livox SDK" << std::endl;
        LivoxLidarSdkUninit();
        throw std::runtime_error("Failed to start Livox SDK");
    }
    std::cout << "SDK started successfully." << std::endl;
    
    // Wait for the specified timeout period for connection to complete
    std::this_thread::sleep_for(std::chrono::seconds(timeout_seconds));
    
    // Prepare result
    py::dict result;
    {
        std::lock_guard<std::mutex> lock(connect_mutex);
        result["success"] = connect_success;
        
        if (!connect_success) {
            result["error"] = connect_error_message.empty() ? "No device found or connection failed" : connect_error_message;
            LivoxLidarSdkUninit();
            return result;
        }
    }
    
    // Get the connected device info
    std::lock_guard<std::mutex> lock(device_mutex);
    if (discovered_devices.empty() || connected_handles.empty()) {
        result["success"] = false;
        result["error"] = "No device found or connected";
        LivoxLidarSdkUninit();
        return result;
    }
    
    // Return information about the connected device
    const auto& device = discovered_devices[0];
    result["dev_type"] = static_cast<int>(device.dev_type);
    result["sn"] = device.sn;
    result["ip"] = device.lidar_ip;
    result["handle"] = connected_handles[0];
    
    // Get device type name
    std::string device_type_name;
    switch (device.dev_type) {
        case kLivoxLidarTypeMid360:
            device_type_name = "Mid-360";
            break;
        case kLivoxLidarTypeHub:
            device_type_name = "Hub";
            break;
        case kLivoxLidarTypeMid40:
            device_type_name = "Mid-40";
            break;
        case kLivoxLidarTypeTele:
            device_type_name = "Tele";
            break;
        case kLivoxLidarTypeHorizon:
            device_type_name = "Horizon";
            break;
        case kLivoxLidarTypeMid70:
            device_type_name = "Mid-70";
            break;
        case kLivoxLidarTypeAvia:
            device_type_name = "Avia";
            break;
        case kLivoxLidarTypeIndustrialHAP:
            device_type_name = "Industrial HAP";
            break;
        case kLivoxLidarTypeHAP:
            device_type_name = "HAP";
            break;
        case kLivoxLidarTypePA:
            device_type_name = "PA";
            break;
        default:
            device_type_name = "Unknown";
            break;
    }
    result["dev_type_name"] = device_type_name;
    result["handle"] = connected_handles.empty() ? 0 : connected_handles[0];
    
    return result;
}

// Function to connect to a specific Livox LiDAR by IP address has been removed
// due to compilation errors in Ubuntu. Use auto_connect instead.

// Function to start the LiDAR (change work mode to Normal)
bool start_lidar(std::string host_ip = "", int timeout_seconds = 5) {
    // Reset work mode change status
    {
        std::lock_guard<std::mutex> lock(work_mode_mutex);
        work_mode_changed = false;
        work_mode_error_message.clear();
    }
    
    create_config_file(host_ip.empty() ? "192.168.1.10" : host_ip);
    std::cout << "Initializing SDK..." << std::endl;
    if (!LivoxLidarSdkInit(CONFIG_FILE_PATH.c_str())) {
        std::cerr << "Failed to initialize Livox SDK" << std::endl;
        work_mode_error_message = "Failed to initialize Livox SDK";
        return false;
    }
    std::cout << "SDK initialized successfully." << std::endl;

    SetLivoxLidarInfoChangeCallback(DeviceInfoChangeCallback, nullptr);
    std::cout << "Callback registered." << std::endl;

    if (!LivoxLidarSdkStart()) {
        std::cerr << "Failed to start Livox SDK" << std::endl;
        LivoxLidarSdkUninit();
        work_mode_error_message = "Failed to start Livox SDK";
        return false;
    }
    std::cout << "SDK started successfully." << std::endl;
    
    // Wait for device discovery
    std::this_thread::sleep_for(std::chrono::seconds(timeout_seconds));
    
    // Check if any devices were discovered
    {
        std::lock_guard<std::mutex> lock(device_mutex);
        if (discovered_devices.empty()) {
            LivoxLidarSdkUninit();
            work_mode_error_message = "No LiDAR devices discovered";
            return false;
        }
        
        // Get the handle of the first discovered device
        uint32_t handle = 0;
        for (const auto& handle_pair : connected_handles) {
            handle = handle_pair;
            break;
        }
        
        if (handle == 0) {
            for (const auto& device : discovered_devices) {
                // We don't have the handle directly from discovery, 
                // so we'll set the work mode for any detected device by IP
                struct in_addr addr;
                addr.s_addr = inet_addr(device.lidar_ip);
                handle = addr.s_addr;
                break;
            }
        }
        
        if (handle == 0) {
            LivoxLidarSdkUninit();
            work_mode_error_message = "Could not determine LiDAR device handle";
            return false;
        }
        
        // Set work mode to Normal
        SetLivoxLidarWorkMode(handle, kLivoxLidarNormal, WorkModeCallback, nullptr);
    }
    
    // Wait for the work mode change to complete
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    // Check if work mode change was successful
    {
        std::lock_guard<std::mutex> lock(work_mode_mutex);
        if (!work_mode_changed) {
            LivoxLidarSdkUninit();
            if (work_mode_error_message.empty()) {
                work_mode_error_message = "Failed to change LiDAR work mode to Normal";
            }
            return false;
        }
    }
    
    // Cleanup SDK resources
    LivoxLidarSdkUninit();
    
    return true;
}



// Function to stop the LiDAR (change work mode to Sleep)
bool stop_lidar(std::string host_ip = "", int timeout_seconds = 5) {
    // Reset work‐mode change state
    {
        std::lock_guard<std::mutex> lk(work_mode_mutex);
        work_mode_changed = false;
        work_mode_error_message.clear();
    }

    create_config_file(host_ip.empty() ? "192.168.1.10" : host_ip);
    std::cout << "Initializing SDK to send stop command..." << std::endl;
    if (!LivoxLidarSdkInit(CONFIG_FILE_PATH.c_str())) {
        std::cerr << "Failed to initialize Livox SDK" << std::endl;
        work_mode_error_message = "Failed to initialize Livox SDK";
        return false;
    }

    // Register the device discovery callback
    SetLivoxLidarInfoChangeCallback(DeviceInfoChangeCallback, nullptr);
    std::cout << "Registered callback for device discovery..." << std::endl;

    // Start the SDK to discover devices
    if (!LivoxLidarSdkStart()) {
        std::cerr << "Failed to start Livox SDK" << std::endl;
        LivoxLidarSdkUninit();
        work_mode_error_message = "Failed to start Livox SDK";
        return false;
    }
    std::cout << "SDK started successfully for device discovery..." << std::endl;

    // Wait for discovery
    std::cout << "Discovering LiDAR devices to stop..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(timeout_seconds));

    // Choose the first discovered device handle
    uint32_t handle = 0;
    {
        std::lock_guard<std::mutex> lk(device_mutex);
        if (discovered_devices.empty()) {
            LivoxLidarSdkUninit();
            work_mode_error_message = "No LiDAR devices discovered";
            return false;
        }
        // Prefer an existing connection handle, else use IP→uint32_t
        if (!connected_handles.empty()) {
            handle = connected_handles.front();
        } else {
            struct in_addr addr;
            addr.s_addr = inet_addr(discovered_devices.front().lidar_ip);
            handle = addr.s_addr;
        }
    }

    if (handle == 0) {
        LivoxLidarSdkUninit();
        work_mode_error_message = "Could not determine LiDAR device handle";
        return false;
    }

    // Command the LiDAR into Sleep mode (previously Standby)
    std::cout << "Sending command to stop LiDAR (Sleep mode)..." << std::endl;
    SetLivoxLidarWorkMode(handle, kLivoxLidarSleep, WorkModeCallback, nullptr);

    // Give it a moment to complete
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Check result
    {
        std::lock_guard<std::mutex> lk(work_mode_mutex);
        if (!work_mode_changed) {
            LivoxLidarSdkUninit();
            if (work_mode_error_message.empty()) {
                work_mode_error_message = "Failed to change LiDAR work mode to Sleep";
            }
            std::cout << "Failed to stop LiDAR: " << work_mode_error_message << std::endl;
            return false;
        }
    }

    std::cout << "LiDAR stopped successfully!" << std::endl;
    // Clean up the SDK
    LivoxLidarSdkUninit();
    return true;
}

// Function to start recording point cloud data
bool start_recording(std::string host_ip = "") {
    // Reset recording state
    {
        std::lock_guard<std::mutex> lock(point_cloud_mutex);
        point_cloud_buffer.clear();
        reflectivity_buffer.clear();
        recording_error_message.clear();
    }
    
    create_config_file(host_ip.empty() ? "192.168.1.10" : host_ip);
    std::cout << "Initializing SDK..." << std::endl;
    if (!LivoxLidarSdkInit(CONFIG_FILE_PATH.c_str())) {
        std::cerr << "Failed to initialize Livox SDK" << std::endl;
        recording_error_message = "Failed to initialize Livox SDK";
        return false;
    }
    std::cout << "SDK initialized successfully." << std::endl;

    // Register point cloud callback
    SetLivoxLidarPointCloudCallBack(PointCloudRecordCallback, nullptr);
    
    // Register device discovery callback to connect to the LiDAR
    SetLivoxLidarInfoChangeCallback(ConnectionCallback, nullptr);
    std::cout << "Callback registered." << std::endl;

    // Start the SDK
    if (!LivoxLidarSdkStart()) {
        std::cerr << "Failed to start Livox SDK" << std::endl;
        LivoxLidarSdkUninit();
        recording_error_message = "Failed to start Livox SDK";
        return false;
    }
    std::cout << "SDK started successfully." << std::endl;
    
    // Set the recording flag to true to start capturing points
    is_recording.store(true);
    
    return true;
}

// Function to stop recording point cloud data
bool stop_recording() {
    // Set the recording flag to false to stop capturing points
    is_recording.store(false);
    
    // Wait a moment to ensure any pending callbacks have completed
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Cleanup
    LivoxLidarSdkUninit();
    
    return true;
}

// Function to retrieve the recorded point cloud data
py::tuple get_point_cloud() {
    std::lock_guard<std::mutex> lock(point_cloud_mutex);
    
    // Create numpy arrays for the point cloud and reflectivity
    size_t num_points = point_cloud_buffer.size();
    
    // Create arrays with proper dimensions
    std::vector<pybind11::ssize_t> shape = {static_cast<pybind11::ssize_t>(num_points), 3};
    std::vector<pybind11::ssize_t> reflectivity_shape = {static_cast<pybind11::ssize_t>(num_points)};
    
    py::array_t<float> points_array(shape);
    py::array_t<uint8_t> reflectivity_array(reflectivity_shape);
    
    auto points_ptr = points_array.mutable_data();
    auto reflectivity_ptr = reflectivity_array.mutable_data();
    
    for (size_t i = 0; i < num_points; i++) {
        *points_ptr++ = point_cloud_buffer[i][0];
        *points_ptr++ = point_cloud_buffer[i][1];
        *points_ptr++ = point_cloud_buffer[i][2];
        *reflectivity_ptr++ = i < reflectivity_buffer.size() ? reflectivity_buffer[i] : 0;
    }
    
    // Return a tuple containing both arrays
    return py::make_tuple(points_array, reflectivity_array);
}

// Function to get recording status
bool is_recording_active() {
    return is_recording.load();
}

// Function to get the last recording error message
std::string get_recording_error() {
    return recording_error_message;
}

// Function to get the last error message from work mode change operations
std::string get_work_mode_error() {
    std::lock_guard<std::mutex> lock(work_mode_mutex);
    return work_mode_error_message;
}

// Function to initialize the SDK once
bool init_sdk(std::string host_ip = "") {
    create_config_file(host_ip.empty() ? "192.168.1.10" : host_ip);
    std::cout << "Initializing SDK..." << std::endl;
    if (!LivoxLidarSdkInit(CONFIG_FILE_PATH.c_str())) {
        std::cerr << "Failed to initialize Livox SDK" << std::endl;
        recording_error_message = "Failed to initialize Livox SDK";
        return false;
    }
    std::cout << "SDK initialized successfully." << std::endl;

    // Register point cloud callback
    SetLivoxLidarPointCloudCallBack(PointCloudRecordCallback, nullptr);
    
    // Register device discovery callback to connect to the LiDAR
    SetLivoxLidarInfoChangeCallback(ConnectionCallback, nullptr);
    std::cout << "Callbacks registered." << std::endl;

    // Start the SDK
    if (!LivoxLidarSdkStart()) {
        std::cerr << "Failed to start Livox SDK" << std::endl;
        LivoxLidarSdkUninit();
        recording_error_message = "Failed to start Livox SDK";
        return false;
    }
    std::cout << "SDK started successfully." << std::endl;
    return true;
}

// Function to start recording without initializing SDK again
bool start_recording_without_init() {
    // Reset point cloud buffers
    {
        std::lock_guard<std::mutex> lock(point_cloud_mutex);
        point_cloud_buffer.clear();
        reflectivity_buffer.clear();
        recording_error_message.clear();
    }
    
    // Set the recording flag to true to start capturing points
    is_recording.store(true);
    return true;
}

// Function to stop recording without uninitializing SDK
bool stop_recording_without_uninit() {
    // Set the recording flag to false to stop capturing points
    is_recording.store(false);
    
    // Wait a moment to ensure any pending callbacks have completed
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return true;
}

// Function to uninitialize the SDK
bool uninit_sdk() {
    LivoxLidarSdkUninit();
    return true;
}

// Define Python module
PYBIND11_MODULE(openpylivoxv2, m) {
    m.doc() = "Python wrapper for Livox SDK2";
    
    // Register discover function
    m.def("discover", &discover, 
          py::arg("host_ip") = "",
          py::arg("timeout_seconds") = 5,
          "Discover Livox LiDAR devices in the network.\n"
          "Args:\n"
          "    host_ip: Optional IP address of the host computer. If not provided, default will be used.\n"
          "    timeout_seconds: Time to wait for device discovery in seconds.\n"
          "Returns:\n"
          "    List of discovered devices with their information."
    );
    
    // Register auto_connect function
    m.def("auto_connect", &auto_connect,
          py::arg("host_ip") = "",
          py::arg("timeout_seconds") = 5,
          "Automatically connect to the first available Livox LiDAR device.\n"
          "Args:\n"
          "    host_ip: Optional IP address of the host computer. If not provided, default will be used.\n"
          "    timeout_seconds: Time to wait for connection in seconds.\n"
          "Returns:\n"
          "    Dictionary with connection status and device information if successful."
    );
      // Connect function has been removed due to compilation errors in Ubuntu.
    // Use auto_connect instead.
    
    // Register start_lidar function
    m.def("start_lidar", &start_lidar,
          py::arg("host_ip") = "",
          py::arg("timeout_seconds") = 5,
          "Start the LiDAR by setting its work mode to Normal.\n"
          "Args:\n"
          "    host_ip: Optional IP address of the host computer. If not provided, default will be used.\n"
          "    timeout_seconds: Time to wait for device discovery in seconds.\n"
          "Returns:\n"
          "    bool: True if successful, False otherwise."
    );
    
    // Register stop_lidar function
    m.def("stop_lidar", &stop_lidar,
          py::arg("host_ip") = "",
          py::arg("timeout_seconds") = 5,
          "Stop the LiDAR by setting its work mode to Standby.\n"
          "Args:\n"
          "    host_ip: Optional IP address of the host computer. If not provided, default will be used.\n"
          "    timeout_seconds: Time to wait for device discovery in seconds.\n"
          "Returns:\n"
          "    bool: True if successful, False otherwise."
    );
    
    // Register get_work_mode_error function
    m.def("get_work_mode_error", &get_work_mode_error,
          "Get the last error message from work mode change operations.\n"
          "Returns:\n"
          "    str: The error message."
    );
      // Register point cloud recording functions
    m.def("start_recording", &start_recording,
          py::arg("host_ip") = "",
          "Start recording point cloud data from the LiDAR.\n"
          "Args:\n"
          "    host_ip: Optional IP address of the host computer. If not provided, default will be used.\n"
          "Returns:\n"
          "    bool: True if recording started successfully, False otherwise."
    );
    
    m.def("stop_recording", &stop_recording,
          "Stop recording point cloud data from the LiDAR.\n"
          "Returns:\n"
          "    bool: True if recording stopped successfully, False otherwise."
    );
    
    m.def("get_point_cloud", &get_point_cloud,
          "Get the recorded point cloud data.\n"
          "Returns:\n"
          "    tuple: (points, reflectivity) where points is a numpy array of shape (n, 3) and reflectivity is a numpy array of shape (n,)."
    );
    
    m.def("is_recording_active", &is_recording_active,
          "Check if point cloud recording is active.\n"
          "Returns:\n"
          "    bool: True if recording is active, False otherwise."
    );
    
    m.def("get_recording_error", &get_recording_error,
          "Get the last error message from point cloud recording.\n"
          "Returns:\n"
          "    str: The error message."
    );
    
    // Register SDK initialization functions
    m.def("init_sdk", &init_sdk,
          py::arg("host_ip") = "",
          "Initialize the Livox SDK only once.\n"
          "Args:\n"
          "    host_ip: Optional IP address of the host computer. If not provided, default will be used.\n"
          "Returns:\n"
          "    bool: True if initialization was successful, False otherwise."
    );

    m.def("start_recording_without_init", &start_recording_without_init,
          "Start recording point cloud data without initializing the SDK again.\n"
          "Returns:\n"
          "    bool: True if recording started successfully, False otherwise."
    );

    m.def("stop_recording_without_uninit", &stop_recording_without_uninit,
          "Stop recording point cloud data without uninitializing the SDK.\n"
          "Returns:\n"
          "    bool: True if recording stopped successfully, False otherwise."
    );

    m.def("uninit_sdk", &uninit_sdk,
          "Uninitialize the Livox SDK.\n"
          "Returns:\n"
          "    bool: True if uninitialization was successful, False otherwise."
    );
}
