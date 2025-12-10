/*
 * Test case for CSV recording functionality in AllReduce plugin
 * This test simulates the environment and verifies CSV file generation
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <cstring>

// Mock the necessary components for testing
namespace tensorrt_llm {
namespace common {
    enum DataType {
        kFLOAT = 0,
        kHALF = 1,
        kINT32 = 2,
        kINT8 = 3
    };
    
    constexpr static size_t getDTypeSize(DataType type) {
        switch (type) {
            case kFLOAT: return 4;
            case kHALF: return 2;
            case kINT32: return 4;
            case kINT8: return 1;
            default: return 4;
        }
    }
}
}

// Function to create directory recursively
bool create_directory_recursive(const std::string& path) {
    size_t pos = 0;
    std::string dir;
    
    while ((pos = path.find('/', pos)) != std::string::npos) {
        dir = path.substr(0, pos++);
        if (dir.empty()) continue;
        
        if (mkdir(dir.c_str(), 0755) != 0 && errno != EEXIST) {
            std::cerr << "Failed to create directory: " << dir << " - " << strerror(errno) << std::endl;
            return false;
        }
    }
    
    if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
        std::cerr << "Failed to create directory: " << path << " - " << strerror(errno) << std::endl;
        return false;
    }
    
    return true;
}

// Function to check if file exists
bool file_exists(const std::string& filename) {
    std::ifstream file(filename);
    return file.good();
}

// Test function that mimics the CSV recording logic from allreducePlugin.cpp
void test_csv_recording(size_t size, tensorrt_llm::common::DataType mType) {
    // Calculate Allreduce input data size
    size_t data_size = size * tensorrt_llm::common::getDTypeSize(mType);
    
    // Get environment variables
    const char* batch_size_env = std::getenv("BATCH_SIZE");
    const char* sequence_length_env = std::getenv("SEQUENCE_LENGTH");
    const char* su_algo_env = std::getenv("SU_ALGO");
    const char* nccl_proto_env = std::getenv("NCCL_PROTO");
    
    std::cout << "Environment variables:" << std::endl;
    std::cout << "  BATCH_SIZE: " << (batch_size_env ? batch_size_env : "not set") << std::endl;
    std::cout << "  SEQUENCE_LENGTH: " << (sequence_length_env ? sequence_length_env : "not set") << std::endl;
    std::cout << "  SU_ALGO: " << (su_algo_env ? su_algo_env : "not set") << std::endl;
    std::cout << "  NCCL_PROTO: " << (nccl_proto_env ? nccl_proto_env : "not set") << std::endl;
    
    if (batch_size_env && sequence_length_env && su_algo_env) {
        std::string batch_size(batch_size_env);
        std::string sequence_length(sequence_length_env);
        std::string su_algo(su_algo_env);
        std::string nccl_proto = nccl_proto_env ? nccl_proto_env : "";
        
        // Create CSV filename
        std::stringstream csv_filename;
        csv_filename << "/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication/comm_" 
                     << batch_size << "_" << sequence_length << "_" << su_algo;
        if (!nccl_proto.empty()) {
            csv_filename << "_" << nccl_proto;
        }
        csv_filename << ".csv";
        
        std::cout << "Generating CSV file: " << csv_filename.str() << std::endl;
        
        // Create directory if it doesn't exist
        std::string dir_path = "/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication";
        if (!create_directory_recursive(dir_path)) {
            std::cerr << "Failed to create directory structure" << std::endl;
            return;
        }
        
        // Check if file exists to determine if we need to write headers
        bool exists = file_exists(csv_filename.str());
        
        // Open CSV file for appending
        std::ofstream csv_file(csv_filename.str(), std::ios::app);
        if (csv_file.is_open()) {
            // Write headers if file is new
            if (!exists) {
                csv_file << "Algorithm,Batch_size,Sequence_length,Communication\n";
                std::cout << "Added CSV headers" << std::endl;
            }
            
            // Write data row
            if (nccl_proto.empty()) {
                csv_file << su_algo << ",";
            } else {
                csv_file << su_algo << "_" << nccl_proto << ",";
            }
            csv_file << batch_size << "," 
                     << sequence_length << "," 
                     << data_size << "\n";
            
            csv_file.close();
            
            std::cout << "Successfully recorded data: ";
            if (nccl_proto.empty()) {
                std::cout << su_algo << ",";
            } else {
                std::cout << su_algo << "_" << nccl_proto << ",";
            }
            std::cout << batch_size << "," << sequence_length << "," << data_size << std::endl;
            
            // Verify file was created
            if (file_exists(csv_filename.str())) {
                std::cout << "✓ CSV file successfully created: " << csv_filename.str() << std::endl;
            } else {
                std::cerr << "✗ CSV file was not created!" << std::endl;
            }
        } else {
            std::cerr << "Failed to open CSV file for writing: " << csv_filename.str() << std::endl;
        }
    } else {
        std::cerr << "Missing required environment variables. Required: BATCH_SIZE, SEQUENCE_LENGTH, SU_ALGO" << std::endl;
    }
}

// Function to read and display CSV file contents
void display_csv_contents(const std::string& filename) {
    std::ifstream file(filename);
    if (file.is_open()) {
        std::cout << "\nContents of " << filename << ":" << std::endl;
        std::string line;
        int line_num = 0;
        while (std::getline(file, line)) {
            std::cout << "  Line " << ++line_num << ": " << line << std::endl;
        }
        file.close();
    } else {
        std::cerr << "Failed to open " << filename << " for reading" << std::endl;
    }
}

// Function to list all CSV files in the directory
void list_csv_files() {
    std::string dir_path = "/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication";
    std::cout << "\nListing CSV files in " << dir_path << ":" << std::endl;
    
    std::string command = "ls -la " + dir_path + "/*.csv 2>/dev/null || echo 'No CSV files found'";
    system(command.c_str());
}

int main() {
    std::cout << "=== AllReduce CSV Recording Test ===" << std::endl;
    std::cout << "Current working directory: " << getcwd(nullptr, 0) << std::endl;
    
    // Test Case 1: With NCCL_PROTO
    std::cout << "\n--- Test Case 1: With NCCL_PROTO ---" << std::endl;
    setenv("BATCH_SIZE", "32", 1);
    setenv("SEQUENCE_LENGTH", "512", 1);
    setenv("SU_ALGO", "NCCL", 1);
    setenv("NCCL_PROTO", "LL", 1);
    
    test_csv_recording(1024, tensorrt_llm::common::kFLOAT);  // 1024 elements * 4 bytes = 4096 bytes
    
    // Test Case 2: Without NCCL_PROTO (empty)
    std::cout << "\n--- Test Case 2: Without NCCL_PROTO ---" << std::endl;
    setenv("BATCH_SIZE", "64", 1);
    setenv("SEQUENCE_LENGTH", "256", 1);
    setenv("SU_ALGO", "ONESHOT", 1);
    unsetenv("NCCL_PROTO");  // Make NCCL_PROTO empty
    
    test_csv_recording(2048, tensorrt_llm::common::kHALF);  // 2048 elements * 2 bytes = 4096 bytes
    
    // Test Case 3: Different data types
    std::cout << "\n--- Test Case 3: Different data types ---" << std::endl;
    setenv("BATCH_SIZE", "16", 1);
    setenv("SEQUENCE_LENGTH", "1024", 1);
    setenv("SU_ALGO", "TWOSHOT", 1);
    setenv("NCCL_PROTO", "SIMPLE", 1);
    
    test_csv_recording(4096, tensorrt_llm::common::kINT32);  // 4096 elements * 4 bytes = 16384 bytes
    
    // List all generated CSV files
    list_csv_files();
    
    // Display generated CSV files
    std::cout << "\n--- Displaying Generated CSV Files ---" << std::endl;
    display_csv_contents("/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication/comm_32_512_NCCL_LL.csv");
    display_csv_contents("/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication/comm_64_256_ONESHOT.csv");
    display_csv_contents("/root/autodl-tmp/TensorRT-LLM/mybenchmark/results/communication/comm_16_1024_TWOSHOT_SIMPLE.csv");
    
    std::cout << "\n=== Test completed ===" << std::endl;
    
    return 0;
}