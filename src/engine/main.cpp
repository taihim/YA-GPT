#include <iostream>
#include <chrono>
#include <cblas.h>
#include <fstream>
#include <vector>
#include <array>
#include <cstdint>
#include <cmath>

#include "blahs.h"

struct Model {
    uint8_t version, has_bias;
    uint32_t layers, rows, cols;
    std::vector<float> W;
    float b;
};

Model load_model(const char* path) {
    std::ifstream f(path, std::ios::binary);
    std::array<char,4> magic{};
    f.read(magic.data(), 4);

    Model m{};
    f.read(reinterpret_cast<char*>(&m.version), 1);
    f.read(reinterpret_cast<char*>(&m.layers), 4);
    f.read(reinterpret_cast<char*>(&m.has_bias), 1);
    f.read(reinterpret_cast<char*>(&m.rows), 4);
    f.read(reinterpret_cast<char*>(&m.cols), 4);

    m.W.resize(static_cast<size_t>(m.rows) * m.cols);
    f.read(reinterpret_cast<char*>(m.W.data()), m.W.size() * sizeof(float));
    if (m.has_bias) f.read(reinterpret_cast<char*>(&m.b), sizeof(float));
    return m;
}

float sigmoid(float x){
    return 1.0f / (1.0f + std::exp(-x));
}


int main() {
    Model model = load_model("slp.tai");
    std::cout << model.rows << " " << model.cols << "\n";
    std::cout << model.W[0] << "\n";
    std::cout << model.W[1] << "\n";
    std::cout << model.b << "\n";
    
    int M = 2, N = model.rows, P = model.cols;
    double total_gflops_ops = (2.0 * M * N * P) / 1e9;

    // std::vector<float> mat1 = gen_matrix(M, N);
    // std::vector<float> mat2 = gen_matrix(N, P);

    std::vector<float> input{1.0, 1.0, 0.0, 0.0};
    
    std::vector<float> res(M * P, 0);
    std::vector<float> res2(M * P, 0);

    auto start = std::chrono::steady_clock::now();
    
    matmul(input.data(), model.W.data(), res.data(), M, N, P);

    for(int i = 0; i < M * P; i++) {
        res[i] = sigmoid(res[i] + model.b);
        std::cout << res[i] << std::endl;
    }

    
    auto end = std::chrono::steady_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double my_gflops = total_gflops_ops / (ns / 1000.0);

    // std::cout << std::endl;
    
    // auto start1 = std::chrono::steady_clock::now();
    // cblas_sgemm(
    //     CblasRowMajor,
    //     CblasNoTrans,
    //     CblasNoTrans,
    //     M,
    //     P,
    //     N,
    //     1.0f,
    //     input.data(),
    //     N,
    //     model.W.data(),
    //     P,
    //     0.0f,
    //     res2.data(),
    //     P
    // );

    // auto end1 = std::chrono::steady_clock::now();
    // auto ns1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    // double cblas_gflops = total_gflops_ops / (ns1 / 1000.0);

    // std::cout << "Time taken matmul outerproduct: " << ns << std::endl;
    // std::cout << "Time taken matmul cblas: " << ns1 << std::endl;

    // std::cout << "My Kernel Performance: " << my_gflops << " GFLOPS" << std::endl;
    // std::cout << "CBLAS Performance: " << cblas_gflops << " GFLOPS" << std::endl;
    // std::cout << std::endl;

    // const auto eps = 1e-5f;
    // for (int i = 0; i < M * P; i++) {
    //     float diff = std::abs(res2[i] - res[i]);
    //     float largest = std::max(std::abs(res2[i]), std::abs(res[i]));
        
    //     // If numbers are very close to 0, use absolute error; otherwise, use relative error
    //     if (diff > eps && (diff / largest) > eps) {
    //         std::cout << "Results not equal at index " << i 
    //                 << " (res1: " << res2[i] << ", res2: " << res[i] << ")" << std::endl;
    //         return 0;
    // }
    // }
    // std::cout << "Results equal" << std::endl;
}