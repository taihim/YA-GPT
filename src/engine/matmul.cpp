#include <iostream>
#include <vector>
#include <chrono>
#include <random>
// #include <cblas.h>

// The key intuition is just one sentence:
// Element (i, j) of the result is the dot product of row i from A and column j from B.
// That's the entire algorithm. The three loops follow directly:
// - i selects the row from A
// - j selects the column from B
// - k walks across both simultaneously (across A's row, down B's column)


// if M x N and N x P matrices
// 1 2 3     7  8  9  10
// 4 5 6     11 12 13 14
//           15 16 17 18


// M x N * N x P -> M x P
// i -> M
// j -> P
// k-> N

// std::vector<float> sigmoid(std::vector<float> vector) {
    
// }

std::vector<float> transpose(const std::vector<float>& matrix, int rows, int cols) {
    std::vector<float> tposed(rows*cols, 0); 
    for(int j = 0; j < cols; j++) {
        for(int i = 0; i < rows; i++) {
            tposed[j * rows + i] = matrix[i * cols + j];
        }
    }
    return tposed;
}

std::vector<float> gen_matrix(const int rows, const int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<float> result(rows*cols, 0);

    for(float& x: result) x = dist(gen);

    return result;
}


int main() {
    int M = 4, N = 4, P = 4;
    std::vector<float> mat1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; // 4x4
    std::vector<float> mat2 = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}; // 4x4
    
    // int M = 64, N = 64, P = 64;
    
    // std::vector<float> mat1 = gen_matrix(M, N);
    // std::vector<float> mat2 = gen_matrix(N, P);

    std::vector<float> res1(M * P, 0);
    std::vector<float> res2(M * P, 0);
    std::vector<float> res3(M * P, 0);


    // auto start1 = std::chrono::steady_clock::now();
    // cblas_sgemm(
    //     CblasRowMajor,
    //     CblasNoTrans,
    //     CblasNoTrans,
    //     M,
    //     P,
    //     N,
    //     1.0f,
    //     mat1.data(),
    //     N,
    //     mat2.data(),
    //     P,
    //     0.0f,
    //     res1.data(),
    //     P
    // );
    // auto end1 = std::chrono::steady_clock::now();
    // auto ns1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count();


    // took 3320710669 ns for int M = 512, N = 1024, P = 1536;
    // auto start1 = std::chrono::steady_clock::now();
    // // M = 2 N = 3 P = 4
    // for(int i = 0; i < M; i ++) {
    //     for(int j = 0; j < P; j++) {
    //         for(int k = 0; k < N; k++) {
    //             res1[i * P + j] += mat1[i * N + k] * mat2[k * P + j];
    //         };
    //     };
    // };
    // auto end1 = std::chrono::steady_clock::now();
    // auto ns1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1).count();
    

    // took 2356132504ns for int M = 512, N = 1024, P = 1536;
    // auto start2 = std::chrono::steady_clock::now();
    // std::vector<float> mat3 = transpose(mat2, N, P);
    // // multiply with transposed matrix
    // for(int i = 0; i < M; i ++) {
    //     for(int j = 0; j < P; j++) {
    //         for(int k = 0; k < N; k++) {
    //             res2[i * P + j] += mat1[i * N + k] * mat3[j * N + k];
    //         };
    //     };
    // };
    // auto end2 = std::chrono::steady_clock::now();
    // auto ns2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start2).count();

    const int tile_size = 2;
    auto start3 = std::chrono::steady_clock::now();
    std::vector<float> mat4 = transpose(mat2, N, P);
    // tile the matrix, no transpose
    auto tiledM = M / tile_size;
    auto tiledN = N / tile_size;
    auto tiledP = P / tile_size;

    for(int i = 0; i < tiledM; i ++) {
        for(int j = 0; j < tiledP; j++) {
            for(int k = 0; k < tiledN; k++) {
                // auto tileA = i * tiledN + k;
                // auto tileB = k * tiledP + j;

                // std::cout << "Tile A:" << tileA << std::endl;
                // std::cout << "Tile B:" << tileB << std::endl;

                for(int ii = 0; ii < tile_size; ii++) {
                    for(int jj = 0; jj < tile_size; jj++) {
                        for(int kk = 0; kk < tile_size; kk++) {
                            // general formula for an element at row r and col c in a matrix with num_cols columns: r * num_cols + c
                            // for row, i * tile_size gives us the row the tile is in and then ii tells us which row within the tile
                            // so i * tile_size + ii -> row of element
                            // for col, k * tile_size + kk -> col of element
                            std::cout << "A: " << (i*tile_size + ii) * N + (k*tile_size + kk) << "| B: " << " " << std::endl;
                            // std::cout << "A: " << ( tileA * tile_size ) + ii*tile_size + kk << "| B: " << ((tileB * tile_size) + kk * tile_size + jj) << std::endl;
                            
                        }
                    }
                }
                std::cout << "-------------------------------\n";
            };
        };
    };
    auto end3 = std::chrono::steady_clock::now();
    auto ns3 = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - start3).count();

    // for (const auto& x : res1) {
    //     std::cout << x << " ";
    // };
    // std::cout << std::endl;
    // std::cout << "Time taken matmul standard: " << ns1 << std::endl;

    // for (const auto& x : res2) {
    //     std::cout << x << " ";
    // };
    
    // std::cout << std::endl;
    // std::cout << "Time taken matmul cblas: " << ns1 << std::endl;

    // std::cout << std::endl;
    // std::cout << "Time taken matmul transposed: " << ns2 << std::endl;

    // std::cout << std::endl;
    // std::cout << "Time taken matmul transposed + tiled: " << ns3 << std::endl;
    
    // const auto eps = 1e-5f;
    // for (int i = 0; i < M * P; i++) {
    //     if(abs(res1[i] - res3[i]) <= eps){
    //         continue;
    //     }
    //     std::cout << "Results not equal" << std::endl;
    //     return 0;
    // };
    // std::cout << "Results correct" << std::endl;
    
    return 0;
}



// Square matrix
// int main() {
//     std::cout << "Matrix Multiplication is nice" << std::endl;

//     // suppose 2x2 matrix stored in row-major order
//     std::vector<float> mat1 = {1, 2, 3, 4};
//     std::vector<float> mat2 = {5, 6, 7, 8};
//     std::vector<float> res(4, 0);
//     int rows = 2;
//     int cols = 2;

//     for(int i = 0; i < rows; i ++) {
//         for(int j = 0; j < cols; j++) {
//             float temp = 0.0;
//             for(int k = 0; k < cols; k++) {
//                 temp += mat1[i * cols + k] * mat2[k * cols + j];
//             };
//             res[i * cols + j] = temp;
//         };
//     };

//     for (const auto& x : res) {
//         std::cout << x << " ";
//     };

//     return 0;
// }

// 1 2 3 4
// 5 6 7 8

// 1 2  x  5 6
// 3 4  x  7 8

// 1 * 5 + 2 * 7    1 * 6 + 2 * 8
// 3 * 5 + 4 * 7    3 * 6 + 4 * 8

// 19 22
// 43 50

// (i * cols + j) * (j * cols + k)   |   (j * cols + k) * ()       

// 0 0  1 2   |   0 1  1 3
// 2 0  3 2   |   2 1  3 3

//0 1 0 1
//2 3 2 3


// 0 0 0 0 1 1 1 1
// 0 0 1 1 0 0 1 1
// 0 1 0 1 0 1 0 1

// 0 * 2 + 0 0 * 2 + 0       0 * 2 + 1  1 * 2 + 0
