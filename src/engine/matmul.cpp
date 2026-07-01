#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>
#include <cblas.h>

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

template <int M, int N, int P, 
          int MACRO_TILE = 256, 
          int MICRO_TILE = 32> // This matches your tile_size
void run_tiled_matmul(const float* mat1, const float* mat2, float* res2) {

    // Collapse the two outer macro loops across OpenMP threads
    #pragma omp parallel for collapse(2) num_threads(8)
    for(int tileI = 0; tileI < M; tileI += MACRO_TILE){
        for(int tileJ = 0; tileJ < P; tileJ += MACRO_TILE) {
            
            // Loop over K in blocks of MICRO_TILE
            for(int tileK = 0; tileK < N; tileK += MICRO_TILE) {
                
                // --- THE MICRO KERNEL ---
                int iEnd = std::min(M, tileI + MACRO_TILE);
                int jEnd = std::min(P, tileJ + MACRO_TILE);
                int kEnd = std::min(N, tileK + MICRO_TILE);
                float packedB[MICRO_TILE * MACRO_TILE];
                
                for(int k = tileK; k < kEnd; k++) {
                    int packed_row_offset = (k - tileK) * MACRO_TILE;
                    int mat2_row_offset = k * P;
                    
                    for(int j = tileJ; j < jEnd; j++) {
                        packedB[packed_row_offset + (j - tileJ)] = mat2[mat2_row_offset + j];
                    }
                }


                for(int i = tileI; i < iEnd; i++) {
                    
    
                    for(int k = tileK; k < kEnd; k++) {
                        
                        float mat1_val = mat1[i * N + k];
                        int packed_row_offset = (k - tileK) * MACRO_TILE;
                        int res_row_offset = i * P;

                        for(int j = tileJ; j < jEnd; j++) {
                            // res2[i * P + j] += mat1_val * mat2[k * P + j];
                            res2[res_row_offset + j] += mat1_val * packedB[packed_row_offset + (j - tileJ)];
                        }
                    }
                }
            }
        }
    }
}


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
    // int M = 4, N = 4, P = 4;
    // std::vector<float> mat1 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; // 4x4
    // std::vector<float> mat2 = {17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}; // 4x4
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int P = 4096;
    // int M = 4096, N = 4096, P = 4096;
    
    std::vector<float> mat1 = gen_matrix(M, N);
    std::vector<float> mat2 = gen_matrix(N, P);

    std::vector<float> res1(M * P, 0);
    std::vector<float> res2(M * P, 0);
    std::vector<float> res3(M * P, 0);


    auto start1 = std::chrono::steady_clock::now();
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        M,
        P,
        N,
        1.0f,
        mat1.data(),
        N,
        mat2.data(),
        P,
        0.0f,
        res1.data(),
        P
    );
    auto end1 = std::chrono::steady_clock::now();
    auto ns1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();


    // took 3320710669 ns for int M = 512, N = 1024, P = 1536;
    // auto start1 = std::chrono::steady_clock::now();
    // for(int i = 0; i < M; i ++) {
    //     for(int j = 0; j < P; j++) {
    //         for(int k = 0; k < N; k++) {
    //             res1[i * P + j] += mat1[i * N + k] * mat2[k * P + j];
    //         };
    //     };
    // };
    // auto end1 = std::chrono::steady_clock::now();
    // auto ns1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();

    // outer product method
    auto start2 = std::chrono::steady_clock::now();
    const int tile_size = 32;
    const int macro_tile_size = 256;

    auto tiledM = M / tile_size;
    auto tiledN = N / tile_size;
    auto tiledP = P / tile_size;

    // #pragma omp parallel for collapse(2) num_threads(8)
    // for(int tileI = 0; tileI < M; tileI += macro_tile_size){
    //     for(int tileJ = 0; tileJ < P; tileJ += macro_tile_size) {
    //         for(int tileK = 0; tileK < N; tileK+=tile_size) {
    //             auto iEnd = std::min(M, tileI + macro_tile_size);
    //             auto jEnd = std::min(P, tileJ + macro_tile_size);
    //             auto kEnd = std::min(N, tileK + tile_size);
    //             for(int i = tileI; i < iEnd; i++) {
    //                 for(int k = tileK; k < kEnd; k++) {
    //                     auto mat1_val = mat1[i * N + k];
    //                     for(int j = tileJ; j < jEnd; j++) {
    //                         res2[i * P + j] += mat1_val * mat2[k * P + j];
    //                     };
    //                 };
    //             };
    //         };
    //     };
    // };
    run_tiled_matmul<M, N, P>(mat1.data(), mat2.data(), res2.data());
    auto end2 = std::chrono::steady_clock::now();
    auto ns2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count();
    

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
    auto start3 = std::chrono::steady_clock::now();
    std::vector<float> mat3 = transpose(mat2, N, P);

    // tile the matrix, with transpose



    #pragma omp parallel for collapse(2)
    for(int i = 0; i < tiledM; i ++) {
        for(int j = 0; j < tiledP; j++) {
            for(int k = 0; k < tiledN; k++) {
                // i, j, k iterate through the tiles now instead of through every element in the matrix. for 4x4 matrix with tile_size 2, that means 4 tiles
                // i gives us the row of a tile in matrix A
                // j gives us the column of a tile in matrix B
                // k iterates through A's columns and through B's rows

                // auto tileA = i * tiledN + k;
                // auto tileB = k * tiledP + j;
                // std::cout << "Tile A:" << tileA << std::endl;
                // std::cout << "Tile B:" << tileB << std::endl;
    
                for(int ii = 0; ii < tile_size; ii++) {
                    for(int jj = 0; jj < tile_size; jj++) {
                        const int aBase = (i * tile_size + ii) * N + k * tile_size;
                        const int bBase = (j * tile_size + jj) * N + k * tile_size;
                        float sum = res3[(i * tile_size + ii) * P + (j * tile_size + jj)];
                        
                        // #pragma omp simd reduction(+:sum)
                        for(int kk = 0; kk < tile_size; kk++) {
                            // general formula for an element at row r and col c in a matrix with num_cols columns: r * num_cols + c
                            // for row, i * tile_size gives us the row the tile is in and then ii tells us which row within the tile
                            // so i * tile_size + ii -> row of element
                            // for col, k * tile_size + kk -> col of element

                            // for matrix B, j chooses the col and k chooses the row so
                            // col = j * tile_size + jj
                            // row = k * tile_size + kk
                            // index = row * P + col

                            // output will have P columns
                            // so r * P + c
                            // output has row of A and col of B

                            // std::cout << "A: " << (i*tile_size + ii) * N + (k*tile_size + kk) << "| B: " << (k*tile_size + kk) * P + (j*tile_size + jj) << std::endl;
                            // res3[(i*tile_size+ii) * P + (j*tile_size+jj)] += mat1[(i*tile_size + ii) * N + (k*tile_size + kk)] * mat2[(k*tile_size + kk) * P + (j*tile_size + jj)];
                            // sum += mat1[(i*tile_size + ii) * N + (k*tile_size + kk)] * mat3[(j*tile_size + jj) * N + (k*tile_size + kk)];
                            sum += mat1[aBase + kk] * mat3[bBase + kk];
                        }
                        res3[(i*tile_size+ii) * P + (j*tile_size+jj)] = sum;
                        // res3[row * P + column] = sum;
                    }
                }

                // for(int ii = 0; ii < tile_size; ii++) {
                //     for(int kk = 0; kk < tile_size; kk++) {
                //         float a = mat1[(i*tile_size + ii) * N + (k*tile_size + kk)];
                //         for(int jj = 0; jj < tile_size; jj++) {
                //             res3[(i*tile_size+ii) * P + (j*tile_size+jj)] += a * mat2[(k*tile_size + kk) * P + (j*tile_size + jj)];
                //         }
                //     }
                // }

                // std::cout << "-------------------------------\n";
            }
        }
    }
    auto end3 = std::chrono::steady_clock::now();
    auto ns3 = std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3).count();

    // for (const auto& x : res3) {
    //     std::cout << x << " ";
    // };
    // std::cout << std::endl;
    // std::cout << "Time taken matmul standard: " << ns1 << std::endl;

    // for (const auto& x : res2) {
    //     std::cout << x << " ";
    // };
    
    std::cout << std::endl;
    std::cout << "Time taken matmul cblas: " << ns1 << std::endl;

    std::cout << std::endl;
    std::cout << "Time taken matmul outerproduct: " << ns2 << std::endl;

    std::cout << std::endl;
    std::cout << "Time taken matmul tiled: " << ns3 << std::endl;
    
    const auto eps = 1e-5f;
    for (int i = 0; i < M * P; i++) {
        float diff = std::abs(res2[i] - res3[i]);
        float largest = std::max(std::abs(res2[i]), std::abs(res3[i]));
        
        // If numbers are very close to 0, use absolute error; otherwise, use relative error
        if (diff > eps && (diff / largest) > eps) {
            std::cout << "Results not equal at index " << i 
                    << " (res2: " << res2[i] << ", res3: " << res3[i] << ")" << std::endl;
            return 0;
    }
}
    std::cout << "Results equal" << std::endl;
    
    return 0;
}
