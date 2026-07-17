#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>

#include "blahs.h"

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

void matmul(const float* mat1, const float* mat2, float* res, const int M, const int N, const int P) {
    // if (M % 32 != 0) {
    //     return;
    // }
    
    if((M + N + P) < 4096) {
        for(int i = 0; i < M; i ++) {
            for(int j = 0; j < P; j++) {
                for(int k = 0; k < N; k++) {
                    res[i * P + j] += mat1[i * N + k] * mat2[k * P + j];
                }
            }
        }
    }
    else {
        const int tile_size = 32;
        const int macro_tile_size = 256;

        auto tiledM = M / tile_size;
        auto tiledN = N / tile_size;
        auto tiledP = P / tile_size;

        #pragma omp parallel for collapse(2) schedule(guided, 4)
        for (int tileI = 0; tileI < M; tileI += macro_tile_size) {
            for (int tileJ = 0; tileJ < P; tileJ += macro_tile_size) {
                for (int tileK = 0; tileK < N; tileK += tile_size) {
                    
                    int iEnd = std::min(M, tileI + macro_tile_size);
                    int jEnd = std::min(P, tileJ + macro_tile_size);
                    int kEnd = std::min(N, tileK + tile_size);
                    
                    // --- AVX2 Register Tiled Kernel (6 rows x 8 columns) ---
                    int i = tileI;
                    for (; i <= iEnd - 6; i += 6) {
                        int j = tileJ;
                        for (; j <= jEnd - 8; j += 8) {
                            
                            // 1. Load the 6x8 block of accumulation data into 256-bit registers
                            __m256 c0 = _mm256_loadu_ps(&res[(i + 0) * P + j]);
                            __m256 c1 = _mm256_loadu_ps(&res[(i + 1) * P + j]);
                            __m256 c2 = _mm256_loadu_ps(&res[(i + 2) * P + j]);
                            __m256 c3 = _mm256_loadu_ps(&res[(i + 3) * P + j]);
                            __m256 c4 = _mm256_loadu_ps(&res[(i + 4) * P + j]);
                            __m256 c5 = _mm256_loadu_ps(&res[(i + 5) * P + j]);

                            // 2. Inner K loop: Accumulate entirely inside registers
                            for (int k = tileK; k < kEnd; k++) {
                                // Only issue the prefetch instruction every 4 loops to avoid pipeline bloat
                                if ((k & 3) == 0) {
                                    _mm_prefetch((const char*)&mat2[(k + 4) * P + j], _MM_HINT_T0);
                                }
                                
                                // Load 8 elements from mat2 (Row k)
                                __m256 b = _mm256_loadu_ps(&mat2[k * P + j]);

                                // Broadcast 1 element from mat1 for each of the 6 rows
                                __m256 a0 = _mm256_set1_ps(mat1[(i + 0) * N + k]);
                                __m256 a1 = _mm256_set1_ps(mat1[(i + 1) * N + k]);
                                __m256 a2 = _mm256_set1_ps(mat1[(i + 2) * N + k]);
                                __m256 a3 = _mm256_set1_ps(mat1[(i + 3) * N + k]);
                                __m256 a4 = _mm256_set1_ps(mat1[(i + 4) * N + k]);
                                __m256 a5 = _mm256_set1_ps(mat1[(i + 5) * N + k]);

                                // AVX2 Fused Multiply-Add
                                c0 = _mm256_fmadd_ps(a0, b, c0);
                                c1 = _mm256_fmadd_ps(a1, b, c1);
                                c2 = _mm256_fmadd_ps(a2, b, c2);
                                c3 = _mm256_fmadd_ps(a3, b, c3);
                                c4 = _mm256_fmadd_ps(a4, b, c4);
                                c5 = _mm256_fmadd_ps(a5, b, c5);
                            }

                            // 3. Write back to memory ONCE
                            _mm256_storeu_ps(&res[(i + 0) * P + j], c0);
                            _mm256_storeu_ps(&res[(i + 1) * P + j], c1);
                            _mm256_storeu_ps(&res[(i + 2) * P + j], c2);
                            _mm256_storeu_ps(&res[(i + 3) * P + j], c3);
                            _mm256_storeu_ps(&res[(i + 4) * P + j], c4);
                            _mm256_storeu_ps(&res[(i + 5) * P + j], c5);
                        }

                        // Cleanup loop for J if it doesn't align to 8 elements
                        for (; j < jEnd; j++) {
                            for (int r = 0; r < 6; r++) {
                                float sum = 0.0f;
                                for (int k = tileK; k < kEnd; k++) {
                                    sum += mat1[(i + r) * N + k] * mat2[k * P + j];
                                }
                                res[(i + r) * P + j] += sum;
                            }
                        }
                    }

                    // Cleanup loop for I if it doesn't align to 6 elements
                    for (; i < iEnd; i++) {
                        for (int k = tileK; k < kEnd; k++) {
                            float mat1_val = mat1[i * N + k];
                            int j = tileJ;
                            for (; j <= jEnd - 8; j += 8) {
                                __m256 c = _mm256_loadu_ps(&res[i * P + j]);
                                __m256 b = _mm256_loadu_ps(&mat2[k * P + j]);
                                __m256 a = _mm256_set1_ps(mat1_val);
                                c = _mm256_fmadd_ps(a, b, c);
                                _mm256_storeu_ps(&res[i * P + j], c);
                            }
                            for (; j < jEnd; j++) {
                                res[i * P + j] += mat1_val * mat2[k * P + j];
                            }
                        }
                    }
                }
            }
        }
    }
}


    // for avx512
    // #pragma omp parallel for collapse(2)
    // for (int tileI = 0; tileI < M; tileI += macro_tile_size) {
    //     for (int tileJ = 0; tileJ < P; tileJ += macro_tile_size) {
    //         for (int tileK = 0; tileK < N; tileK += tile_size) {
                
    //             int iEnd = std::min(M, tileI + macro_tile_size);
    //             int jEnd = std::min(P, tileJ + macro_tile_size);
    //             int kEnd = std::min(N, tileK + tile_size);
                
    //             // --- Register Tiled Kernel (4 rows at a time, 16 columns at a time) ---
    //             int i = tileI;
    //             for (; i <= iEnd - 8; i += 8) {
    //                 int j = tileJ;
    //                 for (; j <= jEnd - 16; j += 16) {
                        
    //                     // 1. Load the 4x16 block of accumulation data into 512-bit registers
    //                     __m512 c0 = _mm512_loadu_ps(&res2[(i + 0) * P + j]);
    //                     __m512 c1 = _mm512_loadu_ps(&res2[(i + 1) * P + j]);
    //                     __m512 c2 = _mm512_loadu_ps(&res2[(i + 2) * P + j]);
    //                     __m512 c3 = _mm512_loadu_ps(&res2[(i + 3) * P + j]);

    //                     __m512 c4 = _mm512_loadu_ps(&res2[(i + 4) * P + j]);
    //                     __m512 c5 = _mm512_loadu_ps(&res2[(i + 5) * P + j]);
    //                     __m512 c6 = _mm512_loadu_ps(&res2[(i + 6) * P + j]);
    //                     __m512 c7 = _mm512_loadu_ps(&res2[(i + 7) * P + j]);

    //                     // 2. Inner K loop: Accumulate entirely inside registers
    //                     for (int k = tileK; k < kEnd; k++) {
    //                         // Load 16 elements from mat2 (Row k)
    //                         __m512 b = _mm512_loadu_ps(&mat2[k * P + j]);

    //                         // Broadcast 1 element from mat1 for each of the 4 rows
    //                         __m512 a0 = _mm512_set1_ps(mat1[(i + 0) * N + k]);
    //                         __m512 a1 = _mm512_set1_ps(mat1[(i + 1) * N + k]);
    //                         __m512 a2 = _mm512_set1_ps(mat1[(i + 2) * N + k]);
    //                         __m512 a3 = _mm512_set1_ps(mat1[(i + 3) * N + k]);

    //                         __m512 a4 = _mm512_set1_ps(mat1[(i + 4) * N + k]);
    //                         __m512 a5 = _mm512_set1_ps(mat1[(i + 5) * N + k]);
    //                         __m512 a6 = _mm512_set1_ps(mat1[(i + 6) * N + k]);
    //                         __m512 a7 = _mm512_set1_ps(mat1[(i + 7) * N + k]);

    //                         // Fused Multiply-Add: c = a * b + c
    //                         c0 = _mm512_fmadd_ps(a0, b, c0);
    //                         c1 = _mm512_fmadd_ps(a1, b, c1);
    //                         c2 = _mm512_fmadd_ps(a2, b, c2);
    //                         c3 = _mm512_fmadd_ps(a3, b, c3);
    //                         c4 = _mm512_fmadd_ps(a4, b, c4);
    //                         c5 = _mm512_fmadd_ps(a5, b, c5);
    //                         c6 = _mm512_fmadd_ps(a6, b, c6);
    //                         c7 = _mm512_fmadd_ps(a7, b, c7);
    //                     }

    //                     // 3. Write the finalized 4x16 block back to memory ONCE
    //                     _mm512_storeu_ps(&res2[(i + 0) * P + j], c0);
    //                     _mm512_storeu_ps(&res2[(i + 1) * P + j], c1);
    //                     _mm512_storeu_ps(&res2[(i + 2) * P + j], c2);
    //                     _mm512_storeu_ps(&res2[(i + 3) * P + j], c3);

    //                     _mm512_storeu_ps(&res2[(i + 4) * P + j], c4);
    //                     _mm512_storeu_ps(&res2[(i + 5) * P + j], c5);
    //                     _mm512_storeu_ps(&res2[(i + 6) * P + j], c6);
    //                     _mm512_storeu_ps(&res2[(i + 7) * P + j], c7);
    //                 }

    //                 // // Extra cleanup loop for J if macro_tile_size or P is not a multiple of 16
    //                 // for (; j < jEnd; j++) {
    //                 //     for (int r = 0; r < 4; r++) {
    //                 //         float sum = 0.0f;
    //                 //         for (int k = tileK; k < kEnd; k++) {
    //                 //             sum += mat1[(i + r) * N + k] * mat2[k * P + j];
    //                 //         }
    //                 //         res2[(i + r) * P + j] += sum;
    //                 //     }
    //                 // }
    //             }

    //             // // Extra cleanup loop for I if macro_tile_size or M is not a multiple of 4
    //             // for (; i < iEnd; i++) {
    //             //     for (int k = tileK; k < kEnd; k++) {
    //             //         float mat1_val = mat1[i * N + k];
    //             //         int j = tileJ;
    //             //         for (; j <= jEnd - 16; j += 16) {
    //             //             __m512 c = _mm512_loadu_ps(&res2[i * P + j]);
    //             //             __m512 b = _mm512_loadu_ps(&mat2[k * P + j]);
    //             //             __m512 a = _mm512_set1_ps(mat1_val);
    //             //             c = _mm512_fmadd_ps(a, b, c);
    //             //             _mm512_storeu_ps(&res2[i * P + j], c);
    //             //         }
    //             //         for (; j < jEnd; j++) {
    //             //             res2[i * P + j] += mat1_val * mat2[k * P + j];
    //             //         }
    //             //     }
    //             // }

    //         }
    //     }
    // }
