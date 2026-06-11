#include <iostream>
#include <vector>

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

std::vector<float> transpose(const std::vector<float>& matrix, int rows, int cols) {
    std::vector<float> tposed(rows*cols, 0); 
    for(int j = 0; j < cols; j++) {
        for(int i = 0; i < rows; i++) {
            tposed[j * rows + i] = matrix[i * cols + j];
        }
    }
    return tposed;
}


int main() {
    std::cout << "Matrix Multiplication is nice" << std::endl;

    std::vector<float> mat1 = {1, 2, 3, 4, 5, 6}; // 2x3
    std::vector<float> mat2 = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}; // 3x4
    int M = 2, N = 3, P = 4;
    std::vector<float> mat3 = transpose(mat2, N, P);
    
    std::vector<float> res1(M * P, 0);
    std::vector<float> res2(M * P, 0);

    // std::vector<float> res = transpose(mat2, N, P);
    
    // M = 2 N = 3 P = 4
    for(int i = 0; i < M; i ++) {
        for(int j = 0; j < P; j++) {
            float temp = 0.0;
            for(int k = 0; k < N; k++) {
                temp += mat1[i * N + k] * mat2[k * P + j];
            };
            res1[i * P + j] = temp;
        };
    };
    
    // multiply with transposed matrix
    for(int i = 0; i < M; i ++) {
        for(int j = 0; j < P; j++) {
            float temp = 0.0;
            for(int k = 0; k < N; k++) {
                temp += mat1[i * N + k] * mat3[j * N + k];
            };
            res2[i * P + j] = temp;
        };
    };


    for (const auto& x : res1) {
        std::cout << x << " ";
    };

    for (const auto& x : res2) {
        std::cout << x << " ";
    };

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
