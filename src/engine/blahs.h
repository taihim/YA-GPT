#pragma once
#include <vector>

std::vector<float> gen_matrix(const int rows, const int cols);
void matmul(const float* mat1, const float* mat2, float* res, int M, int N, int P);
