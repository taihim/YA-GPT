#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <omp.h>


int main() {
    std::vector<float> a(3,4);

    for(float elem : a) {
        std::cout << elem << std::endl;
    }


    return 0;
}