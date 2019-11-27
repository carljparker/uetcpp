#include <iostream>
#include <string>
#include <random>
#include <cmath>
#include <algorithm>
#include <vector>
#include <ctime>
#include <string>
#include <queue>
#include <fstream>
#include <sstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <cstdio>
#include <iomanip>
#include <Dense>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <unordered_set>

using Eigen::MatrixXi;
using Eigen::MatrixXd;
using namespace std;

int main() {
    Eigen::initParallel();
    int nrows = 150;
    int nTrees = 100000;
    int counter = 0;    

    #pragma omp parallel for reduction(+:counter)
    for (int i = 0; i < 100000; i++) {
        counter += 1;
    }
    cout << counter << endl;

    // MatrixXi matrix = MatrixXi::Zero(nrows, nrows);
    // #pragma omp critical
    // for (int i = 0; i < nTrees; i++) {
    //     MatrixXi a = MatrixXi::Ones(nrows,nrows);
    //     matrix += a;
    // }

    // // MatrixXd matrix2 = matrix/nTrees;
    // for (int i = 0; i < nrows; i++) {
    //     for (int j = 0; j < nrows; j++ ) {
    //         cout << matrix(i,j) << endl;
    //     }
    // }
    return 0;
}