#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>  // for high_resolution_clock
#include <omp.h>

using namespace std;

cv::Vec3b hor(int i, int j, cv::Mat_<cv::Vec3b> source, vector<float> vec) {
    cv::Vec3b res;
    int size = (vec.size() - 1) / 2;

    int kini = j - size;
    if (kini < 0) kini = 0;
    int kend = j + size;
    if (kend >= source.cols) kend = source.cols - 1;

    float total = 0;
    for (int k = kini; k < kend; k++) {
        res += source(i, k) * vec[k - kini];
        total += vec[k - kini];
    }
    res /= total;
    return res;
}

cv::Vec3b ver(int i, int j, cv::Mat_<cv::Vec3b> source, vector<float> vec) {
    cv::Vec3b res;
    int size = (vec.size() - 1) / 2;

    int kini = i - size;
    if (kini < 0) kini = 0;
    int kend = i + size;
    if (kend >= source.rows) kend = source.rows - 1;

    float total = 0;
    for (int k = kini; k < kend; k++) {
        res += source(k, j) * vec[k - kini];
        total += vec[k - kini];
    }
    res /= total;
    return res;
}

vector< vector<float> > getMatrix(int ksize, float sigma) {
    typedef vector< vector<float> > Matrix;
    typedef vector<float> Row;
    Matrix matrix;

    for (size_t i = 0; i < ksize; ++i) {
        Row row(ksize);
        matrix.push_back(row);
    }

    float r;
    float s = 2.0 * sigma * sigma;
    int val = (ksize - 1) / 2;

    // sum is for normalization
    float sum = 0.0;

    // generating 5x5 kernel
    for (int x = -val; x <= val; x++) {
        for (int y = -val; y <= val; y++) {
            r = sqrt(x * x + y * y);
            matrix[x + val][y + val] = (exp(-(r * r) / s)) / (3.14159265359 * s);
            sum += matrix[x + val][y + val];
        }
    }

    for (int i = 0; i < ksize; ++i)
        for (int j = 0; j < ksize; ++j)
            matrix[i][j] /= sum;
    return matrix;
}


int main(int argc, char** argv)
{
    cv::Mat_<cv::Vec3b> source = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat_<cv::Vec3b> tmp(source.rows, source.cols);
    cv::Mat_<cv::Vec3b> destination(source.rows, source.cols);
    string mode(argv[2]);

    int ksize = atoi(argv[2]);
    float sigma = atof(argv[3]);
    typedef vector< vector<float> > Matrix;
    typedef vector<float> Row;
    Matrix matrix = getMatrix(ksize, sigma);
    Row vect(ksize, 0.0);
    for (int i = 0; i < matrix.size(); i++) {
        for (int j = 0; j < matrix[0].size(); j++) {
            vect[i] += matrix[i][j];
        }
    }

    cv::imshow("Source Image", source);

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 50;
    #pragma omp parallel for
    for (int it = 0; it < iter; it++) {
        #pragma omp parallel for
        for (int i = 0; i < source.rows; i++) {
            #pragma omp parallel for
            for (int j = 0; j < source.cols; j++) {
                tmp(i, j) = hor(i, j, source, vect);
            }
        }
        #pragma omp parallel for
        for (int j = 0; j < source.cols; j++) {
            #pragma omp parallel for
            for (int i = 0; i < source.rows; i++) {
                destination(i, j) = ver(i, j, tmp, vect);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - begin;

    cv::imshow("Processed Image", destination);

    cout << "Total time: " << diff.count() << " s" << endl;
    cout << "Time for 1 iteration: " << diff.count() / iter << " s" << endl;
    cout << "IPS: " << iter / diff.count() << endl;

    cv::waitKey();
    return 0;
}
