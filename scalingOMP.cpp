#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>  // for high_resolution_clock
#include <omp.h>

using namespace std;

float forcedCeil(float x) {
    if (int(x) == x) return x + 1.0;
    return ceil(x);
}


cv::Vec3b scaling(cv::Mat_<cv::Vec3b> source, float scaleFactor, int i, int j) {
    float X = i / scaleFactor;
    float Y = j / scaleFactor;
    cv::Vec3f TLVal = source(int(X), int(Y));
    cv::Vec3f TRVal = source(int(X + 1), int(Y + 1));
    cv::Vec3b res = TRVal * (X - int(X)) + TLVal * (int(X + 1) - X);
    return res;
}

int main(int argc, char** argv)
{
    cv::Mat_<cv::Vec3b> source = cv::imread(argv[1], cv::IMREAD_COLOR);

    int scaleFactor = atof(argv[2]);

    cv::Mat_<cv::Vec3b> destination(floor((source.rows - 1) * scaleFactor), floor((source.cols - 1) * scaleFactor));

    cv::imshow("Source Image", source);

    auto begin = chrono::high_resolution_clock::now();

    const int iter = 50;
    #pragma omp parallel for
    for (int it = 0; it < iter; it++) {
        #pragma omp parallel for
        for (int i = 0; i < destination.rows; i++) {
            #pragma omp parallel for
            for (int j = 0; j < destination.cols; j++) {
                destination(i, j) = scaling(source, scaleFactor, i, j);
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
