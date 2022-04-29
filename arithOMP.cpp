#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>  // for high_resolution_clock

using namespace std;

cv::Vec3b mult(cv::Vec3b source1, cv::Vec3b source2) {
    float b = float(source1[0]) / 255 * float(source2[0]) / 255;
    float g = float(source1[1]) / 255 * float(source2[1]) / 255;
    float r = float(source1[2]) / 255 * float(source2[2]) / 255;

    cv::Vec3b res = cv::Vec3b(uchar(b * 255), uchar(g * 255), uchar(r * 255));
    return res;
}

cv::Vec3b div(cv::Vec3b source1, cv::Vec3b source2) {
    float b = (float(source1[0]) / 255) / (float(source2[0]) / 255);
    float g = (float(source1[1]) / 255) / (float(source2[1]) / 255);
    float r = (float(source1[2]) / 255) / (float(source2[2]) / 255);

    cv::Vec3b res = cv::Vec3b(uchar(b * 255), uchar(g * 255), uchar(r * 255));
    return res;
}


int main(int argc, char** argv)
{
    cv::Mat_<cv::Vec3b> source1 = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat_<cv::Vec3b> source2 = cv::imread(argv[2], cv::IMREAD_COLOR);
    cv::Mat_<cv::Vec3b> destination(source1.rows, source1.cols);

    string opp(argv[3]);

    cout << "argc: " << argc << endl;

    cv::imshow("Source Image", source1);
    cv::imshow("Source Image 2", source2);

    auto begin = chrono::high_resolution_clock::now();

    const int iter = 10;
    #pragma omp parallel for
    for (int it = 0; it < iter; it++) {
        if (opp == "+") {
            #pragma omp parallel for
            for (int i = 0; i < source1.rows; i++)
                for (int j = 0; j < source1.cols; j++)
                    destination(i, j) = source1(i, j) + source2(i, j);
        }
        else if (opp == "-") {
            #pragma omp parallel for
            for (int i = 0; i < source1.rows; i++)
                for (int j = 0; j < source1.cols; j++)
                    destination(i, j) = source1(i, j) - source2(i, j);
        }
        else if (opp == "*") {
            #pragma omp parallel for
            for (int i = 0; i < source1.rows; i++)
                for (int j = 0; j < source1.cols; j++)
                    destination(i, j) = mult(source1(i, j), source2(i, j));
        }
        else {
            #pragma omp parallel for
            for (int i = 0; i < source1.rows; i++)
                for (int j = 0; j < source1.cols; j++)
                    destination(i, j) = div(source1(i, j), source2(i, j));
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
