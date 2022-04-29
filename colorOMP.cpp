#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <math.h>
#include <chrono>  // for high_resolution_clock
#include <omp.h>

using namespace std;

cv::Vec3b colorTransformation(cv::Vec3b RGB, float rad) {
    float var_R = (float(RGB[2]) / 255);
    float var_G = (float(RGB[1]) / 255);
    float var_B = (float(RGB[0]) / 255);

    //cout << "Original RGB: " << float(RGB[2]) << " : " << float(RGB[1]) << " : " << float(RGB[0]);

    if (var_R > 0.04045) var_R = pow(((var_R + 0.055) / 1.055), 2.4);
    else var_R = var_R / 12.92;
    if (var_G > 0.04045) var_G = pow(((var_G + 0.055) / 1.055), 2.4);
    else var_G = var_G / 12.92;
    if (var_B > 0.04045) var_B = pow(((var_B + 0.055) / 1.055), 2.4);
    else var_B = var_B / 12.92;

    var_R *= 100;
    var_G *= 100;
    var_B *= 100;

    float XYZ[3];

    //Observer = 2°, Illuminant = D65
    XYZ[0] = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805;
    XYZ[1] = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722;
    XYZ[2] = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505;

    //cout << "Original XYZ: " << XYZ[0] << " : " << XYZ[1] << " : " << XYZ[2] << endl;

    XYZ[0] = float(XYZ[0]) / 95.047;
    XYZ[1] = float(XYZ[1]) / 100.0;
    XYZ[2] = float(XYZ[2]) / 108.883;

    for (int i = 0; i < 3; ++i) {
        float value = XYZ[i];
        if (value > 0.008856) {
            value = pow(value, 0.3333333333333333);
        }
        else {
            value = (7.787 * value) + (16 / 116);
        }
        XYZ[i] = value;
    }

    float L = (116 * XYZ[1]) - 16;
    float a = 500 * (XYZ[0] - XYZ[1]);
    float b = 200 * (XYZ[1] - XYZ[2]);

    //cout << "Original Lab: " << L << " : " << a << " : " << b << endl;

    float C = sqrt((a * a) + (b * b));
    float hub = atan(b / a);

    float na = cos(hub + rad) * C;
    float nb = sin(hub + rad) * C;

    //cout << "C: " << C << " hub: " << hub << endl;
    //cout << "na: " <<  na << " nb: " << nb << " hub: "<< hub + rad << endl;

    float nY = (L + 16) / 116;
    float nX = (na / 500) + nY;
    float nZ = nY - (nb / 200);

    if (pow(nY, 3) > 0.008856) nY = pow(nY, 3);
    else                       nY = (nY - (16 / 116)) / 7.787;
    if (pow(nX, 3) > 0.008856) nX = pow(nX, 3);
    else                       nX = (nX - (16 / 116)) / 7.787;
    if (pow(nZ, 3) > 0.008856) nZ = pow(nZ, 3);
    else                       nZ = (nZ - (16 / 116)) / 7.787;

    nX *= 95.047;
    nY *= 100.0;
    nZ *= 108.883;

    //cout << "Final XYZ: " << nX << " : " << nY << " : " << nZ << endl;

    float nR = (nX / 100) * 3.2406 + (nY / 100) * -1.5372 + (nZ / 100) * -0.4986;
    float nG = (nX / 100) * -0.9689 + (nY / 100) * 1.8758 + (nZ / 100) * 0.0415;
    float nB = (nX / 100) * 0.0557 + (nY / 100) * -0.2040 + (nZ / 100) * 1.0570;

    if (nR > 0.0031308) nR = 1.055 * (pow(nR, (1 / 2.4))) - 0.055;
    else                nR = 12.92 * nR;
    if (nG > 0.0031308) nG = 1.055 * (pow(nG, (1 / 2.4))) - 0.055;
    else                nG = 12.92 * nG;
    if (nB > 0.0031308) nB = 1.055 * (pow(nB, (1 / 2.4))) - 0.055;
    else                nB = 12.92 * nB;

    cv::Vec3b res = cv::Vec3b(uchar(nB * 255), uchar(nG * 255), uchar(nR * 255));
    //cout << res << endl;
    return res;
}

int main(int argc, char** argv)
{
    float M_PI = 3.14159265358979323846;
    cv::Mat_<cv::Vec3b> source = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat_<cv::Vec3b> destination(source.rows, source.cols);
    int deg = atoi(argv[2]);
    cv::imshow("Source Image", source);
    float rad = float(deg) * (M_PI / 180);
    cout << "rad: " << rad << " deg: " << deg << endl;
    auto begin = chrono::high_resolution_clock::now();
    const int iter = 50;
    #pragma omp parallel for
    for (int it = 0; it < iter; it++) {
        #pragma omp parallel for
        for (int i = 0; i < source.rows; i++) {
            for (int j = 0; j < source.cols; j++) {
                destination(i, j) = colorTransformation(source(i, j), rad);
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
