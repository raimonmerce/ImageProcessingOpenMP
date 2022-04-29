#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>  // for high_resolution_clock
#include <omp.h>

using namespace std;

bool smaller(cv::Vec3b a, cv::Vec3b b) {
    float fa = float(a[0]) + float(a[1]) + float(a[2]);
    float fb = float(b[0]) + float(b[1]) + float(b[2]);
    return fa < fb;
}

cv::Vec3b knn(int i, int j, cv::Mat_<cv::Vec3b> source, int ksize, int percent) {
    int size2 = (ksize - 1) / 2;
    int kini = i - size2;
    if (kini < 0) kini = 0;
    int kend = i + size2;
    if (kend >= source.rows) kend = source.rows - 1;
    int lini = j - size2;
    if (lini < 0) lini = 0;
    int lend = j + size2;
    if (lend >= source.cols) lend = source.cols - 1;

    vector<cv::Vec3b> tmpV((kend - kini) * (lend - lini));

    int count = 0;
    for (int k = kini; k < kend; k++) {
        for (int l = lini; l < lend; l++) {
            tmpV[count] = source(k, l);
            count++;
        }
    }

    vector<cv::Vec3b> sorted(tmpV.size());
    vector<bool> visited(tmpV.size(), true);

    for (int k = 0; k < sorted.size(); k++) {
        int pos;
        bool ini = true;
        for (int l = 0; l < tmpV.size(); l++) {
            if (visited[l]) {
                if (ini) {
                    ini = false;
                    pos = l;
                }
                else if (smaller(tmpV[l], tmpV[pos])) {
                    pos = l;
                }
            }
        }
        sorted[k] = tmpV[pos];
        visited[pos] = false;
    }

    int hallf = floor(((ceil(sorted.size() / 2) - 1) * percent) / 100);
    int ini = floor(sorted.size() / 2) - hallf;
    int end = (sorted.size() / 2) + hallf;
    if (sorted.size() % 2 == 0) ++end;

    float b = 0;
    float g = 0;
    float r = 0;
    for (int k = ini; k <= end; k++) {
        b += float(sorted[k][0]);
        g += float(sorted[k][1]);
        r += float(sorted[k][2]);
    }
    int div = end - ini + 1;
    cv::Vec3b res = cv::Vec3b(uchar(b / div), uchar(g / div), uchar(r / div));
    return res;
}

int main(int argc, char** argv)
{
    cv::Mat_<cv::Vec3b> source = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat_<cv::Vec3b> destination(source.rows, source.cols);
    string mode(argv[2]);

    int ksize = atoi(argv[2]);
    int percent = atoi(argv[3]);

    cv::imshow("Source Image", source);

    auto begin = chrono::high_resolution_clock::now();
    const int iter = 50;
    #pragma omp parallel for
    for (int it = 0; it < iter; it++) {
        #pragma omp parallel for
        for (int i = 0; i < source.rows; i++) {
            #pragma omp parallel for
            for (int j = 0; j < source.cols; j++) {
                destination(i, j) = knn(i, j, source, ksize, percent);
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
