#include <iostream>
#include <string>
#include <vector>
#include "mosaic.h"

int main()
{

    auto start = std::chrono::high_resolution_clock::now();
    Mosaic m(
        "/Users/mac/Downloads/2018_08_20 Эмблема АМА (1) (1).jpg",
        "/Users/mac/Downloads/CompCars/data/filenames.txt",
        230000,
        16);

    // std::vector<std::string> arr_s = {
    //   "/Users/mac/Downloads/Снимок экрана 2025-05-24 в 17.05.png",
    //   "/Users/mac/Downloads/флагсердце (2).jpg"
    //   // "/Users/mac/Downloads/Toyota Supra with Falken Tires p.jpg",
    //   // "/Users/mac/Downloads/yuyuy.jpg",
    //   // "/Users/mac/Downloads/car.jpg",
    // };

    m.get_ts();
    m.start();
    m.build_mosaic(0);

    // for (int i = 0; i < arr_s.size(); ++i)
    // {
    //     auto start_s = std::chrono::high_resolution_clock::now();
    //     m.re_c(arr_s[i], i + 1, 1);
    //     auto end_s = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double> duration_s = end_s - start_s;
    //     std::cout << "Время выполнения: " << duration_s.count() << " секунд\n";
    // }

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> duration = end - start;

    // std::cout << "Время выполнения: " << duration.count() << " секунд\n";
    // m.c_t();

    // std::ofstream out("out.txt");
    // for (int i  = 0; i < arr_s.size(); ++i)
    // {
    //     cv::Mat jj = cv::imread("/Users/mac/Desktop/mosaic/testsout/" + std::to_string(i + 1) + ".png");
    //     cv::Mat yy = cv::imread(arr_s[i]);

    //     std::vector<double> sun_rgb = {0.0, 0.0, 0.0};
    //     for (int y = 0; y < yy.rows; ++y)
    //     {
    //         for (int x = 0; x < yy.cols; ++x)
    //         {
    //             cv::Vec3b colorj = jj.at<cv::Vec3b>(y, x);
    //             cv::Vec3b colory = yy.at<cv::Vec3b>(y, x);

    //             for (int i = 0; i < 3; ++i)
    //             {
    //                 sun_rgb[i] += std::sqrt((double)(colorj[i] - colory[i])*(colorj[i] - colory[i]));
    //             }
    //         }
    //     }
    //     for (int i = 0; i < 3; ++i)
    //         std::cout << sun_rgb[i]/(yy.cols*yy.rows) << " ";
    //     std::cout << std::endl;
    // }

    // out.close();
}
