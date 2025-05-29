#ifndef MOSAIC_H
#define MOSAIC_H

#include <vector>
#include <string>
#include <thread>
#include <fstream>
#include <chrono>
#include <limits>
#include <random>
#include <algorithm>
#include <utility>

#include "opencv2/opencv.hpp"


struct approx_im;
class Mosaic;

struct approx_im{

    float ave_r{0.0};
    float ave_g{0.0};
    float ave_b{0.0};
    float grad_r{0.0};
    float grad_g{0.0};
    float grad_b{0.0};

    int x{std::numeric_limits<int>::max()}, y{std::numeric_limits<int>::max()};
    cv::Mat im;

    approx_im() = default;
    float& operator [](int index);
    const float operator[](int index) const;
};

struct d3_tree{
    std::pair<approx_im, int> center_p;
    int axis;
    d3_tree *left = nullptr, *right = nullptr;
};

struct d6_tree{
    approx_im* a_m;
    int axis;
    d6_tree *left = nullptr, *right = nullptr;
};

class Mosaic{

    cv::Mat mosaic;
    int NUM_THREADS{1};
    std::string path_mosaic;
    std::string path_list;
    int multi_p {0};

    std::string cur_dir{"/Users/mac/Desktop/mosaic/testsout/"};

    std::vector<std::thread> t_pool;
    std::vector<std::string> paths_img;
    std::vector<approx_im> approx_bs;
    std::vector<approx_im> approx_ts;


    std::vector<std::pair<approx_im, int>> centers_p;
    std::vector<approx_im> centers;
    std::vector<std::vector<approx_im*>> clusters;

    d3_tree* c_3d_t{nullptr};
    std::vector<d6_tree*> sup_ts;

    int D;
    int k{32};

    void get_approx(approx_im& a_im);
    void init_t(int s, int e);
    void init_ta(int s, int e);
    void init_kmeans(int num);
    void kmeans_clusters(int num, int max_iters);

    d3_tree* build(std::vector<std::pair<approx_im, int>>& centers_p, int left, int right, int depth);
    d6_tree* b_d6(std::vector<approx_im*>& cluster, int left, int right, int depth);
    d3_tree* nearest_d3(d3_tree* node, const approx_im& target, d3_tree*& best_node, float& best_dist, int depth = 0);
    d6_tree* nearest_d6(d6_tree* node, const approx_im& target, d6_tree*& best_node, float& best_dist, int x, int y, int min_d = 64, int depth = 0);
    void part(int start_y, int end_y);
public:

    Mosaic(std::string path_mosaic, std::string path_list, int D, int k);
    void get_ts();
    void get_ta(int num);
    float eu_d(const approx_im& a, const approx_im& b, int iters);
    void start();
    void build_mosaic(int test_num);
    void concatGrid(int x, int y, cv::Mat& img, cv::Mat& grid);
    void re_c(std::string path, int test_num, int multi_p);
    void c_t(){ t_pool.clear();}
};


#endif  //MOSAIC_H
