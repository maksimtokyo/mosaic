#include "mosaic.h"

float& approx_im::operator [](int index)
{
    switch(index)
    {
        case 0: return ave_r;
        case 1: return ave_g;
        case 2: return ave_b;
        case 3: return grad_r;
        case 4: return grad_g;
        case 5: return grad_b;
        default: throw std::out_of_range("Index out of the range" + std::to_string(index));
    }

}

const float approx_im::operator[](int index) const
{
    switch(index)
    {
        case 0: return ave_r;
        case 1: return ave_g;
        case 2: return ave_b;
        case 3: return grad_r;
        case 4: return grad_g;
        case 5: return grad_b;
        default: throw std::out_of_range("Index out of the range" + std::to_string(index));
    }
}

float Mosaic::eu_d(const approx_im& a, const approx_im& b, int iters)
{
    float dist = 0.0;
    for (int i = 0; i < iters; ++i)
    {
        float s = (b[i] - a[i]);
        dist += (s*s);
    }
    return dist;
}

Mosaic::Mosaic(std::string path_mosaic, std::string path_list, int D, int k)
    : path_mosaic(path_mosaic), path_list(path_list), D(D), k(k)
{
    NUM_THREADS =  std::thread::hardware_concurrency();
    mosaic = cv::imread(path_mosaic);

    paths_img.resize(D);
    approx_ts.resize(D, {});
    approx_bs.resize((mosaic.rows*mosaic.cols)/(k*k), {});
    t_pool.resize(NUM_THREADS);

    std::ifstream file(path_list);
    for (int i = 0; i < paths_img.size() && std::getline(file, paths_img[i]); ++i);
    file.close();
}


void Mosaic::get_ts()
{
    int step = D/NUM_THREADS;
    int b_step = mosaic.rows/NUM_THREADS;
    int s_step = 0; int bs_step = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < NUM_THREADS; ++t)
    {

        int e_step = (t == NUM_THREADS - 1) ? D : s_step + step;
        int be_step = (t == NUM_THREADS - 1) ? mosaic.rows : bs_step + b_step;
        t_pool[t] = std::thread([=, this]{
            init_t(s_step, e_step);
            part(bs_step, be_step);
        });
        s_step = e_step;
        bs_step = be_step;
    }

    for (auto& t : t_pool)
        if (t.joinable()) t.join();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Время выполнения: " << duration.count() << " секунд\n";
}

void Mosaic::init_t(int s, int e)
{
    for (int i = s; i < e; ++i)
    {

        while(paths_img[i].back() != 'g' && !paths_img[i].empty()) paths_img[i].pop_back();
        cv::Mat imgh = cv::imread(paths_img[i]);
        cv::resize(imgh, approx_ts[i].im, cv::Size(k, k));
        get_approx(approx_ts[i]);
    }

    std::cout << "\n Thread finished work -> " << std::this_thread::get_id() << "\n";
}

void Mosaic::get_approx(approx_im& a_im)
{
    int count = (k-2)*(k-2);
    for (int y = 1; y < k - 1; ++y)
    {
        const cv::Vec3b* prev = a_im.im.ptr<cv::Vec3b>(y - 1);
        const cv::Vec3b* curr = a_im.im.ptr<cv::Vec3b>(y);
        const cv::Vec3b* next = a_im.im.ptr<cv::Vec3b>(y + 1);

        for (int x = 1; x < k - 1; ++x)
        {
            for (int c = 0; c < 3; ++c)
            {
                float val =curr[x][c]/255.0;
                a_im[c] += val;

                float gx = prev[x + 1][c] - prev[x - 1][c]
                            + 2 * (curr[x + 1][c] - curr[x - 1][c])
                            + next[x + 1][c] - next[x - 1][c];
                float gy = prev[x - 1][c] + 2 * prev[x][c] + prev[x + 1][c]
                            - next[x - 1][c] - 2 * next[x][c] - next[x + 1][c];

                float mag = std::sqrt(((gx * gx) + (gy * gy))/(255.0*255.0));
                a_im[c + 3] += mag;
            }
        }
    }

    for (int i = 0; i < 6; ++i)
        a_im[i]/=count;
}

void Mosaic::init_kmeans(int num)
{

    centers.resize(num, {});
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, D - 1);
    centers[0] = approx_ts[dis(gen)];
    std::vector<float> dist(D, std::numeric_limits<float>::max());

    for (int center_count = 1; center_count < num; ++center_count)
    {
        float dist_sum = 0.0;

        for (size_t i = 0; i < D; ++i)
        {
            float d = eu_d(approx_ts[i], centers[center_count - 1], 3);
            if (d < dist[i]) dist[i] = d;
            dist_sum +=  dist[i];
        }

        std::uniform_real_distribution<float> dis_prob(0, dist_sum);
        float target = dis_prob(gen);
        float cumulative = 0.0;
        size_t next_center_idx = 0;

        for (size_t i = 0; i < D; ++i)
        {
            cumulative += dist[i];
            if(cumulative >= target)
            {
                next_center_idx = i;
                break;
            }
        }
        centers[center_count] = approx_ts[next_center_idx];
    }
}

d3_tree* Mosaic::build(std::vector<std::pair<approx_im, int>>& centers_p, int left, int right, int depth)
{
    if (left >= right) return nullptr;
    int axis  = depth%3;

    std::sort(centers_p.begin() + left, centers_p.begin() + right, [axis](const std::pair<approx_im, int>& a, const std::pair<approx_im, int>& b){
        return a.first[axis] < b.first[axis];
    });

    d3_tree* node =  new d3_tree;
    int mid = (left + right)/2;
    node->center_p = centers_p[mid];
    node->axis = axis;

    node->left = build(centers_p, left, mid, depth + 1);
    node->right = build(centers_p, mid + 1, right, depth + 1);
    return node;
}

d6_tree* Mosaic::b_d6(std::vector<approx_im*>& cluster, int left, int right, int depth)
{
    if (left >= right) return nullptr;

    int axis = depth%6;

    std::sort(cluster.begin() + left, cluster.begin() + right, [axis](const approx_im* a, const approx_im* b){
        return (*a)[axis] < (*b)[axis];
    });

    d6_tree* node = new d6_tree;
    int mid = (left + right)/2;
    node->axis = axis;
    node->a_m = cluster[mid];

    node->left = b_d6(cluster, left, mid, depth + 1);
    node->right = b_d6(cluster, mid + 1, right, depth + 1);

    return node;
}

void Mosaic::part(int start_y, int end_y)
{
    for (int y = start_y; y + k <= end_y; y+=k)
    {
        for (int x = 0; x + k <=  mosaic.cols; x += k)
        {
            cv::Rect block_rect(x, y, k, k);
            int block_x = x / k;
            int block_y = y / k;
            int blocks_per_row = mosaic.cols / k;
            int index = block_y * blocks_per_row + block_x;
            approx_bs[index].im = mosaic(block_rect);
            approx_bs[index].x = x; approx_bs[index].y = y;
            get_approx(approx_bs[index]);
        }
    }
}

void Mosaic::kmeans_clusters(int num, int max_iters)
{

    clusters.resize(num, {});
    bool changed = true;
    int iter = 0;

    while (changed && (iter < max_iters))
    {
        changed = false;
        ++iter;

        for (auto& cluster : clusters) cluster.clear();

        for (auto& approx_t : approx_ts)
        {
            float min_dist = eu_d(approx_t, centers[0], 3);
            int best_idx = 0;

            for (int i = 1; i < num; ++i)
            {
                float dist = eu_d(approx_t, centers[i], 3);
                if (dist < min_dist)
                {
                    min_dist = dist;
                    best_idx = i;
                }
            }
            clusters[best_idx].push_back(&approx_t);
        }

        for (int i = 0; i < num; ++i)
        {
            if (clusters[i].empty()) continue;

            float ave_r = 0, ave_g = 0, ave_b = 0;

            for (auto approx_t : clusters[i])
            {
                ave_r += approx_t->ave_r;
                ave_g += approx_t->ave_g;
                ave_b += approx_t->ave_b;
            }

            int n = clusters[i].size();
            ave_r /= n;
            ave_g /= n;
            ave_b /= n;

            approx_im n_center;
            n_center.ave_r = ave_r;
            n_center.ave_g = ave_g;
            n_center.ave_b = ave_b;

            if (eu_d(n_center, centers[i], 3) > 1e-4)
            {
                centers[i] = n_center;
                changed = true;
            }
        }
    }

    centers_p.resize(num, {});
    for (int i = 0; i < num; ++i)
    {
        centers_p[i].first = centers[i];
        centers_p[i].second = i;
    }
    c_3d_t = build(centers_p, 0, centers_p.size(), 0);
}

void Mosaic::init_ta(int s, int e)
{
    for (int i = s; i < e; ++i)
    {
        sup_ts[i] = b_d6(clusters[i], 0, clusters[i].size(), 0);
    }
    std::cout << "\n Thread finished work -> " << std::this_thread::get_id() << "\n";
}
void Mosaic::get_ta(int num)
{
    t_pool.clear();
    t_pool.resize(NUM_THREADS);
    sup_ts.resize(num, {});

    int step = num/NUM_THREADS;
    int s_step = 0;

    for (int t = 0; t <  NUM_THREADS; ++t)
    {
        int e_step = (t == NUM_THREADS - 1) ? num : s_step + step;
        t_pool[t] = std::thread([=, this]{
            init_ta(s_step, e_step);
        });
        s_step = e_step;
    }

    for (auto& t : t_pool)
        if (t.joinable()) t.join();
}

void Mosaic::start()
{
    auto start = std::chrono::high_resolution_clock::now();
    init_kmeans(100);
    kmeans_clusters(100, 100);
    get_ta(100);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Время выполнения: " << duration.count() << " секунд\n";

}

d3_tree* Mosaic::nearest_d3(d3_tree* node, const approx_im& target, d3_tree *& best_node, float& best_dist, int depth)
{
    if (!node) return best_node;
    float d  = eu_d(target, node->center_p.first, 3);

    if (d < best_dist)
    {
        best_dist = d;
        best_node = node;
    }

    int axis = node->axis;
    bool goleft = target[axis] < node->center_p.first[axis];
    nearest_d3(goleft ? node->left : node->right, target, best_node, best_dist, depth + 1);

    float delta = target[axis] - node->center_p.first[axis];

    if (delta * delta < best_dist) {
            nearest_d3(goleft ? node->right : node->left, target, best_node, best_dist, depth + 1);
        }
    return best_node;
}

d6_tree* Mosaic::nearest_d6(d6_tree* node, const approx_im& target, d6_tree*& best_node, float& best_dist, int x, int y, int min_d, int depth)
{
    if (!node) return best_node;
    float d  = eu_d(target, *(node->a_m), 6);

    if (d < best_dist && (std::abs(node->a_m->x - x) > min_d || std::abs(node->a_m->y - y) > min_d))
    {
        best_dist = d;
        best_node = node;
    }

    int axis = node->axis;
    bool goleft = target[axis] < (*node->a_m)[axis];
    nearest_d6(goleft ? node->left : node->right, target, best_node, best_dist, x, y, min_d,  depth + 1);

    float delta = target[axis] - (*node->a_m)[axis];
    if (delta * delta < best_dist) {
        nearest_d6(goleft ? node->right : node->left, target, best_node, best_dist, x, y, min_d, depth + 1);
    }
    return best_node;
}
void Mosaic::build_mosaic(int test_num)
{
    cv::Mat grid(mosaic.rows, mosaic.cols, CV_8UC3);
    for (int i = 0; i < approx_bs.size(); ++i)
    {
        d3_tree* u = nullptr;
        float fg = 100000000.0;
        d3_tree* t = nearest_d3(c_3d_t, approx_bs[i], u, fg);

        float fgп = 100000000.0;
        d6_tree* uu = nullptr;
        d6_tree* ans = nearest_d6(sup_ts[t->center_p.second], approx_bs[i], uu, fgп, approx_bs[i].x, approx_bs[i].y, k*multi_p);
        if (ans == nullptr) std::cout << "Fuck: " << ans << " " << i <<  std::endl;
        ans->a_m->x = approx_bs[i].x;
        ans->a_m->y = approx_bs[i].y;
        concatGrid(approx_bs[i].x, approx_bs[i].y, ans->a_m->im, grid);
    }

    std::string path = cur_dir + std::to_string(test_num) + ".png";
    cv::imwrite(path, grid);
}

void Mosaic::concatGrid(int x, int y, cv::Mat& img, cv::Mat& grid)
{
    img.copyTo(grid(cv::Rect(x, y, k, k)));
}


void Mosaic::re_c(std::string path, int test_num, int multi_p)
{

    this->multi_p = multi_p;
    mosaic = cv::imread(path);
    approx_bs.clear();
    approx_bs.resize((mosaic.rows*mosaic.cols)/(k*k), {});
    int b_step = mosaic.rows/NUM_THREADS;
    int s_s = 0;
    t_pool.clear();
    t_pool.resize(NUM_THREADS);

    for (int t = 0; t < NUM_THREADS; ++t)
    {
        int e_s = (t  == NUM_THREADS - 1) ? mosaic.rows : s_s + b_step;
        t_pool[t] = std::thread([=, this]{
            part(s_s, e_s);
        });
        s_s = e_s;
    }

    for (auto& t : t_pool)
        if (t.joinable()) t.join();

    build_mosaic(test_num);
}
