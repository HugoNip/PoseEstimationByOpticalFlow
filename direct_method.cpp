#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;
// path
std::string left_file = "../data/left.png";
std::string disparity_file = "../data/disparity.png";
boost::format fmt_others("../data/%06d.png");

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

// class for accumulator jacobians in parallel
class JacobianAccumulator {
public:
    JacobianAccumulator(
            const cv::Mat &img1_,
            const cv::Mat &img2_,
            const VecVector2d &px_ref_,
            const std::vector<double> depth_ref_,
            Sophus::SE3d &T21_) :
            img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d (px_ref.size(), Eigen::Vector2d(0, 0));
    }

    // accumulate jacobians in a range
    void accumulate_jacobian(const cv::Range &range);

    // get hessian matrix
    Matrix6d hessian() const {  return H;   }

    // get bias
    Vector6d bias() const { return b;   }

    // get total cost
    double cost_func() const {  return cost;    }

    // get projected points
    VecVector2d projected_points() const {  return projection;  }

    // reset h, b, cost to zero
    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const std::vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const std::vector<double> depth_ref,
        Sophus::SE3d &T21
);

/**
 * pose estimation using direct method
 * @param img1
 * @param img2
 * @param px_ref
 * @param depth_ref
 * @param T21
 */
void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const std::vector<double> depth_ref,
        Sophus::SE3d &T21
);

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar* data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
            (1 - xx) * (1 - yy) * data[0] +
            xx * (1 - yy) * data[1] +
            (1 - xx) * yy * data[img.step] +
            xx * yy * data[img.step + 1]
    );
}

int main(int argc, char** argv) {
    cv::Mat left_img = cv::imread(left_file, 0);
    cv::Mat disparity_img = cv::imread(disparity_file, 0);

    // randomly pixel pixels in the first image and
    // generate some 3d points in the first image's frame
    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    std::vector<double> depth_ref;

    // generate pixels in ref and load depth data
    // real value
    for (int i = 0; i < nPoints; i++) {
        // don't pick pixels close to boarder
        int x = rng.uniform(boarder, left_img.cols - boarder);
        int y = rng.uniform(boarder, left_img.rows - boarder);
        int disparity = disparity_img.at<uchar>(y, x);
        // this disparity is to depth
        double depth = fx * baseline / disparity;
        // record coordinate of picked up pixels
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    // estimates 01-05.png's pose using this information
    Sophus::SE3d T_cur_ref;

    for (int j = 1; j < 6; j++) {
        cv::Mat img = cv::imread((fmt_others % j).str(), 0);
        // single layer
        // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
        DirectPoseEstimationMultiLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    }

    return 0;
}

void DirectPoseEstimationSingleLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref, // camera1, pixel coordinate
        const std::vector<double> depth_ref, // camera1, depth
        Sophus::SE3d &T21) {

    const int iterations = 10;
    double cost = 0, lastCost = 0;
    auto t1 = std::chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, px_ref.size()),
                          std::bind(&JacobianAccumulator::accumulate_jacobian,
                                    &jaco_accu, std::placeholders::_1));
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0])) {
            // sometimes occured when we have a black or white patch and H is irreversible
            std::cout << "update is nan" << std::endl;
            break;
        }

        if (iter > 0 && cost > lastCost) {
            std::cout << "cost increased: " << cost << ", " << lastCost << std::endl;
            break;
        }

        if (update.norm() < 1e-3) {
            // converge
            break;
        }

        lastCost = cost;
        std::cout << "iteration: " << iter << ", cost: " << lastCost << std::endl;
    }

    std::cout << "T21 = \n" << T21.matrix() << std::endl;
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "direct method for single layer: " << time_used.count() << std::endl;

    // plot the projected pixels here
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, cv::COLOR_GRAY2BGR);
    VecVector2d projection = jaco_accu.projected_points();
    for (size_t i = 0; i < px_ref.size(); i++) {
        auto p_ref = px_ref[i];
        auto p_cur = projection[i];
        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2,
                       cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]),
                     cv::Point2f(p_cur[0], p_cur[1]), cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("current", img2_show);
    cv::waitKey();
}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {
    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    for (size_t i = range.start; i < range.end; i++) {

        // compute the projection in the second image
        // ZP_uv = KTP_w
        Eigen::Vector3d point_ref =
                depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1); // P = Z_1 * p_1
        Eigen::Vector3d point_cur = T21 * point_ref; // camera2: [X, Y, Z], TP
        if (point_cur[2] < 0) // depth invalid
            continue;

        // pixel coordinate
        float u = fx * point_cur[0] / point_cur[2] + cx; // u = fx * X / Z + cx
        float v = fy * point_cur[1] / point_cur[2] + cy; // v = fy * Y / Z + cy

        if (u < half_patch_size ||
            u > img2.cols - half_patch_size ||
            v < half_patch_size ||
            v > img2.rows - half_patch_size)
            continue;

        projection[i] = Eigen::Vector2d(u, v); // reprojection
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2], Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // compute error abd jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++) {

                // e = I_1(p_1) - I_2(p_2)
                // (8.13)
                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;


                // (8.18)
                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                        0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                        0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
                );

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose(); // (8.19)

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }

    if (cnt_good) {
        // set hessian, bias ans cost
        std::unique_lock<std::mutex> lck(hessian_mutex);
        H += hessian;
        b += bias;
        cost += cost_tmp / cnt_good;
    }
}

void DirectPoseEstimationMultiLayer(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const VecVector2d &px_ref,
        const std::vector<double> depth_ref,
        Sophus::SE3d &T21) {

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    std::vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG =cx, cyG = cy;
    for (int level = pyramids - 1; level >= 0; level--) {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px: px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }
}

/*
 * /media/nipnie/DATA/SLAMLearning/slambook2_myPC/ch8/cmake-build-debug/direct_method
iteration: 0, cost: 2.28102e+06
iteration: 1, cost: 970487
iteration: 2, cost: 385210
iteration: 3, cost: 251684
cost increased: 267418, 251684
T21 =
   0.999991  0.00226116  0.00368398 -0.00837304
-0.00227106    0.999994  0.00268373 0.000714081
-0.00367789 -0.00269207     0.99999   -0.725078
          0           0           0           1
direct method for single layer: 0.0990815
iteration: 0, cost: 254962
cost increased: 257153, 254962
T21 =
    0.999989   0.00302295   0.00351096 -0.000253747
 -0.00303088     0.999993   0.00225443   0.00641454
 -0.00350412  -0.00226505     0.999991    -0.728387
           0            0            0            1
direct method for single layer: 0.0357507
iteration: 0, cost: 365605
iteration: 1, cost: 339886
cost increased: 341426, 339886
T21 =
   0.999991  0.00253135  0.00342118 -0.00132494
-0.00253932    0.999994  0.00232837  0.00249305
-0.00341526 -0.00233703    0.999991   -0.735066
          0           0           0           1
direct method for single layer: 0.0565418
iteration: 0, cost: 504948
iteration: 1, cost: 478198
cost increased: 479599, 478198
T21 =
   0.999991  0.00248078  0.00344423 -0.00392992
 -0.0024883    0.999995  0.00218168    0.003183
 -0.0034388 -0.00219023    0.999992   -0.732291
          0           0           0           1
direct method for single layer: 0.0587187
iteration: 0, cost: 1.8897e+06
iteration: 1, cost: 1.28271e+06
iteration: 2, cost: 836443
iteration: 3, cost: 535967
iteration: 4, cost: 393341
iteration: 5, cost: 350874
cost increased: 361249, 350874
T21 =
   0.999971  0.00109427  0.00748034   0.0115242
 -0.0011281    0.999989  0.00452061  0.00222311
-0.00747531 -0.00452892    0.999962    -1.45931
          0           0           0           1
direct method for single layer: 0.122874
iteration: 0, cost: 505052
iteration: 1, cost: 489807
cost increased: 490736, 489807
T21 =
   0.999971  0.00115785  0.00752577  0.00570163
-0.00118816    0.999991  0.00402387  0.00389147
-0.00752105 -0.00403269    0.999964    -1.46931
          0           0           0           1
direct method for single layer: 0.0497232
iteration: 0, cost: 622563
cost increased: 627829, 622563
T21 =
     0.99997  0.000726776   0.00767328  -0.00137749
-0.000756731     0.999992   0.00390169   0.00330944
 -0.00767038  -0.00390738     0.999963     -1.47992
           0            0            0            1
direct method for single layer: 0.117415
iteration: 0, cost: 702430
iteration: 1, cost: 668958
T21 =
    0.999971  0.000698818   0.00758967  -0.00249717
-0.000727093     0.999993   0.00372327   0.00399917
 -0.00758701  -0.00372868     0.999964     -1.48134
           0            0            0            1
direct method for single layer: 0.139963
iteration: 0, cost: 1.89805e+06
iteration: 1, cost: 1.52114e+06
iteration: 2, cost: 1.24731e+06
iteration: 3, cost: 1.05443e+06
iteration: 4, cost: 804361
iteration: 5, cost: 743973
iteration: 6, cost: 563640
iteration: 7, cost: 535972
iteration: 8, cost: 520780
iteration: 9, cost: 520651
T21 =
    0.99994  0.00134682   0.0108648   0.0360145
-0.00140942    0.999982  0.00575629  0.00964109
 -0.0108568 -0.00577126    0.999924    -2.18858
          0           0           0           1
direct method for single layer: 0.140172
iteration: 0, cost: 681160
iteration: 1, cost: 654720
iteration: 2, cost: 647091
cost increased: 647763, 647091
T21 =
   0.999936  0.00138725   0.0111837   0.0246449
-0.00144823    0.999984  0.00544606  0.00198196
 -0.0111759 -0.00546191    0.999923    -2.21631
          0           0           0           1
direct method for single layer: 0.14412
iteration: 0, cost: 904857
iteration: 1, cost: 885543
iteration: 2, cost: 865051
cost increased: 882505, 865051
T21 =
   0.999935  0.00153241   0.0112658   0.0186298
-0.00159428    0.999984   0.0054852 -0.00533671
 -0.0112572 -0.00550281    0.999921    -2.23498
          0           0           0           1
direct method for single layer: 0.146037
iteration: 0, cost: 1.29266e+06
iteration: 1, cost: 1.19385e+06
iteration: 2, cost: 1.17269e+06
iteration: 3, cost: 1.16042e+06
iteration: 4, cost: 1.15458e+06
T21 =
    0.999934    0.0012471    0.0114065   0.00246521
 -0.00130725     0.999985   0.00526708 -0.000489021
  -0.0113997  -0.00528165     0.999921     -2.24049
           0            0            0            1
direct method for single layer: 0.199021
iteration: 0, cost: 2.06548e+06
iteration: 1, cost: 1.85611e+06
iteration: 2, cost: 1.50776e+06
iteration: 3, cost: 1.31913e+06
iteration: 4, cost: 1.11262e+06
iteration: 5, cost: 983659
iteration: 6, cost: 856684
iteration: 7, cost: 793141
iteration: 8, cost: 722321
iteration: 9, cost: 710257
T21 =
   0.999873 -0.00024726   0.0159652   0.0247768
0.000133189    0.999974  0.00714566 -0.00384354
 -0.0159665 -0.00714262    0.999847    -2.95941
          0           0           0           1
direct method for single layer: 0.201973
iteration: 0, cost: 906825
iteration: 1, cost: 866319
iteration: 2, cost: 860644
cost increased: 865467, 860644
T21 =
    0.999866 -0.000286031    0.0163589    0.0139422
 0.000177415     0.999978   0.00664065   -0.0046732
  -0.0163604  -0.00663686     0.999844     -3.00582
           0            0            0            1
direct method for single layer: 0.0493113
iteration: 0, cost: 1.19277e+06
iteration: 1, cost: 1.14702e+06
iteration: 2, cost: 1.1347e+06
iteration: 3, cost: 1.13144e+06
T21 =
    0.999864 -0.000222788    0.0164684   0.00164523
 0.000119784      0.99998   0.00625535  -0.00434143
  -0.0164695  -0.00625252     0.999845     -3.01752
           0            0            0            1
direct method for single layer: 0.102753
iteration: 0, cost: 1.81301e+06
iteration: 1, cost: 1.76628e+06
iteration: 2, cost: 1.73562e+06
iteration: 3, cost: 1.73417e+06
iteration: 4, cost: 1.72741e+06
T21 =
    0.999863  0.000263449    0.0165249   -0.0123927
-0.000360389     0.999983   0.00586363  0.000315694
   -0.016523  -0.00586878     0.999846     -3.02755
           0            0            0            1
direct method for single layer: 0.0738702
iteration: 0, cost: 2.35383e+06
iteration: 1, cost: 2.17638e+06
iteration: 2, cost: 1.84716e+06
iteration: 3, cost: 1.44386e+06
iteration: 4, cost: 1.18609e+06
iteration: 5, cost: 1.04537e+06
iteration: 6, cost: 971768
iteration: 7, cost: 957879
iteration: 8, cost: 952288
iteration: 9, cost: 945971
T21 =
    0.999799  0.000605813     0.020016    0.0360904
-0.000748249     0.999974   0.00710937    0.0137726
  -0.0200112  -0.00712293     0.999774     -3.76018
           0            0            0            1
direct method for single layer: 0.107959
iteration: 0, cost: 1.39585e+06
iteration: 1, cost: 1.33145e+06
iteration: 2, cost: 1.30529e+06
iteration: 3, cost: 1.28185e+06
cost increased: 1.28187e+06, 1.28185e+06
T21 =
    0.999783  0.000677586    0.0208336   0.00372023
-0.000820023     0.999976   0.00682912   0.00786539
  -0.0208285  -0.00684472      0.99976     -3.83407
           0            0            0            1
direct method for single layer: 0.0581535
iteration: 0, cost: 1.8475e+06
iteration: 1, cost: 1.75777e+06
iteration: 2, cost: 1.74386e+06
cost increased: 1.75118e+06, 1.74386e+06
T21 =
   0.999779  0.00113171   0.0210117 -0.00619981
-0.00126317     0.99998  0.00624437   0.0117463
 -0.0210042 -0.00626953     0.99976    -3.85464
          0           0           0           1
direct method for single layer: 0.139324
iteration: 0, cost: 2.45907e+06
iteration: 1, cost: 2.42193e+06
iteration: 2, cost: 2.40223e+06
cost increased: 2.40312e+06, 2.40223e+06
T21 =
   0.999786  0.00138017   0.0206633 -0.00312465
-0.00150567    0.999981  0.00605893  0.00849709
 -0.0206545 -0.00608874    0.999768    -3.86073
          0           0           0           1
direct method for single layer: 0.155657
 */

