#include "utils.h"
#include <regex>
#include <pcl/registration/gicp.h>

M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);


V3D calculateStdDev(const std::vector<V3D> &measurements, const V3D &mean)
{
    V3D sum = Zero3d;
    for (const auto &measurement : measurements)
    {
        V3D diff = measurement - mean;
        sum += diff.cwiseProduct(diff); 
    }
    return (sum / measurements.size()).cwiseSqrt(); 
}

double calculateMean(const std::vector<double> &values)
{
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / values.size();
}

double calculateMedian(std::vector<double> values)
{
    std::sort(values.begin(), values.end()); // sort the vector

    if (values.size() % 2 == 0)
    {
        // even number of elements, average the two middle values
        return (values[values.size() / 2 - 1] + values[values.size() / 2]) / 2.0;
    }
    else
    {
        return values[values.size() / 2];
    }
}

V3D gnss::computeWeightedAverage(const std::vector<V3D> &measurements, const std::vector<V3D> &covariances)
{
    V3D weighted_sum(0.0, 0.0, 0.0);
    V3D weight_sum(0.0, 0.0, 0.0);

    for (size_t i = 0; i < measurements.size(); ++i)
    {
        V3D weight = covariances[i].cwiseInverse();
        weighted_sum += measurements[i].cwiseProduct(weight);
        weight_sum += weight;
    }

    return weighted_sum.cwiseQuotient(weight_sum);
}

void computeTransformation(const std::vector<V3D> &gnss_points,
                           const std::vector<V3D> &mls_points,
                           M3D &R, V3D &t)
{

    assert(gnss_points.size() == mls_points.size() && "computeTransformation -> Point sets must have the same size");

    int n = gnss_points.size();

    V3D centroid_gnss = Zero3d;
    V3D centroid_mls = Zero3d;

    for (int i = 0; i < n; ++i)
    {
        centroid_gnss += gnss_points[i];
        centroid_mls += mls_points[i];
    }

    centroid_gnss /= n;
    centroid_mls /= n;

    std::vector<V3D> centered_gnss(n), centered_mls(n);
    for (int i = 0; i < n; ++i)
    {
        centered_gnss[i] = gnss_points[i] - centroid_gnss;
        centered_mls[i] = mls_points[i] - centroid_mls;
    }

    // Compute covariance matrix
    M3D covariance = Eigen::Matrix3d::Zero();
    for (int i = 0; i < n; ++i)
    {
        covariance += centered_gnss[i] * centered_mls[i].transpose();
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    M3D U = svd.matrixU();
    M3D V = svd.matrixV();

    // Compute rotation
    R = V * U.transpose();

    // Handle the special case of reflection
    if (R.determinant() < 0)
    {
        V.col(2) *= -1;
        R = V * U.transpose();
    }

    // Compute translation
    t = centroid_mls - R * centroid_gnss;
}

void TransformPoints(const M3D &R, const V3D &T, PointCloudXYZI::Ptr &points)
{
    if (!points || points->empty())
        return;

    tbb::parallel_for(tbb::blocked_range<int>(0, points->size()),
                      [&](const tbb::blocked_range<int> &r)
                      {
                          for (int i = r.begin(); i < r.end(); ++i)
                          {
                              V3D point(points->points[i].x,
                                        points->points[i].y,
                                        points->points[i].z);
                              V3D transformed_position = R * point + T;
                              points->points[i].x = transformed_position.x();
                              points->points[i].y = transformed_position.y();
                              points->points[i].z = transformed_position.z();
                          }
                      });
}

void TransformPoints(const Sophus::SE3 &T, pcl::PointCloud<PointType>::Ptr &cloud)
{
    tbb::parallel_for(tbb::blocked_range<int>(0, cloud->size()),
                      [&](tbb::blocked_range<int> r)
                      {
                          for (int i = r.begin(); i < r.end(); ++i)
                          {
                              V3D p_src(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
                              V3D p_transformed = T * p_src;

                              cloud->points[i].x = p_transformed.x();
                              cloud->points[i].y = p_transformed.y();
                              cloud->points[i].z = p_transformed.z();
                          }
                      });
}

std::vector<std::string> expandBagPattern(const std::string &pattern_path)
{
    namespace fs = boost::filesystem;
    std::vector<std::string> results;

    fs::path p(pattern_path);
    fs::path dir = p.parent_path();               // directory part
    std::string filename = p.filename().string(); // e.g. "1_hesai-CPT_2024-*.bag"

    if (dir.empty())
        dir = ".";

    // Convert shell-style * into regex
    std::string regex_pattern = std::regex_replace(filename, std::regex("\\*"), ".*");
    std::regex re(regex_pattern);

    if (!fs::exists(dir) || !fs::is_directory(dir))
    {
        std::cerr << "Directory does not exist: " << dir << std::endl;
        return results;
    }

    for (auto &entry : fs::directory_iterator(dir))
    {
        if (fs::is_regular_file(entry.path()))
        {
            std::string fname = entry.path().filename().string();
            if (std::regex_match(fname, re))
            {
                results.push_back(entry.path().string());
            }
        }
    }

    std::sort(results.begin(), results.end());
    return results;
}

