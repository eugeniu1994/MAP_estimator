#include "utils.h"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <pcl/registration/gicp.h>

M3D Eye3d(M3D::Identity());
M3F Eye3f(M3F::Identity());
V3D Zero3d(0, 0, 0);
V3F Zero3f(0, 0, 0);

float ekf::calc_dist(PointType p1, PointType p2)
{
    float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    return d;
}

V3D calculateStdDev(const std::vector<V3D> &measurements, const V3D &mean)
{
    V3D sum = Zero3d;
    for (const auto &measurement : measurements)
    {
        V3D diff = measurement - mean;
        sum += diff.cwiseProduct(diff); // Element-wise square of the difference
    }
    return (sum / measurements.size()).cwiseSqrt(); // Return the square root of the mean of squared differences
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
        // odd number of elements, return the middle value
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

double gnss::findAngle(const V3D &gps0, const V3D &gps1, const V3D &imu0, const V3D &imu1)
{
    // GPS p1 and p2
    Eigen::Vector2d A0(gps0[0], gps0[1]);
    Eigen::Vector2d A1(gps1[0], gps1[1]);

    // MLS p1 and p2
    Eigen::Vector2d B0(imu0[0], imu0[1]);
    Eigen::Vector2d B1(imu1[0], imu1[1]);

    // Calculate direction vectors
    Eigen::Vector2d gps_line = A1 - A0;
    Eigen::Vector2d imu_line = B1 - B0;

    double dot_product = gps_line.dot(imu_line);
    double magnitude_d1 = gps_line.norm();
    double magnitude_d2 = imu_line.norm();

    double cos_theta = dot_product / (magnitude_d1 * magnitude_d2);
    // Ensure the value is within the valid range for acos due to floating-point precision issues
    // cos_theta = std::max(-1.0, std::min(1.0, cos_theta));
    // Ensure the cosine value is within valid range for acos
    cos_theta = std::clamp(cos_theta, -1.0, 1.0);
    double theta_radians = std::acos(cos_theta); // in radians
    double theta_degrees = theta_radians * (180.0 / M_PI);

    return theta_degrees;
}

double gnss::checkAlignment(const Eigen::Vector2d &vec1, const Eigen::Vector2d &vec2)
{
    double dot_product = vec1.dot(vec2);
    double norm1 = vec1.norm();
    double norm2 = vec2.norm();

    double cos_sim = dot_product / (norm1 * norm2);
    cos_sim = std::clamp(cos_sim, -1.0, 1.0);
    double cos_sim_radians = std::acos(cos_sim);
    return cos_sim_radians * (180.0 / M_PI);
}

Eigen::Vector2d gnss::rotateVector(const Eigen::Vector2d &vec, double angle_deg)
{
    double angle_rad = angle_deg * M_PI / 180.0;
    Eigen::Matrix2d rotation_matrix;
    rotation_matrix << std::cos(angle_rad), -std::sin(angle_rad),
        std::sin(angle_rad), std::cos(angle_rad);

    return rotation_matrix * vec;
}

double gnss::verifyAngle(const double &theta_degrees, const V3D &gps0, const V3D &gps1, const V3D &imu0, const V3D &imu1)
{
    // GPS p1 and p2
    Eigen::Vector2d A0(gps0[0], gps0[1]);
    Eigen::Vector2d A1(gps1[0], gps1[1]);

    // IMU p1 and p2
    Eigen::Vector2d B0(imu0[0], imu0[1]);
    Eigen::Vector2d B1(imu1[0], imu1[1]);

    // Calculate direction vectors
    Eigen::Vector2d gps_line = A1 - A0;
    Eigen::Vector2d imu_line = B1 - B0;

    Eigen::Vector2d rotated_gps_positive = rotateVector(gps_line, theta_degrees);
    Eigen::Vector2d rotated_gps_negative = rotateVector(gps_line, -theta_degrees);

    double alignment_positive = checkAlignment(rotated_gps_positive, imu_line);
    double alignment_negative = checkAlignment(rotated_gps_negative, imu_line);
    std::cout << "alignment_positive:" << alignment_positive << ", alignment_negative:" << alignment_negative << std::endl;

    return (fabs(alignment_positive) > fabs(alignment_negative)) ? -theta_degrees : theta_degrees;
}

// Function to fit a line to two points
gnss::LineModel gnss::fitLine(const std::pair<double, double> &p1, const std::pair<double, double> &p2)
{
    LineModel model;
    model.m = (p2.second - p1.second) / (p2.first - p1.first);
    model.b = p1.second - model.m * p1.first;
    return model;
}

// Function to compute the distance from a point to a line
double gnss::pointToLineDistance(double x, double y, const gnss::LineModel &model)
{
    return std::abs(model.m * x - y + model.b) / std::sqrt(model.m * model.m + 1);
}

// RANSAC algorithm to fit a line
gnss::LineModel gnss::ransacFitLine(const std::vector<double> &x, const std::vector<double> &y, int iterations, double threshold)
{
    size_t numPoints = x.size();
    if (numPoints < 2)
    {
        throw std::runtime_error("Need at least two points to fit a line.");
    }

    gnss::LineModel bestModel;
    int maxInliers = 0;
    double min_todal_distance = 9999999;

    // std::srand(static_cast<unsigned>(std::time(0))); // this will not be deterministic
    std::srand(12345); // Fixed seed for reproducibility

    for (int i = 0; i < iterations; ++i)
    {
        // Randomly select two points
        size_t idx1 = std::rand() % numPoints;
        size_t idx2 = std::rand() % numPoints;
        while (idx2 == idx1)
        {
            idx2 = std::rand() % numPoints;
        }

        std::pair<double, double> p1(x[idx1], y[idx1]);
        std::pair<double, double> p2(x[idx2], y[idx2]);

        LineModel model = fitLine(p1, p2);

        // Count inliers and compute total distance
        int inliers = 1;
        double totalDistance = 0.0;
        for (size_t j = 0; j < numPoints; ++j)
        {
            double distance = gnss::pointToLineDistance(x[j], y[j], model);
            if (distance < threshold)
            {
                ++inliers;
            }
            totalDistance += distance;
        }
        totalDistance /= inliers;

        if (inliers > maxInliers && totalDistance < min_todal_distance)
        {
            maxInliers = inliers;
            min_todal_distance = totalDistance;
            bestModel = model;
        }
    }
    std::cout << "RANSAC maxInliers:" << maxInliers << "/" << x.size() << ", min_todal_distance:" << min_todal_distance << std::endl;

    bestModel.m = bestModel.m * 180. / M_PI;

    return bestModel;
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

void TransformPoints(const Sophus::SE3 &T, std::vector<V3D> &points)
{
    tbb::parallel_for(tbb::blocked_range<int>(0, points.size()),
                      [&](tbb::blocked_range<int> r)
                      {
                          for (int i = r.begin(); i < r.end(); ++i)
                          {
                              points[i] = T * points[i];
                          }
                      });
}

void Eigen2PCL(PointCloudXYZI::Ptr &pcl_cloud, const std::vector<V3D> &eigen_cloud)
{
    // Reserve space for points to avoid reallocation
    pcl_cloud->points.resize(eigen_cloud.size());

    for (size_t i = 0; i < eigen_cloud.size(); ++i)
    {
        const auto &vec = eigen_cloud[i];
        PointType &point = pcl_cloud->points[i];

        point.x = vec.x();
        point.y = vec.y();
        point.z = vec.z();
        // point.intensity = 0.0f; // Set to a default value
        // point.time = 0.0f;      // Set to a default value
    }

    // Set cloud dimensions
    pcl_cloud->width = pcl_cloud->points.size();
    pcl_cloud->height = 1; // Unorganized point cloud
    pcl_cloud->is_dense = true;
}

void PCL2EIGEN(const PointCloudXYZI::Ptr &pcl_cloud, std::vector<V3D> &eigen_cloud)
{
    int n = pcl_cloud->size();
    eigen_cloud.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        V3D &vec = eigen_cloud[i];
        const PointType &point = pcl_cloud->points[i];

        vec[0] = point.x;
        vec[1] = point.y;
        vec[2] = point.z;
    }
}

sensor_msgs::PointField GetTimestampField(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    sensor_msgs::PointField timestamp_field;
    for (const auto &field : msg->fields)
    {
        if ((field.name == "t" || field.name == "timestamp" || field.name == "time"))
        {
            timestamp_field = field;
        }
    }
    if (!timestamp_field.count)
    {
        throw std::runtime_error("Field 't', 'timestamp', or 'time'  does not exist");
    }
    return timestamp_field;
}

std::vector<double> NormalizeTimestamps(const std::vector<double> &timestamps)
{
    const auto [min_it, max_it] = std::minmax_element(timestamps.cbegin(), timestamps.cend());
    const double min_timestamp = *min_it;
    const double max_timestamp = *max_it;

    std::vector<double> timestamps_normalized(timestamps.size());
    std::transform(timestamps.cbegin(), timestamps.cend(), timestamps_normalized.begin(),
                   [&](const auto &timestamp)
                   {
                       return (timestamp - min_timestamp) / (max_timestamp - min_timestamp);
                   });
    return timestamps_normalized;
}

#include <pcl/visualization/pcl_visualizer.h>

Sophus::SE3 registerClouds(pcl::PointCloud<PointType>::Ptr &src, pcl::PointCloud<PointType>::Ptr &tgt,
                           pcl::PointCloud<PointType>::Ptr &cloud_aligned)
{
    std::cout << "src:" << src->size() << ", tgt:" << tgt->size() << std::endl;
    std::cout << "\033[31mStart registration...\033[0m" << std::endl;
    pcl::GeneralizedIterativeClosestPoint<PointType, PointType> icp;

    icp.setInputSource(src);
    icp.setInputTarget(tgt);

    icp.setMaximumIterations(200);
    icp.setMaxCorrespondenceDistance(1.); // m

    // Perform the alignment
    icp.align(*cloud_aligned);

    auto init_T = Sophus::SE3();

    if (icp.hasConverged())
    {
        std::cout << "ICP converged." << std::endl
                  << "The score is " << icp.getFitnessScore() << std::endl;
        Eigen::Matrix4f transformation = icp.getFinalTransformation();

        Eigen::Matrix4d T = transformation.cast<double>();
        Sophus::SE3 refinement_T(T.block<3, 3>(0, 0), T.block<3, 1>(0, 3));
        std::cout << "refinement_T:\n" << refinement_T.log().transpose() << std::endl;

        init_T = refinement_T; // vux 2 mls
    }
    else
    {
        std::cout << "\033[31mICP did not converge...Handle this\033[0m" << std::endl;
        // TODO - handle this
    }

    std::cout << "curr als_to_mls:" << init_T.log().transpose() << std::endl;

    if (false)
    {
        pcl::visualization::PCLVisualizer viewer("ICP Example");
        // Original point clouds
        pcl::visualization::PointCloudColorHandlerCustom<PointType> color_source(src, 255, 0, 0);            // Red
        pcl::visualization::PointCloudColorHandlerCustom<PointType> color_target(tgt, 0, 255, 0);            // Green
        pcl::visualization::PointCloudColorHandlerCustom<PointType> color_aligned(cloud_aligned, 0, 0, 255); // Blue

        viewer.addPointCloud(src, color_source, "cloud_source");
        viewer.addPointCloud(tgt, color_target, "cloud_target");
        viewer.addPointCloud(cloud_aligned, color_aligned, "cloud_aligned");

        viewer.spin();
    }

    return init_T;
}