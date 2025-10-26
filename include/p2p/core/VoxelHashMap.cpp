#include "VoxelHashMap.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <queue>
#include <tuple>

namespace
{
    struct ResultTuple
    {
        ResultTuple(std::size_t n)
        {
            source.reserve(n);
            target.reserve(n);
        }
        std::vector<V3D_4> source;
        std::vector<V3D_4> target;
    };

    struct ResultPlaneTuple
    {
        ResultPlaneTuple(std::size_t n)
        {
            source.reserve(n);
            target.reserve(n);
        }
        std::vector<V3D_4> source;
        std::vector<Eigen::Matrix<double, 4, 1>> target;
    };

    /*
    plane equation: Ax + By + Cz + D = 0
    convert to: A/D*x + B/D*y + C/D*z = -1
    solve: A0*x0 = b0
    where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
    normvec:  normalized x0
    */
    template <typename T>
    bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const Vector3dVector &points, const T &threshold)
    {
        Eigen::Matrix<T, NUM_MATCH_POINTS, 3> A;
        Eigen::Matrix<T, NUM_MATCH_POINTS, 1> b;
        A.setZero();
        b.setOnes();
        b *= -1.0f;

        // A/Dx + B/Dy + C/Dz + 1 = 0
        for (int j = 0; j < NUM_MATCH_POINTS; j++)
        {
            A(j, 0) = points[j].x();
            A(j, 1) = points[j].y();
            A(j, 2) = points[j].z();
        }

        Eigen::Matrix<T, 3, 1> normvec = A.colPivHouseholderQr().solve(b);
        T n = normvec.norm();

        pca_result(0) = normvec(0) / n;
        pca_result(1) = normvec(1) / n;
        pca_result(2) = normvec(2) / n;
        pca_result(3) = 1.0 / n;

        for (int j = 0; j < NUM_MATCH_POINTS; j++)
        {
            if (fabs(pca_result(0) * points[j].x() + pca_result(1) * points[j].y() + pca_result(2) * points[j].z() + pca_result(3)) > threshold)
            {
                return false;
            }
        }
        return true;
    }

    struct Neighbor
    {
        double distance;
        V3D_4 point;

        // Custom comparison operator to make the priority queue a max-heap by distance
        bool operator<(const Neighbor &other) const
        {
            return distance < other.distance; // Max-heap (largest distance at the top)
        }
    };

} // namespace

namespace p2p
{
    Vector3dVectorTuple VoxelHashMap::GetPointCorrespondences(
        const Vector3dVector &points, double max_correspondance_distance) const
    {
        // Lambda Function to obtain the KNN of one point, maybe refactor
        auto GetClosestNeighboor = [&](const V3D_4 &point)
        {
            const auto &voxel = PointToVoxel(point, inv_voxel_size_);
            size_t current_index = 0;
            std::vector<Voxel> voxels(27);
            for (int i = voxel.x() - 1; i <= voxel.x() + 1; ++i)
            {
                for (int j = voxel.y() - 1; j <= voxel.y() + 1; ++j)
                {
                    for (int k = voxel.z() - 1; k <= voxel.z() + 1; ++k)
                    {
                        voxels[current_index++] = Voxel(i, j, k);
                    }
                }
            }

            V3D_4 closest_neighbor = Zero3d;
            double closest_distance2 = std::numeric_limits<double>::max(), distance;
            std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &query_voxel)
                          {
                auto search = map_.find(query_voxel);
                if (search != map_.end()) {
                    const auto &points = search->second.points;
                    const V3D_4 &neighbor = *std::min_element(
                        points.cbegin(), points.cend(), [&](const auto &lhs, const auto &rhs) {
                            return (lhs - point).squaredNorm() < (rhs - point).squaredNorm();
                        });
                    distance = (neighbor - point).squaredNorm();
                    if (distance < closest_distance2) {
                        closest_neighbor = neighbor;
                        closest_distance2 = distance;
                    }
                } });

            return std::make_tuple(closest_neighbor, closest_distance2);
        };

        using points_iterator = std::vector<V3D_4>::const_iterator;
        double max_correspondance_distance_2 = max_correspondance_distance * max_correspondance_distance;
        const auto [source, target] = tbb::parallel_reduce(
            // Range
            tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
            // Identity
            ResultTuple(points.size()),
            // 1st lambda: Parallel computation
            [max_correspondance_distance_2, &GetClosestNeighboor](
                const tbb::blocked_range<points_iterator> &r, ResultTuple res) -> ResultTuple
            {
                auto &[src, tgt] = res;
                src.reserve(r.size());
                tgt.reserve(r.size());
                for (const auto &point : r)
                {
                    const auto &[closest_neighboors, closest_sq_distance] = GetClosestNeighboor(point);
                    if (closest_sq_distance < max_correspondance_distance_2)
                    {
                        src.emplace_back(point);
                        tgt.emplace_back(closest_neighboors);
                    }
                }
                return res;
            },
            // 2nd lambda: Parallel reduction
            [](ResultTuple a, const ResultTuple &b) -> ResultTuple
            {
                auto &[src, tgt] = a;
                const auto &[srcp, tgtp] = b;
                src.insert(src.end(), //
                           std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
                tgt.insert(tgt.end(), //
                           std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
                return a;
            });

        return std::make_tuple(source, target);
    }

    std::vector<V3D_4> VoxelHashMap::Pointcloud() const
    {
        std::vector<V3D_4> points; // try to use index here
        points.reserve(max_points_per_voxel_ * map_.size());
        for (const auto &[voxel, voxel_block] : map_)
        {
            (void)voxel;
            for (const auto &point : voxel_block.points)
            {
                points.emplace_back(point);
            }
        }
        return points;
    }

    void VoxelHashMap::Update(const Vector3dVector &points, const V3D_4 &origin)
    {
        AddPoints(points);

        // if(timer_%20 == 0)
        {
            RemovePointsFarFromLocation(origin);
        }

        // timer_++;
    }

    void VoxelHashMap::Update(const Vector3dVector &points, const Sophus::SE3 &pose)
    {
        Vector3dVector points_transformed = points;
        TransformPoints(pose, points_transformed);
        const V3D_4 &origin = pose.translation();
        Update(points_transformed, origin);
    }

    void VoxelHashMap::AddPoints(const std::vector<V3D_4> &points)
    {
        std::cout << "ALS map AddPoints: " << points.size() << std::endl;
        for (int i = 0; i < points.size(); i++)
        {
            const auto &voxel = PointToVoxel(points[i], inv_voxel_size_);

            auto search = map_.find(voxel);
            if (search != map_.end())
            {
                auto &voxel_block = search.value();
                voxel_block.AddPoint(points[i]);
            }
            else
            {
                map_.insert({voxel, VoxelBlock{{points[i]}, max_points_per_voxel_}});
            }
        }
    }

    void VoxelHashMap::RemovePointsFarFromLocation(const V3D_4 &origin)
    {
        std::cout << "RemovePointsFarFromLocation:" << std::endl;
        const auto max_distance2 = max_distance_ * max_distance_;
        for (const auto &[voxel, voxel_block] : map_)
        {
            const auto &pt = voxel_block.points.front();
            if ((pt - origin).squaredNorm() > max_distance2)
            {
                map_.erase(voxel);
            }
        }
    }

    Vector3dNormalTuple VoxelHashMap::GetPlaneCorrespondences(
        const Vector3dVector &points, double max_correspondance_distance) const
    {
        // Lambda Function to obtain the KNN of one point
        auto GetClosestNeighboor = [&](const V3D_4 &point, const double &max_sq_dist)
        {
            const auto &voxel = PointToVoxel(point, inv_voxel_size_);
            size_t current_index = 0;
            std::vector<Voxel> voxels(27);
            for (int i = voxel.x() - 1; i <= voxel.x() + 1; ++i)
            {
                for (int j = voxel.y() - 1; j <= voxel.y() + 1; ++j)
                {
                    for (int k = voxel.z() - 1; k <= voxel.z() + 1; ++k)
                    {
                        voxels[current_index++] = Voxel(i, j, k);
                    }
                }
            }

            std::priority_queue<Neighbor> max_heap; // Max-heap to keep closest points
            double distance;
            // Loop over neighboring voxels
            std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &query_voxel)
                          {
                auto search = map_.find(query_voxel);
                if (search != map_.end()) {
                    const auto& points = search->second.points;
                    for (const auto& neighbor : points) {
                        distance = (neighbor - point).squaredNorm();
                        if(distance < max_sq_dist){ //point is within the threshold
                            // If we have fewer than NUM_MATCH_POINTS neighbors, add to the heap
                            if (max_heap.size() < NUM_MATCH_POINTS) {
                                max_heap.emplace(Neighbor{distance, neighbor});
                            }
                            // If the current neighbor is closer than the farthest in our top 5, replace it
                            else if (distance < max_heap.top().distance) {
                                max_heap.pop();
                                max_heap.emplace(Neighbor{distance, neighbor});
                            }
                        }
                    }
                } });

            Eigen::Matrix<double, 4, 1> pabcd;
            bool rv = false;
            if (max_heap.size() == NUM_MATCH_POINTS)
            {
                // if (max_heap.top().first < max_sq_dist)
                //{
                Vector3dVector plane(max_heap.size()); // plane has the points sorted by dist
                for (int i = max_heap.size() - 1; i >= 0; --i)
                {
                    plane[i] = max_heap.top().point;
                    max_heap.pop();
                }
                rv = true;
                if (esti_plane(pabcd, plane, .1)) // good plane
                // if(false)
                {
                    pabcd(3) = pabcd(0) * point.x() + pabcd(1) * point.y() + pabcd(2) * point.z() + pabcd(3); // p2plane residual
                }
                else
                {
                    rv = false; // this does not use p2p yet
                    const V3D_4 &closest = plane.front();
                    pabcd(0) = closest.x();
                    pabcd(1) = closest.y();
                    pabcd(2) = closest.z();
                    pabcd(3) = -100;
                }
                //}
            }

            return std::make_tuple(pabcd, rv);
        };

        using points_iterator = std::vector<V3D_4>::const_iterator;
        double max_correspondance_distance_2 = max_correspondance_distance * max_correspondance_distance;
        const auto [source, target] = tbb::parallel_reduce(
            // Range
            tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
            // Identity
            ResultPlaneTuple(points.size()),
            // 1st lambda: Parallel computation
            [max_correspondance_distance_2, &GetClosestNeighboor](
                const tbb::blocked_range<points_iterator> &r, ResultPlaneTuple res) -> ResultPlaneTuple
            {
                auto &[src, tgt] = res;
                src.reserve(r.size());
                tgt.reserve(r.size());
                for (const auto &point : r)
                {
                    const auto &[closest_neighboors, rv] = GetClosestNeighboor(point, max_correspondance_distance_2);
                    if (rv)
                    {
                        src.emplace_back(point);
                        tgt.emplace_back(closest_neighboors);
                    }
                }
                return res;
            },
            // 2nd lambda: Parallel reduction
            [](ResultPlaneTuple a, const ResultPlaneTuple &b) -> ResultPlaneTuple
            {
                auto &[src, tgt] = a;
                const auto &[srcp, tgtp] = b;
                src.insert(src.end(), //
                           std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
                tgt.insert(tgt.end(), //
                           std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
                return a;
            });

        return std::make_tuple(source, target);
    }

} // namespace p2p