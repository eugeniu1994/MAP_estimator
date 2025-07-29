#include "RIEKF.hpp"

using namespace ekf;

PointCloudXYZI::Ptr normvec(new PointCloudXYZI(50000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(50000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(50000, 1));
std::vector<bool> point_selected_surf(50000, 1);
std::vector<double> normvec_var(50000, 0);
std::vector<double> corr_normvec_var(50000, 0);
std::vector<std::vector<double>> Nearest_Points_Weights;

std::vector<M3D> tgt_covs(50000, M3D::Zero());
std::vector<V3D> laserCloudTgt(50000, V3D::Zero());

std::vector<M3D> corr_tgt_covs(50000, M3D::Zero());
std::vector<V3D> corr_laserCloudTgt(50000, V3D::Zero());

std::vector<M3D> src_covs(50000, M3D::Zero());
std::vector<V3D> laserCloudSrc(50000, V3D::Zero());

#define USE_STATIC_KDTREE 1

#if USE_STATIC_KDTREE == 0

void RIEKF::lidar_observation_model(residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
                                    KD_TREE<PointType> &ikdtree, std::vector<PointVector> &Nearest_Points, bool extrinsic_est)
{
    int feats_down_size = feats_down_body->points.size();
    laserCloudOri->clear();
    corr_normvect->clear();

#ifdef MP_EN
    // omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType point_world;

        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);

        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 1 ? false
                                                                                                                                : true;
        }
        if (!point_selected_surf[i])
            continue;

        Eigen::Matrix<float, 4, 1> pabcd;
        point_selected_surf[i] = false;

#ifdef ADAPTIVE_KERNEL
        if (ekf::esti_plane(pabcd, points_near, 0.2f))
#else
        if (ekf::esti_plane(pabcd, points_near, 0.1f))
#endif
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            // float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
            float s = 1 - 0.9 * fabs(pd2) / point_body.intensity;

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
            }
        }
    }

    effct_feat_num = 0;
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 5)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        std::cout << "effct_feat_num:" << effct_feat_num << std::endl;
        return;
    }

    // Calculation of Jacobian matrix H and residual vector
    ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, noise_size); // N * 12
    ekfom_data.innovation.resize(effct_feat_num);

#ifdef ADAPTIVE_KERNEL
    double kernel = 3. * sigma;
    auto Weight = [&](double residual2)
    {
        return square(kernel) / square(kernel + residual2);
    };
#endif

#ifdef MP_EN
    // omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {
        V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_);
        V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
        M3D point_I_crossmat;
        point_I_crossmat << SKEW_SYM_MATRX(point_I_);

        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        V3D C(x_.rot.matrix().transpose() * norm_vec);
        V3D A(point_I_crossmat * C);
        if (extrinsic_est)
        {
            V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
            ekfom_data.h_x.block<1, noise_size>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 6>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A); //, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        ekfom_data.innovation(i) = -norm_p.intensity;

#ifdef ADAPTIVE_KERNEL
        double w = Weight(square(-norm_p.intensity));
        ekfom_data.innovation(i) *= w;
        ekfom_data.h_x(i) *= w;
#endif
    }
}

void RIEKF::observtion_model_parallel(residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
                                      KD_TREE<PointType> &ikdtree, std::vector<PointVector> &Nearest_Points, bool extrinsic_est)
{
    int feats_down_size = feats_down_body->points.size();
    laserCloudOri->clear();
    corr_normvect->clear();

    tbb::parallel_for(tbb::blocked_range<int>(0, feats_down_size),
                      [&](tbb::blocked_range<int> r)
                      {
                          for (int i = r.begin(); i < r.end(); i++)
                          {
                              PointType &point_body = feats_down_body->points[i];
                              PointType point_world;
                              V3D p_body(point_body.x, point_body.y, point_body.z);
                              V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);
                              point_world.x = p_global(0);
                              point_world.y = p_global(1);
                              point_world.z = p_global(2);
                              point_world.intensity = point_body.intensity;

                              std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
                              auto &points_near = Nearest_Points[i];

                              if (ekfom_data.converge)
                              {
                                  ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);

                                  point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 1 ? false
                                                                                                                                                      : true;
                              }

                              if (!point_selected_surf[i])
                                  continue; // If the point does not meet the conditions, do not proceed to the following steps

                              Eigen::Matrix<float, 4, 1> pabcd; // plane point information
                              point_selected_surf[i] = false;   // Set the point as an invalid point to determine whether the condition is met
// Fit the plane equation ax+by+cz+d=0 and solve the point-to-plane distance
#ifdef ADAPTIVE_KERNEL
                              if (ekf::esti_plane(pabcd, points_near, 0.2f))
#else
							  if (ekf::esti_plane(pabcd, points_near, 0.1f))
#endif
                              {
                                  float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                                  // float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
                                  float s = 1 - 0.9 * fabs(pd2) / point_body.intensity;

                                  if (s > 0.9)
                                  {
                                      point_selected_surf[i] = true;
                                      normvec->points[i].x = pabcd(0);
                                      normvec->points[i].y = pabcd(1);
                                      normvec->points[i].z = pabcd(2);
                                      normvec->points[i].intensity = pd2;
                                  }
                              }
                          }
                      });

    effct_feat_num = 0;
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 5)
    {
        ekfom_data.valid = false;
        ROS_WARN("No - ekf Effective Points! \n");
        std::cout << "effct_feat_num:" << effct_feat_num << std::endl;
        return;
    }

    // Calculation of Jacobian matrix H and residual vector
    ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, noise_size); // derivative of the observation model  N x 12
    ekfom_data.innovation.resize(effct_feat_num);                       // residuals

#ifdef ADAPTIVE_KERNEL
    double kernel = 3. * sigma;
    auto Weight = [&](double residual2)
    {
        return square(kernel) / square(kernel + residual2);
    };
#endif

    tbb::parallel_for(tbb::blocked_range<int>(0, effct_feat_num),
                      [&](tbb::blocked_range<int> r)
                      {
                          for (int i = r.begin(); i < r.end(); i++)
                          // for (int i = 0; i < effct_feat_num; i++)
                          {
                              V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
                              M3D point_crossmat;
                              point_crossmat << SKEW_SYM_MATRX(point_);
                              V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
                              M3D point_I_crossmat;
                              point_I_crossmat << SKEW_SYM_MATRX(point_I_);

                              // Get the normal vector of the corresponding plane
                              const PointType &norm_p = corr_normvect->points[i];
                              V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

                              // Calculate the Jacobian matrix H
                              V3D C(x_.rot.matrix().transpose() * norm_vec); // rotate norm vector from world to local sensor frame
                              V3D A(point_I_crossmat * C);                   // P(IMU)^ [R(imu <-- w) * normal_w]
                              if (extrinsic_est)
                              {
                                  // B = lidar_p^ R(L <-- I) * corr_normal_I
                                  // B = lidar_p^ R(L <-- I) * R(I <-- W) * normal_W
                                  V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
                                  // ekfom_data.h_x.block<1, state_size>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
                                  ekfom_data.h_x.block<1, noise_size>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
                              }
                              else
                              {
                                  ekfom_data.h_x.block<1, 6>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A); //, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
                              }
                              // Residuals: point-to-plane distance
                              ekfom_data.innovation(i) = -norm_p.intensity;

#ifdef ADAPTIVE_KERNEL
                              double w = Weight(square(-norm_p.intensity));
                              ekfom_data.innovation(i) *= w;
                              ekfom_data.h_x(i) *= w;
#endif
                          }
                      });
}

bool RIEKF::update(double R, PointCloudXYZI::Ptr &feats_down_body, KD_TREE<PointType> &ikdtree,
                   std::vector<PointVector> &Nearest_Points, int maximum_iter, bool extrinsic_est)
{
    normvec->resize(int(feats_down_body->points.size()));

    residual_struct status;
    status.valid = true;
    status.converge = true;
    int t = 0;

    if (ikdtree.size() < 5)
    {
        std::cerr << "Error: ikdtree.size() Point cloud is empty! : " << ikdtree.size() << std::endl;
        status.valid = false;
        status.converge = false;
        return false;
    }

    state x_propagated = x_;
    vectorized_state dx_new = vectorized_state::Zero(); // 24X1

#ifdef ADAPTIVE_KERNEL
    sigma = adaptive_threshold.ComputeThreshold();
    // std::cout << "sigma:" << sigma << ", max_dist:" << 3.0 * sigma << std::endl;
    sigma = std::max(sigma, .5);
#endif

    for (int i = -1; i < maximum_iter; i++)
    {
        status.valid = true;
        // z = y - h(x)
        // lidar_observation_model(status, feats_down_body, ikdtree, Nearest_Points, extrinsic_est);
        observtion_model_parallel(status, feats_down_body, ikdtree, Nearest_Points, extrinsic_est);

        if (!status.valid)
        {
            // no features found
            break;
        }

        vectorized_state dx;
        dx_new = boxminus(x_, x_propagated); // x^k - x^

        // noise_size = 12
        //  Since the H matrix is sparse, only the first 12 columns have non-zero elements,
        //   and the last 12 columns are zero, so the calculation is performed in the form of
        //   a block matrix to reduce the amount of calculation
        auto H = status.h_x;   // m X 12 the matrix,  where m is the number of feature points
        cov HTH = cov::Zero(); // matrix H^T * H   state_size = 24

        HTH.block<noise_size, noise_size>(0, 0) = H.transpose() * H;

        auto K_front = (HTH / R + P_.inverse()).inverse(); // 24x24      formula 20 beginning here
        // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;			 // this should be 24 x m
        Eigen::Matrix<double, state_size, Eigen::Dynamic> K;
        K = K_front.block<state_size, noise_size>(0, 0) * H.transpose() / R; //   Kalman Gain;, Here R is treated as a constant

        cov KH = cov::Zero(); //  matrix K * H
        KH.block<state_size, noise_size>(0, 0) = K * H;
        // K*z is positive since z is used as -z
        // K*z + (K*H-I)*(x_-x_pred)

        // K​=P*​H^T​(H*​P*​H^T​+Rk​)−1  the standard ekf K gain    K_k = P_k * H_k.T * inv( H_k * P_k * H_k.T + R_k )
        // x = x + (K*(z−h(x)) + (K*H − I)*(x-x_k))   ikf        status.innovation = z−h(x)

        //  K is     K = (H^T*R^−1*H  +  P^−1)*H^T*R^-1

        vectorized_state dx_ = K * status.innovation + (KH - cov::Identity()) * dx_new;
        x_ = boxplus(x_, dx_); // dx_ is the delta corection that should be applied

        status.converge = true;
        for (int j = 0; j < state_size; j++)
        {
            if (std::fabs(dx_[j]) > ESTIMATION_THRESHOLD_) // If dx_>ESTIMATION_THRESHOLD_ no convergence is considered
            {
                status.converge = false;
                break;
            }
        }

        if (status.converge)
            t++;

        if (!t && i == maximum_iter - 2)
        {
            status.converge = true;
        }

        if (t > 1 || i == maximum_iter - 1)
        {
            P_ = (cov::Identity() - KH) * P_;
            break;
        }
    }

#ifdef ADAPTIVE_KERNEL
    Sophus::SE3 initial_guess(x_propagated.rot, x_propagated.pos);
    Sophus::SE3 new_pose(x_.rot, x_.pos);
    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold.UpdateModelDeviation(model_deviation);
#endif

    return status.valid;
}

#else

// pcl::KdTreeFLANN<PointType>::Ptr localKdTree_map(new pcl::KdTreeFLANN<PointType>());

void RIEKF::lidar_observation_model(residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
                                    PointCloudXYZI::Ptr &map, std::vector<PointVector> &Nearest_Points, bool extrinsic_est)
{

    // std::cout<<"lidar_observation_model- extrinsic_est = "<<extrinsic_est<<std::endl;
    int feats_down_size = feats_down_body->points.size();
    laserCloudOri->clear();
    corr_normvect->clear();

#ifdef MP_EN
    // omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType point_world;

        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);

        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        std::vector<int> pointSearchInd(NUM_MATCH_POINTS);
        std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];
        if (ekfom_data.converge)
        {
            if (localKdTree_map->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
            {
                // localKdTree_map->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis);
                point_selected_surf[i] = pointSearchSqDis[NUM_MATCH_POINTS - 1] > 1 ? false : true; // true if distance smaller than 1

                points_near.clear();
                for (int j = 0; j < pointSearchInd.size(); j++)
                {
                    // points_near.push_back(map->points[pointSearchInd[j]]);
                    points_near.insert(points_near.begin(), map->points[pointSearchInd[j]]);
                }
            }
            else
            {
                point_selected_surf[i] = false;
            }
        }
        if (!point_selected_surf[i])
            continue;

        Eigen::Matrix<float, 4, 1> pabcd;
        point_selected_surf[i] = false;
#ifdef ADAPTIVE_KERNEL
        if (ekf::esti_plane(pabcd, points_near, 0.2f))
#else
        if (ekf::esti_plane(pabcd, points_near, 0.1f))
#endif
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            // float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
            float s = 1 - 0.9 * fabs(pd2) / point_body.intensity;

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
            }
        }
    }

    effct_feat_num = 0;
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            effct_feat_num++;
        }
    }

    if (effct_feat_num < 5)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        std::cout << "effct_feat_num:" << effct_feat_num << std::endl;
        return;
    }

    // Calculation of Jacobian matrix H and residual vector
    ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, noise_size); // N * 12
    ekfom_data.innovation.resize(effct_feat_num);

#ifdef ADAPTIVE_KERNEL
    double kernel = 3. * sigma;
    auto Weight = [&](double residual2)
    {
        return square(kernel) / square(kernel + residual2);
    };
#endif

#ifdef MP_EN
    // omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {
        V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_);
        V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
        M3D point_I_crossmat;
        point_I_crossmat << SKEW_SYM_MATRX(point_I_);

        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        V3D C(x_.rot.matrix().transpose() * norm_vec);
        V3D A(point_I_crossmat * C);
        if (extrinsic_est)
        {
            V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
            ekfom_data.h_x.block<1, noise_size>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 6>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A); //, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        ekfom_data.innovation(i) = -norm_p.intensity;

#ifdef ADAPTIVE_KERNEL
        double w = Weight(square(-norm_p.intensity));
        ekfom_data.innovation(i) *= w;
        ekfom_data.h_x(i) *= w;
#endif
    }
}

bool RIEKF::update(double R, PointCloudXYZI::Ptr &feats_down_body, PointCloudXYZI::Ptr &map,
                   std::vector<PointVector> &Nearest_Points, int maximum_iter, bool extrinsic_est)
{
    normvec->resize(int(feats_down_body->points.size()));

    residual_struct status;
    status.valid = true;
    status.converge = true;
    int t = 0;

    state x_propagated = x_;
    vectorized_state dx_new = vectorized_state::Zero(); // 24X1

    if (map->points.size() < 5)
    {
        std::cerr << "Error: map Point cloud is empty! : " << map->points.size() << std::endl;
        status.valid = false;
        status.converge = false;
        return false;
    }

    localKdTree_map->setInputCloud(map);

#ifdef ADAPTIVE_KERNEL
    sigma = adaptive_threshold.ComputeThreshold();
    // std::cout << "sigma:" << sigma << ", max_dist:" << 3.0 * sigma << std::endl;
    sigma = std::max(sigma, .5);

#endif
    for (int i = -1; i < maximum_iter; i++) // maximum_iter
    {
        status.valid = true;
        // z = y - h(x)
        lidar_observation_model(status, feats_down_body, map, Nearest_Points, extrinsic_est);
        // observtion_model_parallel(status, feats_down_body, ikdtree, Nearest_Points, extrinsic_est);

        if (!status.valid)
        {
            break;
        }

        vectorized_state dx;
        dx_new = boxminus(x_, x_propagated); // x^k - x^ in formula (18)

        auto H = status.h_x;   // m X 12 the matrix,  where m is the number of feature points
        cov HTH = cov::Zero(); // matrix H^T * H   state_size = 24

        HTH.block<noise_size, noise_size>(0, 0) = H.transpose() * H;

        auto K_front = (HTH / R + P_.inverse()).inverse(); // 24x24      formula 20 beginning here
        // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> K;			 // this should be 24 x m
        Eigen::Matrix<double, state_size, Eigen::Dynamic> K;
        K = K_front.block<state_size, noise_size>(0, 0) * H.transpose() / R; //   Kalman Gain;, Here R is treated as a constant

        cov KH = cov::Zero(); //  matrix K * H
        KH.block<state_size, noise_size>(0, 0) = K * H;
        // K*z is positive since z is used as -z
        // K*z + (K*H-I)*(x_-x_pred)

        // K​=P*​H^T​(H*​P*​H^T​+Rk​)−1  the standard ekf K gain    K_k = P_k * H_k.T * inv( H_k * P_k * H_k.T + R_k )
        // x = x + (K*(z−h(x)) + (K*H − I)*(x-x_k))   ikf        status.innovation = z−h(x)

        // but here K is  K = (H^T*R^−1*H  +  P^−1)*H^T*R^-1

        vectorized_state dx_ = K * status.innovation + (KH - cov::Identity()) * dx_new;
        x_ = boxplus(x_, dx_); // dx_ is the delta corection that should be applied

        status.converge = true;
        for (int j = 0; j < state_size; j++)
        {
            if (std::fabs(dx_[j]) > ESTIMATION_THRESHOLD_) // If dx_>ESTIMATION_THRESHOLD_ no convergence is considered
            {
                status.converge = false;
                break;
            }
        }

        if (status.converge)
            t++;

        if (!t && i == maximum_iter - 2)
        {
            status.converge = true;
        }

        if (t > 1 || i == maximum_iter - 1)
        {
            P_ = (cov::Identity() - KH) * P_;
            break;
        }
    }

#ifdef ADAPTIVE_KERNEL
    Sophus::SE3 initial_guess(x_propagated.rot, x_propagated.pos);
    Sophus::SE3 new_pose(x_.rot, x_.pos);
    const auto model_deviation = initial_guess.inverse() * new_pose;
    adaptive_threshold.UpdateModelDeviation(model_deviation);
#endif

    return status.valid;
}

void RIEKF::lidar_observation_model_tighly_fused(residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
                                                 PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als,
                                                 const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als,
                                                 std::vector<PointVector> &Nearest_Points, bool extrinsic_est)
{

    int feats_down_size = feats_down_body->points.size();
    laserCloudOri->clear();
    corr_normvect->clear();
    // int p_mls = 0, p_als=0;

#ifdef MP_EN
// std::cout<<"run lidar_observation_model_tighly_fused in parallel..."<<std::endl;
// omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType point_world;

        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);

        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        std::vector<int> pointSearchInd(NUM_MATCH_POINTS);
        std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];
        std::vector<double> point_weights;

        // TODO - there is a bug here, if not converged, search new NN
        // and perform p2plane with new data, else compute new cost with prev planes

        if (ekfom_data.converge)
        {
            points_near.clear();
            point_selected_surf[i] = false;

            // search als neighbours
            if (localKdTree_map_als->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
            {
                if (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 1)
                {
                    point_selected_surf[i] = true;
                    for (int j = 0; j < pointSearchInd.size(); j++)
                    {
                        points_near.push_back(map_als->points[pointSearchInd[j]]);
                        point_weights.push_back(100.);
                    }
                    // p_als++;
                }
            }

            // if (point_selected_surf[i] == false) // not valid neighbours has been found
            {
                pointSearchInd.clear();
                pointSearchSqDis.clear();
                // search mls neighbours
                if (localKdTree_map->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
                {
                    if (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 1)
                    {
                        point_selected_surf[i] = true;

                        for (int j = 0; j < pointSearchInd.size(); j++)
                        {
                            points_near.push_back(map_mls->points[pointSearchInd[j]]);
                            // points_near.insert(points_near.begin(), map_mls->points[pointSearchInd[j]]);
                            point_weights.push_back(1.);
                        }
                        // p_mls++;
                    }
                }
            }
            //}

            if (!point_selected_surf[i]) // src point does not have neighbours
                continue;

            Eigen::Matrix<float, 4, 1> pabcd;
            point_selected_surf[i] = false;

            if (ekf::esti_plane_pca(pabcd, points_near, .03, point_weights, true))
            // if (ekf::esti_plane2(pabcd, points_near, .2f)) //.1f
            {
                float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                // float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
                float s = 1 - 0.9 * fabs(pd2) / point_body.intensity;

                if (s > 0.9)
                {
                    point_selected_surf[i] = true;
                    normvec->points[i].x = pabcd(0);
                    normvec->points[i].y = pabcd(1);
                    normvec->points[i].z = pabcd(2);
                    normvec->points[i].intensity = pd2;
                }
            }
        }
    }

    // std::cout<<"p_mls:"<<p_mls<<", p_als:"<<p_als<<std::endl;

    effct_feat_num = 0;
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            effct_feat_num++;
        }
    }

    // std::cout << "effct_feat_num: " << effct_feat_num << "/" << feats_down_size << std::endl;
    if (effct_feat_num < 5)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! cannot find good planes \n");
        std::cout << "effct_feat_num:" << effct_feat_num << " cannot find good planes" << std::endl;
        throw std::runtime_error("Stop here - the system will collapes, no good planes found");
        return;
    }

    // Calculation of Jacobian matrix H and residual vector
    ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, used_state_size); // m * 12
    ekfom_data.innovation.resize(effct_feat_num);

#ifdef MP_EN
    // omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {
        V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_);
        V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
        M3D point_I_crossmat;
        point_I_crossmat << SKEW_SYM_MATRX(point_I_);

        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        V3D C(x_.rot.matrix().transpose() * norm_vec);
        V3D A(point_I_crossmat * C);
        if (extrinsic_est)
        {
            V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 6>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A); //, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        ekfom_data.innovation(i) = -norm_p.intensity;
    }
}

bool RIEKF::update_tighly_fused(double R, PointCloudXYZI::Ptr &feats_down_body,
                                PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als,
                                const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als,
                                std::vector<PointVector> &Nearest_Points, int maximum_iter,
                                bool extrinsic_est)
{
    normvec->resize(int(feats_down_body->points.size()));

    residual_struct status;
    status.valid = true;
    status.converge = true;
    int t = 0;

    state x_propagated = x_;
    vectorized_state dx_new = vectorized_state::Zero(); // 24X1

    std::cout << "map_mls:" << map_mls->size() << ", map_als:" << map_als->size() << std::endl;
    if (map_mls->points.size() < 5)
    {
        std::cerr << "Error: map_mls Point cloud is empty! : " << map_mls->points.size() << std::endl;
        status.valid = false;
        status.converge = false;
        return false;
    }

    localKdTree_map->setInputCloud(map_mls);
    // localKdTree_map_als->setInputCloud(map_als);

    const cov P_inv = P_.inverse();
    for (int i = -1; i < maximum_iter; i++) // maximum_iter
    {
        status.valid = true;

        // std::cout<<"\nekfom_data.converge:"<<status.converge << ", iteration "<<i+1<<std::endl;

        // z = y - h(x)
        lidar_observation_model_tighly_fused(status, feats_down_body, map_mls, map_als, localKdTree_map_als, Nearest_Points, extrinsic_est);

        if (!status.valid)
        {
            break;
        }

        vectorized_state dx;
        dx_new = boxminus(x_, x_propagated); // x^k - x^ in formula (18)

        const auto &H = status.h_x; // m X 12 the matrix,  where m is the number of feature points
        cov HTH = cov::Zero();      // matrix H^T * H   state_size = 24

        HTH.block<used_state_size, used_state_size>(0, 0).noalias() = H.transpose() * H;

        auto K_front = (HTH / R + P_inv).inverse();                                         // 24x24      formula 20 beginning here
        Eigen::Matrix<double, state_size, Eigen::Dynamic> K;                                // 24 x m
        K.noalias() = K_front.block<state_size, used_state_size>(0, 0) * H.transpose() / R; //   Kalman Gain;, Here R is treated as a scalar variance

        cov KH = cov::Zero(); //  matrix K * H
        KH.block<state_size, used_state_size>(0, 0) = K * H;

        // vectorized_state dx_ = K * status.innovation + (KH - cov::Identity()) * dx_new;
        vectorized_state dx_;
        dx_.noalias() = K * status.innovation;
        dx_.noalias() += (KH - cov::Identity()) * dx_new;

        x_ = boxplus(x_, dx_); // dx_ is the delta corection that should be applied

        status.converge = true;
        for (int j = 0; j < state_size; j++)
        {
            if (std::fabs(dx_[j]) > ESTIMATION_THRESHOLD_) // If dx_>ESTIMATION_THRESHOLD_ no convergence is considered
            {
                status.converge = false;
                break;
            }
        }

        if (status.converge)
            t++;

        if (!t && i == maximum_iter - 2)
        {
            status.converge = true;
        }

        if (t > 1 || i == maximum_iter - 1)
        {
            P_ = (cov::Identity() - KH) * P_;
            break;
        }
    }

    return status.valid;
}

#endif

void RIEKF::update(const V3D &gnss_position, const V3D &cov_pos_, int maximum_iter, bool global_error, M3D R)
{
    std::cout << "UPdate EKF with gps..." << std::endl;

    const int gps_dim = 3;

    residual_struct status;
    status.valid = true;
    status.converge = true;
    state x_propagated = x_;                            // the initial guess
    vectorized_state dx_new = vectorized_state::Zero(); // 24X1

    Eigen::Matrix<double, gps_dim, state_size> H_gnss = Eigen::Matrix<double, gps_dim, state_size>::Zero();

    if (global_error)
    {
        // observation model h(x) = R*x+t Construct Jacobian matrix H_gnss      3 x 24
        H_gnss.block<3, 3>(0, 0) = R; // Position part
    }
    else
    {
        // observation model h(x) = I*x Construct Jacobian matrix H_gnss      3 x 24
        H_gnss.block<3, 3>(0, 0) = Eye3d; // Position part
    }

    M3D measurement_cov = Eye3d; // R * Matrix6d::Identity()
    measurement_cov.block<3, 3>(0, 0) = cov_pos_.asDiagonal();

    for (int i = -1; i < maximum_iter; i++)
    {
        // Compute Kalman Gain
        // K_k =  P_k   *   H_k.T * inv( H_k  *  P_k  * H_k.T + R_k )
        // 24X24  *   24x3  * inv(3x24  * 24x24 * 24x3  + 3x3

        Eigen::Matrix<double, state_size, gps_dim> K; // this should be 24 x 3
        K = P_ * H_gnss.transpose() * ((H_gnss * P_ * H_gnss.transpose() + measurement_cov).inverse());

        // Residual calculation // y - h(x)  where y is measurement h(x) is observation model
        V3D residual;
        if (global_error)
            residual = gnss_position - R * x_.pos;
        else
            residual = gnss_position - x_.pos; // y - h(x)  where y is measurement h(x) is observation model

        // for iekf
        dx_new = boxminus(x_, x_propagated); // x^k - x^     // 24X1

        // vectorized_state dx_ = K * residual; //used before for gps_ins
        cov KH = cov::Zero(); //  matrix K * H
        KH = K * H_gnss;      // 24 x 3  * 3 x 24
        vectorized_state dx_ = K * residual + (KH - cov::Identity()) * dx_new;

        x_ = boxplus(x_, dx_);

        status.converge = true;
        for (int j = 0; j < state_size; j++)
        {
            if (std::fabs(dx_[j]) > ESTIMATION_THRESHOLD_)
            {
                status.converge = false;
                break;
            }
        }

        if (status.converge || i == maximum_iter - 1)
        {
            P_ = (cov::Identity() - K * H_gnss) * P_;
            break;
        }
    }
}

//---------------------------------------------------------------------------------------

// #define use_gps_measurements //uncomment this to integrate the gps tighly
/*
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{

    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i];
        PointType &point_world = feats_down_world->points[i];

        //transform to world frame
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        search new NN only if converge
        if (ekfom_data.converge)
        {
            // Find the closest surfaces in the map
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        next will estimate new cost with new pose and same planes
        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
}
*/

void RIEKF::observation_model_test(const double R, residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body, const V3D &gps,
                                   PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als,
                                   const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als,
                                   std::vector<PointVector> &Nearest_Points, bool extrinsic_est)
{

    int feats_down_size = feats_down_body->points.size();
    laserCloudOri->clear();
    corr_normvect->clear();
    corr_normvec_var.clear();
    // int p_mls = 0, p_als=0;

    double R_inv = 1. / R; // 1./0.05;  //R = 0.05 original version
    auto travelled_distance = x_.pos.norm();
    // if(x_.pos.norm() > 5)
    //     R_inv = 1./(9.*corr_normvec_var[i]); // 3 sigma

#ifdef MP_EN
// std::cout<<"run observation_model_test in parallel...MP_PROC_NUM:"<<MP_PROC_NUM<<std::endl;
// omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType point_world;

        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);

        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        std::vector<int> pointSearchInd(NUM_MATCH_POINTS);
        std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        // std::vector<double> point_weights;

        auto &points_near = Nearest_Points[i];
        auto &point_weights = Nearest_Points_Weights[i]; // std vector of weights for point i

        // std::cout<<"observation_model_test ekfom_data.converge:"<<ekfom_data.converge<<std::endl;
        if (ekfom_data.converge)
        {
            points_near.clear();
            point_selected_surf[i] = false;
            point_weights.clear();

            // search als neighbours
            // if (localKdTree_map_als->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
            // {
            //     if (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 1)
            //     {
            //         point_selected_surf[i] = true;
            //         for (int j = 0; j < pointSearchInd.size(); j++)
            //         {
            //             points_near.push_back(map_als->points[pointSearchInd[j]]);
            //             point_weights.push_back(100.);
            //         }
            //         // p_als++;
            //     }
            // }

            // if (point_selected_surf[i] == false) // not valid neighbours has been found
            {
                pointSearchInd.clear();
                pointSearchSqDis.clear();
                // search mls neighbours
                if (localKdTree_map->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
                {
                    if (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 1)
                    {
                        point_selected_surf[i] = true;

                        for (int j = 0; j < pointSearchInd.size(); j++)
                        {
                            points_near.push_back(map_mls->points[pointSearchInd[j]]);
                            // points_near.insert(points_near.begin(), map_mls->points[pointSearchInd[j]]);
                            point_weights.push_back(1.);
                        }
                        // p_mls++;
                    }
                }
            }
            //}
        }
        // added the above and next accolade - search new NN only if converge
        //{
        if (!point_selected_surf[i]) // src point does not have neighbours
            continue;

        Eigen::Matrix<float, 4, 1> pabcd;
        point_selected_surf[i] = false;
        double plane_var = 0;
        // if (ekf::esti_plane_pca(pabcd, points_near, .03, point_weights, true))
        //  if (ekf::esti_plane2(pabcd, points_near, .2f)) //.1f
        // std::cout<<"Before test...points_near:"<<points_near.size()<<",point_weights:"<<point_weights.size() <<std::endl;
        if (ekf::esti_plane_pca(pabcd, points_near, .03, point_weights, plane_var, true))
        {
            // std::cout<<"esti_plane_pca succeeded"<<std::endl;
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            // float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
            float s = 1 - 0.9 * fabs(pd2) / point_body.intensity;

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                // normvec_var[i] = plane_var;

                // std::cout<<"plane_var:"<<plane_var<<std::endl;

                if (travelled_distance > 5)
                    normvec_var[i] = 1. / (9. * plane_var); // 3 sigma
                else
                    normvec_var[i] = R_inv;
            }
        }
        //}
    }

    // std::cout<<"p_mls:"<<p_mls<<", p_als:"<<p_als<<std::endl;
    effct_feat_num = 0;
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            corr_normvec_var[effct_feat_num] = normvec_var[i]; // R represent the variance of the noise for each measurement
            // std::cout<<"var:"<<normvec_var[i]<<std::endl; //var:0.0001251750
            effct_feat_num++;
        }
    }

    // std::cout << "effct_feat_num: " << effct_feat_num << "/" << feats_down_size << std::endl;
    if (effct_feat_num < 5)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! cannot find good planes \n");
        std::cout << "effct_feat_num:" << effct_feat_num << " cannot find good planes" << std::endl;
        throw std::runtime_error("Stop here - the system will collapes, no good planes found");
        return;
    }

// Calculation of Jacobian matrix H and residual vector
#ifdef use_gps_measurements
    int m = effct_feat_num + gps_dim;
#else
    int m = effct_feat_num;
#endif

    ekfom_data.h_x = Eigen::MatrixXd::Zero(m, used_state_size);       // m * 24  =  m x 24   but can be smaller
    ekfom_data.innovation.resize(m);                                  // z_hat
    ekfom_data.H_T_R_inv = Eigen::MatrixXd::Zero(used_state_size, m); // n x m

    // double c = .1;// 1.0;
    // When r2≪c2:
    //  w(r2)≈1/c2(constant weight)
    //  → inliers are treated almost uniformly
    //  When r2≫c2:
    //  w(r2)→0
    //  → outliers are ignored
    //  auto Weight_GMLoss = [&](double r)
    //  {
    //      double denom = c + r*r;
    //      return c*c / (denom*denom);
    //  };

#ifdef MP_EN
    // omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {
        V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_);
        V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
        M3D point_I_crossmat;
        point_I_crossmat << SKEW_SYM_MATRX(point_I_);

        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        V3D C(x_.rot.matrix().transpose() * norm_vec);
        V3D A(point_I_crossmat * C);
        // if (extrinsic_est)
        // {
        //     V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
        //     ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        // }
        // else
        // {
        //     // d_t, d_R
        //     ekfom_data.h_x.block<1, 6>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A);
        // }

        Eigen::Matrix<double, 6, 1> J;                           // Eigen::Vector6d J;                               // 6x1
        J.block<3, 1>(0, 0) = V3D(norm_p.x, norm_p.y, norm_p.z); // df/dt
        J.block<3, 1>(3, 0) = A;                                 // df/dR

        ekfom_data.h_x.block<1, 6>(i, 0) = J;         // 1x6 -> m x 6
        ekfom_data.innovation(i) = -norm_p.intensity; // 0 - h(x)   m x 1

        // corr_normvec_var[i] stores the 1/R_inv
        ekfom_data.H_T_R_inv.block<6, 1>(0, i) = J.transpose() * corr_normvec_var[i]; // 6 x 1 * 1 x 1m

        // double w = 1.0; //this will be robust kernel
        // w = Weight_GMLoss(norm_p.intensity);
        // std::cout<<"w:"<<w<<std::endl;
        // ekfom_data.H_T_R_inv.block<6, 1>(0, i) = w * J.transpose() * R_inv; // 6 x 1 * 1 x 1m
    }

#ifdef use_gps_measurements
    // for gps
    //  Assign to last 3 positions
    ekfom_data.innovation.tail<3>() = gps - x_.pos; // y - h(x)  where y is measurement h(x) is observation model gps_innovation;
    // Assign GNSS measurement Jacobian to the bottom block of h_x
    ekfom_data.h_x.block<gps_dim, gps_dim>(effct_feat_num, 0) = Eye3d; // d e dT is identity put it directly here
                                                                       /// ekfom_data.h_x.block<gps_dim, state_size>(effct_feat_num, 0) = H_gnss;// / .05;  //gps_dim x state_size

    double w = 1.0;
    double R_inv = 1. / 0.05;                                                                        // R = 0.05
    ekfom_data.H_T_R_inv.block<gps_dim, gps_dim>(0, effct_feat_num) = w * Eye3d.transpose() * R_inv; // 6 x 1 * 1 x 1m
#endif
}

bool RIEKF::update_tighly_fused_test(double R, PointCloudXYZI::Ptr &feats_down_body,
                                     PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als,
                                     const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als,
                                     std::vector<PointVector> &Nearest_Points,
                                     const V3D &gps, double R_gps_cov,
                                     int maximum_iter, bool extrinsic_est)
{
    int features = int(feats_down_body->points.size());

    normvec->resize(features);
    normvec_var.resize(features);
    Nearest_Points_Weights.resize(features);
    std::cout << "update_tighly_fused_test..." << std::endl;

    residual_struct status;
    status.valid = true;
    status.converge = true;
    int t = 0;

    state x_propagated = x_;
    vectorized_state dx_new = vectorized_state::Zero(); // 24X1

    std::cout << "map_mls:" << map_mls->size() << ", map_als:" << map_als->size() << std::endl;
    if (map_mls->points.size() < 5)
    {
        std::cerr << "Error: map_mls Point cloud is empty! : " << map_mls->points.size() << std::endl;
        status.valid = false;
        status.converge = false;
        return false;
    }

    localKdTree_map->setInputCloud(map_mls);
    // localKdTree_map_als->setInputCloud(map_als);

    // #define debug_speed

#ifdef debug_speed
    // Initialize running averages
    double avg_time_inverse = 0;
    double avg_time_ldlt = 0;
    double avg_time_llt = 0;

#endif

    // double inv_R = 1.0 / R;
    // double inv_R_gps = 1.0 / R_gps_cov;

    const cov P_inv = P_.inverse();

    for (int i = -1; i < maximum_iter; i++) // maximum_iter
    {
        status.valid = true;
        // z = y - h(x)
        observation_model_test(R, status, feats_down_body, gps, map_mls, map_als, localKdTree_map_als, Nearest_Points, extrinsic_est);

        if (!status.valid)
        {
            break;
        }

        // before average update time 40102.75
        // now

        vectorized_state dx;
        dx_new = boxminus(x_, x_propagated); // x^k - x^

        const auto &H = status.h_x; // (m x n) Jacobian    m x 24

        // Optimized code with noalias
        Eigen::Matrix<double, state_size, Eigen::Dynamic> K; // n x m = 24 x m

        //       Eigen::MatrixXd Ht_Rinv = H.transpose();                       //   ( n x m)
        //       Ht_Rinv.block(0, 0, used_state_size, effct_feat_num) *= inv_R; //(start_row, start_col, num_rows, num_cols) - for the transposed version  runtime, a bit slow
        // #ifdef use_gps_measurements
        //         Ht_Rinv.block(0, effct_feat_num, used_state_size, gps_dim) *= inv_R_gps; // uncomment this to use GPS, i can use the 3d cov here also
        // #endif
        const auto &Ht_Rinv = status.H_T_R_inv; //   ( n x m)

        cov HTH = cov::Zero(); // (n x n)
        HTH.block<used_state_size, used_state_size>(0, 0).noalias() = Ht_Rinv * H;

        // auto K_front = (HTH + P_inv).inverse();
        cov &&K_front = (HTH + P_inv).inverse();

        K.noalias() = K_front.block<state_size, used_state_size>(0, 0) * Ht_Rinv; // direct inversion fastest for small matrices (n < 50)
        cov KH = cov::Zero();                                                     // (n x m) * (m x n) -> (n x n)
        KH.block<state_size, used_state_size>(0, 0).noalias() = K * H;

        // vectorized_state dx_ = K * status.innovation + (KH - cov::Identity()) * dx_new;
        vectorized_state dx_;
        dx_.noalias() = K * status.innovation;
        dx_.noalias() += (KH - cov::Identity()) * dx_new;

        {
#ifdef debug_speed

            // Avoid HTH.inverse() by using another LDLT decomposition
            // Slower than direct inverse for small matrices, faster for large matrices
            // K = HTH.ldlt().solve(Ht_Rinv);
            // K = HTH.selfadjointView<Eigen::Upper>().ldlt().solve(Ht_Rinv);
            // K = HTH.llt().solve(Ht_Rinv);

            // Method 2: LDLT decomposition
            auto start2 = high_resolution_clock::now();
            // Eigen::MatrixXd K_ldlt = HTH.ldlt().solve(Ht_Rinv);
            Eigen::MatrixXd K_ldlt = HTH.selfadjointView<Eigen::Upper>().ldlt().solve(Ht_Rinv);
            auto stop2 = high_resolution_clock::now();
            double time2 = duration_cast<microseconds>(stop2 - start2).count();

            // Method 1: Direct inverse
            auto start1 = high_resolution_clock::now();
            Eigen::MatrixXd K_inverse = HTH.inverse() * Ht_Rinv;
            auto stop1 = high_resolution_clock::now();
            double time1 = duration_cast<microseconds>(stop1 - start1).count();

            // Method 3: LLT decomposition
            auto start3 = high_resolution_clock::now();
            Eigen::MatrixXd K_llt = HTH.llt().solve(Ht_Rinv);
            auto stop3 = high_resolution_clock::now();
            double time3 = duration_cast<microseconds>(stop3 - start3).count();

            // Verify results are numerically equivalent
            double diff = (K_inverse - K_ldlt).norm();

            // Display results
            std::cout << "Kalman Gain Computation Benchmark (n=" << 24 << ", m=" << m << ")\n";
            std::cout << "----------------------------------------\n";
            std::cout << "Numerical difference:      " << diff << " (should be small)\n";

            std::cout << "Symmetry error: " << (HTH - HTH.transpose()).norm() << "\n";

            count++;
            avg_time_inverse += (time1 - avg_time_inverse) / count;
            avg_time_ldlt += (time2 - avg_time_ldlt) / count;
            avg_time_llt += (time3 - avg_time_llt) / count;
            std::cout << "Average time (direct inverse): " << avg_time_inverse << " μs\n";
            std::cout << "Average time (LDLT): " << avg_time_ldlt << " μs\n";
            std::cout << "Average time (LLT): " << avg_time_llt << " μs\n";

            std::cout << "Ratio (inverse/LDLT): " << avg_time_inverse / avg_time_ldlt << "\n";

            Eigen::LDLT<Eigen::MatrixXd> ldlt(HTH);
            if (ldlt.info() != Eigen::Success || !ldlt.isPositive())
            {
                std::cout << "LDLT decomposition failed or not positive definite.\n";
                throw std::runtime_error("Stop");
            }
#endif
        }

        x_ = boxplus(x_, dx_); // dx_ is the delta corection that should be applied

        status.converge = true;
        for (int j = 0; j < state_size; j++)
        {
            if (std::fabs(dx_[j]) > ESTIMATION_THRESHOLD_) // If dx_>ESTIMATION_THRESHOLD_ no convergence is considered
            {
                status.converge = false; // means re-optimize with same NN, no need to search for new NN
                break;
            }
        }

        if (status.converge)
            t++;

        if (!t && i == maximum_iter - 2)
        {
            status.converge = true;
        }

        if (t > 1 || i == maximum_iter - 1)
        {
            // Update covariance (Joseph form is more numerically stable)
            P_ = (cov::Identity() - KH) * P_;
            break;
        }
    }

    return status.valid;
}

void RIEKF::h(residual_struct &ekfom_data,
              double R_lidar_cov, double R_gps_cov,
              const PointCloudXYZI::Ptr &feats_down_body, const V3D &gps_pos,
              PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als,
              const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als, std::vector<PointVector> &Nearest_Points,
              bool extrinsic_est, bool use_gnss, bool use_als, bool tightly_coupled)
{
    int feats_down_size = feats_down_body->points.size();
    laserCloudOri->clear();
    corr_normvect->clear();
    corr_normvec_var.clear();

    double R_inv_lidar = 1. / R_lidar_cov; // 1./0.05;  //R = 0.05 original version

    double travelled_distance = x_.pos.norm();

    // 1)
    // find the new NN planes in MLS or ALS-MLS if converged
    // recompute the residuals with latest estimates
    {
#ifdef MP_EN
// omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
        for (int i = 0; i < feats_down_size; i++)
        {
            PointType &point_body = feats_down_body->points[i];
            PointType point_world;

            // transform to global with latest estimate
            V3D p_body(point_body.x, point_body.y, point_body.z);
            V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);

            point_world.x = p_global(0);
            point_world.y = p_global(1);
            point_world.z = p_global(2);
            point_world.intensity = point_body.intensity;

            std::vector<int> pointSearchInd(NUM_MATCH_POINTS);
            std::vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

            auto &points_near = Nearest_Points[i];
            auto &point_weights = Nearest_Points_Weights[i]; // vector of weights for point i

            if (ekfom_data.converge) // converged - search new planes
            {
                points_near.clear();
                point_selected_surf[i] = false;
                point_weights.clear();

                // search als neighbours
                if(use_als){
                    if (localKdTree_map_als->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
                    {
                        if (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 1)
                        {
                            point_selected_surf[i] = true;
                            for (int j = 0; j < pointSearchInd.size(); j++)
                            {
                                points_near.push_back(map_als->points[pointSearchInd[j]]);
                                point_weights.push_back(100.);
                            }
                        }
                    }
                }
                // if (point_selected_surf[i] == false) // not valid neighbours has been found in ALS
                {
                    pointSearchInd.clear();
                    pointSearchSqDis.clear();
                    // search mls neighbours
                    if (localKdTree_map->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
                    {
                        if (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 1)
                        {
                            point_selected_surf[i] = true;
                            for (int j = 0; j < pointSearchInd.size(); j++)
                            {
                                points_near.push_back(map_mls->points[pointSearchInd[j]]);
                                // points_near.insert(points_near.begin(), map_mls->points[pointSearchInd[j]]);
                                point_weights.push_back(1.);
                            }
                        }
                    }
                }
                //}
            }

            if (!point_selected_surf[i]) // src point does not have neighbours
                continue;

            Eigen::Matrix<float, 4, 1> pabcd;
            point_selected_surf[i] = false;
            double plane_var = 0;
            // calculate the rezidual
            // if (ekf::esti_plane2(pabcd, points_near, .2f)) //.1f
            if (ekf::esti_plane_pca(pabcd, points_near, .03, point_weights, plane_var, true))
            {
                float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
                // float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
                float s = 1 - 0.9 * fabs(pd2) / point_body.intensity; // intensity stores the sqrt(range)

                if (s > 0.9)
                {
                    point_selected_surf[i] = true;
                    normvec->points[i].x = pabcd(0);
                    normvec->points[i].y = pabcd(1);
                    normvec->points[i].z = pabcd(2);
                    normvec->points[i].intensity = pd2;

                    if (travelled_distance > 5)
                        normvec_var[i] = 1./plane_var; // 1. / (9. * plane_var); // 3 sigma 1 sigma is better
                    else
                        normvec_var[i] = R_inv_lidar;
                }
            }
        }

        // solve the DA with     src-tgt norms
        effct_feat_num = 0;
        for (int i = 0; i < feats_down_size; i++)
        {
            if (point_selected_surf[i])
            {
                laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
                corr_normvect->points[effct_feat_num] = normvec->points[i];
                corr_normvec_var[effct_feat_num] = normvec_var[i]; // R represent the variance of the noise for each measurement

                effct_feat_num++;
            }
        }

        // std::cout << "effct_feat_num: " << effct_feat_num << "/" << feats_down_size << std::endl;
        if (effct_feat_num < 5)
        {
            ekfom_data.valid = false;
            ROS_WARN("No Effective Points! cannot find good planes \n");
            std::cout << "effct_feat_num:" << effct_feat_num << " cannot find good planes" << std::endl;
            throw std::runtime_error("Stop here - the system will collapes, no good planes found");
            return;
        }
    }

    // 2)
    //  Calculation of Jacobian matrix H and residual vector
    int m = effct_feat_num;
    if (use_gnss)
    {
        m = effct_feat_num + gps_dim;
    }

    ekfom_data.h_x = Eigen::MatrixXd::Zero(m, used_state_size);       // m * 24 = m x 24
    ekfom_data.innovation.resize(m);                                  // z_hat
    ekfom_data.H_T_R_inv = Eigen::MatrixXd::Zero(used_state_size, m); // n x m

#ifdef MP_EN
    // omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {
        V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
        M3D point_crossmat;
        point_crossmat << SKEW_SYM_MATRX(point_);
        V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
        M3D point_I_crossmat;
        point_I_crossmat << SKEW_SYM_MATRX(point_I_);

        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        V3D C(x_.rot.matrix().transpose() * norm_vec);
        V3D A(point_I_crossmat * C);
        // if (extrinsic_est)
        // {
        //     V3D B(point_crossmat * x_.offset_R_L_I.matrix().transpose() * C);
        //     ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        // }
        // else
        // {
        //     // d_t, d_R
        //     ekfom_data.h_x.block<1, 6>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A);
        // }

        Eigen::Matrix<double, 6, 1> J;  // Eigen::Vector6d J;  // 6x1
        J.block<3, 1>(0, 0) = norm_vec; // df/dt
        J.block<3, 1>(3, 0) = A;        // df/dR
        // Eigen::Matrix<double, 12, 1> J;
        // J.block<3, 1>(6, 0) = B;
        // J.block<3, 1>(9, 0) = C;

        ekfom_data.h_x.block<1, 6>(i, 0) = J;         // 1x6 -> m x 6
        ekfom_data.innovation(i) = -norm_p.intensity; // 0 - h(x)   m x 1

        // corr_normvec_var[i] stores the 1/R_inv
        ekfom_data.H_T_R_inv.block<6, 1>(0, i) = J.transpose() * corr_normvec_var[i]; // 6 x 1 * 1 x 1m
    }

    if (use_gnss)
    {
        //  Assign to last 3 positions
        ekfom_data.innovation.tail<3>() = gps_pos - x_.pos; // y - h(x)  where y is measurement h(x) is observation model;
        // GNSS measurement Jacobian to the bottom block of h_x
        ekfom_data.h_x.block<gps_dim, gps_dim>(effct_feat_num, 0) = Eye3d; // de/dt is identity

        double w = 1.0;                                                                                      // add a robust kernel here
        double R_inv_gps = 1. / R_gps_cov;                                                                   // R = 0.05
        ekfom_data.H_T_R_inv.block<gps_dim, gps_dim>(0, effct_feat_num) = w * Eye3d.transpose() * R_inv_gps; // 3 x 3 * 1 x 1m
    }
}

bool RIEKF::update_final(
    double R_lidar_cov, double R_gps_cov,
    PointCloudXYZI::Ptr &feats_down_body, const V3D &gps_pos,
    PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als,
    const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als,
    std::vector<PointVector> &Nearest_Points,
    int maximum_iter, bool extrinsic_est,
    bool use_gnss, bool use_als, bool tightly_coupled)
{
    if (use_als)
    {
        std::cout << "update_final with MLS-ALS" << std::endl;
    }
    else
    {
        std::cout << "update_final only with MLS" << std::endl;
    }
    int features = int(feats_down_body->points.size());

    normvec->resize(features);    // plane measurements
    normvec_var.resize(features); // plane variances
    Nearest_Points_Weights.resize(features);

    residual_struct status;
    status.valid = true;
    status.converge = true;
    int t = 0;

    state x_propagated = x_;
    vectorized_state dx_new = vectorized_state::Zero(); // 24X1

    std::cout << "map_mls:" << map_mls->size() << ", map_als:" << map_als->size() << std::endl;
    if (map_mls->points.size() < 5)
    {
        std::cerr << "Error: map_mls Point cloud is empty! : " << map_mls->points.size() << std::endl;
        status.valid = false;
        status.converge = false;
        return false;
    }

    localKdTree_map->setInputCloud(map_mls);
    const cov P_inv = P_.inverse();

    for (int i = -1; i < maximum_iter; i++) // maximum_iter
    {
        status.valid = true;
        // z = y - h(x)
        h(status, R_lidar_cov, R_gps_cov, feats_down_body, gps_pos, map_mls, map_als, localKdTree_map_als, Nearest_Points, extrinsic_est, use_gnss, use_als, tightly_coupled);

        if (!status.valid)
        {
            break;
        }

        vectorized_state dx;
        dx_new = boxminus(x_, x_propagated); // x^k - x^

        const auto &H = status.h_x;                          // (m x n) Jacobian    m x 24
        Eigen::Matrix<double, state_size, Eigen::Dynamic> K; // n x m = 24 x m
        const auto &Ht_Rinv = status.H_T_R_inv;              //   ( n x m)

        cov HTH = cov::Zero(); // (n x n)
        HTH.block<used_state_size, used_state_size>(0, 0).noalias() = Ht_Rinv * H;

        cov &&K_front = (HTH + P_inv).inverse(); // auto K_front = (HTH + P_inv).inverse();

        K.noalias() = K_front.block<state_size, used_state_size>(0, 0) * Ht_Rinv; // direct inversion fastest for small matrices (n < 50)
        cov KH = cov::Zero();                                                     // (n x m) * (m x n) -> (n x n)
        KH.block<state_size, used_state_size>(0, 0).noalias() = K * H;

        vectorized_state dx_; // vectorized_state dx_ = K * status.innovation + (KH - cov::Identity()) * dx_new;
        dx_.noalias() = K * status.innovation;
        dx_.noalias() += (KH - cov::Identity()) * dx_new;

        x_ = boxplus(x_, dx_); // dx_ is the delta corection that should be applied

        status.converge = true;
        for (int j = 0; j < state_size; j++)
        {
            if (std::fabs(dx_[j]) > ESTIMATION_THRESHOLD_) // If dx_>ESTIMATION_THRESHOLD_ no convergence is considered
            {
                status.converge = false; // means re-optimize with same NN, no need to search for new NN
                break;
            }
        }

        if (status.converge)
            t++;

        if (!t && i == maximum_iter - 2)
        {
            status.converge = true;
        }

        if (t > 1 || i == maximum_iter - 1)
        {
            // Update covariance (Joseph form is more numerically stable)
            P_ = (cov::Identity() - KH) * P_;
            break;
        }
    }

    return status.valid;
}

//-----------------------------------------------------------------------------------------------------------
M3D regularizeCovarianceMatrix(const M3D &cov, double epsilon = .001)
{
    /*
    Regularize the covariance matrix cov to make it:
    Numerically stable (invertible, positive-definite),

    This matrix can be:
    -Ill-conditioned (almost singular) if the neighborhood is planar,
    -Singular if points lie exactly on a plane (rank-deficient),
    -Noisy due to poor point distribution.
    */

    return cov;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    Eigen::Vector3d eigvals = solver.eigenvalues();
    Eigen::Matrix3d eigvecs = solver.eigenvectors();

    double min_val = .01;
    double max_val = 1.0;

    eigvals = eigvals.cwiseMax(min_val).cwiseMin(max_val);

    return eigvecs * eigvals.asDiagonal() * eigvecs.transpose();

    if (false)
    { // FROBENIUS RegularizationMethod
        M3D C = cov + epsilon * M3D::Identity();
        M3D C_inv = C.inverse();
        return (C_inv / C_inv.norm()).inverse();
    }

    if (true)
    {
        Eigen::JacobiSVD<M3D> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
        V3D values;

        // RegularizationMethod::PLANE
        values = Eigen::Vector3d(1, 1, epsilon);

        // RegularizationMethod::MIN_EIG
        // values = svd.singularValues().array().max(epsilon);

        return svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
    }
}

bool projectPoint(const PointType &pt, int &row, int &col, float ang_res_x = 0.18f, float ang_bottom = 15.0, float ang_res_y = 1., int N_SCAN = 32, int Horizon_SCAN = 1500)
{
    float verticalAngle = atan2(pt.z, sqrt(pt.x * pt.x + pt.y * pt.y)) * 180 / M_PI;
    int row_tmp = (verticalAngle + ang_bottom) / ang_res_y;

    if (row_tmp < 0 || row_tmp >= N_SCAN)
        return false;

    float horizonAngle = atan2(pt.x, pt.y) * 180 / M_PI;
    int col_tmp = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;

    if (col_tmp >= Horizon_SCAN)
        col_tmp -= Horizon_SCAN;
    if (col_tmp < 0 || col_tmp >= Horizon_SCAN)
        return false;

    row = row_tmp;
    col = col_tmp;

    // std::cout<<"projectPoint row:"<<row<<", col:"<<col<<std::endl;

    return true;
}

// Compute covariance matrix from neighbors
M3D RIEKF::computeCovariance(const PointCloudXYZI::Ptr &cloud,
                             const pcl::KdTreeFLANN<PointType>::Ptr &kdtree,
                             const PointType &point,
                             const PointCloudXYZI::Ptr &feats_undistort,
                             Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> Neighbours,
                             V3D &out_mean,
                             int k_neighbors,
                             bool use_radius,
                             float radius)
{
    std::vector<int> indices;
    std::vector<float> distances;

    if (use_radius)
    {
        kdtree->radiusSearch(point, radius, indices, distances);
    }
    else
    {
        indices.resize(k_neighbors);
        distances.resize(k_neighbors);
        if (kdtree->nearestKSearch(point, k_neighbors, indices, distances) == 0)
            return M3D::Identity() * 1e-3;
    }

    if (indices.size() < 5)
        return M3D::Identity() * 1e-3; // Not enough neighbors

    V3D mean = V3D::Zero();
    for (int idx : indices) // add the sum of points from kdtree
    {
        mean += V3D(cloud->points[idx].x, cloud->points[idx].y, cloud->points[idx].z);
    }

    int n_points = indices.size();
    int cloud_size = feats_undistort->size();
    // indices1.insert(indices1.end(), indices2.begin(), indices2.end());
    int row, col;
    std::vector<int> indices_range_image;
    if (projectPoint(point, row, col) && false)
    {
        float range_curr = V3D(point.x, point.y, point.z).norm();

        for (const auto &neigh : density_kernel)
        {
            int y = row + neigh.first;
            int x = col + neigh.second;
            if (y < 0 || y >= 32)
                continue; // out of the rings

            // if (x < 0)
            //     x = 1500 + x;

            // if (x >= 1500)
            //     x = x - 1500;

            if (x < 0 || x >= 1500)
                continue;

            auto &idx = Neighbours(y, x);
            if (idx > 0 && idx < cloud_size)
            {
                auto p = V3D(feats_undistort->points[idx].x, feats_undistort->points[idx].y, feats_undistort->points[idx].z);
                if (abs(p.norm() - range_curr) < 1 && abs(p.z() - point.z) < 1) // 1 m
                {
                    indices_range_image.push_back(Neighbours(y, x));
                    mean += p;
                    n_points++;
                }
            }
        }
    }

    mean /= n_points;

    M3D cov = M3D::Zero();
    for (int idx : indices)
    {
        V3D diff = V3D(cloud->points[idx].x, cloud->points[idx].y, cloud->points[idx].z) - mean;
        cov += diff * diff.transpose();
    }

    for (int idx : indices_range_image)
    {
        V3D diff = V3D(feats_undistort->points[idx].x, feats_undistort->points[idx].y, feats_undistort->points[idx].z) - mean;
        cov += diff * diff.transpose();
    }
    cov /= (n_points - 1);

    cov += 1e-6 * M3D::Identity(); // Regularization

    out_mean = mean;
    return cov;
}

bool check_cov(const M3D &cov, double planar_thresh = 0.01, double edge_thresh = 0.1)
{
    return true;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if (solver.info() != Eigen::Success)
    {
        return false;
    }

    Eigen::Vector3d eigvals = solver.eigenvalues(); // sorted ascending
    double sum = eigvals.sum();
    if (sum <= 0)
    {
        return false;
    }

    double e1 = eigvals(2); // largest
    double e2 = eigvals(1);
    double e3 = eigvals(0); // smallest

    double linearity = (e1 - e2) / e1; // Edge
    double planarity = (e2 - e3) / e1; // Plane
    double scattering = e3 / e1;

    // Or use raw ratios:
    if ((e3 / sum < planar_thresh) || (e2 / sum < edge_thresh))
        return true;

    return false;
}

void RIEKF::observation_model_test2(residual_struct &ekfom_data, PointCloudXYZI::Ptr &feats_down_body,
                                    const PointCloudXYZI::Ptr &feats_undistort,
                                    Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> Neighbours,
                                    const V3D &gps,
                                    PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als,
                                    const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als,
                                    std::vector<PointVector> &Nearest_Points, bool extrinsic_est)
{

    int feats_down_size = feats_down_body->points.size();
    laserCloudOri->clear();

    // corr_tgt_covs.clear(); //there is resize called
    // corr_laserCloudTgt.clear();
    //  int p_mls = 0, p_als=0;
    bool use_radius = true;
#ifdef MP_EN
// std::cout<<"run lidar_observation_model_tighly_fused in parallel..."<<std::endl;
// omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body = feats_down_body->points[i];
        PointType point_world;

        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(x_.rot * (x_.offset_R_L_I * p_body + x_.offset_T_L_I) + x_.pos);

        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        int k_neighbors = 10;
        std::vector<int> pointSearchInd(10); // NUM_MATCH_POINTS
        std::vector<float> pointSearchSqDis(10);

        auto &points_near = Nearest_Points[i];
        std::vector<double> point_weights;

        if (ekfom_data.converge)
        {
            points_near.clear();
            point_selected_surf[i] = false;

            // search als neighbours
            // if (localKdTree_map_als->nearestKSearch(point_world, NUM_MATCH_POINTS, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
            // {
            //     if (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 1)
            //     {
            //         point_selected_surf[i] = true;
            //         for (int j = 0; j < pointSearchInd.size(); j++)
            //         {
            //             points_near.push_back(map_als->points[pointSearchInd[j]]);
            //             point_weights.push_back(100.);
            //         }
            //         // p_als++;
            //     }
            // }

            // if (point_selected_surf[i] == false) // not valid neighbours has been found
            {
                pointSearchInd.clear();
                pointSearchSqDis.clear();
                // search mls neighbours
                if (use_radius)
                {
                    if (localKdTree_map->radiusSearch(point_world, 1., pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
                    {
                        // if (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 1)
                        //{
                        point_selected_surf[i] = true;

                        for (int j = 0; j < pointSearchInd.size(); j++)
                        {
                            points_near.push_back(map_mls->points[pointSearchInd[j]]);
                            // points_near.insert(points_near.begin(), map_mls->points[pointSearchInd[j]]);
                            point_weights.push_back(1.);
                        }
                        // p_mls++;
                        //}
                    }
                }
                else
                {
                    pointSearchInd.resize(k_neighbors);
                    pointSearchSqDis.resize(k_neighbors);
                    if (localKdTree_map->nearestKSearch(point_world, k_neighbors, pointSearchInd, pointSearchSqDis) >= NUM_MATCH_POINTS)
                    {
                        // if (pointSearchSqDis[NUM_MATCH_POINTS - 1] <= 1)
                        {
                            point_selected_surf[i] = true;

                            for (int j = 0; j < pointSearchInd.size(); j++)
                            {
                                points_near.push_back(map_mls->points[pointSearchInd[j]]);
                                // points_near.insert(points_near.begin(), map_mls->points[pointSearchInd[j]]);
                                point_weights.push_back(1.);
                            }
                            // p_mls++;
                        }
                    }
                }
            }
            //}

            if (!point_selected_surf[i]) // src point does not have neighbours
                continue;

            point_selected_surf[i] = false;

            if (ekf::esti_cov(points_near, point_weights, tgt_covs[i], laserCloudTgt[i], true))
            {
                point_selected_surf[i] = true;

                // take the closet point - modify this to get the closest one
                laserCloudTgt[i].x() = points_near[0].x;
                laserCloudTgt[i].y() = points_near[0].y;
                laserCloudTgt[i].z() = points_near[0].z;
            }
            // tgt_covs[i] = regularizeCovarianceMatrix(tgt_covs[i]); // Enforce planarity

            point_selected_surf[i] = check_cov(tgt_covs[i]);
        }
    }

    // std::cout<<"p_mls:"<<p_mls<<", p_als:"<<p_als<<std::endl;
    effct_feat_num = 0;
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_tgt_covs[effct_feat_num] = tgt_covs[i];
            corr_laserCloudTgt[effct_feat_num] = laserCloudTgt[i];

            // corr_tgt_covs.push_back(tgt_covs[i]);
            // corr_laserCloudTgt.push_back(laserCloudTgt[i]);

            effct_feat_num++;
        }
    }

    // std::cout << "effct_feat_num: " << effct_feat_num << "/" << feats_down_size << std::endl;
    if (effct_feat_num < 5)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! cannot find good planes \n");
        std::cout << "effct_feat_num:" << effct_feat_num << " cannot find good planes" << std::endl;
        throw std::runtime_error("Stop here - the system will collapes, no good planes found");
        return;
    }

    // Calculation of Jacobian matrix H and residual vector
    // #ifdef use_gps_measurements
    //     ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num + gps_dim, used_state_size); // (m+3) * n  =  m+3 x 24
    //     ekfom_data.innovation.resize(effct_feat_num + gps_dim);                            // z_hat
    // #else
    //     ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, used_state_size); // m * 24  =  m x 24   but can be smaller
    //     ekfom_data.innovation.resize(effct_feat_num);                            // z_hat
    // #endif

    const int total_dim = 3 * effct_feat_num;

    // jacobian for 1 poit is J in [3 x 6]
    ekfom_data.h_x = Eigen::MatrixXd::Zero(total_dim, used_state_size); // 3m * n
    ekfom_data.innovation.resize(total_dim);                            // 3m residual is 3x1

    ekfom_data.H_T_R_inv = Eigen::MatrixXd::Zero(used_state_size, total_dim); // n x 3m
    // MatrixXd H_T_R_inv(6, effct_feat_num);//same as Eigen::MatrixXd Ht_Rinv = H.transpose(); //   ( n x m)

    const auto &R = x_.rot.matrix();
    // Compute inverse translation
    V3D t_inv = -R.transpose() * x_.pos;

    double kernel = 1.0;
    auto Weight = [&](double residual2)
    {
        return std::pow(kernel, 2) / std::pow(kernel + residual2, 2);
    };

#ifdef MP_EN
    // omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {
        V3D point_(laserCloudOri->points[i].x, laserCloudOri->points[i].y, laserCloudOri->points[i].z);
        V3D out_mean = V3D(0, 0, 0);
        bool use_radius = false; // feats_down_body
        M3D point_cov = computeCovariance(feats_undistort, cloud_tree, laserCloudOri->points[i],
                                          feats_undistort, Neighbours,
                                          out_mean, 10, use_radius, 1.0); // point cov lidar frame

        if (!check_cov(point_cov))
            continue;

        // point_ = out_mean;
        // point_cov = regularizeCovarianceMatrix(point_cov);

        // V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
        // M3D point_I_crossmat;
        // point_I_crossmat << SKEW_SYM_MATRX(point_I_);
        // const PointType &norm_p = corr_normvect->points[i];
        // V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
        // V3D C(x_.rot.matrix().transpose() * norm_vec); //normal transformed from global to local
        // V3D A(point_I_crossmat * C);
        // d_t, d_R
        // ekfom_data.h_x.block<1, 6>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A);
        // ekfom_data.innovation(i) = -norm_p.intensity; // 0 - h(x)

        if (false)
        {
            // the error in local frame---------------------------------------------------------
            V3D point_I_ = x_.offset_R_L_I * point_ + x_.offset_T_L_I;
            M3D cov_src_I_ = x_.offset_R_L_I.matrix() * point_cov * x_.offset_R_L_I.matrix().transpose(); // rotate cov to imu frame
            M3D cov_tgt = R.transpose() * corr_tgt_covs[i] * R;                                           // transform from global to local
            // M3D cov_tgt = corr_tgt_covs[i]; //transform from global to local

            auto tgt_local = R.transpose() * corr_laserCloudTgt[i] + t_inv;

            V3D e = point_I_ - tgt_local;                // res error in local imu frame
            Eigen::Matrix<double, 3, used_state_size> J; // Compute Jacobian (3x6)
            J.block<3, 3>(0, 0) = Eye3d;
            J.block<3, 3>(0, 3) = -R * Sophus::SO3::hat(point_I_); // dR*p/dtheta - this requires the error compute in local frame
            // J.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed);

            M3D cov = cov_tgt + R * cov_src_I_ * R.transpose(); // Matrix3d W = (C_s + C_m).inverse();
            auto R_inv = cov.inverse();

            ekfom_data.h_x.block<3, used_state_size>(3 * i, 0) = J; // 3x6 -> 3m x 6
            ekfom_data.innovation.segment<3>(3 * i) = -e;           // 0 - h(x)

            // double w = 1.0;
            double w = 1.0 / std::sqrt((e.transpose() * R_inv * e).coeff(0));
            ekfom_data.H_T_R_inv.block<used_state_size, 3>(0, 3 * i) = w * J.transpose() * R_inv; // 6 x 3 -> 6 x 3m
        }
        else
        {
            // the error in global frame---------------------------------------------------------

            M3D cov_src = x_.offset_R_L_I.matrix() * point_cov * x_.offset_R_L_I.matrix().transpose(); // rotate cov to imu frame

            V3D p_transformed(x_.rot * (x_.offset_R_L_I * point_ + x_.offset_T_L_I) + x_.pos);
            V3D e = p_transformed - corr_laserCloudTgt[i]; // res error in global frame

            // use used_state_size instead of 6
            Eigen::Matrix<double, 3, used_state_size> J; // Compute Jacobian (3x6)
            J.block<3, 3>(0, 0) = Eye3d;                 // dR*p/dt
            // J.block<3, 3>(0, 3) = -R * Sophus::SO3::hat(point_I_); // dR*p/dtheta - this requires the error compute in local frame
            J.block<3, 3>(0, 3) = -1.0 * Sophus::SO3::hat(p_transformed); // dR*p/dtheta - use this bc the error is computed in globa frame

            // Combined GICP covariance
            M3D cov = corr_tgt_covs[i] + R * cov_src * R.transpose(); // Matrix3d W = (C_s + C_m).inverse();
            auto R_inv = cov.inverse();

            ekfom_data.h_x.block<3, used_state_size>(3 * i, 0) = J; // 3x6 -> 3m x 6
            ekfom_data.innovation.segment<3>(3 * i) = -e;           // 0 - h(x)

            // double w = 1.0;
            // double w = 1.0 / std::sqrt((e.transpose() * R_inv * e).coeff(0));
            double w = Weight(e.squaredNorm());

            ekfom_data.H_T_R_inv.block<used_state_size, 3>(0, 3 * i) = w * J.transpose() * R_inv; // 6 x 3 -> 6 x 3m

            laserCloudSrc[i] = p_transformed;
            src_covs[i] = R * cov_src * R.transpose();
        }
    }

#ifdef use_gps_measurements
    // for gps
    //  Assign to last 3 positions
    ekfom_data.innovation.tail<3>() = gps - x_.pos; // y - h(x)  where y is measurement h(x) is observation model gps_innovation;
    // Assign GNSS measurement Jacobian to the bottom block of h_x
    ekfom_data.h_x.block<gps_dim, gps_dim>(effct_feat_num, 0) = Eye3d; // d e dT is identity put it directly here
/// ekfom_data.h_x.block<gps_dim, state_size>(effct_feat_num, 0) = H_gnss;// / .05;  //gps_dim x state_size
#endif
}

bool RIEKF::update_tighly_fused_test2(double R, PointCloudXYZI::Ptr &feats_down_body, PointCloudXYZI::Ptr &feats_undistort,
                                      Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> Neighbours,
                                      PointCloudXYZI::Ptr &map_mls, PointCloudXYZI::Ptr &map_als,
                                      const pcl::KdTreeFLANN<PointType>::Ptr &localKdTree_map_als,
                                      std::vector<PointVector> &Nearest_Points,
                                      const V3D &gps, double R_gps_cov,
                                      int maximum_iter, bool extrinsic_est)
{

    std::cout << "update_tighly_fused_test2..." << std::endl;

    corr_laserCloudTgt.resize(int(feats_down_body->points.size()));
    corr_tgt_covs.resize(int(feats_down_body->points.size()));

    src_covs.resize(int(feats_down_body->points.size()));
    laserCloudSrc.resize(int(feats_down_body->points.size()));

    residual_struct status;
    status.valid = true;
    status.converge = true;
    int t = 0;

    state x_propagated = x_;
    vectorized_state dx_new = vectorized_state::Zero(); // 24X1

    std::cout << "map_mls:" << map_mls->size() << ", map_als:" << map_als->size() << std::endl;
    if (map_mls->points.size() < 5)
    {
        std::cerr << "Error: map_mls Point cloud is empty! : " << map_mls->points.size() << std::endl;
        status.valid = false;
        status.converge = false;
        return false;
    }

    localKdTree_map->setInputCloud(map_mls);
    // localKdTree_map_als->setInputCloud(map_als);

    // cloud_tree->setInputCloud(feats_down_body);
    cloud_tree->setInputCloud(feats_undistort);

    double inv_R = 1.0 / R;
    double inv_R_gps = 1.0 / R_gps_cov;

    const cov P_inv = P_.inverse();

    for (int i = -1; i < maximum_iter; i++) // maximum_iter
    {
        status.valid = true;
        // z = y - h(x)
        observation_model_test2(status, feats_down_body, feats_undistort, Neighbours, gps, map_mls, map_als, localKdTree_map_als, Nearest_Points, extrinsic_est);

        if (!status.valid)
        {
            break;
        }

        vectorized_state dx;
        dx_new = boxminus(x_, x_propagated); // x^k - x^

        const auto &H = status.h_x; // (m x n) Jacobian    m x 24

        // Optimized code with noalias
        Eigen::Matrix<double, state_size, Eigen::Dynamic> K; // n x m = 24 x m

        // ekfom_data.H_T_R_inv = Eigen::MatrixXd::Zero(used_state_size, total_dim); // n x 3m
        // MatrixXd H_T_R_inv(6, effct_feat_num);//same as Eigen::MatrixXd Ht_Rinv = H.transpose(); //   ( n x m)

        auto &Ht_Rinv = status.H_T_R_inv; //   ( n x m)

        // Eigen::MatrixXd Ht_Rinv = H.transpose();             //   ( n x m)
        Ht_Rinv.block(0, 0, used_state_size, effct_feat_num) *= inv_R; //(start_row, start_col, num_rows, num_cols) - for the transposed version  runtime, a bit slow

        cov HTH = cov::Zero(); // (n x n)
        HTH.block<used_state_size, used_state_size>(0, 0).noalias() = Ht_Rinv * H;

        cov &&K_front = (HTH + P_inv).inverse(); // auto K_front = (HTH + P_inv).inverse();

        K.noalias() = K_front.block<state_size, used_state_size>(0, 0) * Ht_Rinv; // direct inversion fastest for small matrices (n < 50)
        cov KH = cov::Zero();                                                     // (n x m) * (m x n) -> (n x n)
        KH.block<state_size, used_state_size>(0, 0).noalias() = K * H;

        // vectorized_state dx_ = K * status.innovation + (KH - cov::Identity()) * dx_new;
        vectorized_state dx_;
        dx_.noalias() = K * status.innovation;
        dx_.noalias() += (KH - cov::Identity()) * dx_new;

        x_ = boxplus(x_, dx_); // dx_ is the delta corection that should be applied

        status.converge = true;
        for (int j = 0; j < state_size; j++)
        {
            if (std::fabs(dx_[j]) > ESTIMATION_THRESHOLD_) // If dx_>ESTIMATION_THRESHOLD_ no convergence is considered
            {
                status.converge = false;
                break;
            }
        }

        if (status.converge)
            t++;

        if (!t && i == maximum_iter - 2)
        {
            status.converge = true;
        }
        if (t > 1 || i == maximum_iter - 1)
        {
            // Update covariance (Joseph form is more numerically stable)
            P_ = (cov::Identity() - KH) * P_;
            break;
        }
    }

    return status.valid;
}

// i will need the fast matrix inversion for gicp

// // Example for 6x6 matrix
// Eigen::Matrix<double, 6, 6> A, A_inv;
// A.setRandom();

// // Fast direct inversion (OK for small, well-conditioned matrices)
// A_inv = A.inverse();

// // Or better: use LDLT (symmetric positive definite or semi-definite)
// Eigen::LDLT<Eigen::Matrix<double,6,6>> ldlt(A);
// A_inv = ldlt.solve(Eigen::Matrix<double, 6, 6>::Identity());

// // Invert diagonal matrix D efficiently
// Eigen::VectorXd d = D.diagonal();
// Eigen::VectorXd d_inv = d.cwiseInverse();
// Eigen::MatrixXd D_inv = d_inv.asDiagonal();

// Eigen::MatrixXd A, A_pinv;
// A_pinv = (A.transpose() * A).ldlt().solve(A.transpose());  // A⁺ = (AᵀA)⁻¹Aᵀ

////HTH = (HTH + HTH.transpose()) / 2.0;  // Enforce symmetry
// int m = 0;
//  Eigen::MatrixXd stacked_H(m + 3, state_size);
//  stacked_H << H;//, H_gnss;
//  //stacked_H << H_gnss;
//  Eigen::MatrixXd s_stacked_H(m + 3, state_size);
//  s_stacked_H << H / R ;//, H_gnss / R_gps_cov;  //I CAN USE IT THE ACTUAL R matrix format for gps
//  //s_stacked_H << H / R, H_gnss / R_gps_cov;
//  // s_stacked_H << H_gnss / R_gps_cov;
//  Eigen::VectorXd stacked_innovation(m + 3);
//  stacked_innovation << status.innovation;//, gps_innovation;
//  // stacked_innovation << gps_innovation;
//  Eigen::MatrixXd Ht_Rinv = s_stacked_H.transpose();
//  Eigen::MatrixXd HTH = Ht_Rinv * stacked_H + P_.inverse();
//  Eigen::Matrix<double, state_size, Eigen::Dynamic> K; // n x m
//  // K.resize(24, m + 3);                                 // m is runtime-known
//  // direct inversion fastest for small matrices (n < 50)
//  K = HTH.inverse() * Ht_Rinv;
//  cov KH = cov::Zero(); // (n x m) * (m x n) -> (n x n)
//  KH = K * stacked_H;
//  vectorized_state dx_ = K * stacked_innovation + (KH - cov::Identity()) * dx_new;
//{
////Standard EKF Update
// auto R_ = R * Eigen::MatrixXd::Identity(m, m);                 // (m x m) measurement covariance

// //very fucking slow ----------------------------------------------------------
// //Eigen::MatrixXd S = H * P_ * H.transpose() + R_;        // (m x m)
// //Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();   // (n x m)

// //try the ldlt decomposition - still slow ------------------------------------
// Eigen::MatrixXd PHt = P_ * H.transpose();  // Compute once, reuse
// Eigen::MatrixXd S = H * PHt + R_;
// Eigen::MatrixXd K = S.ldlt().solve(PHt.transpose()).transpose();
//-------------------------------------------------------------------------------

// auto Rinv = (R * Eigen::MatrixXd::Identity(m,m)).inverse();
// auto Rinv = (1.0 / R) * Eigen::MatrixXd::Identity(m,m);
// }

// const auto &H = status.h_x;       // (m x n) Jacobian    m x 24
//         //int m = status.innovation.size(); // m
//         int m = 0;

//         Eigen::MatrixXd stacked_H(m + 3, state_size);
//         stacked_H << H;//, H_gnss;
//         //stacked_H << H_gnss;

//         Eigen::MatrixXd s_stacked_H(m + 3, state_size);
//         s_stacked_H << H / R ;//, H_gnss / R_gps_cov;  //I CAN USE IT THE ACTUAL R matrix format for gps
//         //s_stacked_H << H / R, H_gnss / R_gps_cov;
//         // s_stacked_H << H_gnss / R_gps_cov;

//         Eigen::VectorXd stacked_innovation(m + 3);
//         stacked_innovation << status.innovation;//, gps_innovation;
//         // stacked_innovation << gps_innovation;

//         Eigen::MatrixXd Ht_Rinv = s_stacked_H.transpose();

//         Eigen::MatrixXd HTH = Ht_Rinv * stacked_H + P_.inverse();

//         Eigen::Matrix<double, state_size, Eigen::Dynamic> K; // n x m
//         // K.resize(24, m + 3);                                 // m is runtime-known

//         // direct inversion fastest for small matrices (n < 50)
//         K = HTH.inverse() * Ht_Rinv;

//         cov KH = cov::Zero(); // (n x m) * (m x n) -> (n x n)
//         KH = K * stacked_H;

//         vectorized_state dx_ = K * stacked_innovation + (KH - cov::Identity()) * dx_new;