#include "RIEKF.hpp"

using namespace ekf;

PointCloudXYZI::Ptr normvec(new PointCloudXYZI(50000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(50000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(50000, 1));
std::vector<bool> point_selected_surf(50000, 1);

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

    int effct_feat_num = 0;
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
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
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

    int effct_feat_num = 0;
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
                                  ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
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

    int effct_feat_num = 0;
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
            ekfom_data.h_x.block<1, 12>(i, 0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
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

#endif

void RIEKF::update(const V3D &gnss_position, const V3D &cov_pos_, int maximum_iter, bool global_error, M3D R)
{
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