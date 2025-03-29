
#define test1_alignment_problem_time_sync
#ifdef test1_alignment_problem_time_sync

                curr_mls = Sophus::SE3(state_point.rot, state_point.pos);
                curr_mls_time = time_of_day_sec;

                // std::cout << "\n\n prev_mls_time:" << prev_mls_time << ", Pose:" << prev_mls.log().transpose() << "\n\n"
                //           << std::endl;
                // std::cout << "\n\n curr_mls_time:" << curr_mls_time << ", Pose:" << curr_mls.log().transpose() << "\n\n"
                //           << std::endl;

                // Build KDTree for the reference map
                std::cout << "kdtree set input points: " << laserCloudSurfMap->size() << std::endl;
                kdtree->setInputCloud(laserCloudSurfMap); // take this from mls

                while (gnss_vux_data[tmp_index].gps_tod <= time_of_day_sec && tmp_index < gnss_vux_data.size())
                {
                    ros::spinOnce();
                    if (flg_exit || !ros::ok())
                        break;

                    double diff = fabs(gnss_vux_data[tmp_index].gps_tod - time_of_day_sec);
                    std::cout << " vux-gnss:" << gnss_vux_data[tmp_index].gps_tod << ", time_of_day_sec:" << time_of_day_sec << ", diff:" << diff << std::endl;

                    const double &msg_time = gnss_vux_data[tmp_index].gps_tod;
                    Sophus::SE3 p_vux_in_mls = als2mls * gnss_vux_data[tmp_index].se3;
                    // rotate vux orientation to mls orientation
                    // p_vux_in_mls = Sophus::SE3(p_vux_in_mls.so3().matrix() * Rz, p_vux_in_mls.translation());

                    // rotate to ENU but shifted to zero
                    Sophus::SE3 p_vux_local = Sophus::SE3(gnss_vux_data[tmp_index].se3.so3(), gnss_vux_data[tmp_index].se3.translation() - first_vux_pose.translation());
                    p_vux_local = T_vux2mls * p_vux_local; // SHIFT TO MLS using the init guess
                    // Sophus::SE3 p_vux_local = first_vux_pose.inverse() * gnss_vux_data[tmp_index].se3;

                    // publish_ppk_gnss(p_vux_in_mls, msg_time);
                    publish_ppk_gnss(p_vux_local, msg_time);

                    tmp_index++;
                    rate.sleep();

                    while (readVUX.next(next_line))
                    {
                        if (next_line->empty())
                            break;

                        const auto &cloud_time = next_line->points[0].time;
                        // from the future w.r.t. curr gnss time
                        if (cloud_time > gnss_vux_data[tmp_index].gps_tod)
                            break;

                        some_index++;

                        if (time_aligned && some_index % 5 == 0)
                        {
                            auto interpolated_pose = interpolateSE3(gnss_vux_data, cloud_time, tmp_index, false);

                            Sophus::SE3 pose_local = als2mls * interpolated_pose;
                            pose_local = Sophus::SE3(interpolated_pose.so3(), interpolated_pose.translation() - first_vux_pose.translation());
                            // pose_local = first_vux_pose.inverse() * interpolated_pose;

                            pcl::PointCloud<VUX_PointType>::Ptr downsampled_cloud(new pcl::PointCloud<VUX_PointType>);

                            downSizeFilter_vux.setInputCloud(next_line);
                            downSizeFilter_vux.filter(*downsampled_cloud);

                            Sophus::SE3 interpolated_pose_mls = interpolateSE3(prev_mls, prev_mls_time, curr_mls, curr_mls_time, cloud_time);
                            Sophus::SE3 pose4georeference = interpolated_pose_mls;

                            pose4georeference = pose_local;

                            // vux to imu first
                            // georeference with ppk gnss
                            // rotate to mls frame

                            //Sophus::SE3 pose4georeference = T_vux2mls * pose_local * Sophus::SE3(Eigen::Quaterniond(R_vux2imu), t_vux2imu); // rotate local pose to MLS

                            if (false) // registration here
                            {
                                std::cout << "start registration" << std::endl;

                                // Initial guess for extrinsic transformation (Scanner -> IMU)
                                Eigen::Quaterniond q_extrinsic(R_refined);
                                V3D t_extrinsic = t_refined;

                                double prev_cost = std::numeric_limits<double>::max(); // Initialize with a large value
                                double current_cost = prev_cost;
                                double cost_threshold = .01; // Threshold for stopping criterion
                                for (int iter_num = 0; iter_num < 50; iter_num++)
                                {
                                    if (flg_exit || !ros::ok())
                                        break;

                                    // if (iter_num > 3)
                                    //     break;

                                    // find extrinsics------------------------------
                                    ceres::Problem problem;
                                    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                                    // Ensure the quaternion stays valid during optimization
                                    ceres::LocalParameterization *q_parameterization =
                                        new ceres::EigenQuaternionParameterization();
                                    int points_used_for_registration = 0;

                                    double q_param[4] = {q_extrinsic.x(), q_extrinsic.y(), q_extrinsic.z(), q_extrinsic.w()};
                                    double t_param[3] = {t_extrinsic.x(), t_extrinsic.y(), t_extrinsic.z()};

                                    // Add the quaternion parameter block with the local parameterization
                                    // to Ensure the quaternion stays valid during optimization
                                    problem.AddParameterBlock(q_param, 4, q_parameterization);
                                    problem.AddParameterBlock(t_param, 3); // Add the translation parameter block

                                    for (const auto &raw_point : downsampled_cloud->points) // for each scanner raw point
                                    {
                                        V3D p_src(raw_point.x, raw_point.y, raw_point.z);
                                        // transform to global with best T for NN search
                                        V3D p_transformed = pose4georeference * (R_refined * p_src + t_refined);
                                        // Nearest neighbor search
                                        PointType search_point;
                                        search_point.x = p_transformed.x();
                                        search_point.y = p_transformed.y();
                                        search_point.z = p_transformed.z();

                                        bool p2plane = false; // true;
                                        if (p2plane)
                                        {
                                            std::vector<int> point_idx(5);
                                            std::vector<float> point_dist(5);

                                            if (kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0) // there are neighbours
                                            {
                                                if (point_dist[4] < threshold_nn) // not too far
                                                {
                                                    Eigen::Matrix<double, 5, 3> matA0;
                                                    Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                                                    for (int j = 0; j < 5; j++)
                                                    {
                                                        matA0(j, 0) = laserCloudSurfMap->points[point_idx[j]].x;
                                                        matA0(j, 1) = laserCloudSurfMap->points[point_idx[j]].y;
                                                        matA0(j, 2) = laserCloudSurfMap->points[point_idx[j]].z;
                                                    }

                                                    // find the norm of plane
                                                    V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                                                    double negative_OA_dot_norm = 1 / norm.norm();
                                                    norm.normalize();

                                                    bool planeValid = true;
                                                    for (int j = 0; j < 5; j++)
                                                    {
                                                        if (fabs(norm(0) * laserCloudSurfMap->points[point_idx[j]].x +
                                                                 norm(1) * laserCloudSurfMap->points[point_idx[j]].y +
                                                                 norm(2) * laserCloudSurfMap->points[point_idx[j]].z + negative_OA_dot_norm) > 0.1)
                                                        {
                                                            planeValid = false;
                                                            break;
                                                        }
                                                    }

                                                    if (planeValid)
                                                    {
                                                        ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(p_src, norm, negative_OA_dot_norm, pose4georeference);
                                                        problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                                        points_used_for_registration++;
                                                    }
                                                }
                                            }
                                        }
                                        else // p2p
                                        {
                                            std::vector<int> point_idx(1);
                                            std::vector<float> point_dist(1);

                                            if (kdtree->nearestKSearch(search_point, 1, point_idx, point_dist) > 0) // there are neighbours
                                            {
                                                if (point_dist[0] < threshold_nn) // not too far
                                                {
                                                    const PointType &nearest_neighbor = laserCloudSurfMap->points[point_idx[0]];
                                                    points_used_for_registration++;
                                                    V3D target_map(nearest_neighbor.x, nearest_neighbor.y, nearest_neighbor.z);

                                                    ceres::CostFunction *cost_function = LidarDistanceFactor::Create(p_src, target_map, pose4georeference);
                                                    problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                                }
                                            }
                                        }
                                    }

                                    std::cout << "Registration with " << points_used_for_registration << "/" << downsampled_cloud->size() << " points" << std::endl;
                                    // Solve the problem
                                    ceres::Solver::Options options;
                                    options.linear_solver_type = ceres::DENSE_QR; // options.max_num_iterations = 100;
                                    options.minimizer_progress_to_stdout = true;
                                    ceres::Solver::Summary summary;
                                    ceres::Solve(options, &problem, &summary);

                                    // std::cout << summary.FullReport() << std::endl;
                                    // std::cout << summary.BriefReport() << std::endl;

                                    // Output the refined extrinsic transformation
                                    Eigen::Quaterniond refined_q(q_param[3], q_param[0], q_param[1], q_param[2]);
                                    t_refined = V3D(t_param[0], t_param[1], t_param[2]);
                                    R_refined = refined_q.toRotationMatrix();
                                    // std::cout << "Refined Rotation (Quaternion): " << refined_q.coeffs().transpose() << std::endl;
                                    std::cout << "Refined Translation: " << t_refined.transpose() << ", prev t:" << t_vux2mls.transpose() << "\n\n"
                                              << std::endl;
                                    std::cout << "R_refined:\n"
                                              << R_refined << std::endl;

                                    current_cost = summary.final_cost;
                                    std::cout << "Iteration " << iter_num << " - Cost: " << current_cost << std::endl;

                                    // Check if the cost function change is small enough to stop
                                    if (std::abs(prev_cost - current_cost) < cost_threshold)
                                    {
                                        std::cout << "Stopping optimization: Cost change below threshold.\n";
                                        break;
                                    }

                                    prev_cost = current_cost;
                                }
                            }

                            if (true && perform_registration_refinement)
                            {
                                lines.push_back(downsampled_cloud);
                                line_poses.push_back(pose4georeference);
                                if (state_point.pos.norm() > 50) // drove enough
                                {
                                    std::cout << "Start registration with all the lines====================================" << std::endl;
                                    // Initial guess for extrinsic transformation (Scanner -> IMU)
                                    Eigen::Quaterniond q_extrinsic(R_refined);
                                    V3D t_extrinsic = t_refined;
                                    double prev_cost = std::numeric_limits<double>::max(); // Initialize with a large value
                                    double current_cost = prev_cost;
                                    double cost_threshold = .01; // Threshold for stopping criterion
                                    for (int iter_num = 0; iter_num < 75; iter_num++)
                                    {
                                        if (flg_exit || !ros::ok())
                                            break;

                                        feats_undistort->clear();
                                        // find extrinsics------------------------------
                                        ceres::Problem problem;
                                        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                                        // Ensure the quaternion stays valid during optimization
                                        ceres::LocalParameterization *q_parameterization =
                                            new ceres::EigenQuaternionParameterization();
                                        int points_used_for_registration = 0;

                                        double q_param[4] = {q_extrinsic.x(), q_extrinsic.y(), q_extrinsic.z(), q_extrinsic.w()};
                                        double t_param[3] = {t_extrinsic.x(), t_extrinsic.y(), t_extrinsic.z()};

                                        // Add the quaternion parameter block with the local parameterization
                                        // to Ensure the quaternion stays valid during optimization
                                        problem.AddParameterBlock(q_param, 4, q_parameterization);
                                        problem.AddParameterBlock(t_param, 3); // Add the translation parameter block

                                        int total_points = 0;
                                        for (int l = 0; l < lines.size(); l++)
                                        {
                                            const auto &line_cloud = lines[l];
                                            const auto &line_pose = line_poses[l];
                                            for (const auto &raw_point : line_cloud->points) // for each scanner raw point
                                            {
                                                total_points++;
                                                V3D p_src(raw_point.x, raw_point.y, raw_point.z);
                                                // transform to global with best T for NN search
                                                V3D p_transformed = line_pose * (R_refined * p_src + t_refined);

                                                // Nearest neighbor search
                                                PointType search_point;
                                                search_point.x = p_transformed.x();
                                                search_point.y = p_transformed.y();
                                                search_point.z = p_transformed.z();

                                                feats_undistort->push_back(search_point);

                                                bool p2plane = true;
                                                if (p2plane)
                                                {
                                                    std::vector<int> point_idx(5);
                                                    std::vector<float> point_dist(5);

                                                    if (kdtree->nearestKSearch(search_point, 5, point_idx, point_dist) > 0) // there are neighbours
                                                    {
                                                        if (point_dist[4] < threshold_nn) // not too far
                                                        {
                                                            Eigen::Matrix<double, 5, 3> matA0;
                                                            Eigen::Matrix<double, 5, 1> matB0 = -1 * Eigen::Matrix<double, 5, 1>::Ones();

                                                            for (int j = 0; j < 5; j++)
                                                            {
                                                                matA0(j, 0) = laserCloudSurfMap->points[point_idx[j]].x;
                                                                matA0(j, 1) = laserCloudSurfMap->points[point_idx[j]].y;
                                                                matA0(j, 2) = laserCloudSurfMap->points[point_idx[j]].z;
                                                            }

                                                            // find the norm of plane
                                                            V3D norm = matA0.colPivHouseholderQr().solve(matB0);
                                                            double negative_OA_dot_norm = 1 / norm.norm();
                                                            norm.normalize();

                                                            bool planeValid = true;
                                                            for (int j = 0; j < 5; j++)
                                                            {
                                                                if (fabs(norm(0) * laserCloudSurfMap->points[point_idx[j]].x +
                                                                         norm(1) * laserCloudSurfMap->points[point_idx[j]].y +
                                                                         norm(2) * laserCloudSurfMap->points[point_idx[j]].z + negative_OA_dot_norm) > 0.1)
                                                                {
                                                                    planeValid = false;
                                                                    break;
                                                                }
                                                            }

                                                            if (planeValid)
                                                            {
                                                                ceres::CostFunction *cost_function = LidarPlaneNormFactor::Create(p_src, norm, negative_OA_dot_norm, line_pose);
                                                                problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                                                points_used_for_registration++;
                                                            }
                                                        }
                                                    }
                                                }
                                                else // p2p
                                                {
                                                    std::vector<int> point_idx(1);
                                                    std::vector<float> point_dist(1);

                                                    if (kdtree->nearestKSearch(search_point, 1, point_idx, point_dist) > 0) // there are neighbours
                                                    {
                                                        if (point_dist[0] < threshold_nn) // not too far
                                                        {
                                                            const PointType &nearest_neighbor = laserCloudSurfMap->points[point_idx[0]];
                                                            points_used_for_registration++;
                                                            V3D target_map(nearest_neighbor.x, nearest_neighbor.y, nearest_neighbor.z);

                                                            ceres::CostFunction *cost_function = LidarDistanceFactor::Create(p_src, target_map, line_pose);

                                                            // problem.AddResidualBlock(cost_function, nullptr, q_param, t_param);
                                                            problem.AddResidualBlock(cost_function, loss_function, q_param, t_param);
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        publish_frame_debug(pubLaserCloudDebug, feats_undistort);
                                        ros::spinOnce();
                                        rate.sleep();

                                        // Solve the problem
                                        ceres::Solver::Options options;
                                        options.linear_solver_type = ceres::DENSE_QR; // options.max_num_iterations = 100;
                                        options.minimizer_progress_to_stdout = true;
                                        ceres::Solver::Summary summary;
                                        ceres::Solve(options, &problem, &summary);

                                        // std::cout << summary.FullReport() << std::endl;
                                        // std::cout << summary.BriefReport() << std::endl;

                                        // Output the refined extrinsic transformation
                                        Eigen::Quaterniond refined_q(q_param[3], q_param[0], q_param[1], q_param[2]);
                                        t_refined = V3D(t_param[0], t_param[1], t_param[2]);
                                        R_refined = refined_q.toRotationMatrix();
                                        std::cout << "Registration done with " << points_used_for_registration << "/" << total_points << " points" << std::endl;
                                        // std::cout << "Refined Rotation (Quaternion): " << refined_q.coeffs().transpose() << std::endl;
                                        std::cout << "Refined Translation: " << t_refined.transpose() << ", prev t:" << t_vux2mls.transpose() << "\n\n"
                                                  << std::endl;
                                        std::cout << "R_refined:\n"
                                                  << R_refined << std::endl;

                                        current_cost = summary.final_cost;
                                        std::cout << "Iteration " << iter_num << " - Cost: " << current_cost << " \n\n"
                                                  << std::endl;

                                        // Check if the cost function change is small enough to stop
                                        if (std::abs(prev_cost - current_cost) < cost_threshold)
                                        {
                                            std::cout << "Stopping optimization: Cost change below threshold.\n";
                                            break;
                                        }

                                        prev_cost = current_cost;
                                    }

                                    std::cout << "Refined solution:==========================================" << std::endl;
                                    std::cout << "T:" << t_refined.transpose() << std::endl;
                                    std::cout << "R:\n"
                                              << R_refined << std::endl;

                                    // throw std::runtime_error("registration done");
                                    // break;
                                    perform_registration_refinement = false;
                                }
                            }

                            pcl::PointCloud<VUX_PointType>::Ptr transformed_cloud(new pcl::PointCloud<VUX_PointType>);
                            *transformed_cloud = *downsampled_cloud;

                            for (size_t i = 0; i < transformed_cloud->size(); i++)
                            {
                                V3D point_scanner(transformed_cloud->points[i].x, transformed_cloud->points[i].y, transformed_cloud->points[i].z);

                                V3D point_global;

                                if (false) // georeference with ppk gnss-imu
                                {
                                    // transform to vux gnss-imu  and georeference with ppk data
                                    // point_global = R_vux2imu * point_scanner + t_vux2imu;
                                    // point_global = pose_local * point_global;

                                    // point_global = pose4georeference * (delta_pose * point_scanner);
                                }

                                if (true) // georeference with mls and extrinsic init guess
                                {
                                    // point_global = R_vux2mls * point_scanner; // now in mls frame
                                    // point_global += t_vux2mls;

                                    // transform to mls
                                    point_global = R_refined * point_scanner + t_refined;

                                    // georeference with mls
                                    point_global = pose4georeference * point_global;

                                    //point_global = pose4georeference * point_scanner;
                                }

                                transformed_cloud->points[i].x = point_global.x();
                                transformed_cloud->points[i].y = point_global.y();
                                transformed_cloud->points[i].z = point_global.z();
                            }

                            publishPointCloud_vux(transformed_cloud, point_cloud_pub);
                        }
                    }
                }

                // to georeference the vux point cloud with ppk gnss-imu
                // Sophus::SE3 pose4georeference = interpolateSE3(prev_mls, prev_mls_time, curr_mls, curr_mls_time, gnss_vux_data[tmp_index].gps_tod);
                // Sophus::SE3 tmp_pose = (als2mls * gnss_vux_data[tmp_index].se3) * Sophus::SE3(Eigen::Quaterniond(R_vux2imu), t_vux2imu);
                // Sophus::SE3 delta_pose = pose4georeference.inverse() * tmp_pose;
                // R_refined = delta_pose.so3().matrix();
                // //t_refined = delta_pose.translation();
                // std::cout << "R_refined:\n"
                //           << R_refined << std::endl;
                // std::cout << "t_refined:" << t_refined.transpose() << std::endl;

                std::cout << "Extrinsic init offset" << std::endl;
                auto vux_t = (als2mls * gnss_vux_data[tmp_index].se3).translation();
                V3D off;
                off[0] = fabs(state_point.pos[0]) - fabs(vux_t[0]);
                off[1] = fabs(state_point.pos[1]) - fabs(vux_t[1]);
                off[1] = fabs(state_point.pos[2]) - fabs(vux_t[2]);
                std::cout << "off:" << off.transpose() << std::endl;
#endif