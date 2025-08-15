#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
import cv2.aruco as aruco
import os 

#how to run:  rosrun s_mls ros_cam.py _camera_topic:=/thermal_camera_Flir_Center_0161886/image_raw
#            is_thermal:=true

save_dir = "/media/eugeniu/T7/calibration/saved_data_raw/test" 

class ChessboardDetectorNode:
    def __init__(self):
        rospy.init_node("chessboard_detector_node", anonymous=True)

        self.camera_topic = rospy.get_param("~camera_topic", "/image_raw")

        self.pattern_cols = rospy.get_param("~pattern_cols", 10)  # number of inner corners per chessboard row (columns)
        self.pattern_rows = rospy.get_param("~pattern_rows", 7)  # number of inner corners per chessboard column (rows)
        self.square_size = rospy.get_param("~square_size", 0.1)  # meters (used when camera_info available)
        self.show_window = rospy.get_param("~show_window", True)
        self.debug = rospy.get_param("~debug", True)
        self.is_aruco = rospy.get_param("~is_aruco", False)
        self.is_thermal = rospy.get_param("~is_thermal", True)
        
        print("self.is_thermal:",self.is_thermal)
        print("self.is_aruco:",self.is_aruco)

        self.intrinsic_path = rospy.get_param("~intrinsic_path", "")

        self.image_counter = 0

        if self.is_aruco:
            self.createCharucoBoard()

        # internal
        self.bridge = CvBridge()
        self.cam_K = None   # camera matrix 3x3
        self.dist_coefs = None  # distortion coefficients
        self.have_camera_info = False

        # subscribers
        rospy.Subscriber(self.camera_topic, Image, self.image_cb, queue_size=1, buff_size=3)
        
        rospy.loginfo(f"Subscribed to image: {self.camera_topic}")
        if self.show_window:
            cv2.namedWindow("chessboard_detector", cv2.WINDOW_NORMAL)

        self.pattern_size = (self.pattern_cols, self.pattern_rows)
        print("pattern_size:",self.pattern_size)

        if self.intrinsic_path == "":
            print('No intrinsic path - save the data')
        else:
            print("self.intrinsic_path:",self.intrinsic_path)
            self.test_intrinsic(self.intrinsic_path)

        rospy.on_shutdown(self.on_shutdown)
        rospy.spin()

    def createCharucoBoard(self, squaresY = 9, squaresX = 12, squareLength = .06, markerLength = 0.045, display=True):
        self.ARUCO_DICT = aruco.Dictionary_get(aruco.DICT_5X5_1000)
        self.CHARUCO_BOARD = aruco.CharucoBoard_create(
            squaresX=squaresX, squaresY=squaresY,
            squareLength=squareLength,
            markerLength=markerLength,
            dictionary=self.ARUCO_DICT)

        self.pattern_cols = squaresX
        self.pattern_rows = squaresY
        self.square_size = squareLength

        if display:
            imboard = self.CHARUCO_BOARD.draw((900, 700))
            cv2.imshow('CharucoBoard target', imboard)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def caminfo_cb(self, msg: CameraInfo):
        """Store camera intrinsics from CameraInfo message."""
        K = np.array(msg.K).reshape((3, 3))
        D = np.array(msg.D) if msg.D is not None else np.zeros((5,))  # could be empty
        self.cam_K = K.astype(np.float64)
        self.dist_coefs = D.astype(np.float64)
        self.have_camera_info = True
        if self.debug:
            rospy.loginfo(f"Camera intrinsics got: K=\n{self.cam_K}\nD={self.dist_coefs}")
    
    def test_intrinsic(self, path):
        dict = np.load(path, allow_pickle=True)
        K = dict["K"]
        D = dict["dist"]
        self.cam_K = K.astype(np.float64)
        self.dist_coefs = D.astype(np.float64)
        self.have_camera_info = True
        if self.debug:
            rospy.loginfo(f"Camera intrinsics got: K=\n{self.cam_K}\nD={self.dist_coefs}")

    def image_cb(self, msg: Image):
        """Main image callback: detect chessboard, draw results, optionally project 3D overlay."""
        try:
            if self.is_thermal:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough") #img16
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        if self.is_thermal:
            #Normalize using histogram stretch---------------------------------------------------
            # Compute min and max ignoring 0 (bad pixels)
            valid_pixels = cv_image[cv_image > 0]
            min_val = np.percentile(valid_pixels, 1)   #1% of the pixels are darker (colder) than this.
            max_val = np.percentile(valid_pixels, 99)  #99% of the pixels are darker than this

            # print("min_val:",min_val, ", max_val:",max_val)

            # Clip to [min_val, max_val] and normalize to 8-bit
            img_clipped = np.clip(cv_image, min_val, max_val)

            # minv = np.min(img_clipped)
            # maxv = np.max(img_clipped)
            # print("img_clipped minv:",minv, ", maxv:",maxv)

            img8 = ((img_clipped - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8) 

            # minv = np.min(img8)
            # maxv = np.max(img8)
            # print("img8        minv:",minv, ", maxv:",maxv)

            gray = np.array(255.0 - img8, dtype='uint8') #invert the thermal camera

            # minv = np.min(gray)
            # maxv = np.max(gray)
            # print("gray        minv:",minv, ", maxv:",maxv)

        else: 
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        cv_image_show = cv2.cvtColor(gray.copy(),cv2.COLOR_GRAY2RGB)

        if self.is_aruco:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.ARUCO_DICT)

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(cv_image_show, corners, ids)

                # Refine with ChArUco detection
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=self.CHARUCO_BOARD
                )

                if retval > 0:
                    cv2.aruco.drawDetectedCornersCharuco(cv_image_show, charuco_corners, charuco_ids)

                    # Pose estimation if intrinsics are available
                    if self.have_camera_info and charuco_ids is not None and len(charuco_ids) > 3:
                        success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            charuco_corners,
                            charuco_ids,
                            self.CHARUCO_BOARD,
                            self.cam_K,
                            self.dist_coefs
                        )
                        if success:
                            cv2.drawFrameAxes(cv_image_show, self.cam_K, self.dist_coefs, rvec, tvec, 0.1)
            else:
                if self.debug:
                    rospy.logdebug("No ArUco markers detected.")

            # Show image
            if self.show_window:
                cv2.imshow("charuco_detector", cv_image_show)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    rospy.signal_shutdown("User pressed ESC")
            
        else:
            pattern_size = (self.pattern_cols, self.pattern_rows)
            # find chessboard corners
            found, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            cv_image_show = cv2.cvtColor(gray.copy(),cv2.COLOR_GRAY2RGB)

            if found:
                corners_sub = cv2.cornerSubPix(gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                # draw detected corners
                cv2.drawChessboardCorners(cv_image_show, pattern_size, corners_sub, found)

                if self.have_camera_info:
                    object_points = self._make_object_points()
                    # solvePnP expects (N,3) and (N,1,2) or (N,2)
                    success, rvec, tvec = cv2.solvePnP(object_points, corners_sub.reshape(-1, 2),
                                                    self.cam_K, self.dist_coefs,
                                                    flags=cv2.SOLVEPNP_ITERATIVE)
                    if success:
                        # project axes (3 endpoints) for visualization
                        axis_len = self.square_size * max(self.pattern_cols, self.pattern_rows)
                        axes3d = np.float32([
                            [axis_len, 0, 0],
                            [0, axis_len, 0],
                            [0, 0, -axis_len]
                        ]).reshape(-1, 3)
                        axes2d, _ = cv2.projectPoints(axes3d, rvec, tvec, self.cam_K, self.dist_coefs)

                        origin = tuple(corners_sub[0].ravel().astype(int))
                        # X axis - red, Y - green, Z - blue (by conventional OpenCV)
                        cv2.line(cv_image_show, origin, tuple(axes2d[0].ravel().astype(int)), (0, 0, 255), 3)
                        cv2.line(cv_image_show, origin, tuple(axes2d[1].ravel().astype(int)), (0, 255, 0), 3)
                        cv2.line(cv_image_show, origin, tuple(axes2d[2].ravel().astype(int)), (255, 0, 0), 3)

                        cube_height = self.square_size * 2.0
                        cube_pts_3d = []
                        cols = self.pattern_cols
                        rows = self.pattern_rows
                        idx_tl = 0
                        idx_tr = cols - 1
                        idx_bl = (rows - 1) * cols
                        idx_br = rows * cols - 1
                        # their 3D positions (these are inner corners)
                        def corner_3d_at(idx):
                            col = idx % cols
                            row = idx // cols
                            return np.array([col * self.square_size, row * self.square_size, 0.0], dtype=np.float32)

                        base_pts = [corner_3d_at(i) for i in (idx_tl, idx_tr, idx_br, idx_bl)]
                        # bottom (top of cube) is offset in -Z (since z axis from board to camera depending on convention)
                        top_pts = [p + np.array([0, 0, -cube_height], dtype=np.float32) for p in base_pts]
                        cube_pts_3d = base_pts + top_pts
                        cube_pts_3d = np.array(cube_pts_3d, dtype=np.float32)

                        cube_pts_2d, _ = cv2.projectPoints(cube_pts_3d, rvec, tvec, self.cam_K, self.dist_coefs)
                        cube_pts_2d = cube_pts_2d.reshape(-1, 2).astype(int)

                        # draw cube edges
                        # base rectangle
                        for i in range(4):
                            a = tuple(cube_pts_2d[i])
                            b = tuple(cube_pts_2d[(i + 1) % 4])
                            cv2.line(cv_image_show, a, b, (0, 255, 255), 2)
                        # top rectangle
                        for i in range(4, 8):
                            a = tuple(cube_pts_2d[i])
                            b = tuple(cube_pts_2d[4 + ((i - 4 + 1) % 4)])
                            cv2.line(cv_image_show, a, b, (0, 255, 255), 2)
                        # vertical edges
                        for i in range(4):
                            a = tuple(cube_pts_2d[i])
                            b = tuple(cube_pts_2d[4 + i])
                            cv2.line(cv_image_show, a, b, (0, 255, 255), 2)
                    else:
                        rospy.logwarn_throttle(5.0, "solvePnP failed to find a pose.")
                else:
                    # if no camera_info, at least draw a translucent polygon on convex hull of corners
                    pts = corners_sub.reshape(-1, 2).astype(np.int32)
                    hull = cv2.convexHull(pts)
                    overlay = cv_image_show.copy()
                    cv2.drawContours(overlay, [hull], -1, (0, 128, 255), thickness=cv2.FILLED)
                    alpha = 0.25
                    cv_image_show = cv2.addWeighted(overlay, alpha, cv_image_show, 1 - alpha, 0)

            else:
                if self.debug:
                    rospy.logdebug("Chessboard not found in this frame.")

            # show image
            if self.show_window:
                if self.is_thermal:
                    cv2.imshow("chessboard_detector", cv_image_show)
                else:
                    cv2.imshow('chessboard_detector', cv2.resize(cv_image_show, None, fx=.4, fy=.4))
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    rospy.signal_shutdown("User pressed ESC")

                elif key == ord('s') or key == ord('S'):
                    if found:
                        filename = os.path.join(save_dir, f"image_{self.image_counter:04d}.npy")
                        np.save(filename, cv_image)
                        print(f"Saved: {filename}")
                        self.image_counter += 1

    def _make_object_points(self):
        """Create 3D object points for the chessboard inner corners in the board coordinate system.
           Points are (x,y,0) with origin at top-left inner corner, x to right, y down.
        """
        objp = np.zeros((self.pattern_rows * self.pattern_cols, 3), dtype=np.float32)
        # row-major: y changes per row, x per column
        for r in range(self.pattern_rows):
            for c in range(self.pattern_cols):
                objp[r * self.pattern_cols + c] = (c * self.square_size, r * self.square_size, 0.0)
        return objp

    def on_shutdown(self):
        if self.show_window:
            cv2.destroyAllWindows()
        rospy.loginfo("chessboard_detector_node shutting down.")

if __name__ == "__main__":
    try:
        ChessboardDetectorNode()
    except rospy.ROSInterruptException:
        pass
