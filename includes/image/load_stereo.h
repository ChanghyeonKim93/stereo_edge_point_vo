#ifndef _LOAD_STEREO_H_
#define _LOAD_STEREO_H_
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include "Eigen/Dense"

#include "../camera/stereo_cameras.h"

using namespace std;

namespace load_stereo {
	void loadStereoInforms(const string& strPathCalib, StereoCameras*& stereo_cams) {

		// Load calibration files
		// Read rectification parameters
		cv::FileStorage fsSettings(strPathCalib, cv::FileStorage::READ);
		if (!fsSettings.isOpened())
		{
			cerr << "ERROR: Wrong path to settings" << endl;
		}
		cv::Mat K_l, K_r, T_il, T_ir, D_l, D_r;
		fsSettings["LEFT.K"] >> K_l;
		fsSettings["RIGHT.K"] >> K_r;
		fsSettings["LEFT.T_BS"] >> T_il;
		fsSettings["RIGHT.T_BS"] >> T_ir;
		fsSettings["LEFT.D"] >> D_l;
		fsSettings["RIGHT.D"] >> D_r;
		int rows_l = fsSettings["LEFT.height"];
		int cols_l = fsSettings["LEFT.width"];
		int rows_r = fsSettings["RIGHT.height"];
		int cols_r = fsSettings["RIGHT.width"];
	
		fsSettings.release();

		if (K_l.empty() || K_r.empty() || T_il.empty() || T_ir.empty() || D_l.empty() || D_r.empty() ||
			rows_l == 0 || rows_r == 0 || cols_l == 0 || cols_r == 0)
		{
			cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
		}
		cout << "stereo information is being loaded..." << endl;

		// Fill out stereocameras (raw intrinsics)

		stereo_cams->left->cvK_raw = K_l;
		stereo_cams->left->cvD = D_l;
		stereo_cams->left->n_cols = cols_l;
		stereo_cams->left->n_rows = rows_l;
		cv::cv2eigen(K_l,stereo_cams->left->K_raw);
		stereo_cams->left->Kinv_raw = stereo_cams->left->K_raw.inverse();
		stereo_cams->left->undist_map_x = cv::Mat::zeros(rows_l, cols_l, CV_32FC1);
		stereo_cams->left->undist_map_y = cv::Mat::zeros(rows_l, cols_l, CV_32FC1);

		stereo_cams->right->cvK_raw = K_r;
		stereo_cams->right->cvD = D_r;
		stereo_cams->right->n_cols = cols_r;
		stereo_cams->right->n_rows = rows_r;
		cv::cv2eigen(K_r, stereo_cams->right->K_raw);
		stereo_cams->right->Kinv_raw = stereo_cams->right->K_raw.inverse();
		stereo_cams->right->undist_map_x = cv::Mat::zeros(rows_l, cols_l, CV_32FC1);
		stereo_cams->right->undist_map_y = cv::Mat::zeros(rows_l, cols_l, CV_32FC1);

		// raw extrinsics (relatively represented w.r.t. IMU frame);
		cv::cv2eigen(T_il, stereo_cams->T_il);
		cv::cv2eigen(T_ir, stereo_cams->T_ir);

		Eigen::Matrix4d temp = stereo_cams->T_il.inverse()*stereo_cams->T_ir;
		stereo_cams->T_clcr = temp;
		stereo_cams->T_crcl = temp.inverse();

		// Calculate and fill out rectified intrinsics
		// '0' means the reference cooridnate frame (left camera).
		double scale = 1.95;

		Eigen::Matrix4d T_0l = Eigen::Matrix4d::Identity();
		Eigen::Matrix4d T_0r = stereo_cams->T_clcr;
		Eigen::Matrix3d R_0l = T_0l.block<3, 3>(0, 0);
		Eigen::Matrix3d R_0r = T_0r.block<3, 3>(0, 0);

		Eigen::Vector3d k_l = R_0l.block<3, 1>(0, 2);
		Eigen::Vector3d k_r = R_0r.block<3, 1>(0, 2);
		Eigen::Vector3d k_n = (k_l + k_r)*0.5;
		k_n /= k_n.norm();

		Eigen::Vector3d i_n = stereo_cams->T_clcr.block<3, 1>(0, 3);
		i_n /= i_n.norm();

		Eigen::Vector3d j_n = k_n.cross(i_n);
		j_n /= j_n.norm();

		Eigen::Matrix3d R_0n;
		R_0n.block<3, 1>(0, 0) = i_n;
		R_0n.block<3, 1>(0, 1) = j_n;
		R_0n.block<3, 1>(0, 2) = k_n;

		double f_n = (K_l.at<double>(0, 0) + K_r.at<double>(0, 0))/scale;
		double centu = (double)cols_l*0.5;
		double centv = (double)rows_l*0.5;
		Eigen::Matrix3d K_rect, Kinv_rect;
		K_rect << f_n, 0, centu, 0, f_n, centv, 0, 0, 1;
		Kinv_rect = K_rect.inverse();

		stereo_cams->left->K = K_rect;
		stereo_cams->left->Kinv = Kinv_rect;
		stereo_cams->left->fx = K_rect(0, 0);
		stereo_cams->left->fy = K_rect(1, 1);
		stereo_cams->left->cx = K_rect(0, 2);
		stereo_cams->left->cy = K_rect(1, 2);
		stereo_cams->left->fxinv = 1.0 / stereo_cams->left->fx;
		stereo_cams->left->fyinv = 1.0 / stereo_cams->left->fy;
		cv::eigen2cv(K_rect, stereo_cams->left->cvK);

		stereo_cams->right->K = K_rect;
		stereo_cams->right->Kinv = Kinv_rect;
		stereo_cams->right->fx = K_rect(0, 0);
		stereo_cams->right->fy = K_rect(1, 1);
		stereo_cams->right->cx = K_rect(0, 2);
		stereo_cams->right->cy = K_rect(1, 2);
		stereo_cams->right->fxinv = 1.0 / stereo_cams->right->fx;
		stereo_cams->right->fyinv = 1.0 / stereo_cams->right->fy;
		cv::eigen2cv(K_rect, stereo_cams->right->cvK);

		// interpolation grid calculations.
		float* map_l_x_ptr = nullptr;
		float* map_l_y_ptr = nullptr;
		float* map_r_x_ptr = nullptr;
		float* map_r_y_ptr = nullptr;
		Eigen::Vector3d p_n;
		Eigen::Vector3d P_0, P_l, P_r;
		Eigen::Vector2d p_l, p_r;
		double k1, k2, k3, p1, p2;
		double x, y, r, r2, r4, r6, r_radial, x_dist, y_dist;

		for (int v = 0; v < rows_l; v++) 
		{
			map_l_x_ptr = stereo_cams->left->undist_map_x.ptr<float>(v);
			map_l_y_ptr = stereo_cams->left->undist_map_y.ptr<float>(v);

			map_r_x_ptr = stereo_cams->right->undist_map_x.ptr<float>(v);
			map_r_y_ptr = stereo_cams->right->undist_map_y.ptr<float>(v);

			for (int u = 0; u < cols_l; u++)
			{
				p_n << (double)u, (double)v, 1;
				P_0 = R_0n*Kinv_rect*p_n;

				P_l = R_0l.inverse()*P_0;
				P_l /= P_l(2);

				P_r = R_0r.inverse()*P_0;
				P_r /= P_r(2);

				// left
				p_l << stereo_cams->left->K_raw(0, 0)*P_l(0) + stereo_cams->left->K_raw(0, 2),
					stereo_cams->left->K_raw(1, 1)*P_l(1) + stereo_cams->left->K_raw(1, 2);
				k1 = stereo_cams->left->cvD.at<double>(0, 0);
				k2 = stereo_cams->left->cvD.at<double>(0, 1);
				k3 = stereo_cams->left->cvD.at<double>(0, 4);
				p1 = stereo_cams->left->cvD.at<double>(0, 2);
				p2 = stereo_cams->left->cvD.at<double>(0, 3);
				x = (p_l(0) - stereo_cams->left->K_raw(0, 2)) / stereo_cams->left->K_raw(0, 0);
				y = (p_l(1) - stereo_cams->left->K_raw(1, 2)) / stereo_cams->left->K_raw(1, 1);

				r = sqrt(x*x + y*y);
				r2 = r*r;
				r4 = r2*r2;
				r6 = r4*r2;

				r_radial = 1.0 + k1*r2 + k2*r4 + k3*r6;
				x_dist = x*r_radial + 2 * p1*x*y + p2*(r2 + 2 * x*x);
				y_dist = y*r_radial + p1*(r2 + 2 * y*y) + 2 * p2*x*y;
				*(map_l_x_ptr + u) = stereo_cams->left->K_raw(0, 2) + x_dist*stereo_cams->left->K_raw(0, 0);
				*(map_l_y_ptr + u) = stereo_cams->left->K_raw(1, 2) + y_dist*stereo_cams->left->K_raw(1, 1);

				// right
				p_r << stereo_cams->right->K_raw(0, 0)*P_r(0) + stereo_cams->right->K_raw(0, 2),
					stereo_cams->right->K_raw(1, 1)*P_r(1) + stereo_cams->right->K_raw(1, 2);
				k1 = stereo_cams->right->cvD.at<double>(0, 0);
				k2 = stereo_cams->right->cvD.at<double>(0, 1);
				k3 = stereo_cams->right->cvD.at<double>(0, 4);
				p1 = stereo_cams->right->cvD.at<double>(0, 2);
				p2 = stereo_cams->right->cvD.at<double>(0, 3);

				x = (p_r(0) - stereo_cams->right->K_raw(0, 2)) / stereo_cams->right->K_raw(0, 0);
				y = (p_r(1) - stereo_cams->right->K_raw(1, 2)) / stereo_cams->right->K_raw(1, 1);
				r = sqrt(x*x + y*y);
				r2 = r*r;
				r4 = r2*r2;
				r6 = r4*r2;

				r_radial = 1.0 + k1*r2 + k2*r4 + k3*r6;
				x_dist = x*r_radial + 2 * p1*x*y + p2*(r2 + 2 * x*x);
				y_dist = y*r_radial + p1*(r2 + 2 * y*y) + 2 * p2*x*y;
				*(map_r_x_ptr + u) = stereo_cams->right->K_raw(0, 2) + x_dist*stereo_cams->right->K_raw(0, 0);
				*(map_r_y_ptr + u) = stereo_cams->right->K_raw(1, 2) + y_dist*stereo_cams->right->K_raw(1, 1);
			}
		}	

        // Rectified stereo extrinsic parameters (T_nlnr)
        Eigen::Matrix3d R_ln = R_0l.inverse()*R_0n;
        Eigen::Matrix3d R_rn = R_0r.inverse()*R_0n;
        Eigen::Vector3d t_clcr = stereo_cams->T_clcr.block<3, 1>(0, 3);
        stereo_cams->T_nlnr << Eigen::Matrix3d::Identity(), R_ln.inverse()*t_clcr, 0, 0, 0, 1;
        stereo_cams->T_nrnl = stereo_cams->T_nlnr.inverse();

        fsSettings.release(); // close fileLoader of opencv
	};


	void loadStereoImages(const string& strPathLeft, const string& strPathRight, const string& strPathTimes,
		vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
	{
		// [1] image names
		ifstream fileTimes;
		fileTimes.open(strPathTimes.c_str());
		vTimeStamps.reserve(5000);
		vstrImageLeft.reserve(5000);
		vstrImageRight.reserve(5000);
		while (!fileTimes.eof())
		{
			string s;
			getline(fileTimes, s);
			if (!s.empty())
			{
				stringstream ss;
				ss << s;
				vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
				vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
				double t;
				ss >> t;    
				vTimeStamps.push_back(t / 1e9);
				//cout << vstrImageLeft.back() << endl;
			}
		} // end while
		fileTimes.close();
	}; // end loadStereoImages

	void imreadStereo(const string& strPathLeft, const string& strPathRight, cv::Mat& img_left, cv::Mat& img_right) {
		img_left  = cv::imread(strPathLeft, CV_LOAD_IMAGE_GRAYSCALE); // 8 ~ 10 ms
		if (img_left.data == nullptr) throw std::runtime_error("Left image is empty.\n");
		img_right = cv::imread(strPathRight, CV_LOAD_IMAGE_GRAYSCALE); // 8 ~ 10 ms
		if (img_right.data == nullptr) throw std::runtime_error("Right image is empty.\n");
	};
};

#endif