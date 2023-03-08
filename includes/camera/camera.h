#ifndef _CAMERA_H_
#define _CAMERA_H_

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"  
#include "Eigen/Dense"

class Camera {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	// We ONLY support a pinhole camera model.
	int n_cols, n_rows;

	// Unrectified K and Kinv (distorted)
	Eigen::Matrix3d K_raw;
	Eigen::Matrix3d Kinv_raw;
	cv::Mat cvK_raw;
	cv::Mat cvD;

	// Rectified K and Kinv (no distortion!)
	Eigen::Matrix3d K; 
	Eigen::Matrix3d Kinv;
    float fx, fy; // rectified
    float cx, cy;
    float fxinv, fyinv;
	cv::Mat cvK; // rectified

	// undistortion maps
	cv::Mat undist_map_x; // CV_32FC1
	cv::Mat undist_map_y; // CV_32FC1

public:
	// constructor
	Camera();
	// destructor
	~Camera();

	void undistortImage(const cv::Mat& raw, cv::Mat& rectified);
};


/* ================================================================================
================================= Implementation ==================================
=================================================================================== */
// 持失切.
Camera::Camera() 
	: n_cols(-1), n_rows(-1), fx(-1), fy(-1), cx(-1), cy(-1),
	fxinv(-1), fyinv(-1)
{
	// initialize all things
	K_raw = Eigen::Matrix3d::Identity();
	Kinv_raw = Eigen::Matrix3d::Identity();
	K    = Eigen::Matrix3d::Identity();
	Kinv = Eigen::Matrix3d::Identity();
};

// 社瑚切
Camera::~Camera() {
	// No dynamic allocation exists.
	printf("Camera is deleted.\n");
};

void Camera::undistortImage(const cv::Mat& raw, cv::Mat& rectified) {
	if (raw.empty() || raw.type() != CV_8UC1 ||
		raw.cols != n_cols || raw.rows != n_rows)
		throw std::runtime_error("undistortImage: provided image has not the same size as the camera model or image is not grayscale");

	cv::remap(raw, rectified, this->undist_map_x, this->undist_map_y, CV_INTER_LINEAR);
};
#endif
