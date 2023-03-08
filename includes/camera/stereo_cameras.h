#ifndef _STEREOCAMERAS_H_
#define _STEREOCAMERAS_H_

#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"  
#include "Eigen/Dense"

#include "camera.h"

#include "../Params.h"

class StereoCameras {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	// pointers to left and right cameras.
	Camera* left;
	Camera* right;

	// Extrinsic params.
	Eigen::Matrix4d T_il;
	Eigen::Matrix4d T_ir;
	Eigen::Matrix4d T_clcr;
	Eigen::Matrix4d T_crcl;
    Eigen::Matrix4d T_nlnr; // rectified stereo extrinsic parameter.
    Eigen::Matrix4d T_nrnl; // T_nlnr^-1

	// constructor
	StereoCameras();
	// destructor
	~StereoCameras();
	void undistortImages(const cv::Mat& raw_l, const cv::Mat& raw_r,
		cv::Mat& rect_l, cv::Mat& rect_r);
};



/* ================================================================================
================================= Implementation ==================================
=================================================================================== */
// 持失切.
StereoCameras::StereoCameras()
{
	// Generate both cameras.
	left  = new Camera();
	right = new Camera();
	T_il = Eigen::Matrix4d::Identity();
	T_ir = Eigen::Matrix4d::Identity();
	T_clcr = Eigen::Matrix4d::Identity();
	T_crcl = Eigen::Matrix4d::Identity();
    T_nlnr = Eigen::Matrix4d::Identity();
    T_nrnl = Eigen::Matrix4d::Identity();
};

// 社瑚切
StereoCameras::~StereoCameras() {
	cout << "left:";
	delete left;
	cout << "right:";
	delete right;
	printf("StereoCameras is deleted.\n");
};


void StereoCameras::undistortImages(const cv::Mat& raw_l, const cv::Mat& raw_r,
	cv::Mat& rect_l, cv::Mat& rect_r) 
{
	left->undistortImage(raw_l, rect_l);
	right->undistortImage(raw_r, rect_r);
};
#endif
