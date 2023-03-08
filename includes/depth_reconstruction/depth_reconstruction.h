#ifndef _DEPTHRECON_H_
#define _DEPTHRECON_H_

#include <iostream>
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"  
#include "Eigen/Dense"

#include "../frame/frame.h"
#include "../camera/stereo_cameras.h"
#include "../Params.h"

#include "../image/image_proc.h"
#include "../frame/stereo_frames.h"
#include "../quadtrees/CommonStruct.h"

#include "../tracking/affine_klt_tracker.h"

namespace depthrecon {
	void depthReconStatic(StereoFrame* frame);
	void depthReconStaticPoint(StereoFrame* frame);

    void depthReconTemporal(StereoFrame* frame_k, StereoFrame* frame_c, const Eigen::Matrix4d& T_ck);
    void depthReconTemporalPoint(StereoFrame* frame_k, StereoFrame* frame_c, const Eigen::Matrix4d& T_ck);

    int testBoundary(chk::Point2f& pt_start, chk::Point2f& pt_end, const int& n_cols, const int& n_rows, const int& offset);

    void pointDepthUpdate(StereoFrame* frame_k, chk::AffineKLTTracker* tracker, const Eigen::Matrix4d& T_ck);
};

void depthrecon::depthReconStatic(StereoFrame* frame) {
	// 이 함수에서 업데이트 해야하는 것.
	// (1) 대상이 되는 점의 깊이값, (2) 3차원 좌표 생성, (3) 역수깊이, (4) 깊이 표준편차

	// intrinsic 및 extrinsic load
	int n_cols = frame->getPtrCams()->left->n_cols;
	int n_rows = frame->getPtrCams()->left->n_rows;
    
	Eigen::Matrix3d K = frame->K_pyr[0];
    Eigen::Matrix3d Kinv = frame->Kinv_pyr[0];
    Eigen::Matrix4d T_lr = frame->getPtrCams()->T_nlnr; // (rectified) 3D transform from left to right. 
    // (warping right point to left frame representation)

	// 복원 대상 점 & 그들의 그래디언트 방향
	vector<chk::Point2f>& pts_l  = frame->left->ft_edge[0]->pts;
	vector<chk::Point3f>& grad_l = frame->left->ft_edge[0]->grad;

	size_t n_pts = pts_l.size();

	cout << "  -> reconstruction candidate edge-n_pts : " << n_pts << "\n";

	vector<float> scores_best(n_pts, -1.0f);
	vector<char>  flag_valid(n_pts, 0);
	vector<float> invd_save(n_pts, -1.0f);
	vector<float> std_invd_save(n_pts, -1.0f);

	// 파라미터 설정
	int win_sz = frame->getPtrParams()->recon.edge.fixed.win_sz;
	int fat_sz = frame->getPtrParams()->recon.edge.fixed.fat_sz;

	float angle_thres = frame->getPtrParams()->recon.edge.fixed.angle_thres;
	float cosd_thres = cos(angle_thres / 180.0f*pi);

	float thres_zncc1 = frame->getPtrParams()->recon.edge.fixed.thres_zncc1;
	float thres_zncc2 = frame->getPtrParams()->recon.edge.fixed.thres_zncc2;
	float thres_multipeak = frame->getPtrParams()->recon.edge.fixed.thres_multipeak;

	float eps_edge = frame->getPtrParams()->recon.edge.fixed.eps_edge;
	float eps_epi = frame->getPtrParams()->recon.edge.fixed.eps_epi;
	float eps_bright = frame->getPtrParams()->recon.edge.fixed.eps_bright;
	float eps_edge2 = eps_edge*eps_edge;
	float eps_epi2 = eps_epi*eps_epi;

	// minimum & maximum disparity 값. 해당 disparity 범위 안에서만 서칭.
	// baseline 대비, 0.4 m~ 15 m 범위에서 가능한 disparity 값 범위를 계산해둠
	float baseline = T_lr.block<3, 1>(0, 3).norm();
	float focal = K(0, 0);
	float d_min = frame->getPtrParams()->recon.edge.fixed.d_min;
	float d_max = frame->getPtrParams()->recon.edge.fixed.d_max;
	float bf = baseline*focal;
	float bfinv = 1.0f / bf;
	float max_disp = bf / d_min;
	float min_disp = bf / d_max;

	// 본격적인 깊이 복원
	size_t patch_len = (2 * win_sz + 1)*(2 * fat_sz + 1);
	vector<chk::Point2f> pts_patch;
	pts_patch.reserve(patch_len);
	for (int u = -win_sz; u <= win_sz; u++)
		for (int v = -fat_sz; v <= fat_sz; v++)
			pts_patch.push_back(chk::Point2f(u, v));

	float* patch_l = new float[patch_len];
	float* patch_r = new float[patch_len];
	int* valid_l = new int[patch_len];
	int* valid_r = new int[patch_len];

	int u_min = -1;
	int u_max = -1;
	vector<float> scores_in;
	scores_in.reserve(1000);

	for (int i = 0; i < n_pts; i++) {
		// 현재 i 번째 점의 데이터를 추출한다.
		float u_l = pts_l[i].x;
		float v_l = pts_l[i].y;

        float dmag_l = grad_l[i].z;
		float du_l = grad_l[i].x / dmag_l; // normalize
		float dv_l = grad_l[i].y / dmag_l;

		// gradient direction 이 epipolar line과 이루는 각도가 70도 이내만 복원.
		if ((fabs(du_l) >= cosd_thres) && (u_l > win_sz + 1) && (u_l < n_cols - win_sz)
            && (v_l > win_sz + 1) && (v_l < n_rows - win_sz))
		{
			improc::interpImage(frame->left->img[0], pts_patch, chk::Point2f(u_l, v_l), patch_l, valid_l); // 패치 추출

			if (frame->left->ft_edge[0]->invd[i] > 0) {// 만약 깊이값이 이미 있다면, (이전 워핑된) +-2std 범위내 disp. 서칭
				float invd_temp = frame->left->ft_edge[0]->invd[i];
				float std_temp = frame->left->ft_edge[0]->std_invd[i];
				u_min = (int)floor(u_l - bf*(invd_temp + 3 * std_temp));
				u_max = (int)floor(u_l - bf*(invd_temp - 3 * std_temp));
			}
			else {// 깊이가 없다면 전체범위 탐색.
				u_min = (int)(u_l - max_disp);
				u_max = (int)(u_l - min_disp);
			}
			// 서칭범위를 이미지 내부로 제한해준다.
			if (u_min < win_sz + 1) u_min = win_sz + 1;
			if (u_max > n_cols - win_sz - 1) u_max = n_cols - win_sz - 1;

			// 매칭쌍을 찾는다.
			scores_in.resize(0);
			float score_best = -1e5;
			vector<int> u_over; // thres2를 넘은 점들을 모아둠.
			float u_best = -1.0f; // 매칭된 점의 위치 (subpixel)

			int ind_best = -1;
			int u_r = 0;

			improc::calcNCCstrip_fast2(frame->left->img[0], frame->right->img[0], u_l, v_l, u_min, u_max, win_sz, fat_sz, scores_in);

			for (int j = 0; j < u_max - u_min + 1; j++) {
				u_r = j + u_min;
				if (scores_in[j] > thres_zncc2) u_over.push_back(u_r);
				// 만약, 현재 ZNCC가 이전 최적값보다 크면 바꾼다.
				if (scores_in[j] > score_best) {
					score_best = scores_in[j];
					u_best = u_r;
					ind_best = j;
				} // end if (scores)
			} // end for

			// 최고 점수가 threshold를 넘고, 모든 u_over가 6 pixel 이내이고,
			// 범위 가장자리가 아닌 경우에만 신뢰.
			flag_valid[i] = 1;
			for (int j = 0; j < u_over.size(); j++) {
				if (abs(u_best - u_over[j]) > thres_multipeak) {
					flag_valid[i] = 0;
					break;
				} // end if
			} // end for
			if (flag_valid[i] & (score_best > thres_zncc1) & (u_best > u_min + 2) & (u_best < u_max - 2)) {
				scores_best[i] = score_best; // renew best matched score.
				float s1 = scores_in[ind_best - 1];// subpixel refinement (parabolic fitting)
				float s2 = scores_in[ind_best];
				float s3 = scores_in[ind_best + 1];
				float u_refine = u_best - (s3 - s1) / (s3 + s1 - 2 * s2)*0.5f;

				// inverse depth
				invd_save[i] = (u_l - u_refine)*bfinv;
				//std
				std_invd_save[i] = (1.0f / fabs(du_l)*sqrt(eps_edge2 + eps_epi2*dv_l*dv_l)
					+ 1.414*eps_bright / dmag_l)*bfinv;
			}
			else flag_valid[i] = 0;
			// end if~else
		}
	} // end for (int i = 0) (main loop)

	// update reconstruction results
	for (int k = 0; k < n_pts; k++) {
		if (flag_valid[k] > 0) {
			if (frame->left->ft_edge[0]->invd[k] > 0) {// 퓨전
				frame->left->ft_edge[0]->df[k].updateDF(invd_save[k], std_invd_save[k]);
			}
			else { // 아무것도 없는 경우. (이번에 처음 reconstructed 인 경우)
				float invd_curr = invd_save[k];
				float std_curr = std_invd_save[k];
				frame->left->ft_edge[0]->df[k].set_mu(invd_curr);
				frame->left->ft_edge[0]->df[k].set_sig(std_curr);
				if (frame->left->ft_edge[0]->df[k].zmin() > invd_curr) frame->left->ft_edge[0]->df[k].set_zmin(invd_curr);
				if (frame->left->ft_edge[0]->df[k].zmax() < invd_curr) frame->left->ft_edge[0]->df[k].set_zmax(invd_curr);
				frame->left->ft_edge[0]->is_recon[k] = true;
			} // end if ~ else

			// inverse dpeth를 넣어준다.
			frame->left->ft_edge[0]->invd[k] = frame->left->ft_edge[0]->df[k].mu();
			frame->left->ft_edge[0]->std_invd[k] = frame->left->ft_edge[0]->df[k].sig();
			float depth = 1.0f / frame->left->ft_edge[0]->invd[k];
			frame->left->ft_edge[0]->pts3d[k].x = (Kinv(0, 0)*frame->left->ft_edge[0]->pts[k].x + Kinv(0, 2)) * depth;
			frame->left->ft_edge[0]->pts3d[k].y = (Kinv(1, 1)*frame->left->ft_edge[0]->pts[k].y + Kinv(1, 2)) * depth;
			frame->left->ft_edge[0]->pts3d[k].z = depth;
		}//end if
	}//end for
	 // debug! 되긴하는듯?
	if (0) {
		for (int i = 0; i < frame->left->ft_edge[0]->is_recon.size(); i++) {
			if (frame->left->ft_edge[0]->is_recon[i] > 0) {
				cout << frame->left->ft_edge[0]->pts3d[i].x << ", ";
				cout << frame->left->ft_edge[0]->pts3d[i].y << ", ";
				cout << frame->left->ft_edge[0]->pts3d[i].z << endl;
			}
		}
	}

    // free allocated
    delete[] patch_l;
    delete[] patch_r;
    delete[] valid_l;
    delete[] valid_r;
};

void depthrecon::depthReconStaticPoint(StereoFrame* frame) {
	// 이 함수에서 업데이트 해야하는 것.
	// (1) 대상이 되는 점의 깊이값, (2) 3차원 좌표 생성, (3) 역수깊이, (4) 깊이 표준편차

	// intrinsic 및 extrinsic load
	int n_cols = frame->getPtrCams()->left->n_cols;
	int n_rows = frame->getPtrCams()->left->n_rows;
    Eigen::Matrix3d K = frame->K_pyr[0];
    Eigen::Matrix3d Kinv = frame->Kinv_pyr[0];
    Eigen::Matrix4d T_lr = frame->getPtrCams()->T_nlnr; // (rectified) 3D transform from left to right. (warping right point to left frame representation)

	// 복원 대상 점 & 그들의 그래디언트 방향
	vector<chk::Point2f>& pts_l = frame->left->ft_point[0]->pts;
	size_t n_pts = pts_l.size();
	cout << "  -> reconstructed point-n_pts: "<< n_pts << "\n";

	vector<float> scores_best(n_pts, -1.0f);
	vector<char> flag_valid(n_pts, 0);
	vector<float> invd_save(n_pts, -1.0f);
	vector<float> std_invd_save(n_pts, -1.0f);

	// 파라미터 설정
	int win_sz = frame->getPtrParams()->recon.point.fixed.win_sz;
	int fat_sz = frame->getPtrParams()->recon.point.fixed.fat_sz;
	
	float thres_zncc1 = frame->getPtrParams()->recon.point.fixed.thres_zncc1;
	float thres_zncc2 = frame->getPtrParams()->recon.point.fixed.thres_zncc2;
	float thres_multipeak = frame->getPtrParams()->recon.point.fixed.thres_multipeak;

	float eps_edge = frame->getPtrParams()->recon.point.fixed.eps_edge;
	float eps_bright = frame->getPtrParams()->recon.point.fixed.eps_bright;

	// minimum & maximum disparity 값. 해당 disparity 범위 안에서만 서칭.
	// baseline 대비, 0.4 m~ 15 m 범위에서 가능한 disparity 값 범위를 계산해둠
	float baseline = T_lr.block<3, 1>(0, 3).norm();
	float focal = K(0, 0);
	float d_min = frame->getPtrParams()->recon.point.fixed.d_min;
	float d_max = frame->getPtrParams()->recon.point.fixed.d_max;
	float bf = baseline*focal;
	float bfinv = 1.0f / bf;
	float max_disp = bf / d_min;
	float min_disp = bf / d_max;

	// 본격적인 깊이 복원
	size_t patch_len = (2 * win_sz + 1)*(2 * fat_sz + 1);
	vector<chk::Point2f> pts_patch;
	pts_patch.reserve(patch_len);
	for (int u = -win_sz; u <= win_sz; u++)
		for (int v = -fat_sz; v <= fat_sz; v++)
			pts_patch.push_back(chk::Point2f(u, v));

	float u_l, v_l;
	float* patch_l = new float[patch_len];
	float* patch_r = new float[patch_len];
	int* valid_l = new int[patch_len];
	int* valid_r = new int[patch_len];

	int u_min = -1;
	int u_max = -1;
	vector<float> scores_in;
	scores_in.reserve(1000);

	for (int i = 0; i < n_pts; i++) {
		// 현재 i 번째 점의 데이터를 추출한다.
		u_l = pts_l[i].x;
		v_l = pts_l[i].y;

		// gradient direction 이 epipolar line과 이루는 각도가 70도 이내만 복원.
		if ((u_l > win_sz + 1) & (u_l < n_cols - win_sz)
			& (v_l > win_sz + 1) & (v_l < n_rows - win_sz))
		{
			// 패치 추출
			improc::interpImage(frame->left->img[0], pts_patch, chk::Point2f(u_l, v_l), patch_l, valid_l);

			// 만약 깊이값이 이미 있다면, (이전 워핑된) +-2std 범위내 disp. 서칭
			if (frame->left->ft_point[0]->invd[i] > 0) {
				float invd_temp = frame->left->ft_point[0]->invd[i];
				float std_temp = frame->left->ft_point[0]->std_invd[i];
				u_min = (int)floor(u_l - bf*(invd_temp + 3 * std_temp));
				u_max = (int)floor(u_l - bf*(invd_temp - 3 * std_temp));
			}
			else {
				// 깊이가 없다면 전체범위 탐색.
				u_min = (int)(u_l - max_disp);
				u_max = (int)(u_l - min_disp);
			}
			// 서칭범위를 이미지 내부로 제한해준다.
			if (u_min < win_sz + 1) u_min = win_sz + 1;
			if (u_max > n_cols - win_sz - 1) u_max = n_cols - win_sz - 1;

			// 매칭쌍을 찾는다.
			scores_in.resize(0);
			float score_best = -1e5;
			vector<int> u_over; // thres2를 넘은 점들을 모아둠.
			float u_best = -1.0f; // 매칭된 점의 위치 (subpixel)

			// strip에 대해서 NCC 값을 다 구한다. 
			improc::calcNCCstrip_fast2(frame->left->img[0], frame->right->img[0], u_l, v_l, u_min, u_max, win_sz, fat_sz, scores_in);
			// 범위를 순회하며 서칭.
			int ind_best = -1;
			int u_r = 0;
			for (int j = 0; j < u_max - u_min + 1; j++) {
				u_r = j + u_min;
				if (scores_in[j] > thres_zncc2) u_over.push_back(u_r);
				// 만약, 현재 ZNCC가 이전 최적값보다 쎄면 바꾼다.
				if (scores_in[j] > score_best) {
					score_best = scores_in[j];
					u_best = u_r;
					ind_best = j;
				} // end if (scores)
			} // end for

			  // 최고 점수가 threshold를 넘고, 모든 u_over가 6 pixel 이내이고,
			// 범위 가장자리가 아닌 경우에만 신뢰.
			flag_valid[i] = 1;
			for (int j = 0; j < u_over.size(); j++) {
				if (abs(u_best - u_over[j]) > thres_multipeak) {
					flag_valid[i] = 0;
					break;
				} // end if
			} // end for
			if (flag_valid[i] & (score_best > thres_zncc1) & (u_best > u_min + 2) & (u_best < u_max - 2)) {
				// 최고점수를 저장한다.
				scores_best[i] = score_best;
				// score history 이용해서 2차함수 모델로 subpixel refinement.
				float s1 = scores_in[ind_best - 1];
				float s2 = scores_in[ind_best];
				float s3 = scores_in[ind_best + 1];
				float u_refine = u_best - (s3 - s1) / (s3 + s1 - 2 * s2)*0.5f;

				// inverse depth
				invd_save[i] = (u_l - u_refine)*bfinv;
				//std
				std_invd_save[i] = eps_edge*bfinv;
			}
			else {
				flag_valid[i] = 0;
			}// end if~else
		}
	} // end for (int i = 0) (main loop)

	  // 결과 업데이트 
	for (int k = 0; k < n_pts; k++) {
		if (flag_valid[k] > 0)
		{
			if (frame->left->ft_point[0]->invd[k] > 0)
			{
				// 퓨전
				frame->left->ft_point[0]->df[k].updateDF(invd_save[k], std_invd_save[k]);
			}
			else { // 아무것도 없는 경우. (이번에 처음 reconstructed 인 경우)
				float invd_curr = invd_save[k];
				float std_curr = std_invd_save[k];
				frame->left->ft_point[0]->df[k].set_mu(invd_curr);
				frame->left->ft_point[0]->df[k].set_sig(std_curr);
				if (frame->left->ft_point[0]->df[k].zmin() > invd_curr) frame->left->ft_point[0]->df[k].set_zmin(invd_curr);
				if (frame->left->ft_point[0]->df[k].zmax() < invd_curr) frame->left->ft_point[0]->df[k].set_zmax(invd_curr);
				frame->left->ft_point[0]->is_recon[k] = true;
			} // end if ~ else

			  // inverse dpeth를 넣어준다.
			frame->left->ft_point[0]->invd[k] = frame->left->ft_point[0]->df[k].mu();
			frame->left->ft_point[0]->std_invd[k] = frame->left->ft_point[0]->df[k].sig();
			float depth = 1.0f / frame->left->ft_point[0]->invd[k];
			frame->left->ft_point[0]->pts3d[k].x = (Kinv(0, 0)*frame->left->ft_point[0]->pts[k].x + Kinv(0, 2)) * depth;
			frame->left->ft_point[0]->pts3d[k].y = (Kinv(1, 1)*frame->left->ft_point[0]->pts[k].y + Kinv(1, 2)) * depth;
			frame->left->ft_point[0]->pts3d[k].z = depth;
		}//end if
	}//end for

	 // debug! 되긴하는듯?
	if (0) {
		for (int i = 0; i < frame->left->ft_point[0]->is_recon.size(); i++) {
			if (frame->left->ft_point[0]->is_recon[i] > 0) {
				cout << frame->left->ft_point[0]->pts3d[i].x << ", ";
				cout << frame->left->ft_point[0]->pts3d[i].y << ", ";
				cout << frame->left->ft_point[0]->pts3d[i].z << endl;
			}
		}
	}
};

void depthrecon::depthReconTemporal(StereoFrame* frame_k, StereoFrame* frame_c, const Eigen::Matrix4d& T_ck) {
    // 이 함수에서 업데이트 해야하는 것.
    // (1) 대상이 되는 점의 깊이값, (2) 3차원 좌표 생성, (3) 역수깊이, (4) 깊이 표준편차

    // intrinsic 및 extrinsic load
    int n_cols = frame_k->getPtrCams()->left->n_cols;
    int n_rows = frame_k->getPtrCams()->left->n_rows;

    Eigen::Matrix3d K    = frame_k->K_pyr[0];
    Eigen::Matrix3d Kinv = frame_k->Kinv_pyr[0];
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxinv = 1.0f / fx;
    float fyinv = 1.0f / fy;

    Eigen::Matrix3d R_ck = T_ck.block<3, 3>(0, 0);
    Eigen::Vector3d t_ck = T_ck.block<3, 1>(0, 3);
    Eigen::Matrix3d R_kc = R_ck.inverse();
    Eigen::Vector3d t_kc = R_kc*(-t_ck);

    chk::Point3f nt_ck(t_ck(0), t_ck(1), t_ck(2));
    chk::Point3f nt_kc(t_kc(0), t_kc(1), t_kc(2));
    float t_ck_norm = t_ck.norm(); // t_ck.norm == t_kc.norm.
    nt_ck /= t_ck_norm;
    nt_kc /= t_ck_norm;

    float a = 2.0f*atan(0.25f / fx); // for 3d recon. and its std. value calculation.

    // Fundamental metrix & Essential matrix
    Eigen::Matrix3d F_ck = Kinv.transpose()*Lie::hat(t_ck)*Kinv;

    // load images
    cv::Mat& img_k = frame_k->left->img[0];
    cv::Mat& img_c = frame_c->left->img[0];

    // 복원 대상 점 & 그들의 그래디언트 방향
    vector<chk::Point2f>& pts_k  = frame_k->left->ft_edge[0]->pts;
    vector<chk::Point3f>& grad_k = frame_k->left->ft_edge[0]->grad;

    size_t n_pts = pts_k.size();

    cout << "    -> temporal candidate edge-n_pts : " << n_pts << "\n";

    vector<float> scores_best(n_pts, -1.0f);
    vector<char>  flag_valid(n_pts, 0);
    vector<float> invd_save(n_pts, -1.0f);
    vector<float> std_invd_save(n_pts, -1.0f);

    // 파라미터 설정
    int win_sz = frame_k->getPtrParams()->recon.edge.temporal.win_sz; // patch length
    int fat_sz = frame_k->getPtrParams()->recon.edge.temporal.fat_sz; // patch height 

    float angle_thres = frame_k->getPtrParams()->recon.edge.temporal.angle_thres;
    float cosd_thres  = cosf(angle_thres / 180.0f*pi);

    float thres_zncc1     = frame_k->getPtrParams()->recon.edge.temporal.thres_zncc1;
    float thres_zncc2     = frame_k->getPtrParams()->recon.edge.temporal.thres_zncc2;
    float thres_multipeak = frame_k->getPtrParams()->recon.edge.temporal.thres_multipeak;

    float eps_edge   = frame_k->getPtrParams()->recon.edge.temporal.eps_edge;
    float eps_epi    = frame_k->getPtrParams()->recon.edge.temporal.eps_epi;
    float eps_bright = frame_k->getPtrParams()->recon.edge.temporal.eps_bright;
    float eps_edge2 = eps_edge*eps_edge;
    float eps_epi2  = eps_epi*eps_epi;

    // minimum & maximum disparity 값. 해당 disparity 범위 안에서만 서칭.
    // baseline 대비, 0.4 m~ 15 m 범위에서 가능한 disparity 값 범위를 계산해둠
    float baseline = T_ck.block<3, 1>(0, 3).norm();
    float focal = K(0, 0);
    float d_min_default = frame_k->getPtrParams()->recon.edge.temporal.d_min;
    float d_max_default = frame_k->getPtrParams()->recon.edge.temporal.d_max;
    float bf = baseline*focal;
    float bfinv = 1.0f / bf;


    // 본격적인 깊이 복원
    size_t patch_len = (2 * win_sz + 1)*(2 * fat_sz + 1);
    vector<chk::Point2f> pts_patch;
    pts_patch.reserve(patch_len);
    for (int u = -win_sz; u <= win_sz; u++)
        for (int v = -fat_sz; v <= fat_sz; v++)
            pts_patch.push_back(chk::Point2f(u, v));

    float* patch_l = new float[patch_len];
    float* patch_r = new float[patch_len];
    int* valid_l   = new int[patch_len];
    int* valid_r   = new int[patch_len];

    int u_min = -1;
    int u_max = -1;
    vector<float> scores_in;
    scores_in.reserve(1000);

    for (int i = 0; i < n_pts; i++) {
        if (frame_k->left->ft_edge[0]->std_invd[i] > 0.005) { // std_invd 0.005 -> 2 m 에서 1cm오차에 해당.
            // 4 m 에서 3.6 cm 오차
            // 8 m 에서 14.2 cm 오차에 해당.

            // 현재 i 번째 점의 데이터를 추출한다.
            float uk = pts_k[i].x;
            float vk = pts_k[i].y;

            float dmag_k = grad_k[i].z;
            float du_k = grad_k[i].x / dmag_k; // normalize
            float dv_k = grad_k[i].y / dmag_k;

            // uk,vk 에 대해 current image epiline을 찾고, 
            // On the current image, epipolar line's unit directional vector (l_c)
            Eigen::Vector3d coeff_c;
            coeff_c(0) = F_ck(0, 0)*uk + F_ck(0, 1)*vk + F_ck(0, 2);
            coeff_c(1) = F_ck(1, 0)*uk + F_ck(1, 1)*vk + F_ck(1, 2);
            coeff_c(2) = F_ck(2, 0)*uk + F_ck(2, 1)*vk + F_ck(2, 2);
            chk::Point2f l_c(coeff_c(1), -coeff_c(0));
            l_c /= l_c.norm(); // unit vector.

            Eigen::Vector3d Xkwarp = R_ck*(Kinv*Eigen::Vector3d(uk, vk, 1))*500.0f + t_ck; // 500 m far away assumption.
            Eigen::Vector2d pt_c_arbi(K(0, 0)*Xkwarp(0) / Xkwarp(2) + K(0, 2), K(1, 1)*Xkwarp(1) / Xkwarp(2) + K(1, 2));
            Eigen::Vector3d coeff_k; 
            coeff_k(0) = F_ck(0, 0)*pt_c_arbi(0) + F_ck(1, 0)*pt_c_arbi(1) + F_ck(2, 0);
            coeff_k(1) = F_ck(0, 1)*pt_c_arbi(0) + F_ck(1, 1)*pt_c_arbi(1) + F_ck(2, 1);
            coeff_k(2) = F_ck(0, 2)*pt_c_arbi(0) + F_ck(1, 2)*pt_c_arbi(1) + F_ck(2, 2);
            chk::Point2f l_k(coeff_k(1), -coeff_k(0));
            l_k /= l_k.norm(); // unit vector

            // both lc lk need to be same direciton (dot product > 0)
            if (l_c.dot(l_k) < 0) l_c *= -1;

            float cos_th = l_k.x*du_k + l_k.y*dv_k;
            float sin_th = sqrt(1.0f - cos_th*cos_th);

            // cout << l_c(0) << "," << l_c(1) << " / " << l_k(0) << "," << l_k(1) << endl;
            // reconstruct only if image gradient direction forms under 60 degrees btw. key epiline.
            if ((abs(cos_th) > cosd_thres) &&
                (uk > win_sz + 1) && (uk < n_cols - win_sz - 1) &&
                (vk > win_sz + 1) && (vk < n_rows - win_sz - 1)) 
            {
                // initialize validity flag
                flag_valid[i] = true;

                // determine searching range.
                float d_min, d_max;
                bool has_depth = false;
                if (frame_k->left->ft_edge[0]->invd[i] > 0) {
                    float invd_temp = frame_k->left->ft_edge[0]->invd[i];
                    float std_temp = frame_k->left->ft_edge[0]->std_invd[i];
                    if (std_temp < 0.01) std_temp = 0.01; // 왜 clipping을하지? 
                    d_min = 1.0f / (invd_temp + 2.0f*std_temp);
                    d_max = 1.0f / (invd_temp - 2.0f*std_temp);
                    has_depth = true;
                    //cout << "dminmax: " << d_min << "," << d_max << ", d_nominal:"<<1.0f/invd_temp<<endl;
                }
                else {
                    d_min = d_min_default;
                    d_max = d_max_default;
                    //cout << "dminmax: " << d_min << "," << d_max << endl;
                }
                // chirality test. current image 보다 앞에 위치하는 깊이인지 판단.
                if (d_min < t_ck(2) + d_min_default) d_min = t_ck(2) + d_min_default;
                // maximum 범위보다 밖에 있으면 기각. (TODO: scale이 다른 환경으로 확장한다면 바뀌어야한다.)
                if (d_max > d_max_default) flag_valid[i] = false;
                // 현재 알고있는 depth의 min / max 값을 이용해서 keyframe에서 current로 워핑.
                Eigen::Vector3d X_warp_far   = R_ck*(Kinv*Eigen::Vector3d(uk, vk, 1))*d_max + t_ck;
                Eigen::Vector3d X_warp_close = R_ck*(Kinv*Eigen::Vector3d(uk, vk, 1))*d_min + t_ck;

                // in image test.바운더리에 pt_far, pt_close의 선분이 걸치는지 확인한다.
                // 서칭범위를 이미지 내부 / 최대 & 최소 깊이 정보를 고려하여 범위를 좁힌다.
                chk::Point2f pt_start(fx*X_warp_far(0) / X_warp_far(2) + cx, fy*X_warp_far(1) / X_warp_far(2) + cy);
                chk::Point2f pt_end(fx*X_warp_close(0) / X_warp_close(2) + cx, fy*X_warp_close(1) / X_warp_close(2) + cy);
                int offset = (int)ceil(sqrt(win_sz*win_sz + fat_sz*fat_sz));
                int n_search = testBoundary(pt_start, pt_end, n_cols, n_rows, offset);
                
                // pt_end - pt_start 방향을 l_c 방향에 맞춰준다.
                if (l_c.dot(pt_end - pt_start) < 0) {
                    chk::Point2f pt_temp = pt_start;
                    pt_start = pt_end;
                    pt_end   = pt_temp;
                }

                // 매칭쌍을 찾는다.
                //cout << "lk:" << l_k << ",lc:" << l_c << endl;
                //cout << "ptsk: " << pts_k[i] << endl;

                improc::calcNCCstripDirectional_fast2(img_k, img_c, pts_k[i], pt_start, 
                    l_k, l_c, n_search, win_sz, fat_sz, scores_in);

                //for (int ii = 0; ii < scores_in.size(); ii++) {
                //    cout << scores_in[ii] << endl;
                //}
                //cout << "hasdepth:"<< has_depth<<", n_search:" << n_search << ", pt_start:" << pt_start << ", pt_end:" << pt_end << endl;
                float score_best = -1e5;
                vector<int> idx_over; // thres2를 넘은 점들을 모아둠.
                idx_over.reserve(20);
                int idx_best = -1; // 매칭된 점의 index
                for (int j = 0; j < n_search; j++) {
                    // 현재 스코어
                    float score_now = scores_in[j];
                    if (score_now > thres_zncc2) idx_over.push_back(j); // zncc2보다 크면 일단 keep
                    // 만약 현재 ZNCC가 이전 값 보다 높으면 바꾼다.
                    if (score_now > score_best) {
                        score_best = score_now;
                        idx_best = j;
                    }
                } // end for n_search

                // 최고 점수가 threshold를 넘으면서, 모든 idx_over가 6 pixel 이내이고,
                // 탐색 범위의 가장자리가 아닌 경우에만 매칭을 신뢰한다.
                if ((idx_best < 2) || (idx_best > n_search - 2)) flag_valid[i] = false;
                if (flag_valid[i])
                    for (int j = 0; j < idx_over.size(); j++)
                        if (abs(idx_best - idx_over[j]) > thres_multipeak) {
                            flag_valid[i] = false;
                            break;
                        }

                if (flag_valid[i] && (score_best > thres_zncc1)) {
                    //최고 점수를 저장한다.
                    scores_best[i] = score_best;
                    // score history 이용해서 2차함수 모델로 subpixel refinement 수행.
                    float s1 = scores_in[idx_best - 1];
                    float s2 = scores_in[idx_best];
                    float s3 = scores_in[idx_best + 1];
                    chk::Point2f pt_c_best = pt_start + l_c*idx_best - l_c*((s3-s1)/(s3+s1-2.0f*s2)*0.5);

                    // Calculate a triangulation and standard deviation.
                    chk::Point3f xk((uk-cx)*fxinv,(vk-cy)*fyinv,1.0f);
                    chk::Point3f xc((pt_c_best.x - cx)*fxinv, (pt_c_best.y - cy)*fyinv, 1.0f);
                    xk /= xk.norm();
                    xc /= xc.norm();
                    float alpha = acosf(nt_kc.dot(xk));
                    float beta  = acosf(nt_ck.dot(xc));
                    float parallax = pi - alpha - beta;
                    float Xkp_norm = t_ck_norm*sinf(beta + a) / sinf(pi - alpha - beta - a);
                    float Xk_norm  = t_ck_norm*sinf(beta) / sinf(pi - alpha - beta);

                    float tau_d    = (Xkp_norm - Xk_norm);
                    float tau_invd = abs(tau_d / (Xkp_norm*Xk_norm) );
                    chk::Point3f Xk = xk*Xk_norm;
                    float invz = 1.0f / Xk.z;

                    // cherality check!
                    if (parallax > 0) { // 제대로 복원이 된 경우만 넣는다.
                        invd_save[i] = invz;
                        std_invd_save[i] = tau_invd;
                    }
                    else flag_valid[i] = false;
                    // cout << "taud: " << tau_d << ", d:" << Xk.z <<", tauinvd:"<< tau_invd <<", paral:"<<parallax/pi*180.0f<<" [deg]" << endl;
                }
                else {
                    flag_valid[i] = false;
                }
            }
        }
    } // end for (int i = 0) (main loop)

    // 결과 업데이트 
    for (int k = 0; k < n_pts; k++) {
        if (flag_valid[k] > 0)
        {
            if (frame_k->left->ft_edge[0]->invd[k] > 0)
            {
                // 퓨전
                frame_k->left->ft_edge[0]->df[k].updateDF(invd_save[k], std_invd_save[k]);
            }
            else { // 아무것도 없는 경우. (이번에 처음 reconstructed 인 경우)
                float invd_curr = invd_save[k];
                float std_curr = std_invd_save[k];
                frame_k->left->ft_edge[0]->df[k].set_mu(invd_curr);
                frame_k->left->ft_edge[0]->df[k].set_sig(std_curr);
                if (frame_k->left->ft_edge[0]->df[k].zmin() > invd_curr) frame_k->left->ft_edge[0]->df[k].set_zmin(invd_curr);
                if (frame_k->left->ft_edge[0]->df[k].zmax() < invd_curr) frame_k->left->ft_edge[0]->df[k].set_zmax(invd_curr);
                frame_k->left->ft_edge[0]->is_recon[k] = true;
            } // end if ~ else

              // inverse dpeth를 넣어준다.
            frame_k->left->ft_edge[0]->invd[k] = frame_k->left->ft_edge[0]->df[k].mu();
            frame_k->left->ft_edge[0]->std_invd[k] = frame_k->left->ft_edge[0]->df[k].sig();
            float depth = 1.0f / frame_k->left->ft_edge[0]->invd[k];
            frame_k->left->ft_edge[0]->pts3d[k].x = (Kinv(0, 0)*frame_k->left->ft_edge[0]->pts[k].x + Kinv(0, 2)) * depth;
            frame_k->left->ft_edge[0]->pts3d[k].y = (Kinv(1, 1)*frame_k->left->ft_edge[0]->pts[k].y + Kinv(1, 2)) * depth;
            frame_k->left->ft_edge[0]->pts3d[k].z = depth;
        }//end if
    }//end for
    
    // free allocated
    delete[] patch_l;
    delete[] patch_r;
    delete[] valid_l;
    delete[] valid_r;
};


void depthrecon::depthReconTemporalPoint(StereoFrame* frame_k, StereoFrame* frame_c, const Eigen::Matrix4d& T_ck) {
    // 이 함수에서 업데이트 해야하는 것.
    // (1) 대상이 되는 점의 깊이값, (2) 3차원 좌표 생성, (3) 역수깊이, (4) 깊이 표준편차

    // intrinsic 및 extrinsic load
    int n_cols = frame_k->getPtrCams()->left->n_cols;
    int n_rows = frame_k->getPtrCams()->left->n_rows;

    Eigen::Matrix3d K = frame_k->K_pyr[0];
    Eigen::Matrix3d Kinv = frame_k->Kinv_pyr[0];
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxinv = 1.0f / fx;
    float fyinv = 1.0f / fy;

    Eigen::Matrix3d R_ck = T_ck.block<3, 3>(0, 0);
    Eigen::Vector3d t_ck = T_ck.block<3, 1>(0, 3);
    Eigen::Matrix3d R_kc = R_ck.inverse();
    Eigen::Vector3d t_kc = R_kc*(-t_ck);

    chk::Point3f nt_ck(t_ck(0), t_ck(1), t_ck(2));
    chk::Point3f nt_kc(t_kc(0), t_kc(1), t_kc(2));
    float t_ck_norm = t_ck.norm(); // t_ck.norm == t_kc.norm.
    nt_ck /= t_ck_norm;
    nt_kc /= t_ck_norm;

    float a = 2.0f*atan(0.25f / fx); // for 3d recon. and its std. value calculation.

                                     // Fundamental metrix & Essential matrix
    Eigen::Matrix3d F_ck = Kinv.transpose()*Lie::hat(t_ck)*Kinv;

    // load images
    cv::Mat& img_k = frame_k->left->img[0];
    cv::Mat& img_c = frame_c->left->img[0];

    // 복원 대상 점 & 그들의 그래디언트 방향
    vector<chk::Point2f>& pts_k = frame_k->left->ft_point[0]->pts;

    size_t n_pts = pts_k.size();

    cout << "    -> temporal candidate point-n_pts : " << n_pts << "\n";

    vector<float> scores_best(n_pts, -1.0f);
    vector<char>  flag_valid(n_pts, 0);
    vector<float> invd_save(n_pts, -1.0f);
    vector<float> std_invd_save(n_pts, -1.0f);

    // 파라미터 설정
    int win_sz = frame_k->getPtrParams()->recon.point.temporal.win_sz; // patch length
    int fat_sz = frame_k->getPtrParams()->recon.point.temporal.fat_sz; // patch height 
    
    float thres_zncc1 = frame_k->getPtrParams()->recon.point.temporal.thres_zncc1;
    float thres_zncc2 = frame_k->getPtrParams()->recon.point.temporal.thres_zncc2;
    float thres_multipeak = frame_k->getPtrParams()->recon.point.temporal.thres_multipeak;

    float eps_edge = frame_k->getPtrParams()->recon.point.temporal.eps_edge;
    float eps_epi = frame_k->getPtrParams()->recon.point.temporal.eps_epi;
    float eps_bright = frame_k->getPtrParams()->recon.point.temporal.eps_bright;
    float eps_edge2 = eps_edge*eps_edge;
    float eps_epi2 = eps_epi*eps_epi;

    // minimum & maximum disparity 값. 해당 disparity 범위 안에서만 서칭.
    // baseline 대비, 0.4 m~ 15 m 범위에서 가능한 disparity 값 범위를 계산해둠
    float baseline = T_ck.block<3, 1>(0, 3).norm();
    float focal = K(0, 0);
    float d_min_default = frame_k->getPtrParams()->recon.point.temporal.d_min;
    float d_max_default = frame_k->getPtrParams()->recon.point.temporal.d_max;
    float bf = baseline*focal;
    float bfinv = 1.0f / bf;

    // 본격적인 깊이 복원
    vector<float> scores_in;
    scores_in.reserve(1000);
    for (int i = 0; i < n_pts; i++) {
        if (frame_k->left->ft_point[0]->std_invd[i] > 0.005) { // std_invd 0.005 -> 2 m 에서 1cm오차에 해당.
                                                              // 4 m 에서 3.6 cm 오차
                                                              // 8 m 에서 14.2 cm 오차에 해당.

                                                              // 현재 i 번째 점의 데이터를 추출한다.
            float uk = pts_k[i].x;
            float vk = pts_k[i].y;

            // uk,vk 에 대해 current image epiline을 찾고, 
            // On the current image, epipolar line's unit directional vector (l_c)
            Eigen::Vector3d coeff_c;
            coeff_c(0) = F_ck(0, 0)*uk + F_ck(0, 1)*vk + F_ck(0, 2);
            coeff_c(1) = F_ck(1, 0)*uk + F_ck(1, 1)*vk + F_ck(1, 2);
            coeff_c(2) = F_ck(2, 0)*uk + F_ck(2, 1)*vk + F_ck(2, 2);
            chk::Point2f l_c(coeff_c(1), -coeff_c(0));
            l_c /= l_c.norm(); // unit vector.

            Eigen::Vector3d Xkwarp = R_ck*(Kinv*Eigen::Vector3d(uk, vk, 1))*500.0f + t_ck; // 500 m far away assumption.
            Eigen::Vector2d pt_c_arbi(K(0, 0)*Xkwarp(0) / Xkwarp(2) + K(0, 2), K(1, 1)*Xkwarp(1) / Xkwarp(2) + K(1, 2));
            Eigen::Vector3d coeff_k;
            coeff_k(0) = F_ck(0, 0)*pt_c_arbi(0) + F_ck(1, 0)*pt_c_arbi(1) + F_ck(2, 0);
            coeff_k(1) = F_ck(0, 1)*pt_c_arbi(0) + F_ck(1, 1)*pt_c_arbi(1) + F_ck(2, 1);
            coeff_k(2) = F_ck(0, 2)*pt_c_arbi(0) + F_ck(1, 2)*pt_c_arbi(1) + F_ck(2, 2);
            chk::Point2f l_k(coeff_k(1), -coeff_k(0));
            l_k /= l_k.norm(); // unit vector

                               // both lc lk need to be same direciton (dot product > 0)
            if (l_c.dot(l_k) < 0) l_c *= -1;
        
            // cout << l_c(0) << "," << l_c(1) << " / " << l_k(0) << "," << l_k(1) << endl;
            // reconstruct only if image gradient direction forms under 60 degrees btw. key epiline.
            if ((uk > win_sz + 1) && (uk < n_cols - win_sz - 1) &&
                (vk > win_sz + 1) && (vk < n_rows - win_sz - 1))
            {
                // initialize validity flag
                flag_valid[i] = true;

                // determine searching range.
                float d_min, d_max;
                bool has_depth = false;
                if (frame_k->left->ft_point[0]->invd[i] > 0) {
                    float invd_temp = frame_k->left->ft_point[0]->invd[i];
                    float std_temp = frame_k->left->ft_point[0]->std_invd[i];
                    if (std_temp < 0.01) std_temp = 0.01; // 왜 clipping을하지? 
                    d_min = 1.0f / (invd_temp + 2.0f*std_temp);
                    d_max = 1.0f / (invd_temp - 2.0f*std_temp);
                    has_depth = true;
                    //cout << "dminmax: " << d_min << "," << d_max << ", d_nominal:"<<1.0f/invd_temp<<endl;
                }
                else {
                    d_min = d_min_default;
                    d_max = d_max_default;
                    //cout << "dminmax: " << d_min << "," << d_max << endl;
                }
                // chirality test. current image 보다 앞에 위치하는 깊이인지 판단.
                if (d_min < t_ck(2) + d_min_default) d_min = t_ck(2) + d_min_default;
                // maximum 범위보다 밖에 있으면 기각. (TODO: scale이 다른 환경으로 확장한다면 바뀌어야한다.)
                if (d_max > d_max_default) flag_valid[i] = false;
                // 현재 알고있는 depth의 min / max 값을 이용해서 keyframe에서 current로 워핑.
                Eigen::Vector3d X_warp_far = R_ck*(Kinv*Eigen::Vector3d(uk, vk, 1))*d_max + t_ck;
                Eigen::Vector3d X_warp_close = R_ck*(Kinv*Eigen::Vector3d(uk, vk, 1))*d_min + t_ck;

                chk::Point2f pt_c_far(fx*X_warp_far(0) / X_warp_far(2) + cx, fy*X_warp_far(1) / X_warp_far(2) + cy);
                chk::Point2f pt_c_close(fx*X_warp_close(0) / X_warp_close(2) + cx, fy*X_warp_close(1) / X_warp_close(2) + cy);

                // in image test.바운더리에 pt_far, pt_close의 선분이 걸치는지 확인한다.
                // 서칭범위를 이미지 내부 / 최대 & 최소 깊이 정보를 고려하여 범위를 좁힌다.
                chk::Point2f pt_start = pt_c_far;
                chk::Point2f pt_end = pt_c_close;
                int offset = (int)ceil(sqrt(win_sz*win_sz + fat_sz*fat_sz));
                int n_search = testBoundary(pt_start, pt_end, n_cols, n_rows, offset);

                // pt_end - pt_start 방향을 l_c 방향에 맞춰준다.
                if (l_c.dot(pt_end - pt_start) < 0) {
                    chk::Point2f pt_temp = pt_start;
                    pt_start = pt_end;
                    pt_end = pt_temp;
                }

                // 매칭쌍을 찾는다.
                //cout << "lk:" << l_k << ",lc:" << l_c << endl;
                //cout << "ptsk: " << pts_k[i] << endl;

                improc::calcNCCstripDirectional_fast2(img_k, img_c, pts_k[i], pt_start,
                    l_k, l_c, n_search, win_sz, fat_sz, scores_in);

                //for (int ii = 0; ii < scores_in.size(); ii++) {
                //    cout << scores_in[ii] << endl;
                //}
                //cout << "hasdepth:"<< has_depth<<", n_search:" << n_search << ", pt_start:" << pt_start << ", pt_end:" << pt_end << endl;
                float score_best = -1e5;
                vector<int> idx_over; // thres2를 넘은 점들을 모아둠.
                idx_over.reserve(20);
                int idx_best = -1; // 매칭된 점의 index
                for (int j = 0; j < n_search; j++) {
                    // 현재 스코어
                    float score_now = scores_in[j];
                    if (score_now > thres_zncc2) idx_over.push_back(j); // zncc2보다 크면 일단 keep
                                                                        // 만약 현재 ZNCC가 이전 값 보다 높으면 바꾼다.
                    if (score_now > score_best) {
                        score_best = score_now;
                        idx_best = j;
                    }
                } // end for n_search

                  // 최고 점수가 threshold를 넘으면서, 모든 idx_over가 6 pixel 이내이고,
                  // 탐색 범위의 가장자리가 아닌 경우에만 매칭을 신뢰한다.
                if ((idx_best < 2) || (idx_best > n_search - 2)) flag_valid[i] = false;
                if (flag_valid[i])
                    for (int j = 0; j < idx_over.size(); j++)
                        if (abs(idx_best - idx_over[j]) > thres_multipeak) {
                            flag_valid[i] = false;
                            break;
                        }

                if (flag_valid[i] && (score_best > thres_zncc1)) {
                    //최고 점수를 저장한다.
                    scores_best[i] = score_best;
                    // score history 이용해서 2차함수 모델로 subpixel refinement 수행.
                    float s1 = scores_in[idx_best - 1];
                    float s2 = scores_in[idx_best];
                    float s3 = scores_in[idx_best + 1];
                    chk::Point2f pt_c_best = pt_start + l_c*idx_best - l_c*((s3 - s1) / (s3 + s1 - 2.0f*s2)*0.5);

                    // Calculate a triangulation and standard deviation.
                    chk::Point3f xk((uk - cx)*fxinv, (vk - cy)*fyinv, 1.0f);
                    chk::Point3f xc((pt_c_best.x - cx)*fxinv, (pt_c_best.y - cy)*fyinv, 1.0f);
                    xk /= xk.norm();
                    xc /= xc.norm();
                    float alpha = acosf(nt_kc.dot(xk));
                    float beta = acosf(nt_ck.dot(xc));
                    float parallax = pi - alpha - beta;
                    float Xkp_norm = t_ck_norm*sinf(beta + a) / sinf(pi - alpha - beta - a);
                    float Xk_norm = t_ck_norm*sinf(beta) / sinf(pi - alpha - beta);

                    float tau_d = (Xkp_norm - Xk_norm);
                    float tau_invd = abs(tau_d / (Xkp_norm*Xk_norm));
                    chk::Point3f Xk = xk*Xk_norm;
                    float invz = 1.0f / Xk.z;

                    // cherality check!
                    if (parallax > 0) { // 제대로 복원이 된 경우만 넣는다.
                        invd_save[i]     = invz;
                        std_invd_save[i] = tau_invd;
                    }
                    else flag_valid[i] = false;
                    // cout << "taud: " << tau_d << ", d:" << Xk.z <<", tauinvd:"<< tau_invd <<", paral:"<<parallax/pi*180.0f<<" [deg]" << endl;
                }
                else {
                    flag_valid[i] = false;
                }
            }
        }
    } // end for (int i = 0) (main loop)

      // 결과 업데이트 
    for (int k = 0; k < n_pts; k++) {
        if (flag_valid[k] > 0)
        {
            if (frame_k->left->ft_point[0]->invd[k] > 0)
            {
                // 퓨전
                frame_k->left->ft_point[0]->df[k].updateDF(invd_save[k], std_invd_save[k]);
            }
            else { // 아무것도 없는 경우. (이번에 처음 reconstructed 인 경우)
                float invd_curr = invd_save[k];
                float std_curr = std_invd_save[k];
                frame_k->left->ft_point[0]->df[k].set_mu(invd_curr);
                frame_k->left->ft_point[0]->df[k].set_sig(std_curr);
                if (frame_k->left->ft_point[0]->df[k].zmin() > invd_curr) frame_k->left->ft_point[0]->df[k].set_zmin(invd_curr);
                if (frame_k->left->ft_point[0]->df[k].zmax() < invd_curr) frame_k->left->ft_point[0]->df[k].set_zmax(invd_curr);
                frame_k->left->ft_point[0]->is_recon[k] = true;
            } // end if ~ else

              // inverse dpeth를 넣어준다.
            frame_k->left->ft_point[0]->invd[k] = frame_k->left->ft_point[0]->df[k].mu();
            frame_k->left->ft_point[0]->std_invd[k] = frame_k->left->ft_point[0]->df[k].sig();
            float depth = 1.0f / frame_k->left->ft_point[0]->invd[k];
            frame_k->left->ft_point[0]->pts3d[k].x = (Kinv(0, 0)*frame_k->left->ft_point[0]->pts[k].x + Kinv(0, 2)) * depth;
            frame_k->left->ft_point[0]->pts3d[k].y = (Kinv(1, 1)*frame_k->left->ft_point[0]->pts[k].y + Kinv(1, 2)) * depth;
            frame_k->left->ft_point[0]->pts3d[k].z = depth;
        }//end if
    }//end for
};

int depthrecon::testBoundary(chk::Point2f& pt_start, chk::Point2f& pt_end, const int& n_cols, const int& n_rows, const int& offset) 
{
    // start -> end 방향이 epipolar line 방향이랑 같아야 한다. 함수 밖에서 정렬해주도록하자.
    // start + epi*n_search = end 가 되도록! 
    chk::Point2f NW(offset, offset), NE(n_cols- offset, offset), SW(offset,n_rows- offset), SE(n_cols- offset,n_rows- offset);
    // 점들이 내부에 존재하는지 판단.
    // 점 2개 내부 : 아무것도 할 필요 X
    // 점 1개 내부 : 1개의 절편 탐색->pt_end가 내부인 경우 vs.밖인 경우,
    // 점 0개 내부 : 0개의 절편 or 2개의 절편 존재. 0개 절편의 경우에는 서칭범위 0으로 출력. 
    
    // n_search : 서칭 길이. 최소 서칭길이는 5개 이상이 되도록 설정해준다. (아무리 적어도 5 pixel 이상 비교하도록)
    int n_search = -1;

    int in_pt_start = 0;
    int in_pt_end = 0;
    if ((pt_start.x > offset) && (pt_start.x < n_cols - offset) && (pt_start.y > offset) && (pt_start.y < n_rows- offset)) in_pt_start = 1;
    if ((pt_end.x > offset) && (pt_end.x < n_cols - offset) && (pt_end.y > offset) && (pt_end.y < n_rows- offset)) in_pt_end = 1;

    // # of in image point 
    int in_count = in_pt_start + in_pt_end;
    if (in_count == 2) { // OKAY! both points are in boundary. Nothing to do!
        n_search = round((pt_start - pt_end).norm());
        if (n_search < 5) n_search = 5;
        return n_search;
    }
    else if (in_count == 1) { // 둘 중 점 1개가 내부에 존재
        // boundary와의 intersection -> 1개.
        chk::Point2f epiline_section = pt_end - pt_start;

        chk::Point2f vec2nw = NW - pt_start;
        chk::Point2f vec2sw = SW - pt_start;
        chk::Point2f vec2se = SE - pt_start;
        chk::Point2f vec2ne = NE - pt_start;

        float s_nw = epiline_section.cross2d(vec2nw);
        float s_sw = epiline_section.cross2d(vec2sw);
        float s_se = epiline_section.cross2d(vec2se);
        float s_ne = epiline_section.cross2d(vec2ne);

        chk::Point2f pt_update(0, 0);
        // u = lu/lv*(v-pv)+pu   or   v = lv/lu*(u-pu)+pv;
        // 네 변의 touch 여부 결정. (외적 이용!!)
        if (s_nw*s_sw < 0) { // west side intersect
            pt_update.x = (float)offset;
            pt_update.y = epiline_section.y / epiline_section.x*((float)offset - pt_end.x) + pt_end.y;
        }
        else if (s_ne*s_se < 0) { // east side intersect
            pt_update.x = (float)(n_cols- offset);
            pt_update.y = epiline_section.y / epiline_section.x*((float)(n_cols- offset) - pt_end.x) + pt_end.y;
        }
        else if (s_nw*s_ne < 0) { // north side intersect
            pt_update.x = epiline_section.x / epiline_section.y*((float)offset - pt_end.y) + pt_end.x;
            pt_update.y = offset;
        }
        else { // south side intersect
            pt_update.x = epiline_section.x / epiline_section.y*((float)(n_rows- offset) - pt_end.y) + pt_end.x;
            pt_update.y = (float)(n_rows- offset);
        }
        if (in_pt_end > 0) pt_start = pt_update; // pt_end 가 내부인경우, pt_start 업데이트.
        else pt_end = pt_update; // pt_start가 내부인경우, pt_end 업데이트.
        n_search = (int)round((pt_end - pt_start).norm());
        if (n_search < 5) n_search = 5;
        return n_search;
    }
    else { // 점 0개가 내부에 있다. ...
        chk::Point2f epiline_section = pt_end - pt_start;

        chk::Point2f vec2nw = NW - pt_start;
        chk::Point2f vec2sw = SW - pt_start;
        chk::Point2f vec2se = SE - pt_start;
        chk::Point2f vec2ne = NE - pt_start;

        float s_nw = epiline_section.cross2d(vec2nw);
        float s_sw = epiline_section.cross2d(vec2sw);
        float s_se = epiline_section.cross2d(vec2se);
        float s_ne = epiline_section.cross2d(vec2ne);

        chk::Point2f pt_update[2];

        // u = lu/lv*(v-pv)+pu   or   v = lv/lu*(u-pu)+pv;
        // 네 변의 touch 여부 결정. (외적 이용!!)
        int cnt = -1;
        if (s_nw*s_sw < 0) { // west side intersect
            ++cnt;
            pt_update[cnt].x = offset;
            pt_update[cnt].y = epiline_section.y / epiline_section.x*(offset - pt_end.x) + pt_end.y;
        }
        if (s_ne*s_se < 0) { // east side intersect
            ++cnt;
            pt_update[cnt].x = (float)n_cols- offset;
            pt_update[cnt].y = epiline_section.y / epiline_section.x*((float)(n_cols- offset) - pt_end.x) + pt_end.y;
        }
        if (s_nw*s_ne < 0) { // north side intersect
            ++cnt;
            pt_update[cnt].x = epiline_section.x / epiline_section.y*((float)offset - pt_end.y) + pt_end.x;
            pt_update[cnt].y = (float)offset;
        }
        if (s_sw*s_se < 0) { // south side intersect
            ++cnt;
            pt_update[cnt].x = epiline_section.x / epiline_section.y*((float)(n_rows- offset)- pt_end.y) + pt_end.x;
            pt_update[cnt].y = (float)n_rows- offset;
        }

        if (cnt > -1) // 2개여야함. 0개인경우, 아무곳도 touch 안하는 경우이므로 배제.
        {
            pt_start = pt_update[0];
            pt_end   = pt_update[1];
            n_search = (int)round((pt_end - pt_start).norm());
            if (n_search < 5) n_search = 5;
            return n_search;
        }
        else {
            pt_start.x = 0; pt_start.y = 0;
            pt_end.x = 0; pt_end.y = 0;

            n_search = 0;
            return n_search;
        }
    }

};


void depthrecon::pointDepthUpdate(StereoFrame* frame_k, chk::AffineKLTTracker* tracker, const Eigen::Matrix4d& T_ck) {
    // 복원 대상 점 & 그들의 그래디언트 방향
    vector<chk::Point2f>& pts_k = frame_k->left->ft_point[0]->pts;
    int n_pts = (int)pts_k.size();

    
    // intrinsic 및 extrinsic load
    int n_cols = frame_k->getPtrCams()->left->n_cols;
    int n_rows = frame_k->getPtrCams()->left->n_rows;

    Eigen::Matrix3d K = frame_k->K_pyr[0];
    Eigen::Matrix3d Kinv = frame_k->Kinv_pyr[0];
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    float fxinv = 1.0f / fx;
    float fyinv = 1.0f / fy;

    Eigen::Matrix3d R_ck = T_ck.block<3, 3>(0, 0);
    Eigen::Vector3d t_ck = T_ck.block<3, 1>(0, 3);

    float baseline = t_ck.norm();
    float bf = baseline*fx;
    float bfinv = 1.0f / bf;
    
    Eigen::Matrix3d R_ckKinv = R_ck*Kinv;
    Eigen::Vector3d Rf1, f2, pts1_homo, pts2_homo;
    Eigen::Vector3d crossRf1f2, crossf2t21, crossRf1t21;
    Eigen::Vector3d point_temp;

    vector<int> flag_valid(n_pts, 1);
    vector<float> invd_save(n_pts, -1.0f);
    vector<float> std_invd_save(n_pts, -1.0f);

    // tracking results
    chk::Vec2* p_track = tracker->track_ref->pts_p_tracked;
    chk::Vec2* p_pts_k = tracker->track_ref->pts_p;
    int* p_mask = tracker->mask;

    // update depth.
    double lam1, lam2, invlam1, invlam2, invnormcross_Rf1f2;
    for (int i = 0; i < n_pts; ++i) {
        if (!*(p_mask + i)) {
            flag_valid[i] = 0;
            continue;
        }

        pts1_homo << (*(p_pts_k + i))(0), (*(p_pts_k + i))(1), 1.0;
        pts2_homo << (*(p_track+i))(0), (*(p_track + i))(1), 1.0;
        Rf1 = R_ckKinv*pts1_homo;
        Rf1 /= (Rf1.norm());

        f2 = Kinv*pts2_homo;
        f2 /= (f2.norm());

        crossRf1f2 = Rf1.cross(f2);
        crossf2t21 = f2.cross(t_ck);
        crossRf1t21 = Rf1.cross(t_ck);

        // lambdas
        invnormcross_Rf1f2 = 1.0f / (crossRf1f2.norm());
        lam1 = (crossf2t21.norm())  * invnormcross_Rf1f2;
        lam2 = (crossRf1t21.norm()) * invnormcross_Rf1f2;

        // invlam1 = 1 / lam1;
        // invlam2 = 1 / lam2;
        // point_temp = (invlam1*t21 + Rf1 + f2) / (invlam1 + invlam2);

        point_temp = lam2*(t_ck + lam1*(Rf1 + f2)) / (lam1 + lam2);
        // 위 식과 같은 결과를 도출하지만, 나눗셈 연산의 갯수가 적어진다.

        invd_save[i] = 1.0f / (float)point_temp(2);
        std_invd_save[i] = 1.0 * bfinv;

    };

    // 결과 업데이트 
    for (int k = 0; k < n_pts; k++) {
        if (flag_valid[k])
        {
            if (frame_k->left->ft_point[0]->invd[k] > 0)
            {
                // 퓨전
                frame_k->left->ft_point[0]->df[k].updateDF(invd_save[k], std_invd_save[k]);
            }
            else { // 아무것도 없는 경우. (이번에 처음 reconstructed 인 경우)
                float invd_curr = invd_save[k];
                float std_curr = std_invd_save[k];
                frame_k->left->ft_point[0]->df[k].set_mu(invd_curr);
                frame_k->left->ft_point[0]->df[k].set_sig(std_curr);
                if (frame_k->left->ft_point[0]->df[k].zmin() > invd_curr) frame_k->left->ft_point[0]->df[k].set_zmin(invd_curr);
                if (frame_k->left->ft_point[0]->df[k].zmax() < invd_curr) frame_k->left->ft_point[0]->df[k].set_zmax(invd_curr);
                frame_k->left->ft_point[0]->is_recon[k] = true;
            } // end if ~ else

              // inverse dpeth를 넣어준다.
            frame_k->left->ft_point[0]->invd[k] = frame_k->left->ft_point[0]->df[k].mu();
            frame_k->left->ft_point[0]->std_invd[k] = frame_k->left->ft_point[0]->df[k].sig();
            float depth = 1.0f / frame_k->left->ft_point[0]->invd[k];
            frame_k->left->ft_point[0]->pts3d[k].x = (Kinv(0, 0)*frame_k->left->ft_point[0]->pts[k].x + Kinv(0, 2)) * depth;
            frame_k->left->ft_point[0]->pts3d[k].y = (Kinv(1, 1)*frame_k->left->ft_point[0]->pts[k].y + Kinv(1, 2)) * depth;
            frame_k->left->ft_point[0]->pts3d[k].z = depth;
        }//end if
    }//end for
};
#endif