#ifndef _STEREOFRAME_H_
#define _STEREOFRAME_H_

#include <iostream>
#include <vector>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"  
#include "Eigen/Dense"

#include "../Params.h"
#include "../image/image_proc.h"

#include "../frame/frame.h"
#include "../camera/stereo_cameras.h"


class StereoFrame {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	int id;
	Eigen::Matrix4d pose;
	cv::Mat img_l, img_r; // CV_8UC1
	cv::Mat img_lf, img_rf; // CV_32FC1
    
	Frame* left; // left pyramid frames.
	Frame* right; // right pyramid frames.

	vector<Eigen::Matrix3d> K_pyr;
	vector<Eigen::Matrix3d> Kinv_pyr;
    vector<float> fx_pyr;
    vector<float> fy_pyr;
    vector<float> cx_pyr;
    vector<float> cy_pyr;
    vector<float> fxinv_pyr;
    vector<float> fyinv_pyr;

    // 이미지 패치들
    vector<vector<chk::Point2f>> patch_point; // point patch
    vector<vector<chk::Point2f>> patch_edge;

public:
	StereoFrame( // constructor
        int n_img, Eigen::Matrix4d T_0k, int max_pyr_lvl_input, 
        StereoCameras*& stereo_cams, Params* params_);
	~StereoFrame();// destructor

    StereoCameras* getPtrCams() const { return this->cams; };
    Params* getPtrParams() const { return this->params; };

	// frame을 새로 채워넣거나 or 기존것을 업데이트
	// (frame을 새로만들지않고, 기존에 존재하던 frame_k 또는 frame_c에 대해서 수행하는 것임.)
	void framePyrConstruct(const cv::Mat& img_l, const cv::Mat& img_r, const int& n_img, const Eigen::Matrix4d& pose, const bool& is_keyframe, Params* params);
    void initializePatchPoint(bool flag_sparse);
    void initializePatchEdge();

    // void propagateDepthEdge(frame_k, frame_c);
    // void propagateDepthPoint(frame_k, frame_c);

    

private: // 보통 알고리즘 구동을 위한 것들을 private로 선언한다.
    Params* params;
    StereoCameras* cams; // 외부에서 정의된 것을 받아온다.

    int max_pyr_lvl; // maximum pyramid levels
    vector<float> scores;

    
};


/* ================================================================================
================================= Implementation ==================================
=================================================================================== */
// 생성자.
StereoFrame::StereoFrame(int n_img, Eigen::Matrix4d T_0k, int max_pyr_lvl_input, 
	StereoCameras*& stereo_cams, Params* params_)
	: id(n_img), pose(T_0k), max_pyr_lvl(max_pyr_lvl_input)
{
	// 여기서는 이것저것 하는게 아니라, 그냥 frame 구조체만을 만들어준다.
	// stereoFrameConstruct 함수에서 이미지 + 기타등등 수행!
	id = n_img;
	pose = T_0k;
	cams = stereo_cams;
	params = params_;

	left  = new Frame(stereo_cams->left,  max_pyr_lvl, params_);
	right = new Frame(stereo_cams->right, max_pyr_lvl, params_);

	// imagesRead 함수에서 사용하기 위한 선 할당.
	// cv::Mat이 미리 할당되어있고, 불러와야 할 이미지와 크기와 channel이 같으면
	// 메모리 할당없이 바로 복사가 시작된다.
	img_l = cv::Mat::zeros(left->cam->n_rows, left->cam->n_cols, CV_8UC1);
	img_r = cv::Mat::zeros(left->cam->n_rows, left->cam->n_cols, CV_8UC1);
	img_lf = cv::Mat::zeros(left->cam->n_rows, left->cam->n_cols, CV_32FC1);
	img_rf = cv::Mat::zeros(left->cam->n_rows, left->cam->n_cols, CV_32FC1);
	
	// pyramid Intrinsics
	K_pyr.reserve(max_pyr_lvl);
	Kinv_pyr.reserve(max_pyr_lvl);

	// pyramid 레벨에 따라서 다른 동작을 보여야함.
	// rectified K.
	for (int lvl = 0; lvl < max_pyr_lvl; lvl++) {
		if (lvl == 0) {
			K_pyr.push_back(stereo_cams->left->K);
			Kinv_pyr.push_back(K_pyr[lvl].inverse());
		}
		else {
			Eigen::Matrix3d K_temp;
			K_temp << K_pyr[0](0, 0) / pow(2,lvl), 0, K_pyr[0](0, 2) / pow(2, lvl),
				0, K_pyr[0](1, 1) / pow(2, lvl), K_pyr[0](1, 2) / pow(2, lvl),
				0, 0, 1;
			K_pyr.push_back(K_temp);
			Kinv_pyr.push_back(K_temp.inverse());
		} // end if
	} // end for

    for (int lvl = 0; lvl < max_pyr_lvl; lvl++) 
    {
        fx_pyr.push_back(K_pyr[lvl](0, 0));
        fy_pyr.push_back(K_pyr[lvl](1, 1));
        cx_pyr.push_back(K_pyr[lvl](0, 2));
        cy_pyr.push_back(K_pyr[lvl](1, 2));
        fxinv_pyr.push_back(1.0f / K_pyr[lvl](0, 0));
        fyinv_pyr.push_back(1.0f / K_pyr[lvl](1, 1));
    }

    // 알고리즘을 위한 선할당.
	scores.reserve(50000);

    // 패치 할당.
    bool flag_sparse = true;
    initializePatchPoint(flag_sparse); // make point patch (maximum 4 x 4)
    initializePatchEdge(); // make edge patch (one-pixel strip).
}; // end constructor

// 소멸자
StereoFrame::~StereoFrame() {
	cout << "frames are returned." << endl;
	cout << "  L left:  ";
	delete left;
	cout << "  L right: ";
	delete right;
}; // end destructor


void StereoFrame::initializePatchPoint(bool flag_sparse) {
    int max_lvl = params->pyr.max_lvl;
    patch_point.reserve(max_lvl);
    for (int i = 0; i < max_lvl; i++)
        patch_point.push_back(vector<chk::Point2f>());

    int win_sz, cnt;
    for (int lvl = 0; lvl < max_lvl; lvl++) {
        if (lvl <= 4) {
            if (flag_sparse) {
                /*win_sz = 3;
                len = 2 * win_sz*win_sz + 2;
                cnt = 0;
                for (int i = -win_sz; i < win_sz + 1; i += 2)
                    for (int j = -win_sz; j < win_sz + 1; j += 2)
                        patch_point[lvl].push_back(chk::Point2f(i, j));
                for (int i = -win_sz + 1; i < win_sz; i += 2)
                    for (int j = -win_sz + 1; j < win_sz; j += 2)
                        patch_point[lvl].push_back(chk::Point2f(i, j));*/
                float iscale = 1.0f / powf(1.2f, lvl);
                patch_point[lvl].push_back(chk::Point2f(0,-2 * iscale));
                patch_point[lvl].push_back(chk::Point2f(-1 * iscale, -1 * iscale));
                patch_point[lvl].push_back(chk::Point2f(1 * iscale, -1 * iscale));
                patch_point[lvl].push_back(chk::Point2f(-2 * iscale, 0));
                patch_point[lvl].push_back(chk::Point2f(0, 0));
                patch_point[lvl].push_back(chk::Point2f(2 * iscale, 0));
                patch_point[lvl].push_back(chk::Point2f(-1 * iscale, 1 * iscale));
                patch_point[lvl].push_back(chk::Point2f(0, 2 * iscale));
            } // end if(flag_sparse)
            else {
                win_sz = 2;
                cnt = 0;
                for (int i = -win_sz; i < win_sz + 1; i += 2)
                    for (int j = -win_sz; j < win_sz + 1; j += 2)
                        patch_point[lvl].push_back(chk::Point2f(i, j));
            } // end else
        } // end if(lvl <=1)
        else
        {
            patch_point[lvl].push_back(chk::Point2f(0, 0));
            patch_point[lvl].push_back(chk::Point2f(-0.5, -0.5));
            patch_point[lvl].push_back(chk::Point2f(0.5, -0.5));
            patch_point[lvl].push_back(chk::Point2f(-0.5, 0.5));

        } // end else(lvl > 2)
    }; // end for
};

void StereoFrame::initializePatchEdge() {
    // line patch, win_sz = 3;
    int max_lvl = params->pyr.max_lvl;
    patch_edge.reserve(max_lvl);
    for (int i = 0; i < max_lvl; i++)
        patch_edge.push_back(vector<chk::Point2f>());

    for (int lvl = 0; lvl < max_lvl; lvl++) {
        float scaler = 1.0f / pow(1.4f, lvl);
        patch_edge[lvl].push_back(chk::Point2f(-3.0f * scaler, 0));
        patch_edge[lvl].push_back(chk::Point2f(-2.0f * scaler, 0));
        patch_edge[lvl].push_back(chk::Point2f(-1.0f * scaler, 0));
        patch_edge[lvl].push_back(chk::Point2f(0, 0));
        patch_edge[lvl].push_back(chk::Point2f(1.0f * scaler, 0));
        patch_edge[lvl].push_back(chk::Point2f(2.0f * scaler, 0));
        patch_edge[lvl].push_back(chk::Point2f(3.0f * scaler, 0));
    }; // end for
};


void StereoFrame::framePyrConstruct(
	const cv::Mat& img_l_input, const cv::Mat& img_r_input, 
	const int& n_img, const Eigen::Matrix4d& pose, const bool& is_keyframe, Params* params) 
{	


	int max_lvl  = params->pyr.max_lvl;
	int sz_sobel = params->edge.size_sobel;
	float thres_grad_min = params->edge.th_g_min_active;
	float thres_grad_max = params->edge.th_g_max_active;
	float ratio_reduction_right = params->edge.ratio_reduction_right;
	float overlap = params->edge.overlap;
	int len_min = params->edge.len_min;
	int len_max = params->edge.len_max;
	// bin_pts bin_edges

	// bin_pts 초기화 , bin_edges 초기화
	params->clearBinPts();
	params->clearBinEdges(); // ㅠㅠㅠ 무조건 디버그로 하자...

	// 스테레오 이미지의 gradient& edge를 추출한다.
	double sobel_num = 1;
	if (sz_sobel == 3) sobel_num = 1;
	else if (sz_sobel == 5)	sobel_num = 4.5;

	// 이미지 피라미드 저장.
	img_l_input.copyTo(img_l); // 0.03 ms (lab)
	img_r_input.copyTo(img_r); // 0.03 ms (lab)
	img_l.convertTo(img_lf, CV_32FC1); // 0.1ms (lab)
	img_r.convertTo(img_rf, CV_32FC1); // 0.1ms (lab)

	improc::imagePyramid(img_lf, left->img); // 0.3 ms (lab)
	improc::imagePyramid(img_rf, right->img); // 0.3 ms (lab)

	cv::Mat img_l_temp, img_r_temp;
	for (int lvl = 0; lvl < params->pyr.max_lvl; lvl++) {
		img_l_temp = left->img[lvl];
		img_r_temp = right->img[lvl];

		// 최대 resolution 에 대해서만 point (po int& edge)를 저장한다.
		if (lvl == 0) {
			// Left  [2] 추출한 edge의 saliency를 체크한다.edge / points로 구분.
			left->calcGradientAndCanny(img_l, img_lf, thres_grad_min, thres_grad_max);
			left->findSalientEdges(overlap, len_min, len_max); // 4~10 ms

			int Np_raw = (int)left->cluttered.size(); // point 와 edge의 raw 갯수
			int Ne_raw = (int)left->pts_centers.size();

			// [3] point 채우기.point가 있는 경우
			if (is_keyframe && Np_raw > 0) // 일단 이부분은 keyframe인 경우에만 수행한다!!
			{
				// [3] Bucketing을 통해 bin 별로 score에 따라 2개만 남긴다.
				float score_max;
				scores.resize(0);
				left->calcShiTomasi(2, scores, score_max);
				//left->calcFastScore(scores, score_max);

				vector<int> idx_binned;
				params->bucketingBinPoints(left->cluttered, scores, score_max, idx_binned);
				int Np = idx_binned.size();
				// 만약, bucketing에서 점이 하나도 나오지 않으면, Np_raw <= 0 처럼 동작한다.
				if (Np > 0) {
					// df 와 ft_point를 채워넣는다. (Np 개 만큼)
					left->ft_point[0]->purgeVectors(); // zeroize
					left->ft_point[0]->Np = Np;
					for (int i = 0; i < Np; i++) {
						left->ft_point[0]->pts.emplace_back(left->cluttered[idx_binned[i]].x, left->cluttered[idx_binned[i]].y);
						left->ft_point[0]->scores.emplace_back(scores[idx_binned[i]]);
					}
					improc::interpImage(img_lf, left->ft_point[0]->pts, left->ft_point[0]->brightness, vector<int>());
					for (int i = 0; i < Np; i++) {
						left->ft_point[0]->pts3d.emplace_back(0,0,0);
						left->ft_point[0]->invd.emplace_back(0);
						left->ft_point[0]->std_invd.emplace_back(3);
						left->ft_point[0]->is_recon.emplace_back(0);
						left->ft_point[0]->df.emplace_back(DepthFilter());
					}
				}
				else {
					Np = 0; // 없네; 
					// df 와 ft_point는 비어있다.
					left->ft_point[0]->Np = Np;
					left->ft_point[0]->purgeVectors(); // zeroize
				}
			}
			else {
				int Np = 0; // 없네; 
				// df 와 ft_point는 비어있다.
				left->ft_point[0]->Np = Np;
				left->ft_point[0]->purgeVectors(); // zeroize
			}

			// [4] edge 채우기
			if (Ne_raw > 0) { // edge가 있는 경우
				left->ft_edge[0]->purgeVectors();
				// [5] Warping 할 center 점들
				vector<int> idx_selected;
				params->bucketingBinEdges(left->pts_centers, left->bins_centers, idx_selected);

				int Nec = idx_selected.size(); // # of edge center points
				int Ne  = left->salient_ids.size();
				left->ft_edge[0]->Nec = Nec;
				left->ft_edge[0]->Ne  = Ne;

				// edge reference를 채워넣는다.
                left->fillEdgeSalient(0, 1);

				// edge center를 채워넣는다.
                int idx = 0;
				for (int i = 0; i < Nec; i++) {
					idx = idx_selected[i];
					left->ft_edge[0]->pts.emplace_back(left->pts_centers[idx].x, left->pts_centers[idx].y);
					left->ft_edge[0]->dirs.emplace_back(left->bins_centers[idx]);
					left->ft_edge[0]->pts3d.emplace_back(0,0,0);
					left->ft_edge[0]->invd.emplace_back(0);
					left->ft_edge[0]->std_invd.emplace_back(5);
					left->ft_edge[0]->is_recon.emplace_back(false);
					left->ft_edge[0]->df.emplace_back(DepthFilter());
				}
				vector<float> du_center(Nec);
				vector<float> dv_center(Nec);
				improc::interpImage(left->du[0], left->ft_edge[0]->pts, du_center, vector<int>());
				improc::interpImage(left->dv[0], left->ft_edge[0]->pts, dv_center, vector<int>());
				for (int i = 0; i < Nec; i++) {
                    float dmag_temp = sqrt(du_center[i] * du_center[i] + dv_center[i] * dv_center[i]);
					left->ft_edge[0]->grad.emplace_back(du_center[i], dv_center[i], dmag_temp);
				}
				improc::interpImage(left->img[0], left->ft_edge[0]->pts, left->ft_edge[0]->brightness, vector<int>());
			}
			else { // edge center가 없는 경우.
				int Nec = 0;
				int Ne = left->salient_ids.size();
				left->ft_edge[0]->purgeVectors();
				left->ft_edge[0]->Nec = Nec;
				left->ft_edge[0]->Ne = Ne;
				// edge reference를 채워넣는다.
                left->fillEdgeSalient(0, 1);
			} // edge 채우기 완료 // Left 완료
			
			// [1] right 는 edge(ind)만 채운다.
			// [2] 추출한 edge의 saliency를 체크한다.edge / points로 구분.
			right->calcGradientAndCanny(img_r, img_rf, thres_grad_min*ratio_reduction_right, thres_grad_max*ratio_reduction_right);
			right->findSalientEdges(overlap, len_min, len_max);

			// point는 빈 상태로 채운다.
			right->ft_point[0]->Np = 0;
			right->ft_point[0]->purgeVectors(); // zeroize

			// Edge도 pts는 빈 상태로 채운다.
			// [4] edge 채우기
			int Nec = 0; // 센터 점들의 갯수
			int Ne = (int)right->salient_ids.size();
			right->ft_edge[0]->Nec = Nec;
			right->ft_edge[0]->Ne = Ne;
			right->ft_edge[0]->purgeVectorsCenters();

			// edge reference를 채워넣는다.
            right->fillEdgeSalient(0, 1);
			// right 완료. frame완료.
		}
		else { // lvl != 0
			// 비워두면 되는것. point 일체 / edge centers 일체
			// [1] Level 1에서 얻은 salient pixel들을 사영한다. 2배씩 건너뛰며 샘플링하자! (left right 모두)
			// img는 이미 다 넣어놨고, du dv 피라미드를? 넣어야 할듯?
			int step = (int)pow(2, lvl);
            left->fillEdgeSalient(lvl, step);// Left
            right->fillEdgeSalient(lvl, step);// Right
		} // end else
	}// end for(lvl)
};



#endif
