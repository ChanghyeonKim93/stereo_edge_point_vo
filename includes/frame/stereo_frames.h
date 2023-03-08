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

    // �̹��� ��ġ��
    vector<vector<chk::Point2f>> patch_point; // point patch
    vector<vector<chk::Point2f>> patch_edge;

public:
	StereoFrame( // constructor
        int n_img, Eigen::Matrix4d T_0k, int max_pyr_lvl_input, 
        StereoCameras*& stereo_cams, Params* params_);
	~StereoFrame();// destructor

    StereoCameras* getPtrCams() const { return this->cams; };
    Params* getPtrParams() const { return this->params; };

	// frame�� ���� ä���ְų� or �������� ������Ʈ
	// (frame�� ���θ������ʰ�, ������ �����ϴ� frame_k �Ǵ� frame_c�� ���ؼ� �����ϴ� ����.)
	void framePyrConstruct(const cv::Mat& img_l, const cv::Mat& img_r, const int& n_img, const Eigen::Matrix4d& pose, const bool& is_keyframe, Params* params);
    void initializePatchPoint(bool flag_sparse);
    void initializePatchEdge();

    // void propagateDepthEdge(frame_k, frame_c);
    // void propagateDepthPoint(frame_k, frame_c);

    

private: // ���� �˰��� ������ ���� �͵��� private�� �����Ѵ�.
    Params* params;
    StereoCameras* cams; // �ܺο��� ���ǵ� ���� �޾ƿ´�.

    int max_pyr_lvl; // maximum pyramid levels
    vector<float> scores;

    
};


/* ================================================================================
================================= Implementation ==================================
=================================================================================== */
// ������.
StereoFrame::StereoFrame(int n_img, Eigen::Matrix4d T_0k, int max_pyr_lvl_input, 
	StereoCameras*& stereo_cams, Params* params_)
	: id(n_img), pose(T_0k), max_pyr_lvl(max_pyr_lvl_input)
{
	// ���⼭�� �̰����� �ϴ°� �ƴ϶�, �׳� frame ����ü���� ������ش�.
	// stereoFrameConstruct �Լ����� �̹��� + ��Ÿ��� ����!
	id = n_img;
	pose = T_0k;
	cams = stereo_cams;
	params = params_;

	left  = new Frame(stereo_cams->left,  max_pyr_lvl, params_);
	right = new Frame(stereo_cams->right, max_pyr_lvl, params_);

	// imagesRead �Լ����� ����ϱ� ���� �� �Ҵ�.
	// cv::Mat�� �̸� �Ҵ�Ǿ��ְ�, �ҷ��;� �� �̹����� ũ��� channel�� ������
	// �޸� �Ҵ���� �ٷ� ���簡 ���۵ȴ�.
	img_l = cv::Mat::zeros(left->cam->n_rows, left->cam->n_cols, CV_8UC1);
	img_r = cv::Mat::zeros(left->cam->n_rows, left->cam->n_cols, CV_8UC1);
	img_lf = cv::Mat::zeros(left->cam->n_rows, left->cam->n_cols, CV_32FC1);
	img_rf = cv::Mat::zeros(left->cam->n_rows, left->cam->n_cols, CV_32FC1);
	
	// pyramid Intrinsics
	K_pyr.reserve(max_pyr_lvl);
	Kinv_pyr.reserve(max_pyr_lvl);

	// pyramid ������ ���� �ٸ� ������ ��������.
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

    // �˰����� ���� ���Ҵ�.
	scores.reserve(50000);

    // ��ġ �Ҵ�.
    bool flag_sparse = true;
    initializePatchPoint(flag_sparse); // make point patch (maximum 4 x 4)
    initializePatchEdge(); // make edge patch (one-pixel strip).
}; // end constructor

// �Ҹ���
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

	// bin_pts �ʱ�ȭ , bin_edges �ʱ�ȭ
	params->clearBinPts();
	params->clearBinEdges(); // �ФФ� ������ ����׷� ����...

	// ���׷��� �̹����� gradient& edge�� �����Ѵ�.
	double sobel_num = 1;
	if (sz_sobel == 3) sobel_num = 1;
	else if (sz_sobel == 5)	sobel_num = 4.5;

	// �̹��� �Ƕ�̵� ����.
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

		// �ִ� resolution �� ���ؼ��� point (po int& edge)�� �����Ѵ�.
		if (lvl == 0) {
			// Left  [2] ������ edge�� saliency�� üũ�Ѵ�.edge / points�� ����.
			left->calcGradientAndCanny(img_l, img_lf, thres_grad_min, thres_grad_max);
			left->findSalientEdges(overlap, len_min, len_max); // 4~10 ms

			int Np_raw = (int)left->cluttered.size(); // point �� edge�� raw ����
			int Ne_raw = (int)left->pts_centers.size();

			// [3] point ä���.point�� �ִ� ���
			if (is_keyframe && Np_raw > 0) // �ϴ� �̺κ��� keyframe�� ��쿡�� �����Ѵ�!!
			{
				// [3] Bucketing�� ���� bin ���� score�� ���� 2���� �����.
				float score_max;
				scores.resize(0);
				left->calcShiTomasi(2, scores, score_max);
				//left->calcFastScore(scores, score_max);

				vector<int> idx_binned;
				params->bucketingBinPoints(left->cluttered, scores, score_max, idx_binned);
				int Np = idx_binned.size();
				// ����, bucketing���� ���� �ϳ��� ������ ������, Np_raw <= 0 ó�� �����Ѵ�.
				if (Np > 0) {
					// df �� ft_point�� ä���ִ´�. (Np �� ��ŭ)
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
					Np = 0; // ����; 
					// df �� ft_point�� ����ִ�.
					left->ft_point[0]->Np = Np;
					left->ft_point[0]->purgeVectors(); // zeroize
				}
			}
			else {
				int Np = 0; // ����; 
				// df �� ft_point�� ����ִ�.
				left->ft_point[0]->Np = Np;
				left->ft_point[0]->purgeVectors(); // zeroize
			}

			// [4] edge ä���
			if (Ne_raw > 0) { // edge�� �ִ� ���
				left->ft_edge[0]->purgeVectors();
				// [5] Warping �� center ����
				vector<int> idx_selected;
				params->bucketingBinEdges(left->pts_centers, left->bins_centers, idx_selected);

				int Nec = idx_selected.size(); // # of edge center points
				int Ne  = left->salient_ids.size();
				left->ft_edge[0]->Nec = Nec;
				left->ft_edge[0]->Ne  = Ne;

				// edge reference�� ä���ִ´�.
                left->fillEdgeSalient(0, 1);

				// edge center�� ä���ִ´�.
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
			else { // edge center�� ���� ���.
				int Nec = 0;
				int Ne = left->salient_ids.size();
				left->ft_edge[0]->purgeVectors();
				left->ft_edge[0]->Nec = Nec;
				left->ft_edge[0]->Ne = Ne;
				// edge reference�� ä���ִ´�.
                left->fillEdgeSalient(0, 1);
			} // edge ä��� �Ϸ� // Left �Ϸ�
			
			// [1] right �� edge(ind)�� ä���.
			// [2] ������ edge�� saliency�� üũ�Ѵ�.edge / points�� ����.
			right->calcGradientAndCanny(img_r, img_rf, thres_grad_min*ratio_reduction_right, thres_grad_max*ratio_reduction_right);
			right->findSalientEdges(overlap, len_min, len_max);

			// point�� �� ���·� ä���.
			right->ft_point[0]->Np = 0;
			right->ft_point[0]->purgeVectors(); // zeroize

			// Edge�� pts�� �� ���·� ä���.
			// [4] edge ä���
			int Nec = 0; // ���� ������ ����
			int Ne = (int)right->salient_ids.size();
			right->ft_edge[0]->Nec = Nec;
			right->ft_edge[0]->Ne = Ne;
			right->ft_edge[0]->purgeVectorsCenters();

			// edge reference�� ä���ִ´�.
            right->fillEdgeSalient(0, 1);
			// right �Ϸ�. frame�Ϸ�.
		}
		else { // lvl != 0
			// ����θ� �Ǵ°�. point ��ü / edge centers ��ü
			// [1] Level 1���� ���� salient pixel���� �翵�Ѵ�. 2�辿 �ǳʶٸ� ���ø�����! (left right ���)
			// img�� �̹� �� �־����, du dv �Ƕ�̵带? �־�� �ҵ�?
			int step = (int)pow(2, lvl);
            left->fillEdgeSalient(lvl, step);// Left
            right->fillEdgeSalient(lvl, step);// Right
		} // end else
	}// end for(lvl)
};



#endif
