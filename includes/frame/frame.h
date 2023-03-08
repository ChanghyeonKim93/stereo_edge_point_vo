#ifndef _FRAME_H_
#define _FRAME_H_

#include <iostream>
#include <vector>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"  
#include "Eigen/Dense"

#include "../camera/camera.h"
#include "../depth_reconstruction/depth_filter.h"

#include <stack>
#include "../utils/faststack.h"

#include "../quadtrees/CommonStruct.h"

#include <fstream>

#define D2R 0.01745333333

using namespace std;

string getImgType(const cv::Mat& img);

class Frame {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW



	struct FeaturePoint {
		int Np; // # of points.
		vector<chk::Point2f> pts;
		vector<chk::Point3f> pts3d;
		vector<float> invd;
		vector<float> std_invd;
		vector<bool> is_recon;
		vector<float> brightness;
        vector<float> scores;
        vector<DepthFilter> df;
		// constructor
		FeaturePoint() {
			int Np = 0;
			int n_hor = 60; // 넉넉하게!
			int n_ver = 60;
			int n_reserve = n_hor*n_ver * 2;
			pts.reserve(n_reserve); // sufficiently allocate.
			scores.reserve(n_reserve);
			pts3d.reserve(n_reserve);
			invd.reserve(n_reserve);
			std_invd.reserve(n_reserve);
			is_recon.reserve(n_reserve);
			brightness.reserve(n_reserve);
			df.reserve(n_reserve);
		}
		void purgeVectors() {
			pts.resize(0);
			scores.resize(0);
			pts3d.resize(0);
			invd.resize(0);
			std_invd.resize(0);
			is_recon.resize(0);
			brightness.resize(0);
			df.resize(0);
		}
	};
	struct FeatureEdge {
		int Ne; // # of reference edge points
		int Nec; // # of center points of edgelets.

		vector<chk::Point2f> pts_edge;
		vector<chk::Point2c> bins_edge;
		vector<chk::Point3f> grad_edge;
		vector<int> idx_all; // 현재 레벨에서 pts_edge_all을 가리키는 점 좌표. 걍 따로 저장할까 ㅜ 짱나네

		vector<chk::Point2f> pts;
        vector<chk::Point3f> pts3d;
		vector<chk::Point3f> grad;
        vector<char> dirs;
		vector<float> invd;
		vector<float> std_invd;
		vector<bool> is_recon;
		vector<float> brightness;
		vector<DepthFilter> df;
		// constructor
		FeatureEdge() {
			int Np = 0;
			int n_hor = 60; // 넉넉하게!
			int n_ver = 60;
			int n_reserve = n_hor*n_ver * 2;

			pts_edge.reserve(50000);
			bins_edge.reserve(50000);
			grad_edge.reserve(50000);
			idx_all.reserve(60000);

			pts.reserve(n_reserve); // sufficiently allocate.
			dirs.reserve(n_reserve);
			grad.reserve(n_reserve);
			pts3d.reserve(n_reserve);
			invd.reserve(n_reserve);
			std_invd.reserve(n_reserve);
			is_recon.reserve(n_reserve);
			brightness.reserve(n_reserve);
			df.reserve(n_reserve);
		};
		void purgeVectorsCenters() {
			pts.resize(0);
			dirs.resize(0);
			grad.resize(0);
			pts3d.resize(0);
			invd.resize(0);
			std_invd.resize(0);
			is_recon.resize(0);
			brightness.resize(0);
			df.resize(0);
		};
		void purgeVectors() {
			pts_edge.resize(0);
			bins_edge.resize(0);
			grad_edge.resize(0);
			idx_all.resize(0);
			purgeVectorsCenters();
		};
	};

	int id;
	Eigen::Matrix4d	pose;

	cv::Mat du_s; // CV_16S, Sobel result, max. resolution
	cv::Mat dv_s; // CV_16S, Sobel result, max. resolution

	// point 와 edge를 담는 구조체 벡터 (level 별)
	vector<FeaturePoint*> ft_point;
	vector<FeatureEdge*> ft_edge;

	vector<cv::Mat> img;  // CV_32FC1, undistorted image
	vector<cv::Mat> edge; // CV_8UC1, Canny edge result, max.lvl only.
	vector<cv::Mat> du;   // CV_32FC1, (I(x+1)-I(x-1))*0.5
	vector<cv::Mat> dv;   // CV_32FC1, (I(y+1)-I(y-1))*0.5
	vector<cv::Mat> dmag; // CV_32FC1

    vector<int> n_cols_pyr;
    vector<int> n_rows_pyr;

	// resulting variables from 'findSalientEdges'
	vector<vector<chk::Point2f>> edgelets;
	vector<vector<chk::Point2f>> lines;
	vector<chk::Point2f> cluttered;

	// (Nec) center points of the salient edgelets (edgelets + lines) 
	vector<chk::Point2f> pts_centers;
	vector<char> bins_centers;
	vector<float> evalratios;

	vector<int> salient_ids; // pts_edge_all 중에서 salient edge로 구분된 것들의 인덱스.

	// 전체 edge pixel들.
	vector<chk::Point2f> pts_edge_all;
	vector<chk::Point2c> bins_edge_all;
	vector<chk::Point3f> grad_edge_all;

	char* dir1_img;
	char* dir2_img;
	int* id_img; // pts_edge의 번호를 담고있다.

	int max_lvl; // maximum pyramid levels
	Params* params;
	Camera* cam; // 외부에서 정의된 것을 받아온다.
	// 최고 resolution 에 대한 정보만 저장하고있자.

	Frame();
	Frame(Camera*& cam_input, int max_lvl_input, Params* params_);
	~Frame();
    void calcGradientAndCanny(
        const cv::Mat& img_, const cv::Mat& img_f_, const float& thres_grad_min, const float& thres_grad_max);
	void findSalientEdges(
		const float& overlap, const int& min_length, const int& max_length);
    void fillEdgeSalient(const int& lvl, const int& step_sz);
	void calcShiTomasi(int win_sz, vector<float>& scores, float& score_max);
	void calcFastScore(vector<float>& scores, float& score_max);

private: // functions for findSalientEdges
	void floodFillStack(
		chk::Point2f& pt_q, short& dir_q, const int& max_length, 
		int& n_rows, int& n_cols, char* dir1_img, char* dir2_img, int* id_img,
		vector<int>& edgelet);
	void floodFillStackImproved(int& id, short& dir_q, const int& max_length,
		int& n_rows, int& n_cols, char* dir1_img, char* dir2_img, int* id_img,
		vector<int>& edgelet);
	void floodFillStackImproved2(int& id, short& dir_q, const int& max_length,
		int& n_rows, int& n_cols, char* dir1_img, char* dir2_img, int* id_img,
		vector<int>& edgelet);
	inline void calcMeanAndPrincipleAxis(
		vector<chk::Point2f>& pts, chk::Point2f& pt_mean, 
		float& evalsqr_ratio, chk::Point2f& evec);

	// pre-allocated variables for operating algorithms.
	chk::Point2f fast_patch[16];
	stack<chk::Point2f> st; // depricated.
	stack<int> st_int;      // depricated.
	FastStack<int> fst_int;
	
	// static member variable for direction test.
	static char* lookup_dir12; 
	void makeLookupTables(const float& overlap, const int& min_length, const int& max_length);
	void lookupDirections(const int& du, const int& dv, char& bin1, char& bin2);
};
//static member
char* Frame::lookup_dir12 = nullptr;

/* ================================================================================
================================= Implementation ==================================
=================================================================================== */
// Constructor
Frame::Frame() {};
Frame::Frame(Camera*& cam_input, int max_lvl_input, Params* params_)
{
	max_lvl = max_lvl_input;
	cam = cam_input;
	params = params_;

	// 선할당
	img.reserve(max_lvl);
	edge.reserve(max_lvl);
	du.reserve(max_lvl);
	dv.reserve(max_lvl);
	dmag.reserve(max_lvl);
	ft_point.reserve(max_lvl);
	ft_edge.reserve(max_lvl);

	for (int lvl = 0; lvl < max_lvl; lvl++) {
		if (lvl == 0) {
            n_cols_pyr.push_back(cam->n_cols);
            n_rows_pyr.push_back(cam->n_rows);

			img.emplace_back(cam->n_rows, cam->n_cols, CV_32FC1);
			edge.emplace_back(cam->n_rows, cam->n_cols, CV_8UC1);
			du.emplace_back(cam->n_rows, cam->n_cols, CV_32FC1);
			dv.emplace_back(cam->n_rows, cam->n_cols, CV_32FC1);
			dmag.emplace_back(cam->n_rows, cam->n_cols, CV_32FC1);
			FeaturePoint* ft_p = new FeaturePoint();
			FeatureEdge* ft_e = new FeatureEdge();

			ft_point.push_back(ft_p);
			ft_edge.push_back(ft_e);
		}
		else {
			int n_cols_temp = (int)(cam->n_cols / pow(2, lvl));
			int n_rows_temp = (int)(cam->n_rows / pow(2, lvl));
            n_cols_pyr.push_back(n_cols_temp);
            n_rows_pyr.push_back(n_rows_temp);

			img.emplace_back(n_rows_temp, n_cols_temp, CV_32FC1);
			edge.emplace_back(n_rows_temp,n_cols_temp, CV_8UC1);
			du.emplace_back(n_rows_temp, n_cols_temp, CV_32FC1);
			dv.emplace_back(n_rows_temp, n_cols_temp, CV_32FC1);
			dmag.emplace_back(n_rows_temp, n_cols_temp, CV_32FC1);
			FeaturePoint* ft_p = new FeaturePoint();
			FeatureEdge* ft_e  = new FeatureEdge();

			ft_point.push_back(ft_p);
			ft_edge.push_back(ft_e);
		}
	}
	du_s = cv::Mat::zeros(cam->n_rows, cam->n_cols, CV_16SC1);
	dv_s = cv::Mat::zeros(cam->n_rows, cam->n_cols, CV_16SC1);

	dir1_img = new char[cam->n_rows*cam->n_cols];
	dir2_img = new char[cam->n_rows*cam->n_cols];
	id_img   = new int[cam->n_rows*cam->n_cols]; // pts_edge의 번호를 담고있다.
	// cout << " # of pyramids in Frame: " << img.size() << endl; // 잘 만들어진것같은데 ;; 

	fast_patch[0].x = -3; fast_patch[0].y = 0;
	fast_patch[1].x = -3; fast_patch[1].y = -1;
	fast_patch[2].x = -2; fast_patch[2].y = -2;
	fast_patch[3].x = -1; fast_patch[3].y = -3;
	fast_patch[4].x = 0; fast_patch[4].y = -3;
	fast_patch[5].x = 1; fast_patch[5].y = -3;
	fast_patch[6].x = 2; fast_patch[6].y = -2;
	fast_patch[7].x = 3; fast_patch[7].y = -1;
	fast_patch[8].x = 3; fast_patch[8].y = 0;
	fast_patch[9].x = 3; fast_patch[9].y = 1;
	fast_patch[10].x = 2; fast_patch[10].y = 2;
	fast_patch[11].x = 1; fast_patch[11].y = 3;
	fast_patch[12].x = 0; fast_patch[12].y = 3;
	fast_patch[13].x = -1; fast_patch[13].y = 3;
	fast_patch[14].x = -2; fast_patch[14].y = 2;
	fast_patch[15].x = -3; fast_patch[15].y = 1;

	// 출력해야하는것 edgelets / eval_ratios / pts_centers / bins_centers / salient_ids
	edgelets.reserve(4000); // 총 4000개의 edgelets을 담는 container를 만든다.
	lines.reserve(4000); // 총 4000개의 lines를 담는 container를 만든다.
	cluttered.reserve(60000); // cluttered points를 담는 곳.
	evalratios.reserve(4000);
	pts_centers.reserve(4000);
	bins_centers.reserve(4000);

	// 모든 salient edgelets 의 index를 저장.
	salient_ids.reserve(60000);

	// findSalient 내부에서 사용되는것들
	pts_edge_all.reserve(60000);
	grad_edge_all.reserve(60000);
	bins_edge_all.reserve(60000);

	// lookup tables 만들기
	if (lookup_dir12 == nullptr) {
		int lookup_len = 255 * 4;
		lookup_dir12 = new char[(2 * lookup_len + 1)*(2 * lookup_len + 1) * 2];
		makeLookupTables(params->edge.overlap, params->edge.len_min, params->edge.len_max);
	}
};
Frame::~Frame() {
	cam = nullptr; // 카메라는 밖에서지워준다.
	delete[] dir1_img;
	delete[] dir2_img;
	delete[] id_img;

	// delete dynamic-allocated static member variable
	if (lookup_dir12 != nullptr) {
		delete[] lookup_dir12;
		lookup_dir12 = nullptr;
	}
	cout << "frame is deleted." << endl;
};

void Frame::calcGradientAndCanny(const cv::Mat& img_, const cv::Mat& img_f_, const float& thres_grad_min, const float& thres_grad_max) 
{
	cv::Sobel(img_, this->du_s, CV_16SC1, 1, 0); // 0.1ms
	cv::Sobel(img_, this->dv_s, CV_16SC1, 0, 1); // 0.1ms
    improc::diffImage(img_f_, du[0], 1, 0); // 0.2 ms
    improc::diffImage(img_f_, dv[0], 0, 1); // 0.2 ms
    improc::imagePyramid(du[0], du); // 0.3 ms (lab)
    improc::imagePyramid(dv[0], dv); // 0.3 ms (lab)

	cv::magnitude(this->du[0], this->dv[0], this->dmag[0]); // 0.2ms
	cv::Canny(this->du_s, this->dv_s, this->edge[0], thres_grad_min, thres_grad_max, false); // 1ms
    // total 2.4 ~ 3.0 ms ( per image )
};

void Frame::fillEdgeSalient(const int& lvl, const int& step_sz) {
    if (lvl > 0) { // low resolution
        // 초기화 
        ft_edge[lvl]->idx_all.resize(0);
        ft_edge[lvl]->pts_edge.resize(0);
        ft_edge[lvl]->bins_edge.resize(0);
        ft_edge[lvl]->grad_edge.resize(0);

        int idx = 0;
        int Ne_raw = (int)ft_edge[0]->idx_all.size();
        for (int i = 0; i < Ne_raw; i += step_sz) {
            idx = ft_edge[0]->idx_all[i];
            ft_edge[lvl]->idx_all.emplace_back(idx);// Matlab과는 달리, 필요할때 사영해서 쓰자
        }
        // 현재 level의 pts_edge(2f), bins_edge(2c), grad_edge(3f)를 채운다.
        // 특히, lvl 따라서 좌표를 절반치기해야한다.
        int Ne_raw_lowres = (int)ft_edge[lvl]->idx_all.size();
        float scaler = 1.0f / powf(2.0f, lvl);
        for (int i = 0; i < Ne_raw_lowres; i++) {
            idx = ft_edge[lvl]->idx_all[i];
            ft_edge[lvl]->pts_edge.emplace_back(pts_edge_all[idx]);
            ft_edge[lvl]->pts_edge.back().x *= scaler;
            ft_edge[lvl]->pts_edge.back().y *= scaler;
            ft_edge[lvl]->bins_edge.emplace_back(bins_edge_all[idx]);
            ft_edge[lvl]->grad_edge.emplace_back(grad_edge_all[idx]);
        }
    }
    else { // lvl == 0
        // 초기화
        ft_edge[0]->idx_all.resize(0);
        ft_edge[0]->pts_edge.resize(0);
        ft_edge[0]->bins_edge.resize(0);
        ft_edge[0]->grad_edge.resize(0);

        int idx = 0;
        size_t Ne = salient_ids.size();
        for (int i = 0; i < Ne; i++) {
            idx = salient_ids[i];
            ft_edge[0]->idx_all.emplace_back(idx);
            ft_edge[0]->pts_edge.emplace_back(pts_edge_all[idx]);
            ft_edge[0]->bins_edge.emplace_back(bins_edge_all[idx]);
            ft_edge[0]->grad_edge.emplace_back(grad_edge_all[idx]);
        } // right 완료. frame완료.
    }
};

void Frame::makeLookupTables(const float& overlap, const int& min_length, const int& max_length) {
	cout << "   Making lookup tables...\n";
	tic();
	// overlapping angles
	if (overlap < 0) throw std::runtime_error("lookup: overlap should be larger than 0(float).\n");
	else if (overlap > 22.5) throw std::runtime_error("lookup: overlap should be smaller than 22.5 degrees.\n");

	// 최소, 최대 edgelets 길이.
	if (min_length < 0)	throw std::runtime_error("lookup: min_length needs to be larger than 0(integer).\n");
	if (max_length < 0)	throw std::runtime_error("lookup: min_length needs to be larger than 0(integer).\n");
	if (max_length < min_length) throw std::runtime_error("lookup: max_length needs to be larger than min_length(integer).\n");


	// 1. Classify direction and n_pts
	int t225 = round(tan(22.5*D2R) * 1024);
	int t225p = round(tan((22.5 + overlap)*D2R) * 1024);
	int t225m = round(tan((22.5 - overlap)*D2R) * 1024);
	int t675 = round(tan(67.5*D2R) * 1024);
	int t675p = round(tan((67.5 + overlap)*D2R) * 1024);
	int t675m = round(tan((67.5 - overlap)*D2R) * 1024);

	int du = -1, dv = -1;
	int slp = -1; // slope
	char bin1 = -1, bin2 = -1;

	// pointer of lookup table
	char* pLookup = lookup_dir12;

	for (int dv = -1020; dv < 1021; dv++) {
		for (int du = -1020; du < 1021; du++) {
			bin1 = -1;
			bin2 = -1;
			if (du > 0) {
				slp = (dv << 10) / du;
				if (dv > 0) {
					if (slp < t225) {
						bin1 = 0; if (slp > t225m) bin2 = 1;
					}
					else if (slp < t675) {
						bin1 = 1;
						if (slp < t225p) bin2 = 0;
						else if (slp > t675m) bin2 = 2;
					}
					else {
						bin1 = 2; if (slp < t675p) bin2 = 1;
					}
				} // end if(dv > 0)
				else { // dv <=0
					if (slp > -t225) {
						bin1 = 0; if (slp < -t225m) bin2 = 7;
					}
					else if (slp > -t675) {
						bin1 = 7;
						if (slp > -t225p) bin2 = 0;
						else if (slp < -t675m) bin2 = 6;
					}
					else {
						bin1 = 6; if (slp > -t675p) bin2 = 7;
					}
				} // end else 
			} // end if(du > 0)
			else if (du < 0) {
				slp = (dv << 10) / du;
				if (dv > 0) {
					if (slp < -t675) {
						bin1 = 2; if (slp > -t675p) bin2 = 3;
					}
					else if (slp < -t225) {
						bin1 = 3;
						if (slp < -t675m) bin2 = 2;
						else if (slp > -t225p) bin2 = 4;
					}
					else {
						bin1 = 4; if (slp < -t225m) bin2 = 3;
					}
				}
				else { // dv <=0
					if (slp <t225) {
						bin1 = 4; if (slp > t225m) bin2 = 5;
					}
					else if (slp <t675) {
						bin1 = 5;
						if (slp < t225p) bin2 = 4;
						else if (slp > t675m) bin2 = 6;
					}
					else {
						bin1 = 6; if (slp < t675p) bin2 = 5;
					}
				}
			}
			else { // du == 0
				if (dv > 0) bin1 = 2;
				else if (dv < 0) bin1 = 6;
				else bin1 = -1; // 기울기가 아예 0인경우다 ; 
			}
			*pLookup = bin1;
			*(++pLookup) = bin2;
			++pLookup;
		} // end for u
	} // end for v
	cout << "   lapsed time: " << toc(0) << endl;

	// file output
	if (0) {
		ofstream out("dir1.txt");
		ofstream out2("dir2.txt");
		pLookup = lookup_dir12;
		for (int v = 0; v < 2041; v++) {
			for (int u = 0; u < 2041; u++) {
				out << (int)*pLookup;
				out2 << (int)(*(++pLookup));
				++pLookup;
				if (u < 2040) out << ", ";
				if (u < 2040) out2 << ", ";
			}
			out << endl;
			out2 << endl;
		}
		out.close();
		out2.close();
	}
};

void Frame::lookupDirections(const int& du, const int& dv, char& bin1, char& bin2) 
{
	// int ind = 2041 * (dv + 1020) + (du + 1020);
	int ind = 2082840 + 2041 * dv + du;
	char* pLookup = lookup_dir12 + ind;
	bin1 = *(pLookup);
	bin2 = *(pLookup+1);
};

void Frame::floodFillStack(chk::Point2f& pt_q, short& dir_q, const int& max_length, 
	int& n_rows, int& n_cols, char* dir1_img, char* dir2_img, int* id_img,
	vector<int>& edgelet) 
{
	edgelet.resize(0);
	// stack<chk::Point2f> st;
	st.emplace(pt_q);

	int u, v, ind, cnt;
	cnt = 0;
	while (!st.empty()) {
		u = st.top().x;
		v = st.top().y;
		st.pop();

		ind = u + v*n_cols;
		// 해당 원소를 넣는다.
		edgelet.emplace_back(id_img[ind]);
		++cnt;
		if (cnt < max_length && u > 0 && u < n_cols && v > 0 && v < n_rows) {
			int vn_cols = v*n_cols;
			ind = vn_cols - n_cols + u;
			// 1. (u, v - 1)
			if (dir1_img[ind] == dir_q || dir2_img[ind] == dir_q) {
				dir1_img[ind] = -1;
				dir2_img[ind] = -1;
				st.emplace(chk::Point2f(u, v - 1));
			}
			ind = vn_cols + n_cols + u;
			// 2. (u, v + 1)
			if (dir1_img[ind] == dir_q || dir2_img[ind] == dir_q) {
				dir1_img[ind] = -1;
				dir2_img[ind] = -1;
				st.emplace(chk::Point2f(u, v + 1));
			}
			// 3. (u - 1, v) 
			ind = vn_cols + u - 1;
			if (dir1_img[ind] == dir_q || dir2_img[ind] == dir_q) {
				dir1_img[ind] = -1;
				dir2_img[ind] = -1;
				st.emplace(chk::Point2f(u - 1, v));
			}
			// 4. (u + 1, v)
			ind = vn_cols + u + 1;
			if (dir1_img[ind] == dir_q || dir2_img[ind] == dir_q) {
				dir1_img[ind] = -1;
				dir2_img[ind] = -1;
				st.emplace(chk::Point2f(u + 1, v));
			}

			// 5. (u - 1, v - 1)
			ind = vn_cols - n_cols + u - 1;
			if (dir1_img[ind] == dir_q || dir2_img[ind] == dir_q) {
				dir1_img[ind] = -1;
				dir2_img[ind] = -1;
				st.emplace(chk::Point2f(u - 1, v - 1));
			}
			// 6. (u - 1, v + 1)
			ind = vn_cols + n_cols + u - 1;
			if (dir1_img[ind] == dir_q || dir2_img[ind] == dir_q) {
				dir1_img[ind] = -1;
				dir2_img[ind] = -1;
				st.emplace(chk::Point2f(u - 1, v + 1));
			}
			// 7. (u + 1, v - 1)
			ind = vn_cols - n_cols + u + 1;
			if (dir1_img[ind] == dir_q || dir2_img[ind] == dir_q) {
				dir1_img[ind] = -1;
				dir2_img[ind] = -1;
				st.emplace(chk::Point2f(u + 1, v - 1));
			}
			// 8. (u + 1, v + 1)
			ind = vn_cols + n_cols + u + 1;
			if (dir1_img[ind] == dir_q || dir2_img[ind] == dir_q) {
				dir1_img[ind] = -1;
				dir2_img[ind] = -1;
				st.emplace(chk::Point2f(u + 1, v + 1));
			}
		}
	}
};

void Frame::floodFillStackImproved(int& id, short& dir_q, const int& max_length,
	int& n_rows, int& n_cols, char* dir1_img, char* dir2_img, int* id_img,
	vector<int>& edgelet)
{
	edgelet.resize(0);
	chk::Point2f& pt_q = pts_edge_all[id];
	// stack<chk::Point2f> st;
	st_int.emplace(id);

	int u, v, ind, cnt;
	int id_temp;
	cnt = 0;

	char* dir1_ptr;
	char* dir2_ptr;
	int* id_img_ptr;

	while (!st_int.empty()) {
		id_temp = st_int.top();
		u = pts_edge_all[id_temp].x;
		v = pts_edge_all[id_temp].y;
		st_int.pop();

		ind = u + v*n_cols;
		// 해당 원소를 넣는다.
		edgelet.emplace_back(id_img[ind]);
		if ((++cnt) < max_length && u > 0 && u < n_cols && v > 0 && v < n_rows) {
			ind = (v - 1)*n_cols + (u - 1);
			dir1_ptr = dir1_img + ind;
			dir2_ptr = dir2_img + ind;
			id_img_ptr = id_img + ind;
			// 1. (u - 1, v - 1)
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				st_int.emplace(*id_img_ptr);
			}
			// 2. (u, v - 1)
			dir1_ptr += 1;
			dir2_ptr += 1;
			id_img_ptr += 1;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				st_int.emplace(*id_img_ptr);
			}
			// 3. (u + 1, v - 1)
			dir1_ptr += 1;
			dir2_ptr += 1;
			id_img_ptr += 1;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				st_int.emplace(*id_img_ptr);
			}
			// 4. (u + 1, v)
			dir1_ptr += n_cols;
			dir2_ptr += n_cols;
			id_img_ptr += n_cols;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				st_int.emplace(*id_img_ptr);
			}
			// 5. (u - 1, v)
			dir1_ptr -= 2;
			dir2_ptr -= 2;
			id_img_ptr -= 2;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				st_int.emplace(*id_img_ptr);
			}
			// 6. (u - 1, v + 1)
			dir1_ptr += n_cols;
			dir2_ptr += n_cols;
			id_img_ptr += n_cols;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				st_int.emplace(*id_img_ptr);
			}
			// 7. (u, v + 1)
			dir1_ptr += 1;
			dir2_ptr += 1;
			id_img_ptr += 1;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				st_int.emplace(*id_img_ptr);
			}
			// 8. (u + 1, v + 1)
			dir1_ptr += 1;
			dir2_ptr += 1;
			id_img_ptr += 1;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				st_int.emplace(*id_img_ptr);
			}
		}
	}
};

void Frame::floodFillStackImproved2(int& id, short& dir_q, const int& max_length,
	int& n_rows, int& n_cols, char* dir1_img, char* dir2_img, int* id_img,
	vector<int>& edgelet)
{
	edgelet.resize(0);
	chk::Point2f& pt_q = pts_edge_all[id];
	// stack<chk::Point2f> st;
	fst_int.Push(id);

	int u, v, ind, cnt;
	int id_temp;
	cnt = 0;

	char* dir1_ptr;
	char* dir2_ptr;
	int* id_img_ptr;

	while (!fst_int.empty()) {
		id_temp = fst_int.top();
		u = pts_edge_all[id_temp].x;
		v = pts_edge_all[id_temp].y;
		fst_int.pop();

		ind = u + v*n_cols;
		// 해당 원소를 넣는다.
		edgelet.emplace_back(id_img[ind]);
		if ((++cnt) < max_length && u > 0 && u < n_cols && v > 0 && v < n_rows) {
			ind = (v - 1)*n_cols + (u - 1);
			dir1_ptr = dir1_img + ind;
			dir2_ptr = dir2_img + ind;
			id_img_ptr = id_img + ind;
			// 1. (u - 1, v - 1)
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				fst_int.Push(*id_img_ptr);
			}
			// 2. (u, v - 1)
			dir1_ptr += 1;
			dir2_ptr += 1;
			id_img_ptr += 1;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				fst_int.Push(*id_img_ptr);
			}
			// 3. (u + 1, v - 1)
			dir1_ptr += 1;
			dir2_ptr += 1;
			id_img_ptr += 1;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				fst_int.Push(*id_img_ptr);
			}
			// 4. (u + 1, v)
			dir1_ptr += n_cols;
			dir2_ptr += n_cols;
			id_img_ptr += n_cols;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				fst_int.Push(*id_img_ptr);
			}
			// 5. (u - 1, v)
			dir1_ptr -= 2;
			dir2_ptr -= 2;
			id_img_ptr -= 2;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				fst_int.Push(*id_img_ptr);
			}
			// 6. (u - 1, v + 1)
			dir1_ptr += n_cols;
			dir2_ptr += n_cols;
			id_img_ptr += n_cols;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				fst_int.Push(*id_img_ptr);
			}
			// 7. (u, v + 1)
			dir1_ptr += 1;
			dir2_ptr += 1;
			id_img_ptr += 1;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				fst_int.Push(*id_img_ptr);
			}
			// 8. (u + 1, v + 1)
			dir1_ptr += 1;
			dir2_ptr += 1;
			id_img_ptr += 1;
			if (*dir1_ptr == dir_q || *dir2_ptr == dir_q) {
				*dir1_ptr = -1;
				*dir2_ptr = -1;
				fst_int.Push(*id_img_ptr);
			}
		}
	}
	// set index at 0.
	fst_int.clear();
};

inline void Frame::calcMeanAndPrincipleAxis(vector<chk::Point2f>& pts, chk::Point2f& pt_mean, float& evalsqr_ratio, chk::Point2f& evec) {
	int m_u = 0;
	int m_v = 0;
	int m_uu = 0;
	int m_vv = 0;
	int m_uv = 0;

	float Ninv = 1.0 / (float)pts.size();
	float Ninv2 = Ninv*Ninv;
	for (int i = 0; i < pts.size(); i++) {
		m_u += pts[i].x;
		m_v += pts[i].y;
		m_uu += pts[i].x*pts[i].x;
		m_uv += pts[i].x*pts[i].y;
		m_vv += pts[i].y*pts[i].y;
	}

	pt_mean.x = (float)m_u*Ninv;
	pt_mean.y = (float)m_v*Ninv;

	float a, b, c;
	a = (float)(m_uu*Ninv - m_u*m_u*Ninv2);
	b = (float)(m_uv*Ninv - m_u*m_v*Ninv2);
	c = (float)(m_vv*Ninv - m_v*m_v*Ninv2);

	float disk, lam1, lam2; // lam1이 large eigenvalue
	disk = sqrt((a - c)*(a - c) + 4 * b*b);
	lam1 = 0.5*(a + c + disk);
	lam2 = 0.5*(a + c - disk);

	evalsqr_ratio = lam2 / lam1; // small / large. 클 수록 linearity가 큰 것. 1에 가까우면 원에 가까움.

	float invnorm_evec = 1.0 / sqrt(b*b + (a - lam1)*(a - lam1));
	evec.x = -b;
	evec.y = a - lam1;

	evec.x *= invnorm_evec;
	evec.y *= invnorm_evec;
};

void Frame::findSalientEdges(const float& overlap, const int& min_length, const int& max_length) {
	// 초기화
	edgelets.resize(0, vector<chk::Point2f>(0));
	lines.resize(0, vector<chk::Point2f>(0));
	cluttered.resize(0);
	pts_centers.resize(0);
	bins_centers.resize(0);
	evalratios.resize(0);
	salient_ids.resize(0);
	
	int n_rows = img[0].rows;
	int n_cols = img[0].cols;
	int n_elem = n_rows*n_cols; // 총 원소갯수. // printf("n_rows: %d, n_cols: %d, n_elem: %d\n", n_rows, n_cols, n_elem);
	
	// retrieve the query data pointers
	uchar* edge_img = edge[0].ptr<uchar>(0);
	short* du_img   = du_s.ptr<short>(0);
	short* dv_img   = dv_s.ptr<short>(0);

	// overlapping angles
	if (overlap < 0) throw std::runtime_error("overlap should be larger than 0(float).\n");
	else if (overlap > 22.5) throw std::runtime_error("overlap should be smaller than 22.5 degrees.\n");

	// 최소, 최대 edgelets 길이.
	if (min_length < 0)	throw std::runtime_error("min_length needs to be larger than 0(integer).\n");
	if (max_length < 0)	throw std::runtime_error("min_length needs to be larger than 0(integer).\n");
	if (max_length < min_length) throw std::runtime_error("max_length needs to be larger than min_length(integer).\n");


	// 1. Classify direction and n_pts
	int t225 = round(tan(22.5*D2R) * 1024);
	int t225p = round(tan((22.5 + overlap)*D2R) * 1024);
	int t225m = round(tan((22.5 - overlap)*D2R) * 1024);
	int t675 = round(tan(67.5*D2R) * 1024);
	int t675p = round(tan((67.5 + overlap)*D2R) * 1024);
	int t675m = round(tan((67.5 - overlap)*D2R) * 1024);

	int du = -1, dv = -1;
	int slp = -1; // slope
	char bin1 = -1, bin2 = -1;

	// 엣지 픽셀에 대한 
	pts_edge_all.resize(0);
	grad_edge_all.resize(0);
	bins_edge_all.resize(0);


	// 방향을 담는 container. 이미지 크기와 같다.
	for (int i = 0; i < n_elem; i++) {
		dir1_img[i] = -1;
		dir2_img[i] = -1;
		id_img[i]   = -1;
	}

	// 가장자리 픽셀들은 버리는 식으로~
	int ind = 0;
	int n_pts_edge = 0;
	for (int v = 1; v < n_rows - 1; v++) { // 이부분은 lookup tables로 바꿀 수 있다.
		ind = v*n_cols; // C++ index. //ind = u*n_rows; // MATLAB index. 
		for (int u = 1; u <n_cols - 1; u++) {
			++ind;
			if (edge_img[ind]) { // edge 픽셀인 경우에만.
				du = (int)du_img[ind]; // gradient u
				dv = (int)dv_img[ind]; // gradient v
				//lookupDirections(du, dv, bin1, bin2); // Cache올리는거때매 느리다.

				bin1 = -1;
				bin2 = -1;
				if (du > 0) {
					slp = (dv << 10) / du;
					if (dv > 0) {
						if (slp < t225) {
							bin1 = 0; if (slp > t225m) bin2 = 1;
						}
						else if (slp < t675) {
							bin1 = 1;
							if (slp < t225p) bin2 = 0;
							else if (slp > t675m) bin2 = 2;
						}
						else {
							bin1 = 2; if (slp < t675p) bin2 = 1;
						}
					} // end if(dv > 0)
					else { // dv <=0
						if (slp > -t225) {
							bin1 = 0; if (slp < -t225m) bin2 = 7;
						}
						else if (slp > -t675) {
							bin1 = 7;
							if (slp > -t225p) bin2 = 0;
							else if (slp < -t675m) bin2 = 6;
						}
						else {
							bin1 = 6; if (slp > -t675p) bin2 = 7;
						}
					} // end else 
				} // end if(du > 0)
				else if (du < 0) {
					slp = (dv << 10) / du;
					if (dv > 0) {
						if (slp < -t675) {
							bin1 = 2; if (slp > -t675p) bin2 = 3;
						}
						else if (slp < -t225) {
							bin1 = 3;
							if (slp < -t675m) bin2 = 2;
							else if (slp > -t225p) bin2 = 4;
						}
						else {
							bin1 = 4; if (slp < -t225m) bin2 = 3;
						}
					}
					else { // dv <=0
						if (slp <t225) {
							bin1 = 4; if (slp > t225m) bin2 = 5;
						}
						else if (slp <t675) {
							bin1 = 5;
							if (slp < t225p) bin2 = 4;
							else if (slp > t675m) bin2 = 6;
						}
						else {
							bin1 = 6; if (slp < t675p) bin2 = 5;
						}
					}
				}
				else { // du == 0
					if (dv > 0) bin1 = 2;
					else if (dv < 0) bin1 = 6;
					else bin1 = -1; // 기울기가 아예 0인경우다 ; 
				}

				// 데이터 정리.
				pts_edge_all.emplace_back(u, v);
				bins_edge_all.emplace_back(bin1, bin2);
                float dmag = sqrt((float)(du*du + dv*dv));
				grad_edge_all.emplace_back((float)du, (float)dv, dmag);
			
				// dir1_img, dir2_img에 넣어준다.
				dir1_img[ind] = bin1;
				dir2_img[ind] = bin2;
				id_img[ind] = n_pts_edge;

				++n_pts_edge; // edge 픽셀 갯수 증가.
			}
		}
	}

	// 2. Find salient edgelets
	//// 출력해야하는것 edgelets / eval_ratios / pts_centers / bins_centers / salient_ids
	//edgelets.reserve(4000); // 총 4000개의 edgelets을 담는 container를 만든다.
	//lines.reserve(4000); // 총 4000개의 lines를 담는 container를 만든다.
	//cluttered.reserve(MAX_CAPACITY); // cluttered points를 담는 곳.
	//evalratios.reserve(4000);
	//pts_centers.reserve(4000);
	//bins_centers.reserve(4000);

	//// 모든 salient edgelets 의 index를 저장.
	//salient_ids.reserve(MAX_CAPACITY);

	// 현재 edgelet을 저장하는 임시 저장.
	vector<chk::Point2f> edgelet;
	vector<int> edgelet_id;
	edgelet.reserve(max_length);
	edgelet_id.reserve(max_length);

	// salient edge들을 찾고, 그 edgelets을 cluttered / line / curve 로 구분한다.
	short dir_q;
	int ids[9];
	for (int i = 0; i < pts_edge_all.size(); i++) {
		ind = pts_edge_all[i].x + pts_edge_all[i].y*n_cols; // MATLAB index, C++ index ind = v*n_cols + u;
		if (dir1_img[ind] > -1) { // 엣지가 있는 경우,
			dir_q = dir1_img[ind];
			edgelet_id.resize(0); // id를 저장한다.// find edgelet by queue-based DFS.
			//floodFillStack(pts_edge_all[i], dir_q, max_length, n_rows, n_cols, dir1_img, dir2_img, id_img, edgelet_id);
			floodFillStackImproved2(i, dir_q, max_length, n_rows, n_cols, dir1_img, dir2_img, id_img, edgelet_id);
			// 쬐금 빨라진다? 

			// 최소 길이 조건을 만족 한 경우.
			if (edgelet_id.size() > min_length) {
				edgelet.resize(0); // id에 해당하는 점들을 저장하는 객체이다.
				for (int j = 0; j < edgelet_id.size(); j++) {
					edgelet.emplace_back(pts_edge_all[edgelet_id[j]]);
				}
				// 중심점과 주요 방향을 계산함.
				chk::Point2f pt_mean;
				chk::Point2f pt_mean_round(-1, -1);
				float evalsqr_ratio;
				chk::Point2f evec;

				// 주요 방향을 계산. (Eigenvalue decomposition for symmetric matrix)
				calcMeanAndPrincipleAxis(edgelet, pt_mean, evalsqr_ratio, evec);

				int u_i, v_i, v_in_cols, v_in_colsu_i;
				u_i = (int)pt_mean.x;
				v_i = (int)pt_mean.y;
				v_in_cols = v_i*n_cols;
				v_in_colsu_i = v_in_cols + u_i;
				
				// 검사 순서: 중심 상하좌우 좌상 우상 좌하 우하
				int* int_ptr = id_img + v_in_colsu_i;
				ids[0] = *int_ptr; // center
				ids[1] = *(--int_ptr); // left
				++int_ptr;
				ids[2] = *(++int_ptr); // right
				
				int_ptr = id_img + v_in_colsu_i - n_cols - 1;
				ids[3] = *int_ptr; // left top
				ids[4] = *(++int_ptr); // top
				ids[5] = *(++int_ptr); // right top

				int_ptr = id_img + v_in_colsu_i + n_cols - 1;
				ids[6] = *int_ptr; // left bot
				ids[7] = *(++int_ptr); // bot
				ids[8] = *(++int_ptr); // right bot

				// 3x3 내부에 edge response가 있는지 확인한다. 있다면 그 점으로 대체.
				for (int ii = 0; ii < 9; ii++) {
					if (ids[ii] > 0) {
						pt_mean.x = pts_edge_all[ids[ii]].x;
						pt_mean.y = pts_edge_all[ids[ii]].y;
						break;
					};
				};
				// 젠장!!!!!!!!!!!!!!! 아랫거가 문제였다 ;;;;;
				//if (ids[0] > -1) {
				//	pt_mean.x = pts_edge_all[ids[0]].x;
				//	pt_mean.y = pts_edge_all[ids[0]].y;
				//}
				//else {
				//	if (ids[4] > 0) { // top
				//		pt_mean_round = pts_edge_all[ids[4]];
				//	}
				//	else if (ids[7] > 0) { // bot
				//		pt_mean_round = pts_edge_all[ids[7]];
				//	}
				//	else if (ids[1] > 0) { // left
				//		pt_mean_round = pts_edge_all[ids[1]];
				//	}
				//	else if (ids[2] > 0) { // right
				//		pt_mean_round = pts_edge_all[ids[2]];
				//	}
				//	else {
				//		if (ids[3]> 0) { // tl
				//			pt_mean_round = pts_edge_all[ids[3]];
				//		}
				//		else if (ids[5] > 0) { // tr
				//			pt_mean_round = pts_edge_all[ids[5]];
				//		}
				//		else if (ids[6] > 0) { // bl
				//			pt_mean_round = pts_edge_all[ids[6]];
				//		}
				//		else if (ids[8] > 0) { // br
				//			pt_mean_round = pts_edge_all[ids[8]];
				//		}
				//		else {
				//			// 인접한 edge가 없으면, 그냥 평균값 넣는다...
				//		}
				//	}
				//	if (pt_mean_round.x > -1) {
				//		pt_mean.x = pt_mean_round.x;
				//		pt_mean.y = pt_mean_round.y;
				//	}
				//}

				pt_mean.y = round(pt_mean.y); // epipolar searching이 정수 도메인에서 가능하도록 한다.

				pts_centers.emplace_back(pt_mean);
				bins_centers.emplace_back(dir_q);
				evalratios.emplace_back(evalsqr_ratio);

				if (evalsqr_ratio < 0.003) { // line으로 간주.
					for (int j = 0; j < edgelet_id.size(); j++) {
						salient_ids.emplace_back(edgelet_id[j]);
					}
					lines.emplace_back(edgelet);
				}
				else { // curve로 간주.
					for (int j = 0; j < edgelet_id.size(); j++) {
						salient_ids.emplace_back(edgelet_id[j]);
					}
					edgelets.emplace_back(edgelet);
				}
			}
			else {
				for (int j = 0; j < edgelet_id.size(); j++) {
					cluttered.emplace_back(pts_edge_all[edgelet_id[j]]);
				}
			}
		}
	}

	if (0) {
		cout << "edgelets: " << edgelets.size() << endl;
		cout << "liens: " << lines.size() << endl;
		cout << "cluttered: " << cluttered.size() << endl;
		cout << "pts_centers: " << pts_centers.size() << endl;
		cout << "bins_centers: " << bins_centers.size() << endl;
		cout << "ind: " << salient_ids.size() << endl;
	}
	
	//
};

float calcMinEig(short* du_ptr, short* dv_ptr, int& u, int& v, int& win_sz, int& n_cols) {

	int a = 0, b = 0, c = 0;
	int vn_cols = 0;
	int vn_colsuu = 0;
	short du_temp, dv_temp;
	for (int vv = v - win_sz; vv < v + win_sz; vv++) {
		vn_cols = vv*n_cols;
		for (int uu = u - win_sz; uu < u + win_sz; uu++) {
			vn_colsuu = vn_cols + uu;
			du_temp = du_ptr[vn_colsuu];
			dv_temp = dv_ptr[vn_colsuu];
			a += (du_temp*du_temp);
			b += (du_temp*dv_temp);
			c += (dv_temp*dv_temp);
		}
	}

	return 0.5*((float)(a + c) - sqrt((float)((a - c)*(a - c) + 4 * b*b)));
};

void Frame::calcShiTomasi(int win_sz, vector<float>& scores, float& score_max) {
	// 이미지 얻어오기
	short* du_ptr = du[0].ptr<short>(0);
	short* dv_ptr = dv[0].ptr<short>(0);

	int n_rows = du[0].rows;
	int n_cols = du[0].cols;
	int n_elem = n_rows*n_cols; // 총 원소갯수. 

	// cluttered 점에 대해서 수행! 

	// score를 계산하여 출력한다. 특히, 최댓값도 계산한다.
	score_max = 0;
	size_t n_pts = cluttered.size();
	int u, v;
	for (size_t i = 0; i < n_pts; i++) {
		u = (int)cluttered[i].x;
		v = (int)cluttered[i].y;
		if ((u > win_sz) && (u < n_cols - win_sz) && (v > win_sz) && (v < n_rows - win_sz)) {
			float score_temp = calcMinEig(du_ptr, dv_ptr, u, v, win_sz, n_cols);
			scores.push_back(score_temp);
			if (score_temp > score_max) score_max = score_temp;
		}
		else {
			scores.push_back(0);
		}
	}
};

void Frame::calcFastScore(vector<float>& scores, float& score_max) 
{
	float* img_ptr = img[0].ptr<float>(0);

	int n_rows = du[0].rows;
	int n_cols = du[0].cols;
	int n_elem = n_rows*n_cols; // 총 원소갯수.

	// score를 계산하여 출력한다. 특히, 최댓값도 계산한다.
	score_max = 0;
	int win_sz = 4;
	size_t n_pts = cluttered.size();
	int u, v;
	

	for (size_t i = 0; i < n_pts; i++) {
		u = (int)cluttered[i].x;
		v = (int)cluttered[i].y;
		if ((u > win_sz) && (u < n_cols - win_sz) && (v > win_sz) && (v < n_rows - win_sz)) {
			float score_temp = 0;
			int idx = 0;
			float p = *(img_ptr + u + n_cols*v);
			float th = 10;
			for (int j = 0; j < 16; j++) {
				idx = (u + fast_patch[j].x) + n_cols*(v + fast_patch[j].y);
				score_temp += (abs(img_ptr[idx]-p) - th);
			}
			scores.push_back(score_temp);
			if (score_temp > score_max) score_max = score_temp;
		}
		else {
			scores.push_back(0);
		}
	}
};



string getImgType(const cv::Mat& img)
{
	int numImgTypes = 35; // 7 base types, with five channel options each (none or C1, ..., C4)

	int enum_ints[] = {
		CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
		CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
		CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
		CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
		CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
		CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
		CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4 };

	string enum_strings[] = {
		"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
		"CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
		"CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
		"CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
		"CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
		"CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
		"CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4" };

	int imgTypeInt = img.type();
	for (int i = 0; i < numImgTypes; i++)
	{
		if (imgTypeInt == enum_ints[i]) {
			cout << enum_strings[i] << endl;
			return enum_strings[i];
		}
	}
	cout << "unknown image type" << endl;
	return "unknown image type";
};
#endif