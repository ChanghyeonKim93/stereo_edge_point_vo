#ifndef _PARAMS_H_
#define _PARAMS_H_
#include <iostream>
#include <vector>
#include "includes\quadtrees\CommonStruct.h"

using namespace std;

class Params {
public:
	struct Recon {
		struct Edge {
			struct Fixed {
				int win_sz = 5; 
				int fat_sz = 2;
				float angle_thres = 70.0f;
				float thres_zncc1 = 0.93f;
				float thres_zncc2 = 0.83f;
				int thres_multipeak = 5;
				float eps_edge = 0.5f;
				float eps_epi = 0.1f;
				float eps_bright = 2.0f;
				float d_min = 0.2f; // [m]
				float d_max = 20.0f; // [m]
			};
			struct Temporal {
				int win_sz = 5;
				int fat_sz = 2;
				float angle_thres = 60.0f;
				float thres_zncc1 = 0.93f;
				float thres_zncc2 = 0.83f;
				int thres_multipeak = 5;
				float eps_edge = 0.5f;
				float eps_epi = 0.75f;
				float eps_bright = 2.0f;
				float d_min = 0.2f;
				float d_max = 20.0f;
			};
			Fixed fixed;
			Temporal temporal;
		};
		struct Point {
			struct Fixed {
				int win_sz = 3;
				int fat_sz = 3;
				float thres_zncc1 = 0.85f;
				float thres_zncc2 = 0.75f;
				int thres_multipeak = 6;
				float eps_edge = 1.0f;
				float eps_bright = 2.0f;
				float d_min = 0.2f; // [m]
				float d_max = 20.0f; // [m]
			};
            struct Temporal {
                int win_sz = 3;
                int fat_sz = 3;
                float thres_zncc1 = 0.85f;
                float thres_zncc2 = 0.75f;
                int thres_multipeak = 6;
                float eps_edge = 1.0f;
                float eps_epi = 0.75f;
                float eps_bright = 2.0f;
                float d_min = 0.2f; // [m]
                float d_max = 20.0f; // [m]
            };
			Fixed fixed;
            Temporal temporal;
		};
		Edge edge;
		Point point;
	};
	struct Illumination {
		bool on; // illumination change onoff
	};

	struct Quadtree {
		float thres_dist = 30.0f;
		int max_depth = 5;
		float eps = 0.2f;
		float thres_gradtest = 0.98f;
		float thres_std = 0.03f;
	};
	struct Edge {
		float thres_grad_min = 60.0f;
		float thres_grad_max = 100.0f;
		float th_g_min_active = thres_grad_min;
		float th_g_max_active = thres_grad_max;
		float ratio_reduction_right = 0.95f;
		float thres_scale = 1.0f;
		int size_sobel = 3;
		float overlap = 12.0f;
		int len_min = 15; // 15
		int len_max = 20; // 20
	};
	struct Iter {
        int MAX_ITER[5] = {15,40,40,20,20};
		float thres_rot = 6.0f; // [degrees]
		float thres_rot_z = 3.0f; // [degrees]
		float thres_trans = 0.2f; // [m]
		float thres_overlap = 0.7f;
		float thres_cost_rate = 1e-5f;
        float thres_delxi_norm = 1e-5f;
	};
	struct Bucket_p {
		struct BinPts {
			vector<chk::Point2i> idxs;
			vector<int> occupancy;
			vector<int> v_bound;
			vector<int> u_bound;
			int n_bins_u;
			int n_bins_v;
			int count;
			float quality_scale;
			float score_min;
			float min_dist;
		};
		int n_horizontal = 20;
		int n_vertical = 20;
		float min_dist = 5.0f;
		float quality_scale = 0.01f;
		float score_min = 1e4*2.0f;
		BinPts bin_pts;
	};
	struct Bucket_e {
		struct BinEdges {
			vector<chk::Point2i> idxs;
			vector<int> occupancy;
			vector<int> v_bound;
			vector<int> u_bound;
			int n_bins_u;
			int n_bins_v;
			int count;
			float min_dist;
		};
		int n_horizontal = 15;
		int n_vertical = 15;
		float min_dist = 10.0f;
		BinEdges bin_edges;
	};
	struct Klt {
		int win_sz = 25;
		int max_lvl = 3;
	};
	struct Pyramid {
		int max_lvl = 4;
	};
	struct Weight {
		float nu = 1.5f;
	};

	Recon recon;
	Illumination illumi;
	Quadtree qt;
	Edge edge;
	Iter iter;
	Bucket_p bkt_p;
	Bucket_e bkt_e;
	Klt klt;
	Pyramid pyr;
	Weight weight;

public:
	Params(int n_cols, int n_rows);
	~Params() {};
	void initBinPts(int n_cols, int n_rows, int n_bins_u, int n_bins_v, float quality_scale, float score_min, float min_dist);
	void initBinEdges(int n_cols, int n_rows, int n_bins_u, int n_bins_v, float min_dist);
	void clearBinPts();
	void clearBinEdges();
	void bucketingBinPoints(vector<chk::Point2f>& cluttered, vector<float>& scores, float& score_max,
		vector<int>& idx_selected);
	void bucketingBinEdges(vector<chk::Point2f>& pts_center, vector<char>& dirs_center,
		vector<int>& idx_selected);
};

Params::Params(int n_cols, int n_rows)
{
	// 초기화.
	initBinPts(n_cols, n_rows, bkt_p.n_horizontal, bkt_p.n_vertical, bkt_p.quality_scale, bkt_p.score_min, bkt_p.min_dist);
	initBinEdges(n_cols, n_rows, bkt_e.n_horizontal, bkt_e.n_vertical, bkt_e.min_dist);
};
void Params::initBinPts(int n_cols, int n_rows, int n_bins_u, int n_bins_v, float quality_scale, float score_min, float min_dist)
{
	// 각 축 방향으로 bin의 너비 or 높이.
	int step_u = ceil((float)n_cols / (float)n_bins_u);
	int step_v = ceil((float)n_rows / (float)n_bins_v);

	// 1 5  9 13
	// 2 6 10 14
	// 3 7 11 15
	// 4 8 12 16
	bkt_p.bin_pts.idxs.reserve(n_bins_u*n_bins_v);
	bkt_p.bin_pts.occupancy.reserve(n_bins_u*n_bins_v);
	for (int i = 0; i < n_bins_u*n_bins_v; i++) {
		bkt_p.bin_pts.idxs.push_back(chk::Point2i(-1, -1));
		bkt_p.bin_pts.occupancy.push_back(0);
	}
	bkt_p.bin_pts.u_bound.reserve(n_bins_u + 1);
	bkt_p.bin_pts.v_bound.reserve(n_bins_v + 1);

	// container를 채운다.
	bkt_p.bin_pts.n_bins_u = n_bins_u;
	bkt_p.bin_pts.n_bins_v = n_bins_v;
	bkt_p.bin_pts.count = 0;
	bkt_p.bin_pts.quality_scale = quality_scale;
	bkt_p.bin_pts.score_min = score_min;
	bkt_p.bin_pts.min_dist = min_dist;

	// 각 bin의 boundary 값들!
	for (int i = 0; i < n_bins_u; i++) {
		bkt_p.bin_pts.u_bound.push_back(step_u*i);
	}
	bkt_p.bin_pts.u_bound.push_back(n_cols);
	for (int i = 0; i < n_bins_v; i++) {
		bkt_p.bin_pts.v_bound.push_back(step_v*i);
	}
	bkt_p.bin_pts.v_bound.push_back(n_rows);

	cout << "bin_pts is initialized." << endl;
};
void Params::initBinEdges(int n_cols, int n_rows, int n_bins_u, int n_bins_v, float min_dist)
{
	// 각 축 방향으로 bin의 너비 or 높이.
	int step_u = ceil((float)n_cols / (float)n_bins_u);
	int step_v = ceil((float)n_rows / (float)n_bins_v);

	// 1 5  9 13
	// 2 6 10 14
	// 3 7 11 15
	// 4 8 12 16
	bkt_e.bin_edges.idxs.reserve(n_bins_u*n_bins_v);
	bkt_e.bin_edges.occupancy.reserve(n_bins_u*n_bins_v);
	for (int i = 0; i < n_bins_u*n_bins_v; i++) {
		bkt_e.bin_edges.idxs.push_back(chk::Point2i(-1, -1));
		bkt_e.bin_edges.occupancy.push_back(0);
	}
	bkt_e.bin_edges.u_bound.reserve(n_bins_u + 1);
	bkt_e.bin_edges.v_bound.reserve(n_bins_v + 1);

	// container를 채운다.
	bkt_e.bin_edges.n_bins_u = n_bins_u;
	bkt_e.bin_edges.n_bins_v = n_bins_v;
	bkt_e.bin_edges.count = 0;
	bkt_e.bin_edges.min_dist = min_dist;

	// 각 bin의 boundary 값들!
	for (int i = 0; i < n_bins_u; i++) {
		bkt_e.bin_edges.u_bound.push_back(step_u*i);
	}
	bkt_e.bin_edges.u_bound.push_back(n_cols);
	for (int i = 0; i < n_bins_v; i++) {
		bkt_e.bin_edges.v_bound.push_back(step_v*i);
	}
	bkt_e.bin_edges.v_bound.push_back(n_rows);

	cout << "bin_edges is initialized." << endl;
};

void Params::clearBinPts() {
	// 별 기능 없다. 그냥 bin 을 초기상태로 되돌리는 것.
	for (int i = 0; i < bkt_p.bin_pts.idxs.size(); i++) {
		bkt_p.bin_pts.idxs[i].x = -1;
		bkt_p.bin_pts.idxs[i].y = -1;
		bkt_p.bin_pts.occupancy[i] = 0;
	}
	bkt_p.bin_pts.count = 0;
};

void Params::clearBinEdges() {
	// 별 기능 없다. 그냥 bin 을 초기상태로 되돌리는 것.
	for (int i = 0; i < bkt_e.bin_edges.idxs.size(); i++) {
		bkt_e.bin_edges.idxs[i].x = -1;
		bkt_e.bin_edges.idxs[i].y = -1;
		bkt_e.bin_edges.occupancy[i] = 0;
	}
	bkt_e.bin_edges.count = 0;
};

void Params::bucketingBinEdges(
	vector<chk::Point2f>& pts_center, vector<char>& dirs_center, 
	vector<int>& idx_selected)
{
	idx_selected.resize(0);
	float min_dist = bkt_e.bin_edges.min_dist;

	int n_rows = bkt_e.bin_edges.v_bound.back();
	int n_cols = bkt_e.bin_edges.u_bound.back();

	int den_u = ceil(n_cols / bkt_e.bin_edges.n_bins_u);
	int den_v = ceil(n_rows / bkt_e.bin_edges.n_bins_v);

	int n_pts = pts_center.size();
	// bin 위치를 찾고, occupancy를 확인한다.
	int cnt = 0;
	int u_idx, v_idx, idx_bin;
	for (int i = 0; i < n_pts; i++) {
		u_idx = floor(pts_center[i].x / den_u);
		v_idx = floor(pts_center[i].y / den_v);
		idx_bin = v_idx*bkt_e.bin_edges.n_bins_u + u_idx;

		if (bkt_e.bin_edges.occupancy[idx_bin] < 1) { // 아무것도 없을 때, 그냥 넣는다.
			bkt_e.bin_edges.occupancy[idx_bin] = 1;
			bkt_e.bin_edges.idxs[idx_bin].x = i;
			++cnt;
		}
		else { // 1개이상 있을 때.
			if (bkt_e.bin_edges.occupancy[idx_bin] == 1) {//1개인경우.
				// 방향이 다르면 넣어준다. 
				if (dirs_center[bkt_e.bin_edges.idxs[idx_bin].x] != dirs_center[i]) {
					bkt_e.bin_edges.idxs[idx_bin].y = i;
					bkt_e.bin_edges.occupancy[idx_bin] = 2;
					++cnt;
				}
				/*float dist_temp =
					fabs(pts_center[bkt_e.bin_edges.idxs[idx_bin].x].x - pts_center[i].x)
					+ fabs(pts_center[bkt_e.bin_edges.idxs[idx_bin].x].y - pts_center[i].y);*/
				// 많이 가까운데, 기존 점보다 강하면 그 점을 대체한다. 또는 약하면 버린다.
				//if (dist_temp >= min_dist) {
				//	// 기존 점보다 강하면, 그점을 대체
				//	bkt_e.bin_edges.idxs[idx_bin].x = i;
				//	// 기존점보다 약하면 그냥 버린다 (아무것도 안한다).
				//	bkt_e.bin_edges.occupancy[idx_bin] = 1;
				//}
			}
		}
	}
	// 초기 입력된 점 대비 남은 점 갯수 
	// cout << "# of edge centers : [" << n_pts << "] to [" << cnt << "]" << endl;

	// 전체 bin을 순회하며, 최종적으로 남은 index를 뽑아낸다.
	idx_selected.reserve(cnt);
	for (int i = 0; i < bkt_e.bin_edges.n_bins_u*bkt_e.bin_edges.n_bins_v; i++) {
		if (bkt_e.bin_edges.occupancy[i] == 1)
			idx_selected.push_back(bkt_e.bin_edges.idxs[i].x);
		if (bkt_e.bin_edges.occupancy[i] == 2) {
			idx_selected.push_back(bkt_e.bin_edges.idxs[i].x);
			idx_selected.push_back(bkt_e.bin_edges.idxs[i].y);
		}
	}
	if (cnt != idx_selected.size()) throw std::runtime_error("cnt != idx_selected.size() in bucketingEdges\n");
};

void Params::bucketingBinPoints(
	vector<chk::Point2f>& cluttered, vector<float>& scores, float& score_max,
	vector<int>& idx_selected) 
{
	float min_dist = bkt_p.bin_pts.min_dist;

	int n_rows = bkt_p.bin_pts.v_bound.back();
	int n_cols = bkt_p.bin_pts.u_bound.back();

	int den_u = ceil((float)n_cols / (float)bkt_p.bin_pts.n_bins_u);
	int den_v = ceil((float)n_rows / (float)bkt_p.bin_pts.n_bins_v);

	int n_pts = cluttered.size();
	float quality_scale = bkt_p.bin_pts.quality_scale;
	float score_min = bkt_p.bin_pts.score_min;

	float thres_min = 0;
	if (quality_scale*score_max < score_min)
		thres_min = score_min;
	else
		thres_min = quality_scale*score_max;

	// bin 위치를 찾고, 동시에 score를 비교한다.
	int cnt = 0; 
	int u_idx, v_idx, idx_bin;
	for (int i = 0; i < n_pts; i++) {
		if (scores[i] >= 0*thres_min) {
			u_idx = floor(cluttered[i].x / den_u);
			v_idx = floor(cluttered[i].y / den_v);
			idx_bin = v_idx*bkt_p.bin_pts.n_bins_u + u_idx;
			if (bkt_p.bin_pts.occupancy[idx_bin] < 1) { // 아무것도 없을 때, 그냥 넣는다.
				bkt_p.bin_pts.occupancy[idx_bin] = 1;
				bkt_p.bin_pts.idxs[idx_bin].x = i;
				++cnt;
			}
			else { // 1개이상 있을 때.
				if (bkt_p.bin_pts.occupancy[idx_bin] == 1) {//1개인경우.
					float dist_temp = fabs(cluttered[bkt_p.bin_pts.idxs[idx_bin].x].x - cluttered[i].x)
						+ fabs(cluttered[bkt_p.bin_pts.idxs[idx_bin].x].y - cluttered[i].y);
					float score_temp = scores[bkt_p.bin_pts.idxs[idx_bin].x];
					// 많이 가까운데, 기존 점보다 강하면 그 점을 대체한다. 또는 약하면 버린다.
					if (dist_temp <= min_dist) {
						// 기존 점보다 강하면, 그점을 대체
						if (scores[i] > score_temp) { // 새로 들어오는놈이 크면, 대체한다.
							bkt_p.bin_pts.idxs[idx_bin].x = i;
							// 기존점보다 약하면 그냥 버린다 (아무것도 안한다).
							bkt_p.bin_pts.occupancy[idx_bin] = 1;
						}
					}
				}
			}
		}
	}
	// 초기 입력된 점 대비 남은 점 갯수 
	// cout << "# of points : [" << n_pts << "] to [" << cnt << "]" << endl;

	// 전체 bin을 순회하며, 최종적으로 남은 index를 뽑아낸다.
	idx_selected.reserve(cnt);
	for (int i = 0; i < bkt_p.bin_pts.n_bins_u*bkt_p.bin_pts.n_bins_v; i++) 
		if (bkt_p.bin_pts.occupancy[i] > 0) 
			idx_selected.push_back(bkt_p.bin_pts.idxs[i].x);
};
#endif