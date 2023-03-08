#ifndef _QUADTREEFASTMULTIPLEPOOLED_H_
#define _QUADTREEFASTMULTIPLEPOOLED_H_

#include <iostream>
#include <vector>
#include <memory>
#include "QuadTreeFastPooled.h"

using namespace std;

class QuadTreeFastMultiplePooled {
public:
	QuadTreeFastPooled* trees[8]; // 총 8개의 tree를 가지고있다. 각각 생성하도록 하자.
	float n_cols;
	float n_rows;
public:
    // 기본생성자. 안쓰일거야.
	QuadTreeFastMultiplePooled();

	// Constructor for MATLAB mex function version (chk::)
	QuadTreeFastMultiplePooled(vector<chk::Point2f>& points_input_,
		vector<int>& dirs1_input_,
		vector<int>& dirs2_input_,
		int n_rows_, int n_cols_,
		int max_depth_,
		float eps_,
		float dist_thres_,
		ObjectPool<Node>* objpool_node_, ObjectPool<Elem>* objpool_elem_);

	// Constructor for C++ version (chk::)
    QuadTreeFastMultiplePooled(
        const vector<chk::Point2f>& pts_,
        const vector<chk::Point2c>& bins_,
        const int& n_rows_, const int&  n_cols_,
        const int&  max_depth_,
        const float& eps_,
        const float& dist_thres_,
        ObjectPool<Node>* objpool_node_, ObjectPool<Elem>* objpool_elem_);

    // 소멸자. 모든 메모리 뿌수자.
	~QuadTreeFastMultiplePooled();
	void searchNN( // cached node 없이 그냥 써칭.
		vector<chk::Point2f>& points_query,
		vector<int>& dirs,
		vector<int>& id_matched,
		vector<Node*>& nodes_matched);

    void searchNNSingle(
        const float& u, const float& v, const char& dir,
        int& idx_matched, Node*& node_matched);

    // cached node 이용해서 써칭.
	void searchNNCached( 
		vector<chk::Point2f>& points_query,
		vector<int>& dirs,
		vector<int>& id_matched,
		vector<Node*>& nodes_cached_matched);

    void searchNNCachedSingle(
        const float& u, const float& v, const char& dir,
        int& idx_matched, Node*& node_cached_matched);
};

/*
* ----------------------------------------------------------------------------
*                                IMPLEMENTATION
* ----------------------------------------------------------------------------
*/
QuadTreeFastMultiplePooled::QuadTreeFastMultiplePooled() {
	for (int i = 0; i < 8; i++) {
		trees[i] = nullptr;
	};
	// 아무것도 안하는 생성자.
};

QuadTreeFastMultiplePooled::QuadTreeFastMultiplePooled(vector<chk::Point2f>& points_input_,
	vector<int>& dirs1_input_,
	vector<int>& dirs2_input_,
	int n_rows_, int n_cols_,
	int max_depth_,
	float eps_,
	float dist_thres_,
	ObjectPool<Node>* objpool_node_, ObjectPool<Elem>* objpool_elem_)
{
	// printf("start multiple trees generation...\n");
	// 트리들을 초기화한다.
	// multiple을 위한 추가적인 생성자를 통해 점을 넣지않은 상태로 초기화한다.
	for (int i = 0; i < 8; i++) {
		trees[i] =
			new QuadTreeFastPooled(n_rows_, n_cols_, max_depth_, eps_, dist_thres_, objpool_node_, objpool_elem_);
	}

	n_cols = n_cols_;
	n_rows = n_rows_;

	// 각 direction을 고려해 점들을 해당되는 tree로 넣어준다.
	int dir1 = -1;
	int dir2 = -1;
	int counters[8] = { 0,0,0,0,0,0,0,0 }; // 점이 해당 tree로 들어갈때마다, 1씩 더한다. (id가 된다)
	for (int i = 0; i < points_input_.size(); i++) {
		// 바라건데, dir1은 무조건 0~7 사이일것이다.
		dir1 = dirs1_input_[i];
		dir2 = dirs2_input_[i];

		trees[dir1]->insertPublic(points_input_[i], counters[dir1], i);
		++counters[dir1];

		// dir2가 있는지 확인해보자.
		if (dir2 > -1) { // dir2가 있따면, 여기로도 points_input_을 넣어주자.
			trees[dir2]->insertPublic(points_input_[i], counters[dir2], i);
			++counters[dir2];
		}
	}
	//printf("Multiple tree made done.\n");
	//printf("Point numbers:\n");
	/*for (int i = 0; i < 8; i++) {
	printf("dir [%d] - %d\n", i, counters[i]);
	}*/
	int sum_overlap = 0;
	for (int i = 0; i < 8; i++) sum_overlap += counters[i];

	//printf("multiple tree is successfully generated.\n");
};

QuadTreeFastMultiplePooled::QuadTreeFastMultiplePooled(
    const vector<chk::Point2f>& pts_,
	const vector<chk::Point2c>& bins_,
	const int& n_rows_, const int&  n_cols_,
    const int&  max_depth_,
    const float& eps_,
    const float& dist_thres_,
	ObjectPool<Node>* objpool_node_, ObjectPool<Elem>* objpool_elem_)
{
	// printf("start multiple trees generation...\n");
	// 트리들을 초기화한다.
	// multiple을 위한 추가적인 생성자를 통해 점을 넣지않은 상태로 초기화한다.
	for (int i = 0; i < 8; i++) {
		this->trees[i] = new QuadTreeFastPooled(
            n_rows_, n_cols_, max_depth_, eps_, dist_thres_, 
            objpool_node_, objpool_elem_);
	}

	n_cols = n_cols_;
	n_rows = n_rows_;

	// 각 direction을 고려해 점들을 해당되는 tree로 넣어준다.
	char dir1 = -1, dir2 = -1;
	int cnts[8] = { 0,0,0,0,0,0,0,0 }; // 점이 해당 tree로 들어갈때마다 1씩 더한다. (id가 된다)
	size_t n_pts = pts_.size();
	for (int idx = 0; idx < n_pts; idx++) {
		// dir1은 무조건 0~7 사이일것이다. dir2는 -1도 있다.
		dir1 = bins_[idx].x;
		dir2 = bins_[idx].y;

		// [1] dir1에 삽입.
        this->trees[dir1]->insertPublic(pts_[idx], cnts[dir1], idx);
		++cnts[dir1];

		// dir2가 존재한다면, 삽입.
		if (dir2 > -1) { // dir2가 있따면, 여기로도 points_input_을 넣어주자.
            this->trees[dir2]->insertPublic(pts_[idx], cnts[dir2], idx);
			++cnts[dir2];
		}
	}
	
	//printf(" Multiple trees are successfully generated.\n");
};

QuadTreeFastMultiplePooled::~QuadTreeFastMultiplePooled() {
	for (int i = 0; i < 8; i++) {
		delete this->trees[i];
	}
};

void QuadTreeFastMultiplePooled::searchNN(
	vector<chk::Point2f>& points_query,
	vector<int>& dirs,
	vector<int>& idx_matched,
	vector<Node*>& nodes_matched)
{
	// 각 direction을 고려해 점들을 해당되는 tree에서 검색한다.
	int dir = -2;
	int idx_matched_temp = -2;
	for (int i = 0; i < points_query.size(); i++) {
		// 이미지 내부에 있는 경우, 서칭. 
		dir = dirs[i];
		if (dir> -1 && points_query[i].x > 0 && points_query[i].x < n_cols &&
			points_query[i].y > 0 && points_query[i].y < n_rows) {

			// printf("%d - th match... dir :%d\n",i, dir);
			// 해당 방향에서 서칭을 한다.
			idx_matched_temp = trees[dir]->searchNNSingleQuery(points_query[i], nodes_matched[i]);
            idx_matched[i] = idx_matched_temp;
		}
		else { // 이미지 내부에 있지 않으면, id_matched = -1, nodes_matched = root
            idx_matched[i] = (-2);
			nodes_matched[i] = this->trees[dir]->root;
		}

	}
};

void QuadTreeFastMultiplePooled::searchNNSingle(
    const float& u, const float& v, const char& dir,
    int& idx_matched, Node*& node_matched) 
{
    chk::Point2f pt_temp(u,v);
    if ((dir > -1) && (u > 0) && (u < n_cols) && (v > 0) && (v < n_rows)) {
        idx_matched = trees[dir]->searchNNSingleQuery(pt_temp, node_matched);
    }
    else {
        idx_matched = -1;
        node_matched = trees[dir]->root;
    }
}

void QuadTreeFastMultiplePooled::searchNNCached(
	vector<chk::Point2f>& points_query,
	vector<int>& dirs,
	vector<int>& id_matched,
	vector<Node*>& nodes_cached_matched)
{
	// 각 direction을 고려해 점들을 해당되는 tree에서 검색한다.
	int dir = -2;
	int id_matched_temp = -2;
	for (int i = 0; i < points_query.size(); i++) {// 이미지 내부에 있는 경우, 서칭. 
		dir = dirs[i];
		if (dir > -1 && points_query[i].x > 0 && points_query[i].x < n_cols &&
			points_query[i].y > 0 && points_query[i].y < n_rows) {

			// 해당 방향에서 서칭을 한다.
			id_matched_temp = trees[dir]->searchNNSingleQueryCached(points_query[i], nodes_cached_matched[i]);
			id_matched[i] = (id_matched_temp);
		}
		else { // 이미지 내부에 있지 않으면, id_matched = -1, nodes_matched = root
			id_matched[i] = (-2);
			nodes_cached_matched[i] = trees[dir]->root;
		}
	}
};
void QuadTreeFastMultiplePooled::searchNNCachedSingle(
    const float& u, const float& v, const char& dir,
    int& idx_matched, Node*& node_cached_matched)
{
    chk::Point2f pt_temp(u, v);
    if ((dir > -1) && (u > 0) && (u < n_cols) && (v > 0) && (v < n_rows)) {
        idx_matched = trees[dir]->searchNNSingleQueryCached(pt_temp, node_cached_matched);
    }
    else {
        idx_matched = -1;
        node_cached_matched = trees[dir]->root;
    }
}
#endif