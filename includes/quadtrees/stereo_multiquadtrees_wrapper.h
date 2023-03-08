#ifndef _MULTIQUADTREEWRAPPER_H_
#define _MULTIQUADTREEWRAPPER_H_
#include <iostream>
#include <vector>
#include <exception>
#include "QuadTreeFastMultiplePooled.h"

using namespace std;
class StereoMultiQuadtreesWrapper 
{
public:
	QuadTreeFastMultiplePooled* qts_left; // 현재 이미지 왼쪽 트리 주소.
	QuadTreeFastMultiplePooled* qts_right; // 현재 이미지 오른쪽 트리 주소.

public:
	// constructor
    StereoMultiQuadtreesWrapper();

	// destructor
	~StereoMultiQuadtreesWrapper();

    // Functions for multiple quadtrees
	void multiQuadtreeBuild(
        const vector<chk::Point2f>& pts_l_, const vector<chk::Point2c>& bins_l_,
        const vector<chk::Point2f>& pts_r_, const vector<chk::Point2c>& bins_r_,
        const int& n_rows_, const int&  n_cols_,
        const int&  max_depth_,
        const float& eps_,
        const float& dist_thres_);
    void multiQuadtreeDelete();
	void multiQuadtreeNN();
	void multiQuadtreeCachedNN();

    // Functions for single quadtree
	void quadtreeBuild();
    void quadtreeDelete();
	void quadtreeNN();
	void quadtreeCachedNN();

private: // objectpool 관련해서는 어차피 밖에서 지정 할 필요가 없다. private 설정.
    int max_num_obj_input;

    ObjectPool<Node>* objpool_node; //= new ObjectPool<Node>(MAX_NUM_OBJ);
    ObjectPool<Elem>* objpool_elem; //= new ObjectPool<Elem>(MAX_NUM_OBJ);
    void objectpoolsBuild();
    void objectpoolsDelete();
};


/* ================================================================================
================================= Implementation ==================================
=================================================================================== */
StereoMultiQuadtreesWrapper::StereoMultiQuadtreesWrapper()
{
	max_num_obj_input = 524288; // 2^19개 ...
	// Initialize and memory allocation for objectPools
	objpool_node = nullptr;
	objpool_elem = nullptr;
    objectpoolsBuild();

    // Initialize pointers for left and right trees.
    qts_left = nullptr;
    qts_right = nullptr;

    // Initialize vectors (sufficiently pre-allocate)
    /*idx_left.reserve(10000);
    idx_right.reserve(10000);
    node_left.reserve(10000);
    node_right.reserve(10000);*/
    cout << "quadtree wrapper is initialized.\n";
};

StereoMultiQuadtreesWrapper::~StereoMultiQuadtreesWrapper() {
    objectpoolsDelete();
};
void StereoMultiQuadtreesWrapper::objectpoolsBuild() {
	if(objpool_node == nullptr)
		objpool_node = new ObjectPool<Node>(max_num_obj_input);
	if(objpool_elem == nullptr)
		objpool_elem = new ObjectPool<Elem>(max_num_obj_input);
};
void StereoMultiQuadtreesWrapper::objectpoolsDelete() {
	if (objpool_node != nullptr) delete objpool_node;
	objpool_node = nullptr;
	if (objpool_elem != nullptr) delete objpool_elem;
	objpool_elem = nullptr;
};

void StereoMultiQuadtreesWrapper::multiQuadtreeBuild(
    const vector<chk::Point2f>& pts_l_, const vector<chk::Point2c>& bins_l_, 
    const vector<chk::Point2f>& pts_r_, const vector<chk::Point2c>& bins_r_, 
    const int& n_rows_, const int&  n_cols_,
    const int&  max_depth_,
    const float& eps_,
    const float& dist_thres_)
{
    qts_left = new QuadTreeFastMultiplePooled(
        pts_l_, bins_l_, n_rows_, n_cols_, max_depth_, eps_, dist_thres_,
        objpool_node, objpool_elem);
    qts_right = new QuadTreeFastMultiplePooled(
        pts_r_, bins_r_, n_rows_, n_cols_, max_depth_, eps_, dist_thres_,
        objpool_node, objpool_elem);
};
void StereoMultiQuadtreesWrapper::multiQuadtreeDelete() 
{
    delete qts_left;
    delete qts_right;
};

void StereoMultiQuadtreesWrapper::multiQuadtreeNN()
{

};
void StereoMultiQuadtreesWrapper::multiQuadtreeCachedNN() {
};
void StereoMultiQuadtreesWrapper::quadtreeBuild() {
};
void StereoMultiQuadtreesWrapper::quadtreeNN() {
};
void StereoMultiQuadtreesWrapper::quadtreeCachedNN() {
};
void StereoMultiQuadtreesWrapper::quadtreeDelete() {
};
#endif