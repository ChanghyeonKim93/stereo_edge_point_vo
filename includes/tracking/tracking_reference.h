#ifndef _TRACKINGREFERENCE_H_
#define _TRACKINGREFERENCE_H_

#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"  
#include "Eigen/Dense"

#include "../quadtrees/CommonStruct.h"
#include "../frame/stereo_frames.h"

#include "../custom_memory.h"

using namespace std;

namespace chk {

    typedef Eigen::Matrix<float, 2, 1> Vec2;
    typedef Eigen::Matrix<float, 2, 2> Mat22;
    typedef Eigen::Matrix<float, 3, 1> Vec3;
    typedef Eigen::Matrix<float, 3, 3> Mat33;
    typedef Eigen::Matrix<float, 4, 1> Vec4;
    typedef Eigen::Matrix<float, 4, 4> Mat44;
    typedef Eigen::Matrix<float, 6, 1> Vec6;
    typedef Eigen::Matrix<float, 6, 6> Mat66;
    typedef Eigen::Matrix<float, 8, 1> Vec8;
    typedef Eigen::Matrix<float, 8, 8> Mat88;


    class TrackingReference {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW


        /** Creates an empty TrackingReference with optional preallocation per level. */
        TrackingReference(const int& id_, const int& max_pyr_lvl, const vector<int>& nrows_pyr, const vector<int>& ncols_pyr);
        ~TrackingReference();
        void allocateReference(); // dynamic allocation.
        void initializeAndRenewReference(const StereoFrame* keyframe_); // 채워진 keyframe의 점들로 reference 점을 만든다.
        void updateReference(); // 현재 Xe 점이 0,0,0 인것중에서 현재 들어온 놈의 깊이가 있으면 업데이트
        void purgeAll(); // 모든 Nec, Np 를 0 으로 만든다. (Npe, Npp는 안바뀌니까 놔둬라)

        // if new keyframe incomes, do purgeAll() and initializeAndRenewReference(key).


        int id; // Frame number.
        StereoFrame* keyframe; // 현재 keyframe인 stereoframe의 pointer를 가지고 있다.
        int MAX_PYR_LVL; //사실 maximum level은 5 이하로 가지 않는다. 그보다 작을 수 있으므로 저장해둔다.
        int ncols[5];
        int nrows[5];

        // Reference Points without patch.
        int Nec; // # of edge center points
        int Np; // # of corner points

        // Edge (Nec) pointers for each container
        Vec2* pts_e;
        Vec3* grad_e;
        char* bin_e;
        Vec3* Xe;
        float* std_invd_e;


        int* idx_l;
        int* idx_r;
        Node** node_l; // 포인터 배열의 배열
        Node** node_r;

        // Point (Np) pointers for each container
        Vec2* pts_p;
        Vec3* Xp;
        float* std_invd_p;

        // Reference Points with patch.
        int Npe[5]; // # of pixels in one edge patch
        int Npp[5]; // # of pixels in one corner patch

        // Edge patch (Nec*Npe)  pointers for each container
        Vec2* pts_epat[5]; // 포인트는 필요없는 것 같은데?
        float* illumi_epat[5]; // brightness of pixels

        // Point patch (Nec*Npp) pointers for each container
        Vec2* pts_ppat[5];
        Vec3* Xppat[5]; // Xppat 은 매 레벨별로 가지고 있어야 한다.
        float* illumi_ppat[5];





        // For Affine KLT
        Vec2* pts_p_tracked; // Affine KLT result. (currently tracked)
        Vec2* pts_p_warped;  // Warped points. (before the tracking).
        Vec6* point_params; // from keyframe to current frame.

        float* buf_u_warped; // for one patch. (169 pixels)
        float* buf_v_warped; // 
        float* buf_I_c_warped; // 


        // for Reprojection minimization refinement.
        Vec2* pts_e_matched; // Multi-quadtrees matching result.

        
    private:
        void releaseAll();
    };
};

/* ================================================================================
================================= Implementation ==================================
=================================================================================== */

/// [Desciption] Constructor.
chk::TrackingReference::TrackingReference(const int& id_, const int& max_pyr_lvl,
    const vector<int>& nrows_pyr, const vector<int>& ncols_pyr) // 단순한 생성자.
{
    id = id_;
    MAX_PYR_LVL = max_pyr_lvl;

    keyframe = nullptr;

    Nec = 0;
    Np = 0;

    pts_e = nullptr;
    bin_e = nullptr;
    grad_e = nullptr;
    Xe = nullptr;
    std_invd_e = nullptr;
    idx_l = nullptr;
    idx_r = nullptr;
    node_l = nullptr;
    node_r = nullptr;

    pts_p = nullptr;
    Xp = nullptr;
    std_invd_p = nullptr;

    pts_e_matched = nullptr;
    pts_p_tracked = nullptr;
    pts_p_warped = nullptr;
    point_params  = nullptr;
   
    for (int lvl = 0; lvl < max_pyr_lvl; ++lvl)
    {
        Npe[lvl] = 0;
        Npp[lvl] = 0;
        
        pts_epat[lvl] = nullptr;
        illumi_epat[lvl] = nullptr;

        pts_ppat[lvl] = nullptr;
        Xppat[lvl] = nullptr;
        illumi_ppat[lvl] = nullptr;

        ncols[lvl] = ncols_pyr[lvl];
        nrows[lvl] = nrows_pyr[lvl];
        // cout << "     row,col: [" << nrows[lvl] << ", " << ncols[lvl] << "]" << endl;
    }
    tic();
    allocateReference();
    double dt = toc(0);
    cout << "   Reference is initialized. id:["<<id<<"], max_lvl:[" << MAX_PYR_LVL<<"], elapsed time: "<< dt<<" [ms]\n";
};

/// [Desciption] Dyamically allocates all the memories.
void chk::TrackingReference::allocateReference() {
    int w = ncols[0];
    int h = nrows[0];

    pts_e  = new chk::Vec2[w*h];
    bin_e  = (char*)custom_aligned_malloc(sizeof(char)*w*h);
    grad_e = new chk::Vec3[w*h];
    Xe     = new chk::Vec3[w*h];
    idx_l  = (int*)custom_aligned_malloc(sizeof(int)*w*h);
    idx_r  = (int*)custom_aligned_malloc(sizeof(int)*w*h);
    node_l = (Node**)custom_aligned_malloc(sizeof(Node*)*w*h);
    node_r = (Node**)custom_aligned_malloc(sizeof(Node*)*w*h);
    std_invd_e = (float*)custom_aligned_malloc(sizeof(float)*w*h);

    pts_p = new chk::Vec2[w*h];
    Xp = new chk::Vec3[w*h];
    std_invd_p = (float*)custom_aligned_malloc(sizeof(float)*w*h);

    pts_e_matched = new chk::Vec2[w*h];
    pts_p_tracked = new chk::Vec2[w*h];
    pts_p_warped = new chk::Vec2[w*h];
    point_params  = new chk::Vec6[w*h];

    for (int lvl = 0; lvl < MAX_PYR_LVL; lvl++) 
    {
        //w = ncols[lvl]; // 그냥 넉넉~~~하게 할당하자. 어차피 메모리 많다.
        //h = nrows[lvl];
        
        pts_epat[lvl]    = new chk::Vec2[w*h];
        illumi_epat[lvl] = (float*)custom_aligned_malloc(sizeof(float) * w * h);

        pts_ppat[lvl]    = new chk::Vec2[w*h];
        Xppat[lvl]       = new chk::Vec3[w*h];
        illumi_ppat[lvl] = (float*)custom_aligned_malloc(sizeof(float) * w * h);
    }
};

/// [Desciption] it makes Nec and Np to 0, which can be regarded as all data is removed.
void chk::TrackingReference::purgeAll() { Nec = 0; Np = 0; };

/// [Desciption] Whenever depth values of keyframe are update, it is called.
void chk::TrackingReference::initializeAndRenewReference(const StereoFrame* keyframe_) {
    purgeAll(); // purge all existing points. (not de-allocates memories, but changes the Nec & Np to 0).
    keyframe = (StereoFrame*)keyframe_; // update keyframe pointer by newly incoming keyframe's one.
    
    // fill Npe and Npp
    for (int lvl = 0; lvl < MAX_PYR_LVL; lvl++) {
        Npe[lvl] = (int)keyframe->patch_edge[lvl].size();
        Npp[lvl] = (int)keyframe->patch_point[lvl].size();
    }

    // fill out edge centers
    Nec = keyframe->left->ft_edge[0]->Nec;

    Vec2* p_pts = pts_e;
    char* p_bin = bin_e;
    Vec3* p_grad = grad_e;
    Vec3* p_X    = Xe;
    int* p_idx_l = idx_l;
    int* p_idx_r = idx_r;
    Node** p_node_l = node_l;
    Node** p_node_r = node_r;
    float* p_std_invd_e = std_invd_e;

    Vec2* p_pts_e_matched = pts_e_matched;

    Vec2* p_pts_max = p_pts + Nec;
    int cnt = 0;
    for (; p_pts < p_pts_max;
        cnt++, p_pts++, p_bin++, p_grad++, p_X++, p_idx_l++, p_idx_r++, p_node_l++, p_node_r++, p_std_invd_e++,
        p_pts_e_matched++)
    {
        (*p_pts)[0] = keyframe->left->ft_edge[0]->pts[cnt].x; // x-coordi of this pixel
        (*p_pts)[1] = keyframe->left->ft_edge[0]->pts[cnt].y; // y=coordi of this pixel

        *p_bin = keyframe->left->ft_edge[0]->dirs[cnt]; // orientational bin of thie edge center point.

        (*p_grad)[0] = keyframe->left->ft_edge[0]->grad[cnt].x; // dx
        (*p_grad)[1] = keyframe->left->ft_edge[0]->grad[cnt].y; // dy
        (*p_grad)[2] = keyframe->left->ft_edge[0]->grad[cnt].z; // magnitude
        // only when needed, normalize and use it. 

        (*p_X)[0] = keyframe->left->ft_edge[0]->pts3d[cnt].x; // X
        (*p_X)[1] = keyframe->left->ft_edge[0]->pts3d[cnt].y; // Y
        (*p_X)[2] = keyframe->left->ft_edge[0]->pts3d[cnt].z; // Z
        *p_idx_l = -1;
        *p_idx_r = -1;
        *p_node_l = nullptr;
        *p_node_r = nullptr;

        (*p_std_invd_e) = keyframe->left->ft_edge[0]->std_invd[cnt]; // sig_invd

        (*p_pts_e_matched)(0) = 0.0f;
        (*p_pts_e_matched)(1) = 0.0f;
    }

    // fill out edge patches
    for (int lvl = 0; lvl < MAX_PYR_LVL; lvl++) {
        float cx    = keyframe->cx_pyr[lvl];
        float cy    = keyframe->cy_pyr[lvl];
        float fxinv = keyframe->fxinv_pyr[lvl];
        float fyinv = keyframe->fyinv_pyr[lvl];
        float iscale = 1.0f / powf(2.0f, lvl);
        
        vector<chk::Point2f>& patch = keyframe->patch_edge[lvl]; // edge patch
        Vec2* p_pts_pat     = pts_epat[lvl];
        float* p_illumi_pat = illumi_epat[lvl];

        p_pts  = pts_e; // 좌표가 .... 반땅치기 되어야한다.
        p_grad = grad_e;
        Vec2* p_pts_patmax = p_pts_pat + Nec*Npe[lvl]; // max. pointer for pts_epat
        int cnt_total = 0;
        for (; p_pts_pat < p_pts_patmax; p_pts++, p_grad++)
        { // Nec loops
            float ue   = (*p_pts)[0] * iscale; // 2d center u-coordinate
            float ve   = (*p_pts)[1] * iscale; // 2d center v-coordinate
            float gx   = (*p_grad)[0];
            float gy   = (*p_grad)[1];
            float gmag = (*p_grad)[2];
            gx /= gmag;
            gy /= gmag;

            Vec2* p_patmax = p_pts_pat + Npe[lvl]; // patch length of the current level.
            int cnt_pat = 0;
            for (; p_pts_pat < p_patmax; p_pts_pat++, p_illumi_pat++)
            { // Npe[lvl] loops, i 번째 점에 대한 patch pixel 등등을 모두 채워넣는 작업. 잘된다.
                // 2D edge patch pixel points (rotated to the gradient direction)
                // R = [gx, -gy; gy, gx];
                //  - gy*patch.at(cnt_pat).y = 0;
                float ue_pat = gx*patch[cnt_pat].x + ue;
                float ve_pat = gy*patch[cnt_pat].x + ve;
                (*p_pts_pat)[0] = ue_pat;
                (*p_pts_pat)[1] = ve_pat;

                // illumination interpolation
                *p_illumi_pat = improc::interpImageSingle(keyframe->left->img[lvl], ue_pat, ve_pat);
                              
                ++cnt_pat;
                ++cnt_total;
            } // end for Npe
        } // end for Nec
        // cout << "c total: " << cnt_total <<", Nec*Npe[lvl]: "<< ncols[lvl]*nrows[lvl] << endl;
    } // end for lvl



    // fill out corner points
    Np = keyframe->left->ft_point[0]->Np;

    Vec2* ptr_pts_p = pts_p;
    Vec3* ptr_Xp    = Xp;
    float* p_std_invd_p = std_invd_p;
    
    Vec2* p_pts_p_tracked = pts_p_tracked;
    Vec2* p_pts_p_warped = pts_p_warped;
    Vec6* p_point_params = point_params;
    
    Vec2* ptr_pts_pmax = ptr_pts_p + Np;
    cnt = 0;
    for (; ptr_pts_p < ptr_pts_pmax; cnt++, ptr_pts_p++, ptr_Xp++, p_std_invd_p++, 
        p_pts_p_tracked++, p_pts_p_warped++, p_point_params++)
    {
        (*ptr_pts_p)[0] = keyframe->left->ft_point[0]->pts[cnt].x; // x-coordi of this pixel
        (*ptr_pts_p)[1] = keyframe->left->ft_point[0]->pts[cnt].y; // y=coordi of this pixel
        (*ptr_Xp)[0]    = keyframe->left->ft_point[0]->pts3d[cnt].x; // X
        (*ptr_Xp)[1]    = keyframe->left->ft_point[0]->pts3d[cnt].y; // Y
        (*ptr_Xp)[2]    = keyframe->left->ft_point[0]->pts3d[cnt].z; // Z
        (*p_std_invd_p) = keyframe->left->ft_point[0]->std_invd[cnt]; // std_invd (for point)


        (*p_pts_p_tracked)(0) = 0.0f;
        (*p_pts_p_tracked)(1) = 0.0f;
        (*p_pts_p_warped)(0) = 0.0f;
        (*p_pts_p_warped)(1) = 0.0f;
        (*p_point_params)(0) = 0.0f;
        (*p_point_params)(1) = 0.0f;
        (*p_point_params)(2) = 0.0f;
        (*p_point_params)(3) = 0.0f;
        (*p_point_params)(4) = 0.0f;
        (*p_point_params)(5) = 0.0f;
    }

    // fill out corner point patches
    for (int lvl = 0; lvl < MAX_PYR_LVL; lvl++) {
        float cx = keyframe->cx_pyr[lvl];
        float cy = keyframe->cy_pyr[lvl];
        float fxinv = keyframe->fxinv_pyr[lvl];
        float fyinv = keyframe->fyinv_pyr[lvl];
        float iscale = 1.0f / powf(2.0f, lvl);

        vector<chk::Point2f>& patch = keyframe->patch_point[lvl]; // point patch

        Vec2* p_pts_ppat    = pts_ppat[lvl];
        Vec3* p_Xppat       = Xppat[lvl];
        float* p_illumi_pat = illumi_ppat[lvl];

        p_pts = pts_p;
        Vec2* p_pts_patmax = p_pts_ppat + Np*Npp[lvl]; // max. pointer for pts_ppat
        cnt = 0; // 여기가 완전 잘못되었따? 
        for (; p_pts_ppat < p_pts_patmax; p_pts++)
        { // Nec loops
            float up = (*p_pts)(0) * iscale; // 2d point u-coordinate
            float vp = (*p_pts)(1) * iscale; // 2d point v-coordinate
            float z = keyframe->left->ft_point[0]->pts3d[cnt].z;
            // cout << "lv:" << lvl << ", uv:" << up << "," << vp << ", z:" << z << endl;
            ++cnt;

            Vec2* p_patmax = p_pts_ppat + Npp[lvl]; // patch length of the current level.
            int cnt_pat = 0;
            for (; p_pts_ppat < p_patmax; p_pts_ppat++, p_Xppat++, p_illumi_pat++)
            { 
                float up_pat = patch.at(cnt_pat).x + up;
                float vp_pat = patch.at(cnt_pat).y + vp;

                (*p_pts_ppat)(0) = up_pat;
                (*p_pts_ppat)(1) = vp_pat; // patch 좌표 계산 자체는 잘못된게 없는 것 같다.

                (*p_Xppat)(0) = (up_pat - cx)*fxinv * z; // 이게 뭔가 잘못되었나 ? 아닌것같은데 
                (*p_Xppat)(1) = (vp_pat - cy)*fyinv * z;
                (*p_Xppat)(2) = z;

                // illumination interpolation
                *p_illumi_pat = improc::interpImageSingle(keyframe->left->img[lvl], up_pat, vp_pat);

                ++cnt_pat;
            } // end for Npe
        } // end for Nec
    } // end for lvl
};

/// [Desciption] Whenever 3D points are updated by the depthRecon, references are also update in this func.
void chk::TrackingReference::updateReference() {
    // maintaining Nec & Np, and keyframe_, update depth & std_invd.
    // renew edge centers
    Vec3* p_X = Xe;
    int* p_idx_l = idx_l; // to be initialize.
    int* p_idx_r = idx_r; // to be initialize.
    Node** p_node_l = node_l; // to be initialize.
    Node** p_node_r = node_r; // to be initialize. (root? null)
    float* p_std_invd_e = std_invd_e;

    Vec3* p_X_max = p_X + Nec;
    int cnt = 0;
    for (; p_X < p_X_max;
        cnt++, p_X++, p_idx_l++, p_idx_r++, p_node_l++, p_node_r++, p_std_invd_e++)
    {
        // only when needed, normalize and use it. 
        (*p_X)[0] = keyframe->left->ft_edge[0]->pts3d[cnt].x; // X
        (*p_X)[1] = keyframe->left->ft_edge[0]->pts3d[cnt].y; // Y
        (*p_X)[2] = keyframe->left->ft_edge[0]->pts3d[cnt].z; // Z
        *p_idx_l  = -1;
        *p_idx_r  = -1;
        *p_node_l = nullptr;
        *p_node_r = nullptr;
        *p_std_invd_e = keyframe->left->ft_edge[0]->std_invd[cnt]; // sig_invd
    }

    // renew corner points
    Np = keyframe->left->ft_point[0]->Np;
    Vec3* ptr_Xp = Xp;
    Vec3* p_Xp_max = Xp + Np;
    float* p_std_invd_p = std_invd_p;
    cnt = 0;
    for (; ptr_Xp < p_Xp_max; cnt++, ptr_Xp++, p_std_invd_p++) {
        (*ptr_Xp)[0]  = keyframe->left->ft_point[0]->pts3d[cnt].x; // X
        (*ptr_Xp)[1]  = keyframe->left->ft_point[0]->pts3d[cnt].y; // Y
        (*ptr_Xp)[2]  = keyframe->left->ft_point[0]->pts3d[cnt].z; // Z
        *p_std_invd_p = keyframe->left->ft_point[0]->std_invd[cnt]; // std_invd (for point)
    }

    // fill out corner point patches
    for (int lvl = 0; lvl < MAX_PYR_LVL; lvl++) {
        float cx = keyframe->cx_pyr[lvl];
        float cy = keyframe->cy_pyr[lvl];
        float fxinv = keyframe->fxinv_pyr[lvl];
        float fyinv = keyframe->fyinv_pyr[lvl];
        float iscale = 1.0f / powf(2.0f, lvl);

        vector<chk::Point2f>& patch = keyframe->patch_point[lvl]; // point patch

        Vec2* p_pts_ppat = pts_ppat[lvl];
        Vec3* p_Xppat = Xppat[lvl];
        Vec3* p_Xppat_max = Xppat[lvl] + Np*Npp[lvl];// max. pointer for Xppat
        cnt = 0; // 여기가 완전 잘못되었따? 
        for (; p_Xppat < p_Xppat_max;)
        { // Nec loops
            float z = keyframe->left->ft_point[0]->pts3d[cnt].z;
            ++cnt;

            Vec3* p_Xppat_rowmax = p_Xppat + Npp[lvl];
            int cnt_pat = 0;
            for (; p_Xppat < p_Xppat_rowmax; p_pts_ppat++, p_Xppat++)
            {
                float up_pat = (*p_pts_ppat)(0);
                float vp_pat = (*p_pts_ppat)(1);
                (*p_Xppat)(0) = (up_pat - cx)*fxinv * z; // 이게 뭔가 잘못되었나 ? 아닌것같은데 
                (*p_Xppat)(1) = (vp_pat - cy)*fyinv * z;
                (*p_Xppat)(2) = z;
                ++cnt_pat;
            } // end for Npe
        } // end for Nec
    } // end for lvl
};

/// [Desciption] destructor
chk::TrackingReference::~TrackingReference() {
    releaseAll();
    cout << "  Reference is delete.\n";
};

/// [Desciption] All dynamic allocations are released. (at the end of the program.)
void chk::TrackingReference::releaseAll() {

    if (pts_e != nullptr) delete[] pts_e;
    if (bin_e != nullptr) custom_aligned_free((void*)bin_e);
    if (grad_e != nullptr) delete[] grad_e;
    if (Xe != nullptr) delete[] Xe;
    if (idx_l != nullptr) custom_aligned_free((void*)idx_l);
    if (idx_r != nullptr) custom_aligned_free((void*)idx_r);
    if (node_l != nullptr) custom_aligned_free((void*)node_l);
    if (node_r != nullptr) custom_aligned_free((void*)node_r);
    if (std_invd_e != nullptr) custom_aligned_free((void*)std_invd_e);

    if (pts_p != nullptr) delete[] pts_p;
    if (Xp != nullptr) delete[] Xp;
    if (std_invd_p != nullptr) custom_aligned_free((void*)std_invd_p);

    if (pts_e_matched != nullptr) delete[] pts_e_matched;
    if (point_params != nullptr) delete[] point_params;

    if (pts_p_tracked != nullptr) delete[] pts_p_tracked;
    if (pts_p_warped != nullptr) delete[] pts_p_warped;

    for (int lvl = 0; lvl < MAX_PYR_LVL; ++lvl)
    {     
        if (pts_epat[lvl] != nullptr) delete[] pts_epat[lvl];
        if (illumi_epat[lvl] != nullptr) custom_aligned_free((void*)illumi_epat[lvl]);

        if (pts_ppat[lvl] != nullptr) delete[] pts_ppat[lvl];
        if (Xppat[lvl] != nullptr) delete[] Xppat[lvl];
        if (illumi_ppat[lvl] != nullptr) custom_aligned_free((void*)illumi_ppat[lvl]);
    }
};

#endif