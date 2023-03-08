#ifndef _MOTIONTRACKER_H_
#define _MOTIONTRACKER_H_

#include <iostream>
#include <cstdlib>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"  
#include "Eigen/Dense"

#include "../frame/stereo_frames.h"
#include "../tracking/tracking_reference.h"
#include "../quadtrees/stereo_multiquadtrees_wrapper.h"

using namespace std;
namespace chk {
    // Tracking part


    /* SE3 6-DoF motion + 2-DoF Affine brightness parameters*/
    class SE3AffineBrightTracker {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        
        int MAX_PYR_LVL; //사실 maximum level은 5 이하로 가지 않는다. 그보다 작을 수 있으므로 저장해둔다.
        int ncols[5];
        int nrows[5];

        // camera matrix
        Eigen::Matrix3f K[5];
        Eigen::Matrix3f Kinv[5];

        Mat44 T_nlnr; // left to right extrinsic matrix. (calibrated)

        float fx, fy, cx, cy; // 필요할때 만들어쓰자.
        float fxinv, fyinv;

        // Hessian and Jacobian
        // We just solve Linear system JtWJ*delta_xi = mJtWr; where mJtWr = -J^t*W*r;
        // JtWJ matrix is guaranteed to be P.S.D and symmetric matrix.
        // Thus, we can efficiently solve this equation by Cholesky decomposition.
        // --> JtWJ.ldlt().solve(mJtWr);
        Mat88 JtWJ;
        Vec8 mJtWr;

        Mat88 JtWJ_e;
        Vec8 mJtWr_e;

        Mat88 JtWJ_ppat;
        Vec8 mJtWr_ppat;

        float err;
        float err_e;
        float err_ppat;
        float err_soft;

        float n_valid_edge;
        float n_valid_pointpatch;

        int n_overthres_edge;
        int n_overthres_point;

        // n_overthres / n_valid < 0.6 =  huber thres is is good.
        Params* params;
        StereoFrame* frame_c;
        chk::TrackingReference* track_ref;
        StereoMultiQuadtreesWrapper* tree;

        // level과 무관하게 하나씩만 가지고 있으면 된다.
        float* buf_residual_e; // r_e
        float* buf_residual_ppat; // r_p
        float* buf_residual_epat; // r_ppat

        float* buf_dx_ppat;
        float* buf_dy_ppat;
        float* buf_dx_epat;
        float* buf_dy_epat;

        float* buf_xwarp;
        float* buf_ywarp;
        float* buf_zwarp;

        float* buf_weight_e;
        float* buf_weight_ppat;
        float* buf_weight_epat;
        

        SE3AffineBrightTracker(Params* params_,
            const vector<int>& nrows_pyr, const vector<int>& ncols_pyr, 
            const vector<Eigen::Matrix3d>& K_pyr_, const Eigen::Matrix4d& T_nlnr_);
        ~SE3AffineBrightTracker();

        void linkRef(chk::TrackingReference* track_ref_) { track_ref = track_ref_; };
        void linkTree(StereoMultiQuadtreesWrapper* tree_) { tree = tree_; };
        void linkFrameCur(StereoFrame* frame_) { frame_c = frame_; };
        void detachRef() { track_ref = nullptr; };
        void detachTree() { tree = nullptr; };
        void detachFrameCur() { frame_c = nullptr; };

        void calcResidual(const Eigen::Matrix4d& T_ck, const Vec2& ab_ck_, const int& lvl, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point);
        void se3affineMotionTracking(const Eigen::Matrix4d& T_ck_, const Vec2& ab_ck_, const int& lvl, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point, Vec8& delta, const float& lambda);

    private:

        void fillHJr(const Eigen::Matrix4d& T_ck, const Vec2& ab_ck_, const int& lvl, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point);
        inline void updateEDGE(const Vec8& Jt, const float& r, const float& weight);
        inline void updatePPAT(const Vec8& Jt, const float& r, const float& weight);
        inline void updateSum();

        void solveLMstep(Vec8& delta, const float& lambda);
    };

    void SE3AffineBrightTracker::se3affineMotionTracking(const Eigen::Matrix4d& T_ck_, const Vec2& ab_ck_, const int& lvl, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point, Vec8& delta, const float& lambda) {
        float alpha = 0.0f;
        float beta = 0.0f;
        fillHJr(T_ck_, ab_ck_, lvl, is_cached, huber_scaler_edge, huber_scaler_point); // render A and b

        solveLMstep(delta, lambda);     // solve A*x = b
    };

    void SE3AffineBrightTracker::calcResidual(
        const Eigen::Matrix4d& T_ck_, const Vec2& ab_ck_, const int& lvl, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point) 
    {
        // 이 함수에서 exp(alpha), beta 를 먼저 추정하고, Huber norm을 조절한다.
        if (track_ref == nullptr) throw std::runtime_error("TRACKER: track_ref is not linked\n");
        if (tree == nullptr) throw std::runtime_error("TRACKER: tree is not linked\n");
        if (frame_c == nullptr) throw std::runtime_error("TRACKER: frame_c is not linked\n");

        // initialize
        float huber_thres_edge = 1.5f*huber_scaler_edge;
        float huber_thres_ppat = 5.0f*huber_scaler_point;

        err      = 0;
        err_e    = 0;
        err_ppat = 0;

        n_valid_edge = 0;
        n_valid_pointpatch = 0;

        n_overthres_point = 0;
        n_overthres_edge = 0;

        float fx = K[lvl](0, 0); // intrinsic matrix of the current level lvl.
        float fy = K[lvl](1, 1);
        float cx = K[lvl](0, 2);
        float cy = K[lvl](1, 2);
        int n_cols = frame_c->left->n_cols_pyr[lvl];
        int n_rows = frame_c->left->n_rows_pyr[lvl];

        Mat44 T_ck;
        T_ck << T_ck_(0, 0), T_ck_(0, 1), T_ck_(0, 2), T_ck_(0, 3),
            T_ck_(1, 0), T_ck_(1, 1), T_ck_(1, 2), T_ck_(1, 3),
            T_ck_(2, 0), T_ck_(2, 1), T_ck_(2, 2), T_ck_(2, 3),
            T_ck_(3, 0), T_ck_(3, 1), T_ck_(3, 2), T_ck_(3, 3);

        Mat33 R_ck = T_ck.block<3, 3>(0, 0); // rotation from key to cur.
        Vec3 t_ck = T_ck.block<3, 1>(0, 3); // translation from key to cur.
        Mat33 R_rl = T_nlnr.block<3, 3>(0, 0).inverse();
        Vec3 t_rl = -R_rl*T_nlnr.block<3, 1>(0, 3);

        int Nec = track_ref->Nec; // # of edge center points
        int Npe = track_ref->Npe[lvl];

        Vec3* p_Xe = track_ref->Xe;  // 3-D point edge center points
        Vec3* p_grad_ek = track_ref->grad_e; // gradient.
        char* p_dir = track_ref->bin_e; // edge direction

        int* p_idx_l = track_ref->idx_l;
        int* p_idx_r = track_ref->idx_r;
        Node** p_node_l = track_ref->node_l;
        Node** p_node_r = track_ref->node_r;

        float* p_illumi_epat = track_ref->illumi_epat[lvl];

        Vec3* p_Xe_max = track_ref->Xe + Nec;

        float a11 = 0;
        float a12 = 0;
        float a22 = 0;
        float b1 = 0;
        float b2 = 0;
        for (; p_Xe < p_Xe_max; p_Xe++, p_dir++, p_idx_l++, p_idx_r++, p_node_l++, p_node_r++,
            p_illumi_epat += Npe) {
            // [1] warp and project edge centers onto the current level image.
            if ((*p_Xe)(2) < 0.01) continue;

            Vec3 X_warp_l = R_ck*(*p_Xe) + t_ck; // warped point onto left image.
            Vec3 X_warp_r = R_rl*X_warp_l + t_rl;// warped point onto right image.

            float invz = 1.0f / X_warp_l(2);
            float u_warp_l = fx*X_warp_l(0) * invz + cx;
            float v_warp_l = fy*X_warp_l(1) * invz + cy;
            invz = 1.0f / X_warp_r(2);
            float u_warp_r = fx*X_warp_r(0) * invz + cx;
            float v_warp_r = fy*X_warp_r(1) * invz + cy;

            // (1-a) matching left and right.
            if (is_cached) {
                tree->qts_left->searchNNCachedSingle(u_warp_l, v_warp_l, *p_dir, *p_idx_l, *p_node_l);
                tree->qts_right->searchNNCachedSingle(u_warp_r, v_warp_r, *p_dir, *p_idx_r, *p_node_r);
            }
            else {
                tree->qts_left->searchNNSingle(u_warp_l, v_warp_l, *p_dir, *p_idx_l, *p_node_l);
                tree->qts_right->searchNNSingle(u_warp_r, v_warp_r, *p_dir, *p_idx_r, *p_node_r);
            }
            
            // (1-b) if both images are matched, keep going. Otherwise, continue;
            if ((*p_idx_l < 0) || (*p_idx_r < 0)) continue;

            // [2] find matched gradients
            float gx_cl = frame_c->left->ft_edge[lvl]->grad_edge[*p_idx_l].x;
            float gy_cl = frame_c->left->ft_edge[lvl]->grad_edge[*p_idx_l].y;
            float igmag_cl = 1.0f / frame_c->left->ft_edge[lvl]->grad_edge[*p_idx_l].z;
            gx_cl *= igmag_cl;
            gy_cl *= igmag_cl;

            float gx_cr = frame_c->right->ft_edge[lvl]->grad_edge[*p_idx_r].x;
            float gy_cr = frame_c->right->ft_edge[lvl]->grad_edge[*p_idx_r].y;
            float igmag_cr = 1.0f / frame_c->right->ft_edge[lvl]->grad_edge[*p_idx_r].z;
            gx_cr *= igmag_cr;
            gy_cr *= igmag_cr;

            // (2-a) if both gradients render an angle over 3 degrees, continue;
            float dotprod = gx_cl*gx_cr + gy_cl*gy_cr;
            if (dotprod < params->qt.thres_gradtest) continue;

            // (2-b) if both patches are not similar, ignore it. 그리고 깊이가 갑자기 줄어드는경우, occlusion임.
            // 깊이가 줄어들었다는건 어떻게 판단하나? 아 stereo니까 matched pixel 이용해서 triangulate 가능!

            n_valid_edge += 2;

            // All tests are done!!!!
            float ul = frame_c->left->ft_edge[lvl]->pts_edge[*p_idx_l].x;
            float vl = frame_c->left->ft_edge[lvl]->pts_edge[*p_idx_l].y;
            float ur = frame_c->right->ft_edge[lvl]->pts_edge[*p_idx_r].x;
            float vr = frame_c->right->ft_edge[lvl]->pts_edge[*p_idx_r].y;

            // calculate and push edge residual, Huber weight, Jacobian, and Hessian (따로하자)
            float r_el = gx_cl*(u_warp_l - ul) + gy_cl*(v_warp_l - vl);
            float r_er = gx_cr*(u_warp_r - ur) + gy_cr*(v_warp_r - vr);
            float w_el = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);
            float w_er = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);
            err_e += r_el*r_el*w_el;
            err_e += r_er*r_er*w_er;

            if (w_el < 1.0f) ++n_overthres_point;
            if (w_er < 1.0f) ++n_overthres_point;

            // [2] consider edge centers first, and spontaneously fill out the edge patch terms. 
            vector<chk::Point2f>& patch_e = frame_c->patch_edge[lvl];
            for (int i = 0; i < Npe; i += 2) {
                // patch_e.y = 0 임.
                float uij = gx_cl*patch_e[i].x + ul;
                float vij = gy_cl*patch_e[i].x + vl;

                float Ik = *(p_illumi_epat + i);
                float Ic = improc::interpImageSingle(frame_c->left->img[lvl], uij, vij);
                // float r_ep = Ic - Ik;
                //cout << "edge Ic Ik: " << Ic << ", " << Ik << endl;
                //cout << "edge pt_c, pt_k_warp: " << u_warp_l << ", " << v_warp_l << "/ " << ul << "," << vl << endl;

                a11 += Ik*Ik;
                a12 += Ik;
                a22 += 1;
                b1 += Ik*Ic;
                b2 += Ic;
            }
        }
        // edge 부분에서 한번씩? 튀는 부분이 있는데 왜그런지 확인해보자.

        // [1-2] calculate prior \alpha and \beta values from edge patches. 
        float ea_hat = 1.0f;
        float beta_hat = 0.0f;
        if (n_valid_edge > 10) {
            float iden = 1.0f / (a11*a22 - a12*a12);
            ea_hat = (a22*b1 - a12*b2) * iden;
            beta_hat = (-a12*b1 + a11*b2) * iden;
            // illumination prior values.
            // cout << "ea, beta: " << ea_hat << ", " << beta_hat << "\n";
        }

        // [2] with the prior \alpha and \beta, fill out brigtness-compensated residual and Jacobian terms for point patches.
        int Np = track_ref->Np; // # of edge center points
        int Npp = track_ref->Npp[lvl];

        Vec3* p_Xppat = track_ref->Xppat[lvl];  // 3-D point edge center points
        float* p_illumi_ppat = track_ref->illumi_ppat[lvl];

        Vec3* p_Xppat_max = p_Xppat + Np*Npp;
        for (; p_Xppat < p_Xppat_max; p_Xppat++, p_illumi_ppat++) {
            // [1] warp and project a corner point onto the current level image.
            if ((*p_Xppat)(2) < 0.01) continue;

            Vec3 X_warp = R_ck*(*p_Xppat) + t_ck; // warped point onto left image.
            float xw = X_warp(0);
            float yw = X_warp(1);
            float izw = 1.0f / X_warp(2);

            float u_warp = fx*xw * izw + cx;
            float v_warp = fy*yw * izw + cy;

            // in image test
            if ((u_warp < 1) && (u_warp > n_cols - 1) && (v_warp < 1) && (v_warp > n_rows - 1)) continue;

            // calculate and push edge residual, Huber weight, Jacobian, and Hessian
            float Ic_warp = improc::interpImageSingle(frame_c->left->img[lvl], u_warp, v_warp);
            float dx_warp = improc::interpImageSingle(frame_c->left->du[lvl], u_warp, v_warp);
            float dy_warp = improc::interpImageSingle(frame_c->left->dv[lvl], u_warp, v_warp);

            float Ik = *p_illumi_ppat;
            float r_ppat = Ic_warp - Ik;
            float w_ppat = fabs(r_ppat) < huber_thres_ppat ? 1.0f : huber_thres_ppat / fabs(r_ppat);

            if (w_ppat < 1.0f) ++n_overthres_point;

            err_ppat += r_ppat*r_ppat*w_ppat;
            ++n_valid_pointpatch;
        }

        err_e /= n_valid_edge;
        err_ppat /= n_valid_pointpatch;

        err_e = sqrt(err_e);
        err_ppat = sqrt(err_ppat);
        err = err_e + err_ppat;
    };

    /**
    * Stereo Edge: Fill out Hessian (approximated), Jacobian, and residual vector.
    **/
    inline void SE3AffineBrightTracker::updateEDGE(const Vec8& Jt, const float& r, const float& weight) 
    {
        // cout << "Jt.transpose()*Jt:\n" << Jt*(Jt.transpose()) << endl;
        JtWJ_e.noalias() += (Jt*Jt.transpose())*weight;
        mJtWr_e.noalias() -= Jt*(r*weight);
        err_e += r*r*weight;
    }
    /**
    * Left point patch: Fill out Hessian (approximated), Jacobian, and residual vector.
    **/
    inline void SE3AffineBrightTracker::updatePPAT(const Vec8& Jt, const float& r, const float& weight)
    {
        // cout << "Jt.transpose()*Jt:\n" << Jt*(Jt.transpose()) << endl;
        JtWJ_ppat.noalias() += (Jt*Jt.transpose())*weight;
        mJtWr_ppat.noalias() -= Jt*(r*weight);
        err_ppat += r*r*weight;
    }
    /**
    * Fill out Hessian (approximated), Jacobian, and residual vector.
    **/
    inline void SE3AffineBrightTracker::updateSum()
    {
        // cout << "Jt.transpose()*Jt:\n" << Jt*(Jt.transpose()) << endl;
        JtWJ.noalias() += JtWJ_e + JtWJ_ppat;
        mJtWr.noalias() += mJtWr_e + mJtWr_ppat;
        err_e = sqrt(err_e);
        err_ppat = sqrt(err_ppat);
        err += err_e + err_ppat;
    }

    /**
    * Fill out Hessian (approximated), Jacobian, and residual vector.
    **/
    void SE3AffineBrightTracker::fillHJr(const Eigen::Matrix4d& T_ck_, const Vec2& ab_ck_, const int& lvl, const bool& is_cached,
        const float& huber_scaler_edge, const float& huber_scaler_point)
    {        
        if (track_ref == nullptr) throw std::runtime_error("TRACKER: track_ref is not linked\n");
        if (tree == nullptr) throw std::runtime_error("TRACKER: tree is not linked\n");
        if (frame_c == nullptr) throw std::runtime_error("TRACKER: frame_c is not linked\n");

        // initialize
        float huber_thres_edge = 1.5f*huber_scaler_edge;
        float huber_thres_ppat = 5.0f*huber_scaler_point;
        JtWJ.setZero();
        mJtWr.setZero();

        JtWJ_e.setZero();
        mJtWr_e.setZero();

        JtWJ_ppat.setZero();
        mJtWr_ppat.setZero();

        err = 0;
        err_e = 0;
        err_ppat = 0;
        err_soft = 0;

        n_valid_edge = 0;
        n_valid_pointpatch = 0;        

        float fx = K[lvl](0, 0); // intrinsic matrix of the current level lvl.
        float fy = K[lvl](1, 1);
        float cx = K[lvl](0, 2);
        float cy = K[lvl](1, 2);
        int n_cols = frame_c->left->n_cols_pyr[lvl];
        int n_rows = frame_c->left->n_rows_pyr[lvl];

   
        Mat44 T_ck;
        T_ck << T_ck_(0, 0), T_ck_(0, 1), T_ck_(0, 2), T_ck_(0, 3),
            T_ck_(1, 0), T_ck_(1, 1), T_ck_(1, 2), T_ck_(1, 3),
            T_ck_(2, 0), T_ck_(2, 1), T_ck_(2, 2), T_ck_(2, 3),
            T_ck_(3, 0), T_ck_(3, 1), T_ck_(3, 2), T_ck_(3, 3);

        Mat33 R_ck =  T_ck.block<3, 3>(0, 0); // rotation from key to cur.
        Vec3 t_ck  =  T_ck.block<3, 1>(0, 3); // translation from key to cur.
        Mat33 R_rl =  T_nlnr.block<3, 3>(0, 0).inverse();
        Vec3 t_rl  = -R_rl*T_nlnr.block<3, 1>(0, 3);

        int Nec         = track_ref->Nec; // # of edge center points
        int Npe         = track_ref->Npe[lvl];

        Vec3* p_Xe      = track_ref->Xe;  // 3-D point edge center points
        Vec3* p_grad_ek = track_ref->grad_e; // gradient.
        char* p_dir     = track_ref->bin_e; // edge direction

        int* p_idx_l    = track_ref->idx_l;
        int* p_idx_r    = track_ref->idx_r;
        Node** p_node_l = track_ref->node_l;
        Node** p_node_r = track_ref->node_r;

        float* p_illumi_epat = track_ref->illumi_epat[lvl];

        Vec3* p_Xe_max  = track_ref->Xe + Nec;

        // cout << "Nec: " << Nec << endl;
        float a11 = 0;
        float a12 = 0;
        float a22 = 0;
        float b1 = 0;
        float b2 = 0;

        for (; p_Xe < p_Xe_max; p_Xe++, p_dir++, p_idx_l++, p_idx_r++, p_node_l++, p_node_r++, 
            p_illumi_epat += Npe) {
            // [1] warp and project edge centers onto the current level image.
            if ((*p_Xe)(2) < 0.01) continue;

            Vec3 X_warp_l = R_ck*(*p_Xe) + t_ck; // warped point onto left image.
            Vec3 X_warp_r = R_rl*X_warp_l + t_rl;// warped point onto right image.

            float invz = 1.0f / X_warp_l(2);
            float u_warp_l = fx*X_warp_l(0) * invz + cx;
            float v_warp_l = fy*X_warp_l(1) * invz + cy;
            invz = 1.0f / X_warp_r(2);
            float u_warp_r = fx*X_warp_r(0) * invz + cx;
            float v_warp_r = fy*X_warp_r(1) * invz + cy;
            
            // (1-a) matching left and right.
            if (is_cached) {
                tree->qts_left->searchNNCachedSingle(u_warp_l, v_warp_l, *p_dir, *p_idx_l, *p_node_l);
                tree->qts_right->searchNNCachedSingle(u_warp_r, v_warp_r, *p_dir, *p_idx_r, *p_node_r);
            }
            else {
                tree->qts_left->searchNNSingle(u_warp_l, v_warp_l, *p_dir, *p_idx_l, *p_node_l);
                tree->qts_right->searchNNSingle(u_warp_r, v_warp_r, *p_dir, *p_idx_r, *p_node_r);
            }
            
            // (1-b) if both images are matched, keep going. Otherwise, continue;
            if ((*p_idx_l < 0) || (*p_idx_r < 0)) continue;

            // [2] find matched gradients
            float gx_cl   = frame_c->left->ft_edge[lvl]->grad_edge[*p_idx_l].x;
            float gy_cl   = frame_c->left->ft_edge[lvl]->grad_edge[*p_idx_l].y;
            float igmag_cl = 1.0f / frame_c->left->ft_edge[lvl]->grad_edge[*p_idx_l].z;
            gx_cl *= igmag_cl;
            gy_cl *= igmag_cl;

            float gx_cr   = frame_c->right->ft_edge[lvl]->grad_edge[*p_idx_r].x;
            float gy_cr   = frame_c->right->ft_edge[lvl]->grad_edge[*p_idx_r].y;
            float igmag_cr = 1.0f / frame_c->right->ft_edge[lvl]->grad_edge[*p_idx_r].z;
            gx_cr *= igmag_cr;
            gy_cr *= igmag_cr;

            // (2-a) if both gradients render an angle over 3 degrees, continue;
            float dotprod = gx_cl*gx_cr + gy_cl*gy_cr;
            if (dotprod < params->qt.thres_gradtest) continue;

            // (2-b) if both patches are not similar, ignore it. 그리고 깊이가 갑자기 줄어드는경우, occlusion임.
            // 깊이가 줄어들었다는건 어떻게 판단하나? 아 stereo니까 matched pixel 이용해서 triangulate 가능!

            n_valid_edge += 2;
            
            // All tests are done!!!!
            float ul = frame_c->left->ft_edge[lvl]->pts_edge[*p_idx_l].x;
            float vl = frame_c->left->ft_edge[lvl]->pts_edge[*p_idx_l].y;
            float ur = frame_c->right->ft_edge[lvl]->pts_edge[*p_idx_r].x;
            float vr = frame_c->right->ft_edge[lvl]->pts_edge[*p_idx_r].y;

            // calculate and push edge residual, Huber weight, Jacobian, and Hessian
            float r_el = gx_cl*(u_warp_l - ul) + gy_cl*(v_warp_l - vl);
            float r_er = gx_cr*(u_warp_r - ur) + gy_cr*(v_warp_r - vr);
            float w_el = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);
            float w_er = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);


            float xw = X_warp_l(0);
            float yw = X_warp_l(1);
            float xwyw = xw*yw;
            float izw = 1.0f / X_warp_l(2);
            float xizw = xw*izw;
            float yizw = yw*izw;
            float gxfx = fx*gx_cl;
            float gyfy = fy*gy_cl;

            Vec8 J;
            J(0) = gxfx*izw;
            J(1) = gyfy*izw;
            J(2) = (-gxfx*xizw - gyfy*yizw)*izw; // 여기가 틀렸었네 ; 
            J(3) = -gxfx*xizw*yizw - gyfy*(1.0f + yizw*yizw);
            J(4) = gxfx*(1.0f + xizw*xizw) + gyfy*xizw*yizw;
            J(5) = -gxfx*yizw + gyfy*xizw;
            J(6) = 0;
            J(7) = 0;
            updateEDGE(J, r_el, w_el);

            xw = X_warp_r(0);
            yw = X_warp_r(1);
            xwyw = xw*yw;
            izw  = 1.0f / X_warp_r(2);
            xizw = xw*izw;
            yizw = yw*izw;       
            gxfx = fx*gx_cr;
            gyfy = fy*gy_cr;

            J(0) = gxfx*izw;
            J(1) = gyfy*izw;
            J(2) = (-gxfx*xizw - gyfy*yizw)*izw;
            J(3) = -gxfx*xizw*yizw - gyfy*(1.0f + yizw*yizw);
            J(4) = gxfx*(1.0f + xizw*xizw) + gyfy*xizw*yizw;
            J(5) = -gxfx*yizw + gyfy*xizw;
            J(6) = 0;
            J(7) = 0;
            updateEDGE(J, r_er, w_er);

            // [2] consider edge centers first, and spontaneously fill out the edge patch terms. 
            vector<chk::Point2f>& patch_e = frame_c->patch_edge[lvl];
            for (int i = 0; i < Npe; i += 2) {
                // patch_e.y = 0 임.
                float uij = gx_cl*patch_e[i].x + ul;
                float vij = gy_cl*patch_e[i].x + vl;

                float Ik = *(p_illumi_epat + i);
                float Ic = improc::interpImageSingle(frame_c->left->img[lvl], uij, vij);
                // float r_ep = Ic - Ik;
                //cout << "edge Ic Ik: " << Ic << ", " << Ik << endl;
                //cout << "edge pt_c, pt_k_warp: " << u_warp_l << ", " << v_warp_l << "/ " << ul << "," << vl << endl;

                a11 += Ik*Ik;
                a12 += Ik;
                a22 += 1;
                b1 += Ik*Ic;
                b2 += Ic;
            }
        }

        // edge 부분에서 한번씩? 튀는 부분이 있는데 왜그런지 확인해보자.
        
        // [1-2] calculate prior \alpha and \beta values from edge patches. 
        float ea_hat   = 1.0f;
        float beta_hat = 0.0f;
        if (n_valid_edge > 10) {
            float iden = 1.0f / (a11*a22 - a12*a12);
            ea_hat = (a22*b1 - a12*b2) * iden;
            beta_hat = (-a12*b1 + a11*b2) * iden;
            // illumination prior values.
            // cout << "ea, beta: " << ea_hat << ", " << beta_hat << "\n";
        }

        // [2] with the prior \alpha and \beta, fill out brigtness-compensated residual and Jacobian terms for point patches.
        int Np  = track_ref->Np; // # of edge center points
        int Npp = track_ref->Npp[lvl];

        Vec3* p_Xppat        = track_ref->Xppat[lvl];  // 3-D point edge center points
        float* p_illumi_ppat = track_ref->illumi_ppat[lvl];

        Vec3* p_Xppat_max    = p_Xppat + Np*Npp;
        for (; p_Xppat < p_Xppat_max; p_Xppat++, p_illumi_ppat++) {
            // [1] warp and project a corner point onto the current level image.
            if ((*p_Xppat)(2) < 0.01) continue;

            Vec3 X_warp = R_ck*(*p_Xppat) + t_ck; // warped point onto left image.
            float xw = X_warp(0);
            float yw = X_warp(1);
            float izw = 1.0f / X_warp(2);

            float u_warp = fx*xw * izw + cx;
            float v_warp = fy*yw * izw + cy;

            // in image test
            if ((u_warp < 1) && (u_warp > n_cols - 1) && (v_warp < 1) && (v_warp > n_rows - 1)) continue;

            // calculate and push edge residual, Huber weight, Jacobian, and Hessian
            float Ic_warp = improc::interpImageSingle(frame_c->left->img[lvl], u_warp, v_warp);
            float dx_warp = improc::interpImageSingle(frame_c->left->du[lvl],  u_warp, v_warp);
            float dy_warp = improc::interpImageSingle(frame_c->left->dv[lvl],  u_warp, v_warp);

            float Ik     = *p_illumi_ppat;
            float r_ppat = Ic_warp - ab_ck_(0)*Ik - ab_ck_(1);
            float w_ppat = fabs(r_ppat) < huber_thres_ppat ? 1.0f : huber_thres_ppat / fabs(r_ppat);

            // cout << "Ik,Ic,r_p: " << Ik << "," << Ic_warp <<","<<r_ppat<< " / dudv: " << dx_warp << "," << dy_warp << endl;

            float xwyw = xw*yw;
            float xizw = xw*izw;
            float yizw = yw*izw;
            float dxfx = fx*dx_warp;
            float dyfy = fy*dy_warp;

            Vec8 J;
            J(0) = dxfx*izw;
            J(1) = dyfy*izw;
            J(2) = (-dxfx*xizw - dyfy*yizw)*izw;
            J(3) = -dxfx*xizw*yizw - dyfy*(1.0f + yizw*yizw);
            J(4) = dxfx*(1.0f + xizw*xizw) + dyfy*xizw*yizw;
            J(5) = -dxfx*yizw + dyfy*xizw;
            J(6) = -ab_ck_(0)*Ik;
            J(7) = -1.0f;

            updatePPAT(J, r_ppat, w_ppat);

            ++n_valid_pointpatch;
        }
        // Counts the number of edge center pixels which are in the image region.

        JtWJ_e  /= n_valid_edge;
        mJtWr_e /= n_valid_edge;
        err_e /= n_valid_edge;

        JtWJ_ppat /= n_valid_pointpatch;
        mJtWr_ppat /= n_valid_pointpatch;
        err_ppat /= n_valid_pointpatch;

        updateSum();
    };


    /**
    * Solve the H*delta_xi = b by using robust Cholesky decomposition.
    **/
    void SE3AffineBrightTracker::solveLMstep(Vec8& delta, const float& lambda) {
        JtWJ(0, 0) *= (1 + lambda);
        JtWJ(1, 1) *= (1 + lambda);
        JtWJ(2, 2) *= (1 + lambda);
        JtWJ(3, 3) *= (1 + lambda);
        JtWJ(4, 4) *= (1 + lambda);
        JtWJ(5, 5) *= (1 + lambda);
        JtWJ(6, 6) *= (1 + lambda);
        JtWJ(7, 7) *= (1 + lambda);
        delta = JtWJ.ldlt().solve(mJtWr);
    };

    SE3AffineBrightTracker::SE3AffineBrightTracker(
        Params* params_,
        const vector<int>& nrows_pyr, const vector<int>& ncols_pyr, const vector<Eigen::Matrix3d>& K_pyr_, const Eigen::Matrix4d& T_nlnr_) {
        // Dynamic allocations for buffers. (Sufficiently!!)
           
        params = params_;
        MAX_PYR_LVL = params->pyr.max_lvl;

        for (int lvl = 0; lvl < MAX_PYR_LVL; lvl++) {
            K[lvl] << K_pyr_[lvl](0, 0), K_pyr_[lvl](0, 1), K_pyr_[lvl](0, 2),
                K_pyr_[lvl](1, 0), K_pyr_[lvl](1, 1), K_pyr_[lvl](1, 2), 
                K_pyr_[lvl](2, 0), K_pyr_[lvl](2, 1), K_pyr_[lvl](2, 2);
            Kinv[lvl] = K[lvl].inverse();
        }

        T_nlnr << T_nlnr_(0, 0), T_nlnr_(0, 1), T_nlnr_(0, 2), T_nlnr_(0,3),
            T_nlnr_(1, 0), T_nlnr_(1, 1), T_nlnr_(1, 2), T_nlnr_(1, 3),
            T_nlnr_(2, 0), T_nlnr_(2, 1), T_nlnr_(2, 2), T_nlnr_(2, 3),
            0.0f, 0.0f, 0.0f, 1.0f;

        int w = ncols_pyr[0];
        int h = nrows_pyr[0];

        buf_residual_e    = (float*)custom_aligned_malloc(sizeof(float)*w*h); // r_e
        buf_residual_ppat = (float*)custom_aligned_malloc(sizeof(float)*w*h); // r_p
        buf_residual_epat = (float*)custom_aligned_malloc(sizeof(float)*w*h); // r_ppat

        buf_dx_ppat = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_dy_ppat = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_dx_epat = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_dy_epat = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_xwarp = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_ywarp = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_zwarp = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_weight_e    = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_weight_ppat = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_weight_epat = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        // initialize pointers
        detachRef();
        detachTree();
        detachFrameCur();
    };


    SE3AffineBrightTracker::~SE3AffineBrightTracker() {
        custom_aligned_free((void*)buf_residual_e); // r_e
        custom_aligned_free((void*)buf_residual_ppat); // r_p
        custom_aligned_free((void*)buf_residual_epat); // r_ppat

        custom_aligned_free((void*)buf_dx_ppat);
        custom_aligned_free((void*)buf_dy_ppat);
        custom_aligned_free((void*)buf_dx_epat);
        custom_aligned_free((void*)buf_dy_epat);

        custom_aligned_free((void*)buf_xwarp);
        custom_aligned_free((void*)buf_ywarp);
        custom_aligned_free((void*)buf_zwarp);

        custom_aligned_free((void*)buf_weight_e);
        custom_aligned_free((void*)buf_weight_ppat);
        custom_aligned_free((void*)buf_weight_epat);
        cout << "    SE3AffineBrightTracker tracker is deleted.\n";
    };   
};



#endif