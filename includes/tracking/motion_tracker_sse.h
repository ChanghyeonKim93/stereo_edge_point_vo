#ifndef _MOTIONTRACKER_SSE_H_
#define _MOTIONTRACKER_SSE_H_

#include <iostream>
#include <cstdlib>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv.hpp"  
#include "Eigen/Dense"
#include "immintrin.h"

#include "../frame/stereo_frames.h"
#include "../tracking/tracking_reference.h"
#include "../quadtrees/stereo_multiquadtrees_wrapper.h"

#define GET4(ptr,i,j) (*(ptr + 4 * i + j))
#define SUM4i(ptr,i) (*(ptr + 4*i) + *(ptr + 4*i + 1) + *(ptr + 4*i + 2) + *(ptr + 4*i + 3))
#define SUM4(ptr) (*(ptr) + *(ptr + 1) + *(ptr + 2) + *(ptr + 3))

using namespace std;
namespace chk {
    // Tracking part


    /* SE3 6-DoF motion + 2-DoF Affine brightness parameters*/
    class SE3AffineBrightTrackerSSE {
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

        float err; // err = err_e + err_ppat
        float err_e;
        float err_ppat;

        int n_valid_edge;
        int n_valid_pointpatch;

        int n_overthres_edge;
        int n_overthres_point;

        // n_overthres / n_valid < 0.6 =  huber thres is is good.
        Params* params;
        StereoFrame* frame_c;
        chk::TrackingReference* track_ref;
        StereoMultiQuadtreesWrapper* tree;

       
        SE3AffineBrightTrackerSSE(Params* params_,
            const vector<int>& nrows_pyr, const vector<int>& ncols_pyr,
            const vector<Eigen::Matrix3d>& K_pyr_, const Eigen::Matrix4d& T_nlnr_);
        ~SE3AffineBrightTrackerSSE();

        // linking & detaching functions 
        void linkRef(chk::TrackingReference* track_ref_) { track_ref = track_ref_; };
        void linkTree(StereoMultiQuadtreesWrapper* tree_) { tree = tree_; };
        void linkFrameCur(StereoFrame* frame_) { frame_c = frame_; };
        void detachRef() { track_ref = nullptr; };
        void detachTree() { tree = nullptr; };
        void detachFrameCur() { frame_c = nullptr; };

        void calcResidual(const Eigen::Matrix4d& T_ck_, const Vec2& ab_ck_, const int& lvl, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point);
        void se3affineMotionTracking(const Eigen::Matrix4d& T_ck_, Vec2& ab_ck_, const int& lvl, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point, Vec8& delta, const float& lambda);
    
    private:
        // Hessian and Jacobian
        // We just solve Linear system JtWJ*delta_xi = mJtWr; where mJtWr = -J^t*W*r;
        // JtWJ matrix is guaranteed to be P.S.D and symmetric matrix.
        // Thus, we can efficiently solve this equation by Cholesky decomposition.
        // --> JtWJ.ldlt().solve(mJtWr);
        Mat88 JtWJ; // 8 x 8, = JtWJ_e + JtWJ_ppat
        Vec8 mJtWr; // 8 x 1, = mJtWr_e + mJtWr_ppat

        Mat88 JtWJ_e; // 8 x 8
        Vec8 mJtWr_e; // 8 x 1

        Mat88 JtWJ_ppat; // 8 x 8
        Vec8 mJtWr_ppat; // 8 x 1

        // level과 무관하게 하나씩만 가지고 있으면 된다.
        // For se3affinetracker_sse 
        // (1) edge references.
        float* buf_xel_warped; // warped edge reference to left current image.
        float* buf_yel_warped;
        float* buf_zel_warped;
        float* buf_xer_warped; // warped edge reference to right current image.
        float* buf_yer_warped;
        float* buf_zer_warped;

        float* buf_uel_warped; // warped and reprojected edge on left current image.
        float* buf_vel_warped;
        float* buf_uer_warped; // warped and reprojected edge on right current image.
        float* buf_ver_warped;

        float* buf_uel_matched; // matched edge on left current image.
        float* buf_vel_matched;
        float* buf_uer_matched; // matched edge on right current image.
        float* buf_ver_matched;

        float* buf_gxl_matched; // gradient on the matched left pixels.
        float* buf_gyl_matched;
        float* buf_gxr_matched; // gradient on the matched right pixels.
        float* buf_gyr_matched;

        float* buf_res_l_e; // r_e
        float* buf_weight_l_e;
        float* buf_res_r_e;
        float* buf_weight_r_e;

        // (2) point patches
        float* buf_xppat_warped;
        float* buf_yppat_warped;
        float* buf_zppat_warped;

        float* buf_Ik_ppat;
        float* buf_Ic_ppat_warped;
        float* buf_duc_ppat_warped;
        float* buf_dvc_ppat_warped;

        float* buf_res_ppat; // r_p
        float* buf_weight_ppat;

        float* SSEData; // [4 * 45 ] (for make JtWJ and JtWr)


        void warpAndMatchAndCheckAndAffineBrightAndGenerateEdgeBuffers(const Eigen::Matrix4d& T_ck_, const int& lvl, const bool& is_cached, float& alpha, float& beta);
        void calcEdgeResidualAndWeightSSE(const float& huber_scaler_edge);
        void calcEdgeHessianAndJacobianSSE(const int& lvl);

        void warpPointPatchAndGeneratePointPatchBuffers(const Eigen::Matrix4d& T_ck_, const int& lvl);
        void calcPointPatchResidualAndWeightSSE(const float& huber_scaler_point, const float& alpha, const float& beta);
        void calcPointPatchHessianAndJacobianSSE(const int& lvl, const float& alpha, const float& beta);

        void updateSSE(
            const __m128 &J1, const __m128 &J2, const __m128 &J3, const __m128 &J4, const __m128 &J5, const __m128 &J6, const __m128 &J7, const __m128 &J8,
            const __m128& res, const __m128& weight, Mat88& JtWJ, Vec8& mJtWr, float& err);

        void solveLevenbergMarquardtStepSSE(Vec8& delta, const float& lambda);
    };
    void SE3AffineBrightTrackerSSE::se3affineMotionTracking(const Eigen::Matrix4d& T_ck_, Vec2& ab_ck_, const int& lvl, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point, Vec8& delta, const float& lambda) {
        warpAndMatchAndCheckAndAffineBrightAndGenerateEdgeBuffers(T_ck_, lvl, is_cached, ab_ck_(0), ab_ck_(1));
        calcEdgeResidualAndWeightSSE(huber_scaler_edge);
        calcEdgeHessianAndJacobianSSE(lvl);

        warpPointPatchAndGeneratePointPatchBuffers(T_ck_, lvl);
        calcPointPatchResidualAndWeightSSE(huber_scaler_point, ab_ck_(0), ab_ck_(1));
        calcPointPatchHessianAndJacobianSSE(lvl, ab_ck_(0), ab_ck_(1));

        solveLevenbergMarquardtStepSSE(delta, lambda);
    };

    void SE3AffineBrightTrackerSSE::calcResidual(
        const Eigen::Matrix4d& T_ck_, const Vec2& ab_ck_, const int& lvl, const bool& is_cached, const float& huber_scaler_edge, const float& huber_scaler_point)
    {
        // 이 함수에서 exp(alpha), beta 를 먼저 추정하고, Huber norm을 조절한다.
        if (track_ref == nullptr) throw std::runtime_error("TRACKER: track_ref is not linked\n");
        if (tree == nullptr) throw std::runtime_error("TRACKER: tree is not linked\n");
        if (frame_c == nullptr) throw std::runtime_error("TRACKER: frame_c is not linked\n");

        // initialize
        float huber_thres_edge = 1.5f*huber_scaler_edge;
        float huber_thres_ppat = 5.0f*huber_scaler_point;

        err = 0;
        err_e = 0;
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
        }
        // edge 부분에서 한번씩? 튀는 부분이 있는데 왜그런지 확인해보자.
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
            float Ik = *p_illumi_ppat;
            float r_ppat = Ic_warp - ab_ck_(0)*Ik - ab_ck_(1);
            float w_ppat = fabs(r_ppat) < huber_thres_ppat ? 1.0f : huber_thres_ppat / fabs(r_ppat);

            if (w_ppat < 1.0f) ++n_overthres_point;

            err_ppat += r_ppat*r_ppat*w_ppat;
            ++n_valid_pointpatch;
        }

        err_e /= (float)n_valid_edge;
        err_ppat /= (float)n_valid_pointpatch;

        err_e = sqrt(err_e);
        err_ppat = sqrt(err_ppat);
        err = err_e + err_ppat;
    };

    void SE3AffineBrightTrackerSSE::warpAndMatchAndCheckAndAffineBrightAndGenerateEdgeBuffers(const Eigen::Matrix4d& T_ck_, const int& lvl, const bool& is_cached, float& alpha, float& beta)
    {
        // update buf_{x,y,z}el_warped, buf_{x,y,z}er_warped, buf_{u,v}el_warped, buf_{u,v}er_warped,
        // buf_g{x,y}l_matched, buf_g{x,y}r_matched

        // initialize for edge
        n_valid_edge = 0; // same for matched left edge centers.

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
        for (;p_Xe < p_Xe_max; p_Xe++, p_dir++, p_idx_l++, p_idx_r++, p_node_l++, p_node_r++, 
            p_illumi_epat += Npe)
        {
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
            float uel_matched = frame_c->left->ft_edge[lvl]->pts_edge[*p_idx_l].x;
            float vel_matched = frame_c->left->ft_edge[lvl]->pts_edge[*p_idx_l].y;
            float uer_matched = frame_c->right->ft_edge[lvl]->pts_edge[*p_idx_r].x;
            float ver_matched = frame_c->right->ft_edge[lvl]->pts_edge[*p_idx_r].y;

            // calculate affine brightness parameters
            // [2] consider edge centers first, and spontaneously fill out the edge patch terms. 
            vector<chk::Point2f>& patch_e = frame_c->patch_edge[lvl];
            for (int i = 0; i < Npe; i += 2) {
                // patch_e.y = 0 임.
                float uij = gx_cl*patch_e[i].x + uel_matched; // gy_cl*patch_e[i].y == 0 because of patch_e[i].y == 0.
                float vij = gy_cl*patch_e[i].x + vel_matched;

                float Ik = *(p_illumi_epat + i);
                float Ic = improc::interpImageSingle(frame_c->left->img[lvl], uij, vij);
                // float r_ep = Ic - Ik;

                a11 += Ik*Ik;
                a12 += Ik;
                a22 += 1;
                b1 += Ik*Ic;
                b2 += Ic;
            }
           
            // update buffers
            *(buf_xel_warped + n_valid_edge) = X_warp_l(0);
            *(buf_yel_warped + n_valid_edge) = X_warp_l(1);
            *(buf_zel_warped + n_valid_edge) = X_warp_l(2);
            *(buf_uel_warped + n_valid_edge) = u_warp_l;
            *(buf_vel_warped + n_valid_edge) = v_warp_l;
            *(buf_uel_matched + n_valid_edge) = uel_matched;
            *(buf_vel_matched + n_valid_edge) = vel_matched;
            *(buf_gxl_matched + n_valid_edge) = gx_cl;
            *(buf_gyl_matched + n_valid_edge) = gy_cl;

            *(buf_xer_warped + n_valid_edge) = X_warp_r(0);
            *(buf_yer_warped + n_valid_edge) = X_warp_r(1);
            *(buf_zer_warped + n_valid_edge) = X_warp_r(2);
            *(buf_uer_warped + n_valid_edge) = u_warp_r;
            *(buf_ver_warped + n_valid_edge) = v_warp_r;
            *(buf_uer_matched + n_valid_edge) = uer_matched;
            *(buf_ver_matched + n_valid_edge) = ver_matched;
            *(buf_gxr_matched + n_valid_edge) = gx_cr;
            *(buf_gyr_matched + n_valid_edge) = gy_cr;

            ++n_valid_edge;
        }
        if (!is_cached) {
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
            alpha = ea_hat;
            beta  = beta_hat;
        }
    };

    void SE3AffineBrightTrackerSSE::calcEdgeResidualAndWeightSSE(const float& huber_scaler_edge) 
    {
        n_overthres_edge = 0;

        // SSE version!
        float huber_thres_edge = 1.5f*huber_scaler_edge;
    
        // fourth multiplier? 
        int n_steps = n_valid_edge / 4;
        int n_remainders = n_valid_edge % 4;

        // left first!
        __m128 u_warp4, v_warp4, u_match4, v_match4, gx4, gy4;

        // cout << "n_step:" << n_steps << ", n_remainder:" << n_remainders << endl;
        int idx = 0;
        for (; idx < n_valid_edge - 3; idx += 4) {
            u_warp4 = _mm_load_ps(buf_uel_warped + idx);
            v_warp4 = _mm_load_ps(buf_vel_warped + idx);
            u_match4 = _mm_load_ps(buf_uel_matched + idx);
            v_match4 = _mm_load_ps(buf_vel_matched + idx);
            gx4 = _mm_load_ps(buf_gxl_matched + idx);
            gy4 = _mm_load_ps(buf_gyl_matched + idx);

            // calculate residual left edge
            __m128 rrrr = _mm_add_ps(_mm_mul_ps(gx4, _mm_sub_ps(u_warp4, u_match4)), _mm_mul_ps(gy4, _mm_sub_ps(v_warp4, v_match4)));
            _mm_store_ps(buf_res_l_e + idx, rrrr);

            // calculate weight left edge (Huber weight)
            float r_el = *(buf_res_l_e + idx);
            *(buf_weight_l_e + idx) = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);
            if(*(buf_weight_l_e + idx) < 1.0f) ++n_overthres_edge;

            r_el = *(buf_res_l_e + idx + 1);
            *(buf_weight_l_e + idx + 1) = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);
            if (*(buf_weight_l_e + idx + 1) < 1.0f) ++n_overthres_edge;

            r_el = *(buf_res_l_e + idx + 2);
            *(buf_weight_l_e + idx + 2) = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);
            if (*(buf_weight_l_e + idx + 2) < 1.0f) ++n_overthres_edge;

            r_el = *(buf_res_l_e + idx + 3);
            *(buf_weight_l_e + idx + 3) = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);
            if (*(buf_weight_l_e + idx + 3) < 1.0f) ++n_overthres_edge;
        };
        for (; idx < n_valid_edge; ++idx) {  
            float u_warp_l = *(buf_uel_warped + idx);
            float v_warp_l = *(buf_vel_warped + idx);
            float gx_cl = *(buf_gxl_matched + idx);
            float gy_cl = *(buf_gyl_matched + idx);

            float ul = *(buf_uel_matched + idx);
            float vl = *(buf_vel_matched + idx);

            // calculate and push edge residual, Huber weight, Jacobian, and Hessian (따로하자)
            float r_el = gx_cl*(u_warp_l - ul) + gy_cl*(v_warp_l - vl);
            float w_el = fabs(r_el) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_el);

            *(buf_res_l_e + idx) = r_el;
            *(buf_weight_l_e + idx) = w_el;

            if (w_el < 1.0f) ++n_overthres_edge;
        };


        //right

        idx = 0;
        for (; idx < n_valid_edge - 3; idx += 4) {
            u_warp4 = _mm_load_ps(buf_uer_warped + idx);
            v_warp4 = _mm_load_ps(buf_ver_warped + idx);
            u_match4 = _mm_load_ps(buf_uer_matched + idx);
            v_match4 = _mm_load_ps(buf_ver_matched + idx);
            gx4 = _mm_load_ps(buf_gxr_matched + idx);
            gy4 = _mm_load_ps(buf_gyr_matched + idx);

            // calculate residual left edge
            __m128 rrrr = _mm_add_ps(_mm_mul_ps(gx4, _mm_sub_ps(u_warp4, u_match4)), _mm_mul_ps(gy4, _mm_sub_ps(v_warp4, v_match4)));
            _mm_store_ps(buf_res_r_e + idx, rrrr);

            // calculate weight left edge (Huber weight)
            float r_er = *(buf_res_r_e + idx);
            *(buf_weight_r_e + idx) = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);

            r_er = *(buf_res_r_e + idx + 1);
            *(buf_weight_r_e + idx + 1) = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);

            r_er = *(buf_res_r_e + idx + 2);
            *(buf_weight_r_e + idx + 2) = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);

            r_er = *(buf_res_r_e + idx + 3);
            *(buf_weight_r_e + idx + 3) = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);
        };
        for (; idx < n_valid_edge; ++idx) {
            float u_warp_r = *(buf_uer_warped + idx);
            float v_warp_r = *(buf_ver_warped + idx);
            float gx_cr = *(buf_gxr_matched + idx);
            float gy_cr = *(buf_gyr_matched + idx);

            float ur = *(buf_uer_matched + idx);
            float vr = *(buf_ver_matched + idx);

            // calculate and push edge residual, Huber weight, Jacobian, and Hessian (따로하자)
            float r_er = gx_cr*(u_warp_r - ur) + gy_cr*(v_warp_r - vr);
            float w_er = fabs(r_er) < huber_thres_edge ? 1.0f : huber_thres_edge / fabs(r_er);

            *(buf_res_r_e + idx) = r_er;
            *(buf_weight_r_e + idx) = w_er;
        };

    };

    void SE3AffineBrightTrackerSSE::calcEdgeHessianAndJacobianSSE(const int& lvl) {
        JtWJ_e.setZero();
        mJtWr_e.setZero();
        err_e = 0;

        __m128 ones = _mm_set1_ps(1.0f);
        __m128 zeros = _mm_set1_ps(0.0f);
        __m128 minusones = _mm_set1_ps(-1.0f);

        __m128 J1, J2, J3, J4, J5, J6, J7, J8; // four data at once.        
        __m128 fx4 = _mm_set1_ps(K[lvl](0, 0));
        __m128 fy4 = _mm_set1_ps(K[lvl](1, 1));
        J7 = zeros;
        J8 = zeros;

        int idx = 0;
        for (; idx < n_valid_edge - 3; idx += 4) {
            __m128 x4 = _mm_load_ps(buf_xel_warped + idx);
            __m128 y4 = _mm_load_ps(buf_yel_warped + idx);
             __m128 invz4 = _mm_rcp_ps(_mm_load_ps(buf_zel_warped + idx));
            //__m128 invz4 = _mm_div_ps(ones, _mm_load_ps(buf_zel_warped + idx));

            __m128 xy4 = _mm_mul_ps(x4, y4);
            __m128 xinvz4 = _mm_mul_ps(x4, invz4);
            __m128 yinvz4 = _mm_mul_ps(y4, invz4);

            __m128 gxfx4 = _mm_mul_ps(_mm_load_ps(buf_gxl_matched + idx), fx4);
            __m128 gyfy4 = _mm_mul_ps(_mm_load_ps(buf_gyl_matched + idx), fy4);

            // J(1) = gx*fx/z ;
            J1 = _mm_mul_ps(gxfx4, invz4);
            // J(2) = gy*fy/z ;
            J2 = _mm_mul_ps(gyfy4, invz4);
            // J(3) = (-gx*fx*x/z - gy*fy*y/z)/z;
            J3 = _mm_mul_ps(_mm_sub_ps(zeros, _mm_add_ps(_mm_mul_ps(gxfx4,xinvz4), _mm_mul_ps(gyfy4,yinvz4))), invz4);
            // J(4) = -( gxfx*xizw*yizw + gyfy*(1.0f + yizw*yizw) );
            J4 = _mm_sub_ps(zeros, 
                _mm_add_ps(_mm_mul_ps(_mm_mul_ps(gxfx4,xinvz4),yinvz4), _mm_mul_ps(gyfy4, _mm_add_ps(ones, _mm_mul_ps(yinvz4,yinvz4))))
                );
            // J(5) = gxfx*(1.0f + xizw*xizw) + gyfy*xizw*yizw;
            J5 = _mm_add_ps(_mm_mul_ps(gxfx4, _mm_add_ps(ones,_mm_mul_ps(xinvz4,xinvz4))), _mm_mul_ps(gyfy4 , _mm_mul_ps(xinvz4,yinvz4)));
            // J(6) = gyfy*xizw - gxfx*yizw ;
            J6 = _mm_sub_ps(_mm_mul_ps(gyfy4, xinvz4), _mm_mul_ps(gxfx4, yinvz4));
            // J(7) = zeros; 
            // J(8) = zeros;

            // update
            updateSSE(J1, J2, J3, J4, J5, J6, J7, J8, _mm_load_ps(buf_res_l_e + idx), _mm_load_ps(buf_weight_l_e + idx), JtWJ_e, mJtWr_e, err_e);
        };
        for (; idx < n_valid_edge; ++idx) {
            float x = *(buf_xel_warped + idx);
            float y = *(buf_yel_warped + idx);
            float iz = 1.0f / (*(buf_zel_warped + idx));

            float xy = x*y;
            float xiz = x*iz;
            float yiz = y*iz;

            float gxfx = *(buf_gxl_matched + idx)*K[lvl](0, 0);
            float gyfy = *(buf_gyl_matched + idx)*K[lvl](1, 1);

            Vec8 J;
            J(0) = gxfx*iz;
            J(1) = gyfy*iz;
            J(2) = (-gxfx*xiz - gyfy*yiz)*iz; // 여기가 틀렸었네 ; 
            J(3) = -gxfx*xiz*yiz - gyfy*(1.0f + yiz*yiz);
            J(4) = gxfx*(1.0f + xiz*xiz) + gyfy*xiz*yiz;
            J(5) = -gxfx*yiz + gyfy*xiz;
            J(6) = 0;
            J(7) = 0;

            float r = *(buf_res_l_e + idx);
            float w = *(buf_weight_l_e + idx);
            JtWJ_e.noalias() += (J*J.transpose())*w;
            mJtWr_e.noalias() -= J*(r*w);
            err_e += r*r*w;
        };

        // right
        idx = 0;
        for (; idx < n_valid_edge - 3; idx += 4) {
            __m128 x4 = _mm_load_ps(buf_xer_warped + idx);
            __m128 y4 = _mm_load_ps(buf_yer_warped + idx);
             __m128 invz4 = _mm_rcp_ps(_mm_load_ps(buf_zer_warped + idx));
            //__m128 invz4 = _mm_div_ps(ones, _mm_load_ps(buf_zer_warped + idx));

            __m128 xy4 = _mm_mul_ps(x4, y4);
            __m128 xinvz4 = _mm_mul_ps(x4, invz4);
            __m128 yinvz4 = _mm_mul_ps(y4, invz4);

            __m128 gxfx4 = _mm_mul_ps(_mm_load_ps(buf_gxr_matched + idx), fx4);
            __m128 gyfy4 = _mm_mul_ps(_mm_load_ps(buf_gyr_matched + idx), fy4);

            // J(1) = gx*fx/z ;
            J1 = _mm_mul_ps(gxfx4, invz4);
            // J(2) = gy*fy/z ;
            J2 = _mm_mul_ps(gyfy4, invz4);
            // J(3) = (-gx*fx*x/z - gy*fy*y/z)/z;
            J3 = _mm_mul_ps(_mm_sub_ps(zeros, _mm_add_ps(_mm_mul_ps(gxfx4, xinvz4), _mm_mul_ps(gyfy4, yinvz4))), invz4);
            // J(4) = -( gxfx*xizw*yizw + gyfy*(1.0f + yizw*yizw) );
            J4 = _mm_sub_ps(zeros,
                _mm_add_ps(_mm_mul_ps(_mm_mul_ps(gxfx4, xinvz4), yinvz4), _mm_mul_ps(gyfy4, _mm_add_ps(ones, _mm_mul_ps(yinvz4, yinvz4))))
            );
            // J(5) = gxfx*(1.0f + xizw*xizw) + gyfy*xizw*yizw;
            J5 = _mm_add_ps(_mm_mul_ps(gxfx4, _mm_add_ps(ones, _mm_mul_ps(xinvz4, xinvz4))), _mm_mul_ps(gyfy4, _mm_mul_ps(xinvz4, yinvz4)));
            // J(6) = gyfy*xizw - gxfx*yizw ;
            J6 = _mm_sub_ps(_mm_mul_ps(gyfy4, xinvz4), _mm_mul_ps(gxfx4, yinvz4));
            // J(7) = zeros; 
            // J(8) = zeros;

            // update
            updateSSE(J1, J2, J3, J4, J5, J6, J7, J8, _mm_load_ps(buf_res_r_e + idx), _mm_load_ps(buf_weight_r_e + idx), JtWJ_e, mJtWr_e, err_e);
        };
        for (; idx < n_valid_edge; ++idx) {
            float x = *(buf_xer_warped + idx);
            float y = *(buf_yer_warped + idx);
            float iz = 1.0f / (*(buf_zer_warped + idx));

            float xy = x*y;
            float xiz = x*iz;
            float yiz = y*iz;

            float gxfx = *(buf_gxr_matched + idx)*K[lvl](0, 0);
            float gyfy = *(buf_gyr_matched + idx)*K[lvl](1, 1);

            Vec8 J;
            J(0) = gxfx*iz;
            J(1) = gyfy*iz;
            J(2) = (-gxfx*xiz - gyfy*yiz)*iz; // 여기가 틀렸었네 ; 
            J(3) = -gxfx*xiz*yiz - gyfy*(1.0f + yiz*yiz);
            J(4) = gxfx*(1.0f + xiz*xiz) + gyfy*xiz*yiz;
            J(5) = -gxfx*yiz + gyfy*xiz;
            J(6) = 0;
            J(7) = 0;

            float r = *(buf_res_r_e + idx);
            float w = *(buf_weight_r_e + idx);
            JtWJ_e.noalias() += (J*J.transpose())*w;
            mJtWr_e.noalias() -= J*(r*w);
            err_e += r*r*w;
        };
    };

    void SE3AffineBrightTrackerSSE::updateSSE(
        const __m128 &J1, const __m128 &J2, const __m128 &J3, const __m128 &J4, const __m128 &J5, const __m128 &J6, const __m128 &J7, const __m128 &J8,
        const __m128& res, const __m128& weight, Mat88& JtWJ, Vec8& mJtWr, float& err) 
    {
        //A.noalias() += J * J.transpose() * weight;
        float* p_SSEData = SSEData;
        memset(p_SSEData, 0.0f, sizeof(float) * 4 * 45); // Jacobian 계산은 정상인 것 같은데,,,

        __m128 J1w = _mm_mul_ps(J1, weight);
        _mm_store_ps(p_SSEData, _mm_mul_ps(J1w, J1)); // 걍 이러면 되는거아닌가? 
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J1w, J2)); 
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J1w, J3)); 
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J1w, J4)); 
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J1w, J5)); 
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J1w, J6)); 
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J1w, J7)); 
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J1w, J8)); 
    
        __m128 J2w = _mm_mul_ps(J2, weight);
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J2w, J2));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J2w, J3));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J2w, J4));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J2w, J5));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J2w, J6));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J2w, J7));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J2w, J8));


        __m128 J3w = _mm_mul_ps(J3, weight);
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J3w, J3));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J3w, J4));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J3w, J5));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J3w, J6));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J3w, J7));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J3w, J8));

        __m128 J4w = _mm_mul_ps(J4, weight);
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J4w, J4));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J4w, J5));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J4w, J6));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J4w, J7));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J4w, J8));

        __m128 J5w = _mm_mul_ps(J5, weight);
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J5w, J5));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J5w, J6));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J5w, J7));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J5w, J8));

        __m128 J6w = _mm_mul_ps(J6, weight);
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J6w, J6));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J6w, J7));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J6w, J8));

        __m128 J7w = _mm_mul_ps(J7, weight);
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J7w, J7));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J7w, J8));

        __m128 J8w = _mm_mul_ps(J8, weight);
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(J8w, J8));

        //b.noalias() -= J * (res * weight);
        __m128 resw = _mm_mul_ps(res, weight);
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(resw, J1));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(resw, J2));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(resw, J3));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(resw, J4));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(resw, J5));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(resw, J6));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(resw, J7));
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(resw, J8));

        // error += res*res*weight;
        _mm_store_ps((p_SSEData += 4), _mm_mul_ps(resw, res));

        p_SSEData = SSEData;
        // update JtWJ_sse,
        JtWJ(0, 0) += (SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(1, 0) = (JtWJ(0, 1) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(2, 0) = (JtWJ(0, 2) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(3, 0) = (JtWJ(0, 3) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(4, 0) = (JtWJ(0, 4) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(5, 0) = (JtWJ(0, 5) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(6, 0) = (JtWJ(0, 6) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(7, 0) = (JtWJ(0, 7) += SUM4(p_SSEData)); p_SSEData += 4;
            
        JtWJ(1, 1) += (SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(2, 1) = (JtWJ(1, 2) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(3, 1) = (JtWJ(1, 3) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(4, 1) = (JtWJ(1, 4) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(5, 1) = (JtWJ(1, 5) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(6, 1) = (JtWJ(1, 6) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(7, 1) = (JtWJ(1, 7) += SUM4(p_SSEData)); p_SSEData += 4;
            
        JtWJ(2, 2) += (SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(3, 2) = (JtWJ(2, 3) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(4, 2) = (JtWJ(2, 4) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(5, 2) = (JtWJ(2, 5) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(6, 2) = (JtWJ(2, 6) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(7, 2) = (JtWJ(2, 7) += SUM4(p_SSEData)); p_SSEData += 4;
            
        JtWJ(3, 3) += (SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(4, 3) = (JtWJ(3, 4) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(5, 3) = (JtWJ(3, 5) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(6, 3) = (JtWJ(3, 6) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(7, 3) = (JtWJ(3, 7) += SUM4(p_SSEData)); p_SSEData += 4;
            
        JtWJ(4, 4) += (SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(5, 4) = (JtWJ(4, 5) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(6, 4) = (JtWJ(4, 6) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(7, 4) = (JtWJ(4, 7) += SUM4(p_SSEData)); p_SSEData += 4;
            
        JtWJ(5, 5) += (SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(6, 5) = (JtWJ(5, 6) += SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(7, 5) = (JtWJ(5, 7) += SUM4(p_SSEData)); p_SSEData += 4;
            
        JtWJ(6, 6) += (SUM4(p_SSEData)); p_SSEData += 4;
        JtWJ(7, 6) = (JtWJ(6, 7) += SUM4(p_SSEData)); p_SSEData += 4;
            
        JtWJ(7, 7) += (SUM4(p_SSEData)); p_SSEData += 4;

        // update  mJtWr_sse
        mJtWr(0) -= (SUM4(p_SSEData)); p_SSEData += 4;
        mJtWr(1) -= (SUM4(p_SSEData)); p_SSEData += 4;
        mJtWr(2) -= (SUM4(p_SSEData)); p_SSEData += 4;
        mJtWr(3) -= (SUM4(p_SSEData)); p_SSEData += 4;
        mJtWr(4) -= (SUM4(p_SSEData)); p_SSEData += 4;
        mJtWr(5) -= (SUM4(p_SSEData)); p_SSEData += 4;
        mJtWr(6) -= (SUM4(p_SSEData)); p_SSEData += 4;
        mJtWr(7) -= (SUM4(p_SSEData)); p_SSEData += 4;

        // update err.
        err += (SUM4(p_SSEData));
    };



    void SE3AffineBrightTrackerSSE::warpPointPatchAndGeneratePointPatchBuffers(const Eigen::Matrix4d& T_ck_, const int& lvl) 
    {
        // initialize for edge
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

        Mat33 R_ck = T_ck.block<3, 3>(0, 0); // rotation from key to cur.
        Vec3 t_ck = T_ck.block<3, 1>(0, 3); // translation from key to cur.

        // with the prior \alpha and \beta, fill out brigtness-compensated residual and Jacobian terms for point patches.
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

            *(buf_Ic_ppat_warped + n_valid_pointpatch) = Ic_warp;
            *(buf_duc_ppat_warped + n_valid_pointpatch) = dx_warp;
            *(buf_dvc_ppat_warped + n_valid_pointpatch) = dy_warp;
            *(buf_Ik_ppat + n_valid_pointpatch) = Ik;

            *(buf_xppat_warped + n_valid_pointpatch) = X_warp(0);
            *(buf_yppat_warped + n_valid_pointpatch) = X_warp(1);
            *(buf_zppat_warped + n_valid_pointpatch) = X_warp(2);

            ++n_valid_pointpatch;
        }
    };

    void SE3AffineBrightTrackerSSE::calcPointPatchResidualAndWeightSSE(const float& huber_scaler_point, const float& alpha, const float& beta) {
        n_overthres_point = 0;

        float huber_thres_ppat = 5.0f*huber_scaler_point;
        
        // left first!
        __m128 Icwarp4, Ik4;
        __m128 aaaa = _mm_set1_ps(alpha);
        __m128 bbbb = _mm_set1_ps(beta);;

        // cout << "ppat n_step:" << n_steps << ", n_remainder:" << n_remainders << endl;

        int idx = 0;
        for (; idx < n_valid_pointpatch - 3; idx += 4) {
            Icwarp4 = _mm_load_ps(buf_Ic_ppat_warped + idx);
            Ik4     = _mm_load_ps(buf_Ik_ppat + idx);
            
            // calculate residual left edge

            __m128 rrrr = _mm_sub_ps(Icwarp4, _mm_add_ps(_mm_mul_ps(aaaa, Ik4), bbbb));
            _mm_store_ps(buf_res_ppat + idx, rrrr);

            // calculate weight left edge (Huber weight)
            float r = *(buf_res_ppat + idx);
            *(buf_weight_ppat + idx) = fabs(r) < huber_thres_ppat ? 1.0f : huber_thres_ppat / fabs(r);
            if (*(buf_weight_ppat + idx) < 1.0f) ++n_overthres_point;

            r = *(buf_res_ppat + idx + 1);
            *(buf_weight_ppat + idx + 1) = fabs(r) < huber_thres_ppat ? 1.0f : huber_thres_ppat / fabs(r);
            if (*(buf_weight_ppat + idx + 1) < 1.0f) ++n_overthres_point;

            r = *(buf_res_ppat + idx + 2);
            *(buf_weight_ppat + idx + 2) = fabs(r) < huber_thres_ppat ? 1.0f : huber_thres_ppat / fabs(r);
            if (*(buf_weight_ppat + idx + 2) < 1.0f) ++n_overthres_point;

            r = *(buf_res_ppat + idx + 3);
            *(buf_weight_ppat + idx + 3) = fabs(r) < huber_thres_ppat ? 1.0f : huber_thres_ppat / fabs(r);
            if (*(buf_weight_ppat + idx + 3) < 1.0f) ++n_overthres_point;
        };
        for (; idx < n_valid_pointpatch; ++idx) {
            float Icwarp = *(buf_Ic_ppat_warped + idx);
            float Ik = *(buf_Ik_ppat + idx);
          
            // calculate and push edge residual, Huber weight, Jacobian, and Hessian (따로하자)
            float r = Icwarp - alpha*Ik - beta;
            float w = fabs(r) < huber_thres_ppat ? 1.0f : huber_thres_ppat / fabs(r);

            *(buf_res_ppat + idx) = r;
            *(buf_weight_ppat + idx) = w;

            if (w < 1.0f) ++n_overthres_point;
        };

    };

    void SE3AffineBrightTrackerSSE::calcPointPatchHessianAndJacobianSSE(const int& lvl, const float& alpha, const float& beta) {

        JtWJ_ppat.setZero();
        mJtWr_ppat.setZero();
        err_ppat = 0;

        __m128 ones = _mm_set1_ps(1.0f);
        __m128 zeros = _mm_set1_ps(0.0f);
        __m128 minusones = _mm_set1_ps(-1.0f);

        __m128 J1, J2, J3, J4, J5, J6, J7, J8; // four data at once.        
        __m128 fx4 = _mm_set1_ps(K[lvl](0, 0));
        __m128 fy4 = _mm_set1_ps(K[lvl](1, 1));

        __m128 aaaa = _mm_set1_ps(alpha);
        __m128 bbbb = _mm_set1_ps(beta);
        
        J8 = minusones;

        int idx = 0;
        for (; idx < n_valid_pointpatch - 3; idx += 4) {
            __m128 x4 = _mm_load_ps(buf_xppat_warped + idx);
            __m128 y4 = _mm_load_ps(buf_yppat_warped + idx);
            //__m128 invz4 = _mm_rcp_ps(_mm_load_ps(buf_zppat_warped + idx));
            __m128 invz4 = _mm_div_ps(ones, _mm_load_ps(buf_zppat_warped + idx));

            __m128 xy4 = _mm_mul_ps(x4, y4);
            __m128 xinvz4 = _mm_mul_ps(x4, invz4);
            __m128 yinvz4 = _mm_mul_ps(y4, invz4);

            __m128 dxfx4 = _mm_mul_ps(_mm_load_ps(buf_duc_ppat_warped + idx), fx4);
            __m128 dyfy4 = _mm_mul_ps(_mm_load_ps(buf_dvc_ppat_warped + idx), fy4);

            __m128 Ik4 = _mm_load_ps(buf_Ik_ppat + idx);

            // J(1) = gx*fx/z
            J1 = _mm_mul_ps(dxfx4, invz4);
            // J(2) = gy*fy/z
            J2 = _mm_mul_ps(dyfy4, invz4);
            // J(3) = (-gx*fx*x/z - gy*fy*y/z)/z;
            J3 = _mm_mul_ps(_mm_sub_ps(zeros, _mm_add_ps(_mm_mul_ps(dxfx4, xinvz4), _mm_mul_ps(dyfy4, yinvz4))), invz4);
            // J(4) = -( gxfx*xizw*yizw + gyfy*(1.0f + yizw*yizw) );
            J4 = _mm_sub_ps(zeros,
                _mm_add_ps(_mm_mul_ps(_mm_mul_ps(dxfx4, xinvz4), yinvz4), _mm_mul_ps(dyfy4, _mm_add_ps(ones, _mm_mul_ps(yinvz4, yinvz4))))
            );
            // J(5) = gxfx*(1.0f + xizw*xizw) + gyfy*xizw*yizw;
            J5 = _mm_add_ps(_mm_mul_ps(dxfx4, _mm_add_ps(ones, _mm_mul_ps(xinvz4, xinvz4))), _mm_mul_ps(dyfy4, _mm_mul_ps(xinvz4, yinvz4)));
            // J(6) = gyfy*xizw - gxfx*yizw ;
            J6 = _mm_sub_ps(_mm_mul_ps(dyfy4, xinvz4), _mm_mul_ps(dxfx4, yinvz4));
            // J(7) = -a*Ik; 
            J7 = _mm_sub_ps(zeros, _mm_mul_ps(aaaa, Ik4));
            // J(8) = -1.0f;

            /*float val[4];
            memcpy(val, &aaaa, sizeof(val));

            cout << "aaaa: " << val[0] << "," << val[1] << "," << val[2] << "," << val[3] << endl;
            memcpy(val, &J1, sizeof(val));
            cout << "Jac 1 sse: " << val[0] << "," << val[1] << "," << val[2] << "," << val[3] << endl;
            memcpy(val, &J2, sizeof(val));
            cout << "Jac 2 sse: " << val[0] << "," << val[1] << "," << val[2] << "," << val[3] << endl;
            memcpy(val, &J3, sizeof(val));
            cout << "Jac 3 sse: " << val[0] << "," << val[1] << "," << val[2] << "," << val[3] << endl;
            memcpy(val, &J4, sizeof(val));
            cout << "Jac 4 sse: " << val[0] << "," << val[1] << "," << val[2] << "," << val[3] << endl;
            memcpy(val, &J5, sizeof(val));
            cout << "Jac 5 sse: " << val[0] << "," << val[1] << "," << val[2] << "," << val[3] << endl;
            memcpy(val, &J6, sizeof(val));
            cout << "Jac 6 sse: " << val[0] << "," << val[1] << "," << val[2] << "," << val[3] << endl;
            memcpy(val, &J7, sizeof(val));
            cout << "Jac 7 sse: " << val[0] << "," << val[1] << "," << val[2] << "," << val[3] << endl;
            memcpy(val, &J8, sizeof(val));
            cout << "Jac 8 sse: " << val[0] << "," << val[1] << "," << val[2] << "," << val[3] << endl;*/

            // update
            updateSSE(J1, J2, J3, J4, J5, J6, J7, J8, _mm_load_ps(buf_res_ppat + idx), _mm_load_ps(buf_weight_ppat + idx), JtWJ_ppat, mJtWr_ppat, err_ppat);
        };
        for (; idx < n_valid_pointpatch; ++idx) {
            float x = *(buf_xel_warped + idx);
            float y = *(buf_yel_warped + idx);
            float iz = 1.0f / (*(buf_zel_warped + idx));

            float xy = x*y;
            float xiz = x*iz;
            float yiz = y*iz;

            float dxfx = *(buf_duc_ppat_warped + idx)*K[lvl](0, 0);
            float dyfy = *(buf_dvc_ppat_warped + idx)*K[lvl](1, 1);

            Vec8 J;
            J(0) = dxfx*iz;
            J(1) = dyfy*iz;
            J(2) = (-dxfx*xiz - dyfy*yiz)*iz; // 여기가 틀렸었네 ; 
            J(3) = -dxfx*xiz*yiz - dyfy*(1.0f + yiz*yiz);
            J(4) = dxfx*(1.0f + xiz*xiz) + dyfy*xiz*yiz;
            J(5) = -dxfx*yiz + dyfy*xiz;
            J(6) = -alpha*(*(buf_Ik_ppat + idx));
            J(7) = -1.0f;

            float r = *(buf_res_ppat + idx);
            float w = *(buf_weight_ppat + idx);
            JtWJ_ppat.noalias() += (J*J.transpose())*w;
            mJtWr_ppat.noalias() -= J*(r*w);
            err_ppat += r*r*w;
        };
    };

    void SE3AffineBrightTrackerSSE::solveLevenbergMarquardtStepSSE(Vec8& delta, const float& lambda) {
        JtWJ.setZero();
        mJtWr.setZero();
        err = 0;

        JtWJ_e  /= (float)(2*n_valid_edge);
        mJtWr_e /= (float)(2*n_valid_edge);
        err_e   /= (float)(2*n_valid_edge);

        JtWJ_ppat  /= (float)n_valid_pointpatch;
        mJtWr_ppat /= (float)n_valid_pointpatch;
        err_ppat   /= (float)n_valid_pointpatch;

        JtWJ.noalias() += (JtWJ_e + JtWJ_ppat);
        mJtWr.noalias() += (mJtWr_e + mJtWr_ppat);

        err_e = sqrt(err_e);
        err_ppat = sqrt(err_ppat);
        err += (err_e + err_ppat);

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




    SE3AffineBrightTrackerSSE::SE3AffineBrightTrackerSSE(
        Params* params_,
        const vector<int>& nrows_pyr, const vector<int>& ncols_pyr, const vector<Eigen::Matrix3d>& K_pyr_, const Eigen::Matrix4d& T_nlnr_) 
    {
        // Dynamic allocations for buffers. (Sufficiently!!)

        params = params_;
        MAX_PYR_LVL = params->pyr.max_lvl;

        for (int lvl = 0; lvl < MAX_PYR_LVL; lvl++) {
            K[lvl] << K_pyr_[lvl](0, 0), K_pyr_[lvl](0, 1), K_pyr_[lvl](0, 2),
                K_pyr_[lvl](1, 0), K_pyr_[lvl](1, 1), K_pyr_[lvl](1, 2),
                K_pyr_[lvl](2, 0), K_pyr_[lvl](2, 1), K_pyr_[lvl](2, 2);
            Kinv[lvl] = K[lvl].inverse();
        }

        T_nlnr << T_nlnr_(0, 0), T_nlnr_(0, 1), T_nlnr_(0, 2), T_nlnr_(0, 3),
            T_nlnr_(1, 0), T_nlnr_(1, 1), T_nlnr_(1, 2), T_nlnr_(1, 3),
            T_nlnr_(2, 0), T_nlnr_(2, 1), T_nlnr_(2, 2), T_nlnr_(2, 3),
            0.0f, 0.0f, 0.0f, 1.0f;

        int w = ncols_pyr[0];
        int h = nrows_pyr[0];

        buf_xel_warped     = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_yel_warped     = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_zel_warped     = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_xer_warped     = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_yer_warped     = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_zer_warped     = (float*)custom_aligned_malloc(sizeof(float)*w*h);
                           
        buf_uel_warped     = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_vel_warped     = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_uer_warped     = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_ver_warped     = (float*)custom_aligned_malloc(sizeof(float)*w*h);
                           
        buf_uel_matched    = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_vel_matched    = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_uer_matched    = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_ver_matched    = (float*)custom_aligned_malloc(sizeof(float)*w*h);
                           
        buf_gxl_matched    = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_gyl_matched    = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_gxr_matched    = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_gyr_matched    = (float*)custom_aligned_malloc(sizeof(float)*w*h);
                           
        buf_res_l_e        = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_weight_l_e     = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        
        buf_res_r_e        = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_weight_r_e     = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_xppat_warped   = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_yppat_warped   = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_zppat_warped   = (float*)custom_aligned_malloc(sizeof(float)*w*h);
                           
        buf_Ik_ppat        = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_Ic_ppat_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_duc_ppat_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_dvc_ppat_warped = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        buf_res_ppat       = (float*)custom_aligned_malloc(sizeof(float)*w*h);
        buf_weight_ppat    = (float*)custom_aligned_malloc(sizeof(float)*w*h);

        SSEData = (float*)custom_aligned_malloc(sizeof(float)*4*45);

        // initialize pointers
        detachRef();
        detachTree();
        detachFrameCur();
    };


    SE3AffineBrightTrackerSSE::~SE3AffineBrightTrackerSSE() {
        custom_aligned_free((void*)buf_xel_warped);
        custom_aligned_free((void*)buf_yel_warped);
        custom_aligned_free((void*)buf_zel_warped);
        custom_aligned_free((void*)buf_xer_warped);
        custom_aligned_free((void*)buf_yer_warped);
        custom_aligned_free((void*)buf_zer_warped);

        custom_aligned_free((void*)buf_uel_warped);
        custom_aligned_free((void*)buf_vel_warped);
        custom_aligned_free((void*)buf_uer_warped);
        custom_aligned_free((void*)buf_ver_warped);

        custom_aligned_free((void*)buf_uel_matched);
        custom_aligned_free((void*)buf_vel_matched);
        custom_aligned_free((void*)buf_uer_matched);
        custom_aligned_free((void*)buf_ver_matched);

        custom_aligned_free((void*)buf_gxl_matched);
        custom_aligned_free((void*)buf_gyl_matched);
        custom_aligned_free((void*)buf_gxr_matched);
        custom_aligned_free((void*)buf_gyr_matched);

        custom_aligned_free((void*)buf_res_l_e);
        custom_aligned_free((void*)buf_weight_l_e);
        custom_aligned_free((void*)buf_res_r_e);
        custom_aligned_free((void*)buf_weight_r_e);

        custom_aligned_free((void*)buf_xppat_warped);
        custom_aligned_free((void*)buf_yppat_warped);
        custom_aligned_free((void*)buf_zppat_warped);

        custom_aligned_free((void*)buf_Ik_ppat);
        custom_aligned_free((void*)buf_Ic_ppat_warped);
        custom_aligned_free((void*)buf_duc_ppat_warped);
        custom_aligned_free((void*)buf_dvc_ppat_warped);

        custom_aligned_free((void*)buf_res_ppat);
        custom_aligned_free((void*)buf_weight_ppat);

        cout << "    SE3AffineBrightTrackerSSE tracker is deleted.\n";
    };

};



#endif